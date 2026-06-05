from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from sglang.srt.distributed import get_moe_expert_parallel_world_size
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_is_extend_in_batch,
)
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.base import CombineInput
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


# Recv-tightening coefficient for the CUDA-graph-captured mori decode trunc cap.
# The captured path cannot host-sync to read the device recv, so it caps from the
# host global token count (== origin*ws under MAX_LEN padding). A coefficient < 1
# tightens that cap toward the actual device recv to recover the TPOT the loose
# full-global cap leaves on the table (e2e 2026-06-04T16:52Z: recv 120-152 vs cap
# 256 at conc-64 on the MAIN model -> ~26% regression). The InferenceMINI CI
# reference used ~0.7*global (floor(conc*0.7)*(MTP+1), ROOT_CAUSE_CLUES s4).
#
# DEFAULT IS 1.0 (NO tightening) because 0.7 is NOT recv-safe in general: the
# EAGLE/NextN DRAFT model runs tiny batches where 0.7*global < the device recv
# (e2e 2026-06-04T17:40Z, decode_teamA_t26d telemetry: draft global_tokens=40,
# recv=32, 0.7-cap=28 -> margin -4 OOB), so the captured 0.7 slice drops real
# draft rows and crashes the mori COMBINE quant path ("Fp8BlockwiseQuant only
# supports bf16, got fp8_ocp") during the draft worker's cuda-graph capture.
# Under capture we cannot host-sync to clamp the cap up to recv, so there is no
# safe way to tighten below the full global count for ALL batch sizes. Keep the
# correct, fault-free full-global cap by default; the coefficient stays env-
# overridable for experiments, gated by SGLANG_MORI_RECV_CAP_DEBUG telemetry
# that flags any forward where the would-be cap drops below the measured recv.
_MORI_RECV_CAP_COEFF = float(os.environ.get("SGLANG_MORI_RECV_CAP_COEFF", "1.0"))

# Telemetry for the captured-path heuristic above. The captured cap
# (_MORI_RECV_CAP_COEFF * global tokens) is recv-safe ONLY if the actual device
# recv stays below it on every replay; under capture we cannot read the device
# recv to assert that, so an OOB would be silent until it faults. Enable this
# flag to have the EAGER forwards (which DO read the exact device recv) print
# recv vs the would-be captured cap, so the 0.7 margin can be confirmed under
# real traffic BEFORE relying on it under capture. Off by default (the .item()
# is already paid on the eager path, so the only added cost is the print).
_MORI_RECV_CAP_DEBUG = os.environ.get("SGLANG_MORI_RECV_CAP_DEBUG", "0") == "1"


class AiterQuantType(str, Enum):
    NONE = "No"
    PER_TOKEN = "per_Token"
    PER_128X128 = "per_128x128"
    PER_1X32 = "per_1x32"


@dataclass
class AiterMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    quant_type: AiterQuantType = AiterQuantType.NONE
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    expert_mask: Optional[torch.Tensor] = None
    doweight_stage1: bool = False
    hidden_pad: int = 0
    intermediate_pad: int = 0
    swiglu_limit: float = 0.0


@dataclass
class AiterRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor  # int32
    topk_weights: torch.Tensor  # float32
    # Effective activation quant_type (may differ from quant_info.quant_type
    # after the dispatch-aware decision in mori pre_permute).
    quant_type: AiterQuantType
    # Per-token activation scale produced by an EP dispatcher (mori). Falls
    # back to quant_info.a13_scale when None.
    a1_scale: Optional[torch.Tensor] = None
    # Mori-only fused_moe kwargs.
    num_local_tokens: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None
    # Host-side avg #tokens per local expert under uniform routing. Forwarded
    # to aiter.fused_moe so its get_padded_M tier lookup picks a kernel sized
    # for the realistic workload instead of the (potentially padded) M from
    # topk_ids.shape[0]. None for non-mori callers preserves prior behavior.
    expected_m: Optional[int] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


@dataclass
class AiterRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


_AITER_ACTIVATIONS = {"silu": "Silu", "swiglu": "Swiglu"}


def _aiter_activation(activation: str):
    from aiter import ActivationType

    return getattr(ActivationType, _AITER_ACTIVATIONS.get(activation, "Gelu"))


def _aiter_quant_type(quant_type: AiterQuantType):
    from aiter import QuantType

    return getattr(QuantType, quant_type.value)


class AiterRunnerCore(MoeRunnerCore):
    def run(
        self,
        runner_input: AiterRunnerInput,
        quant_info: AiterMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> AiterRunnerOutput:
        assert not self.config.no_combine, "no_combine=True is not supported by AITER"

        if runner_input.hidden_states.shape[0] == 0:
            return AiterRunnerOutput(hidden_states=runner_input.hidden_states)

        from aiter.fused_moe import fused_moe
        from aiter.ops.flydsl.moe_common import GateMode

        a1_scale = (
            runner_input.a1_scale
            if runner_input.a1_scale is not None
            else quant_info.a13_scale
        )

        extra: dict = {}
        if runner_input.num_local_tokens is not None:
            extra["num_local_tokens"] = runner_input.num_local_tokens
        if runner_input.output_dtype is not None:
            extra["dtype"] = runner_input.output_dtype
        if quant_info.swiglu_limit > 0:
            extra["gate_mode"] = GateMode.INTERLEAVE.value
            extra["swiglu_limit"] = quant_info.swiglu_limit
        if runner_input.expected_m is not None:
            extra["expected_m"] = runner_input.expected_m

        output = fused_moe(
            hidden_states=runner_input.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_weight=runner_input.topk_weights,
            topk_ids=runner_input.topk_ids,
            quant_type=_aiter_quant_type(runner_input.quant_type),
            activation=_aiter_activation(self.config.activation),
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            a1_scale=a1_scale,
            a2_scale=quant_info.a2_scale,
            bias1=quant_info.b13,
            bias2=quant_info.b2,
            expert_mask=quant_info.expert_mask,
            doweight_stage1=quant_info.doweight_stage1,
            hidden_pad=quant_info.hidden_pad,
            intermediate_pad=quant_info.intermediate_pad,
            **extra,
        )
        return AiterRunnerOutput(hidden_states=output)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


# ---------------------------------------------------------------------------
# Pre-permute: dispatch_output -> AiterRunnerInput
# ---------------------------------------------------------------------------


@register_pre_permute("standard", "aiter")
def pre_permute_standard_to_aiter(
    dispatch_output: StandardDispatchOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AiterRunnerInput:
    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(torch.float32)

    if runner_config.apply_router_weight_on_input and not quant_info.doweight_stage1:
        # Pre-scale at the Python level for kernels that don't honor doweight_stage1.
        assert (
            topk_weights.dim() == 2 and topk_weights.shape[-1] == 1
        ), "apply_router_weight_on_input requires topk=1"
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        topk_weights = torch.ones_like(topk_weights)

    return AiterRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights,
        quant_type=quant_info.quant_type,
    )


def _is_mori_dispatch_output(dispatch_output: Any) -> bool:
    # MoriEP{Normal,LL}DispatchOutput carry the post-mori-permute origin_topk_*
    # tensors that the standard DeepEP outputs lack.
    return hasattr(dispatch_output, "origin_topk_ids")


def _resolve_mori_quant_type(
    dispatch_a1_dtype: torch.dtype,
    dispatch_scale: Optional[torch.Tensor],
    weight_quant: AiterQuantType,
) -> AiterQuantType:
    """Pick the activation quant_type for AITER when the dispatch path may have
    pre-quantized hidden_states. Mirrors the original MoriEPMoE.run_moe_core
    decision tree."""
    is_fp8_quant = weight_quant in (
        AiterQuantType.PER_128X128,
        AiterQuantType.PER_TOKEN,
    )
    is_w4a4 = weight_quant == AiterQuantType.PER_1X32
    is_fp4_dispatch = dispatch_a1_dtype == torch.float4_e2m1fn_x2
    has_dispatch_scale = dispatch_scale is not None

    if is_w4a4:
        # W4A4 weights always run as per_1x32; FP8 dispatch is upscaled to BF16
        # before this point so dispatch_scale won't conflict.
        return AiterQuantType.PER_1X32
    if is_fp8_quant:
        return weight_quant
    # BF16 weights: lift to the dispatch-side quant type when scales are provided.
    if has_dispatch_scale and is_fp4_dispatch:
        return AiterQuantType.PER_1X32
    if has_dispatch_scale and not is_fp4_dispatch:
        return AiterQuantType.PER_128X128
    return AiterQuantType.NONE


def _pre_permute_deepep_to_aiter(
    dispatch_output: Union[
        DeepEPNormalDispatchOutput,
        DeepEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
        MoriEPLLDispatchOutput,
    ],
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AiterRunnerInput:
    is_mori = _is_mori_dispatch_output(dispatch_output)

    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids.to(torch.int32)
    topk_weights = dispatch_output.topk_weights.to(torch.float32)
    a1_scale: Optional[torch.Tensor] = None
    num_local_tokens: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None
    # Carried from mori dispatch outputs into AiterRunnerInput so the AITER
    # fused_moe call can use it for kernel-config tier lookup. Replaces the
    # SGLANG_MORI_MOE_MAX_INPUT_TOKENS truncation workaround (PR #22952) --
    # see aiter commit "[fused_moe] add expected_m host-side scheduling hint".
    expected_m: Optional[int] = None
    quant_type = quant_info.quant_type

    if is_mori:
        from sglang.srt.layers.moe.rocm_moe_utils import upscale, upscale_mxfp4

        a1_scale = dispatch_output.hidden_states_scale
        num_local_tokens = dispatch_output.num_recv_tokens_per_expert
        output_dtype = dispatch_output.out_dtype
        expected_m = dispatch_output.expected_m

        # Upscale dispatched activations when there is no AITER kernel for the
        # weight/activation dtype pair.
        weight_quant = quant_info.quant_type
        is_fp8_quant = weight_quant in (
            AiterQuantType.PER_128X128,
            AiterQuantType.PER_TOKEN,
        )
        is_w4a4 = weight_quant == AiterQuantType.PER_1X32
        is_fp4_dispatch = hidden_states.dtype == torch.float4_e2m1fn_x2

        if is_w4a4 and a1_scale is not None and not is_fp4_dispatch:
            # W4A4 weights with FP8 dispatch: dequant FP8->BF16 first; the
            # FP4 per_1x32 path needs BF16 input.
            hidden_states = upscale(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None
        elif is_fp8_quant and is_fp4_dispatch and a1_scale is not None:
            # FP8 weights + FP4 dispatch: no kernel for the fp4x2/fp8 pair;
            # dequant FP4->BF16 and let fused_moe re-quantize to FP8.
            hidden_states = upscale_mxfp4(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None

        quant_type = _resolve_mori_quant_type(
            hidden_states.dtype, a1_scale, weight_quant
        )

        running_state["aiter_combine_topk_ids"] = dispatch_output.origin_topk_ids
        running_state["aiter_combine_topk_weights"] = (
            dispatch_output.origin_topk_weights
        )

        # Truncate the padded mori dispatch tensors back to this decode forward's
        # recv upper bound so fused_moe permute/sort/GEMM/combine scale with the
        # live concurrency instead of the worst-case padded buffer
        # (max_dispatch_tokens_per_rank * ep_size). This lets ONE decode server
        # serve many concurrencies without a per-concurrency env knob. mori
        # dispatch is tail-padded (real tokens in [0, totalRecvTokenNum)), so
        # dropping the tail needs no pad-back; expected_m is orthogonal
        # (kernel-tier lookup). Decode only: prefill's per-rank buffer is the
        # fixed 8192 (not concurrency-derived) and its capture batches are tiny,
        # so truncating there would drop real tokens.
        #
        # The ONLY correctness requirement is cap >= the true recv on this rank,
        # because aiter.fused_moe (moe_sorting / grouped-GEMM / combine) indexes
        # hidden_states rows in [0, num_recv_tokens_per_expert.sum()). That recv
        # is a DEVICE ROUTE COUNT routed in from ALL EP ranks; any cap below it
        # makes fused_moe read rows [cap, recv) that hidden_states[:cap] dropped
        # -> GPU Memory access fault. Everything beyond that bound is perf only.
        #
        # A host-side TOKEN count cannot bound the device ROUTE count here:
        #   * The local origin*ws only equals the global recv when every rank has
        #     the same batch. Under DP-attention SUM_LEN padding (eager warmup/
        #     decode) the per-rank batches are heterogeneous, so a small-origin
        #     rank's origin*ws underflows the global recv -> fault.
        #   * sum(get_dp_global_num_tokens()) (the prior global-token fix) fixed
        #     the heterogeneous-batch case but STILL faulted on-device
        #     (2026-06-04T16:17Z e2e): for this deployment topk == ws == 8, so
        #     the global-token cap equals the *uniform-routing* recv exactly
        #     (margin 0). A rank holding hot experts receives > 1/ws of the
        #     global_total*topk routes, so its recv exceeds the global-token cap.
        #   * During PD-disaggregation warmup the DP global-token metadata is not
        #     populated, so a host-token cap is absent/wrong there too.
        #
        # Therefore size the cap from the ACTUAL recv (num_recv_tokens_per_expert
        # .sum()) -- exact, always OOB-safe, and still shrinks with live
        # concurrency. Reading it needs a device->host sync, which is illegal
        # under CUDA-graph capture; but capture uses MAX_LEN padding (uniform per
        # rank) so the host global-token count there equals recv and bakes a
        # fixed slice byte-identical to the original origin*ws -> the captured
        # decode hot path (the benchmarked path) keeps its perf and never syncs.
        # The eager .item() runs only on warmup / non-graph forwards.
        if not get_is_extend_in_batch():
            ws = get_moe_expert_parallel_world_size()
            buffer_rows = hidden_states.shape[0]
            if torch.cuda.is_current_stream_capturing():
                # Capture: a device->host sync (.item()) is illegal, so the cap
                # must be host-derived. DP attention pads MAX_LEN here so the
                # per-forward global token count is uniform and host-readable.
                global_num_tokens = get_dp_global_num_tokens()
                if global_num_tokens is not None and len(global_num_tokens) == ws:
                    # Recv-tight captured cap: the full global token count is
                    # recv-SAFE but ~1.4-1.7x the actual device recv (e2e
                    # 2026-06-04T16:52Z teamB t5: recv 120-152 vs cap 256 at
                    # conc-64), so slicing to it leaves fused_moe processing
                    # excess rows and regresses TPOT. The InferenceMINI CI
                    # reference used a recv-tight static cap ~0.7*global
                    # (floor(conc*0.7)*(MTP+1), ROOT_CAUSE_CLUES s4) that hit the
                    # 7.4/8.7 TPOT targets. RESIDUAL RISK: under capture we cannot
                    # host-sync to assert cap >= recv, so a replay with
                    # recv > 0.7*global would OOB silently; 0.7 is CI-validated
                    # with measured headroom (152 vs 179) but is not a proven
                    # bound. SGLANG_MORI_RECV_CAP_DEBUG=1 prints the eager recv vs
                    # this cap to confirm the margin under real traffic.
                    cap = int(sum(global_num_tokens) * _MORI_RECV_CAP_COEFF)
                else:
                    # Global metadata absent under capture (the NextN/EAGLE DRAFT
                    # worker's cuda-graph capture, or PD-disagg warmup): do NOT
                    # fall back to the local origin*ws bound. origin*ws under-caps
                    # the global recv under heterogeneous per-rank batches, and
                    # truncating the dispatch buffer to that under-cap corrupts
                    # the mori COMBINE quant path -> the "Fp8BlockwiseQuant only
                    # supports bf16, got fp8_ocp" crash on the draft worker
                    # capture (decode_teamA_t26 2026-06-04T17:19Z; Review 7 item
                    # (a) for the captured path). The full padded buffer is always
                    # recv-safe, so skip truncation here.
                    cap = buffer_rows
            elif num_local_tokens is not None:
                # Eager (warmup / non-graph decode): the .item() sync is legal,
                # so use the EXACT device recv -> OOB-safe regardless of routing
                # imbalance or missing global metadata (fixes PD-warmup too).
                cap = int(num_local_tokens.sum().item())
                if _MORI_RECV_CAP_DEBUG:
                    # Confirm the captured-path 0.7 heuristic is recv-safe under
                    # real traffic: compare this exact device recv against the cap
                    # the captured path WOULD bake for the same global token count.
                    global_num_tokens = get_dp_global_num_tokens()
                    if global_num_tokens is not None and len(global_num_tokens) == ws:
                        gt = sum(global_num_tokens)
                    else:
                        gt = dispatch_output.origin_topk_ids.shape[0] * ws
                    would_be_cap = int(gt * _MORI_RECV_CAP_COEFF)
                    print(
                        f"[MORI_RECV_CAP] recv={cap} global_tokens={gt} "
                        f"coeff={_MORI_RECV_CAP_COEFF} would_be_captured_cap="
                        f"{would_be_cap} margin={would_be_cap - cap} "
                        f"{'OOB!' if would_be_cap < cap else 'safe'}",
                        flush=True,
                    )
            else:
                # No recv count available: do not truncate (full buffer is always
                # recv-safe; truncation is only a perf optimization).
                cap = buffer_rows
            cap = min(cap, buffer_rows)
            # mori dispatch tail-pads all four tensors to the worst-case buffer
            # (real entries in [0, totalRecvTokenNum)); slice them consistently to
            # the cap so fused_moe + combine scale with live concurrency. The cap
            # is recv-safe by construction above, so [0, cap) keeps every real row.
            hidden_states = hidden_states[:cap]
            if a1_scale is not None:
                a1_scale = a1_scale[:cap]
            topk_ids = topk_ids[:cap]
            topk_weights = topk_weights[:cap]
    else:
        # DeepEP marks invalid topk slots with idx == -1; AITER cannot accept
        # negative ids, so reroute them to the sink slot at index
        # num_local_experts (masked off by quant_info.expert_mask which has
        # shape (num_local_experts + 1,)).
        topk_ids = torch.where(
            topk_ids == -1,
            torch.full_like(topk_ids, runner_config.num_local_experts),
            topk_ids,
        )
        running_state["aiter_combine_topk_ids"] = dispatch_output.topk_ids
        running_state["aiter_combine_topk_weights"] = dispatch_output.topk_weights

    running_state["aiter_combine_is_mori"] = is_mori

    return AiterRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        quant_type=quant_type,
        a1_scale=a1_scale,
        num_local_tokens=num_local_tokens,
        output_dtype=output_dtype,
        expected_m=expected_m,
    )


register_pre_permute("deepep_normal", "aiter")(_pre_permute_deepep_to_aiter)
register_pre_permute("deepep_ll", "aiter")(_pre_permute_deepep_to_aiter)


# ---------------------------------------------------------------------------
# Post-permute: AiterRunnerOutput -> CombineInput
# ---------------------------------------------------------------------------


@register_post_permute("aiter", "standard")
def post_permute_aiter_to_standard(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)


def _post_permute_aiter_to_deepep(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
    is_normal: bool,
) -> CombineInput:
    if running_state.get("aiter_combine_is_mori"):
        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPLLCombineInput,
            MoriEPNormalCombineInput,
        )

        cls = MoriEPNormalCombineInput if is_normal else MoriEPLLCombineInput
    else:
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPLLCombineInput,
            DeepEPNormalCombineInput,
        )

        cls = DeepEPNormalCombineInput if is_normal else DeepEPLLCombineInput

    return cls(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["aiter_combine_topk_ids"],
        topk_weights=running_state["aiter_combine_topk_weights"],
    )


@register_post_permute("aiter", "deepep_normal")
def post_permute_aiter_to_deepep_normal(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> CombineInput:
    return _post_permute_aiter_to_deepep(
        runner_output, quant_info, runner_config, running_state, is_normal=True
    )


@register_post_permute("aiter", "deepep_ll")
def post_permute_aiter_to_deepep_ll(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> CombineInput:
    return _post_permute_aiter_to_deepep(
        runner_output, quant_info, runner_config, running_state, is_normal=False
    )
