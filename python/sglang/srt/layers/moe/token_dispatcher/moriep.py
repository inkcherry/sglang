from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple

from sglang.srt.eplb.expert_distribution import (
    _ExpertDistributionRecorderNoop,
    get_global_expert_distribution_recorder,
)
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPPDispatchHooks
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    is_tbo_enabled,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    get_bool_env_var,
    get_int_env_var,
    is_hip,
)

if TYPE_CHECKING:
    from sglang.srt.single_batch_overlap import CombineOverlapArgs
    import mori

from enum import Enum, auto
from functools import lru_cache

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

# Blockwise quantization group sizes: number of elements sharing one scale factor
FP8_BLOCK_SIZE = 128
MXFP4_BLOCK_SIZE = 32

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import QuantType, get_hip_quant

logger = logging.getLogger(__name__)


def _should_record_expert_distribution() -> bool:
    recorder = get_global_expert_distribution_recorder()
    if recorder.recording:
        return True
    # While capturing, only bake in the count kernel if a recorder is actually
    # configured (non-Noop); otherwise it would replay as dead work every decode
    # step. Configured recorders still bake it in, so start_record() works after
    # capture.
    if torch.get_device_module().is_current_stream_capturing():
        return not isinstance(recorder, _ExpertDistributionRecorderNoop)
    return False


class MoriEPPDispatchHooks(DeepEPPDispatchHooks):

    def __call__(self, dispatcher: BaseDispatcher):
        for hook_fun in self.hook_dict.values():
            hook_fun(dispatcher)


class MoriEPNormalDispatchOutput(NamedTuple):
    """Mori EP normal dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]
    origin_topk_ids: torch.Tensor
    origin_topk_weights: torch.Tensor
    out_dtype: torch.dtype

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


class MoriEPLLDispatchOutput(NamedTuple):
    """Mori EP low latency dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]
    origin_topk_ids: torch.Tensor
    origin_topk_weights: torch.Tensor
    out_dtype: torch.dtype

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_LL


assert isinstance(MoriEPNormalDispatchOutput, DispatchOutput)
assert isinstance(MoriEPLLDispatchOutput, DispatchOutput)


class MoriEPNormalCombineInput(NamedTuple):
    """Mori EP combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_NORMAL


class MoriEPLLCombineInput(NamedTuple):
    """Mori EP combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_LL


assert isinstance(MoriEPNormalCombineInput, CombineInput)
assert isinstance(MoriEPLLCombineInput, CombineInput)


class EpMode(Enum):
    INTRA_NODE = "intra_node"
    INTER_NODE = "inter_node"
    LOW_LATENCY = "low_latency"


class DispatchDtype(Enum):
    bf16 = "bfloat16"
    fp8 = "float8_blockwise"
    fp4 = "mxfp4_blockwise"


class CombineDtype(Enum):
    bf16 = "bfloat16"
    fp8 = "float8_blockwise"
    fp8_direct_cast = "float8_direct_cast"
    fp4 = "fp4_blockwise"  # packed E2M1, blockwise-scaled; ~half the combine transport of fp8


@dataclass(frozen=True)
class EpDispatchConfig:
    kernel_type: mori.ops.EpDispatchCombineKernelType
    warp_num_per_block: int
    block_num: int
    rdma_block_num: int


def get_ep_dispatch_configs(num_max_dispatch_tokens_per_rank: int = 4096):
    import mori

    # Selects the inter-node kernel. `InterNodeV1LL` is used if `num_max_dispatch_tokens_per_rank`
    # is less than or equal to the threshold, otherwise `InterNodeV1` is used. The threshold defaults to 256.
    inter_kernel_switch_threshold = get_int_env_var(
        "SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD", 256
    )

    inter_kernel_type = (
        mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
        if num_max_dispatch_tokens_per_rank <= inter_kernel_switch_threshold
        else mori.ops.EpDispatchCombineKernelType.InterNodeV1
    )

    return {
        # TODO(billishyahao): need to tune different configs for intra node async
        # Also could be tuned for different AMD platform
        EpMode.INTRA_NODE: EpDispatchConfig(
            kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
            warp_num_per_block=16,
            block_num=80,
            rdma_block_num=0,
        ),
        EpMode.INTER_NODE: EpDispatchConfig(
            kernel_type=inter_kernel_type,
            warp_num_per_block=8,
            block_num=64,
            rdma_block_num=32,
        ),
        EpMode.LOW_LATENCY: EpDispatchConfig(
            kernel_type=mori.ops.EpDispatchCombineKernelType.AsyncLL,
            warp_num_per_block=8,
            block_num=64,
            rdma_block_num=32,
        ),
    }


# init_mori_op only needs do once in model initial stage
# use lru_cache to reuse the same mori_op instance to avoid the init overhead for mori
@lru_cache(maxsize=4)
def init_mori_op(
    group,
    router_topk,
    num_experts,
    num_local_experts,
    hidden_size,
    params_dtype,
    num_max_dispatch_tokens_per_rank,
    deepep_mode,
    instance_id=0,
    dispatch_dtype=DispatchDtype.bf16,
    combine_dtype=CombineDtype.bf16,
    enable_sdma=False,
    use_external_inp_buf=True,
):

    import mori

    world_size = get_parallel().moe_ep_size
    rank = get_parallel().moe_ep_rank

    gpu_per_node = 8 if world_size >= 8 else world_size

    group_name = f"mori"
    cpu_group = group.cpu_group
    try:
        torch._C._distributed_c10d._register_process_group(group_name, cpu_group)
    except Exception as e:
        if "already registered" in str(e):
            logger.info(
                f"[MORI init] The same process group is already "
                f"registered. Ignoring [{str(e)}]"
            )
        else:
            raise
    else:
        # If new group is newly registered then need to init mori shmem. However
        # if the group is registered already then need to skip init mori shmem
        # and reuse the previous one.
        mori.shmem.shmem_torch_process_group_init(group_name)

    mode = EpMode.INTRA_NODE if world_size <= 8 else EpMode.INTER_NODE
    async_mode = deepep_mode.enable_low_latency() or enable_sdma
    if async_mode:
        mode = EpMode.LOW_LATENCY

    cfg = get_ep_dispatch_configs(num_max_dispatch_tokens_per_rank)[mode]

    kernel_type = cfg.kernel_type
    warp_num_per_block = cfg.warp_num_per_block
    block_num = cfg.block_num
    rdma_block_num = cfg.rdma_block_num

    hidden_dim = hidden_size
    scale_dim = 1
    data_type = fp8_dtype
    scale_type_size = torch.float32.itemsize

    if dispatch_dtype == DispatchDtype.bf16:
        data_type = params_dtype
        scale_dim = 0
    elif dispatch_dtype == DispatchDtype.fp8:
        scale_dim = hidden_size // FP8_BLOCK_SIZE
    elif dispatch_dtype == DispatchDtype.fp4:
        # FP4 kernel still takes the original hidden size and do quantization
        # internally, so hidden_dim is not reduced. The reason is that for FP4
        # quantization, we need to keep the original hidden size to calculate
        # the quantization scale correctly. Don't use packed hidden size for FP4 kernel.
        hidden_dim = hidden_size
        scale_dim = hidden_size // MXFP4_BLOCK_SIZE
        data_type = torch.float4_e2m1fn_x2
        scale_type_size = torch.float8_e8m0fnu.itemsize

        if mode == EpMode.INTRA_NODE:
            if num_max_dispatch_tokens_per_rank < 128:
                block_num = 225
                warp_num_per_block = 5
            else:
                block_num = 256
                warp_num_per_block = 16

    # Fp8 blockwise combine uses its own internal scale_dim driven which can be
    # overridden by env ``MORI_FP8_COMBINE_SCALE_DIM`` (default 56)
    # See https://github.com/ROCm/mori/blob/96ffa169710f214e76e07abe5008d686fe54522b/python/mori/ops/dispatch_combine.py#L81-L84
    combine_quant_type = "none"
    if combine_dtype == CombineDtype.fp8:
        combine_quant_type = "fp8_blockwise"
    elif combine_dtype == CombineDtype.fp8_direct_cast:
        combine_quant_type = "fp8_direct_cast"
    elif combine_dtype == CombineDtype.fp4:
        combine_quant_type = "fp4_blockwise"

    logger.info(
        f"[MORI init] {world_size=} {rank=} {hidden_size=} {params_dtype=} "
        f"{num_max_dispatch_tokens_per_rank=} {num_local_experts=} "
        f"{router_topk=} {mode=} {dispatch_dtype=} {combine_dtype=} "
        f"{use_external_inp_buf=} "
    )

    def check_mori_compatibility(kwargs: dict) -> None:
        """Remove kwargs not accepted by the installed mori's EpDispatchCombineConfig."""
        import dataclasses

        config_cls = mori.ops.EpDispatchCombineConfig
        valid_kwargs = {f.name for f in dataclasses.fields(config_cls)}

        invalid_kwargs = set(kwargs.keys()) - valid_kwargs
        for arg in invalid_kwargs:
            logger.warning(f"[MORI compat] Removing incompatible argument {arg} ")
            del kwargs[arg]

    # Definition refer to https://github.com/ROCm/mori/blob/f9be5ee2e5ac87256b9523399ae9d4d0e8a54f53/python/mori/ops/dispatch_combine.py#L66-L121
    common_kwargs = dict(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_token_type_size=params_dtype.itemsize,
        max_num_inp_token_per_rank=num_max_dispatch_tokens_per_rank,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=router_topk,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        max_total_recv_tokens=get_int_env_var(
            "SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS", 0
        ),
        use_external_inp_buf=use_external_inp_buf,
        kernel_type=kernel_type,
        gpu_per_node=gpu_per_node,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=2,  # Number of queue pairs per processing element
        quant_type=combine_quant_type,
    )

    check_mori_compatibility(common_kwargs)

    mori_config = mori.ops.EpDispatchCombineConfig(**common_kwargs)
    mori_op = mori.ops.EpDispatchCombineOp(mori_config)
    _LIVE_OPS.append(
        _LiveOp(
            op=mori_op,
            group=group,
            rank=rank,
            capacity=num_max_dispatch_tokens_per_rank,
        )
    )
    return mori_op


@dataclass
class _LiveOp:
    op: object
    group: object
    rank: int
    # Tracked here, not read back off the op: a failed resize may already have
    # freed the buffers, and reading them through pybind segfaults the rank.
    capacity: int


# `init_mori_op` is lru_cached, so ops are shared across layers: this holds one
# entry per distinct op (1-2 in practice), not one per MoE layer.
_LIVE_OPS: List[_LiveOp] = []


class MoriA2AResizeError(RuntimeError):
    """The EP group could not agree a new a2a capacity, or refused it."""


class MoriA2AGroupDead(MoriA2AResizeError):
    """At least one rank lost its buffers, so the OLD role cannot dispatch
    either. Unlike every other resize failure this is not recoverable."""


def _is_group_fatal(exc: BaseException) -> bool:
    """Whether mori reports the op as unrecoverable (its `ResizeFatal`).

    Read off the raised class rather than imported, so a mori build without the
    resize outcome types still resizes instead of failing at import.
    """
    return any(c.__name__ == "ResizeFatal" for c in type(exc).__mro__)


def _agree_capacity(target: Optional[int], group) -> Optional[int]:
    """Reduce every rank's proposed per-rank capacity to one group verdict.

    A rank that cannot derive a target votes 0 and the whole group then leaves
    the buffers alone. It must not return early: its peers would enter a
    collective it never joins and the group hangs.
    """
    cpu_group = group.cpu_group
    lo = torch.tensor([target or 0], dtype=torch.int64)
    hi = lo.clone()
    torch.distributed.all_reduce(lo, op=torch.distributed.ReduceOp.MIN, group=cpu_group)
    torch.distributed.all_reduce(hi, op=torch.distributed.ReduceOp.MAX, group=cpu_group)
    lo, hi = int(lo.item()), int(hi.item())
    if lo == 0:
        return None
    if lo != hi:
        raise MoriA2AResizeError(
            f"EP ranks disagree on the a2a capacity ({lo} != {hi}); not resizing"
        )
    return lo


def rebuild_mori_dispatch_buffers(
    target_tokens_per_rank: Optional[int], role: str
) -> Optional[Tuple[int, int]]:
    """Resize every live mori a2a dispatch buffer for `role`.

    Returns the (old, new) per-rank capacity, or None if the group voted to
    leave the buffers alone. Raises on refusal or failure; mori reduces the
    outcome over the group, so every rank raises the same type.
    """
    if not _LIVE_OPS:
        return None
    ceiling = get_int_env_var("SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 4096)
    agreed = _agree_capacity(target_tokens_per_rank, _LIVE_OPS[0].group)
    if agreed is None:
        logger.warning("a2a capacity not derivable on some EP rank; buffers unchanged")
        return None
    if agreed > ceiling:
        # This env var is a per-process allocation ceiling, not a role's size:
        # an instance launched below max(prefill, decode) can never flip out.
        raise MoriA2AResizeError(
            f"{role} needs an a2a capacity of {agreed} tokens/rank but this process "
            f"was launched with SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK={ceiling}; "
            "launch flip-capable instances at the larger of the two roles"
        )
    old = _LIVE_OPS[0].capacity
    for live in _LIVE_OPS:
        try:
            # Must be the shmem group: mori reduces the outcome on a CPU tensor
            # and the default PG is nccl, which raises after the free, before
            # the handles refresh -- leaving the op pointing at released memory.
            live.op.reconfigure(agreed, group=live.group.cpu_group)
        except Exception as e:
            if _is_group_fatal(e):
                raise MoriA2AGroupDead(e) from e
            raise
        live.capacity = agreed
    logger.info(
        "[pd-role-switch] a2a capacity %d -> %d rank=%d role=%s",
        old,
        agreed,
        _LIVE_OPS[0].rank,
        role[0],
    )
    return old, agreed


class CommStreamPool:
    _streams = {}  # key -> torch.cuda.Stream

    @classmethod
    def _make_key(cls, group):
        return (torch.cuda.current_device(), id(group))

    @classmethod
    def get_stream_from_pool(cls, group) -> torch.cuda.Stream:
        key = cls._make_key(group)
        stream = cls._streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(priority=0)
            cls._streams[key] = stream
        return stream

    @classmethod
    def clear_group(cls, group):
        key = (torch.cuda.current_device(), id(group))
        cls._streams.pop(key, None)


class _MoriEPDispatcherImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
        instance_id: int = 0,
    ):
        try:
            import mori  # noqa: F401
        except ImportError:
            raise ImportError("Mori EP is not installed. Please install.")
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode
        self.instance_id = instance_id

        self.num_max_dispatch_tokens_per_rank = get_int_env_var(
            "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 4096
        )

        self.enable_sdma = get_bool_env_var("MORI_ENABLE_SDMA", "false")
        self.use_external_inp_buf = True

        self._mori_op = None
        self.dispatch_dtype = DispatchDtype.bf16
        self.combine_dtype = CombineDtype.bf16

        self.quant_config: Optional[dict] = None

        self.overlap_args: Optional[CombineOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

    @property
    def mori_op(self):
        if self._mori_op is None:
            # If set_quant_config was never called, apply env var override now
            if self.quant_config is None:
                self._apply_dispatch_dtype_override()
            self._mori_op = init_mori_op(
                self.group,
                self.router_topk,
                self.num_experts,
                self.num_local_experts,
                self.hidden_size,
                self.params_dtype,
                self.num_max_dispatch_tokens_per_rank,
                self.deepep_mode,
                self.instance_id,
                self.dispatch_dtype,
                self.combine_dtype,
                self.enable_sdma,
                self.use_external_inp_buf,
            )
        return self._mori_op

    def _apply_dispatch_dtype_override(self):
        """Apply env var override to fp8_dispatch/fp4_dispatch/fp8_combine flags."""
        if "SGLANG_MORI_DISPATCH_DTYPE" in os.environ:
            dispatch_dtype = os.environ["SGLANG_MORI_DISPATCH_DTYPE"].lower()
            if dispatch_dtype != "auto":
                if dispatch_dtype == "bf16":
                    self.dispatch_dtype = DispatchDtype.bf16
                elif dispatch_dtype == "fp8":
                    self.dispatch_dtype = DispatchDtype.fp8
                elif dispatch_dtype == "fp4":
                    self.dispatch_dtype = DispatchDtype.fp4
        elif (
            "SGLANG_MORI_FP8_DISP" in os.environ or "SGLANG_MORI_FP4_DISP" in os.environ
        ):
            # Deprecated: will be removed in a future release
            logger.warning_once(
                "SGLANG_MORI_FP8_DISP and SGLANG_MORI_FP4_DISP are deprecated "
                "and will be removed in a future release. "
                "Use SGLANG_MORI_DISPATCH_DTYPE=auto|bf16|fp8|fp4 instead."
            )
            if get_bool_env_var("SGLANG_MORI_FP8_DISP", "False"):
                self.dispatch_dtype = DispatchDtype.fp8
            if get_bool_env_var("SGLANG_MORI_FP4_DISP", "False"):
                self.dispatch_dtype = DispatchDtype.fp4

        if "SGLANG_MORI_COMBINE_DTYPE" in os.environ:
            combine_dtype = os.environ["SGLANG_MORI_COMBINE_DTYPE"].lower()
            if combine_dtype != "auto":
                if combine_dtype == "fp8":
                    self.combine_dtype = CombineDtype.fp8
                elif combine_dtype == "bf16":
                    self.combine_dtype = CombineDtype.bf16
                elif combine_dtype == "fp8_direct_cast":
                    self.combine_dtype = CombineDtype.fp8_direct_cast
                elif combine_dtype == "fp4":
                    self.combine_dtype = CombineDtype.fp4
        elif "SGLANG_MORI_FP8_COMB" in os.environ:
            # Deprecated: will be removed in a future release
            logger.warning_once(
                "SGLANG_MORI_FP8_COMB is deprecated "
                "and will be removed in a future release. "
                "Use SGLANG_MORI_COMBINE_DTYPE=auto|bf16|fp8|fp4|fp8_direct_cast instead."
            )
            if get_bool_env_var("SGLANG_MORI_FP8_COMB", "False"):
                self.combine_dtype = CombineDtype.fp8

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        # Auto-detect dispatch quantization from weight dtype
        weight_dtype = quant_config.get("weight_dtype", None)
        if weight_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            self.dispatch_dtype = DispatchDtype.fp8
            self.combine_dtype = CombineDtype.bf16
        elif weight_dtype == torch.float4_e2m1fn_x2:
            self.dispatch_dtype = DispatchDtype.fp4
            self.combine_dtype = CombineDtype.fp8
        else:
            self.dispatch_dtype = DispatchDtype.bf16
            self.combine_dtype = CombineDtype.bf16
        # Apply env var override immediately so dispatch_a sees correct flags
        self._apply_dispatch_dtype_override()

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ) -> None:
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        self.overlap_args = None
        self.meta_overlap_args = None

    def _combine_kwargs(self, hidden_states: torch.Tensor) -> dict:
        return {}


class _MoriEPDispatcherImplNormal(_MoriEPDispatcherImplBase):
    def __init__(self, async_finish: bool, **kwargs):
        super().__init__(**kwargs)

        self.async_finish = async_finish
        self.quant_config = {}
        self.fp8_quant_func = get_hip_quant(QuantType.per_1x128)
        self.fp4_quant_func = get_hip_quant(QuantType.per_1x32)
        self.enable_dual_stream = is_tbo_enabled()
        self._comm_stream = None
        if self.enable_dual_stream:
            self._comm_stream = CommStreamPool.get_stream_from_pool(self.group)

    def _capture_event_if_async(self) -> Optional[torch.cuda.Event]:
        assert self.enable_dual_stream, "dual stream must be enabled"
        if not self.async_finish:
            return None
        ev = torch.cuda.Event(blocking=False, interprocess=False)
        ev.record(torch.cuda.current_stream())
        return ev

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

        num_token = hidden_states.shape[0]
        output_dtype = hidden_states.dtype
        scale = None

        if self.dispatch_dtype == DispatchDtype.fp8:
            # FP8 quant
            if num_token > 0:
                # NOTE: aiter is able to handle token=0 case in UT. But for some
                # reason it failed at e2e case. Root cause TBD.
                hidden_states, scale = self.fp8_quant_func(
                    hidden_states, quant_dtype=fp8_dtype
                )
            else:
                hidden_states = torch.empty(
                    hidden_states.shape, dtype=fp8_dtype, device=hidden_states.device
                )
                scale = torch.empty(
                    (0, self.hidden_size // FP8_BLOCK_SIZE),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )

        elif self.dispatch_dtype == DispatchDtype.fp4:
            # FP4 quant
            if num_token > 0:
                hidden_states, scale = self.fp4_quant_func(hidden_states, shuffle=False)
            else:
                hidden_states = torch.empty(
                    (0, self.hidden_size // 2),
                    dtype=torch.float4_e2m1fn_x2,
                    device=hidden_states.device,
                )
                scale = torch.empty(
                    (0, self.hidden_size // MXFP4_BLOCK_SIZE),
                    dtype=torch.float8_e8m0fnu,
                    device=hidden_states.device,
                )

        previous_event = self._capture_event_if_async() if self._comm_stream else None

        return (
            hidden_states,
            topk_weights,
            topk_ids,
            scale,
            output_dtype,
            previous_event,
        )

    def dispatch_b(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        scale,
        output_dtype,
        previous_event,
    ):

        (
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
            done_event,
        ) = self._dispatch_core(
            hidden_states,
            topk_weights,
            topk_ids,
            scale=scale,
            previous_event=previous_event,
        )

        if self._comm_stream and self.async_finish and done_event is not None:
            torch.cuda.current_stream().wait_event(done_event)

        return MoriEPNormalDispatchOutput(
            hidden_states=packed_recv_hidden,
            hidden_states_scale=recv_scales,
            topk_ids=recv_topk_ids,
            topk_weights=recv_topk_weights,
            num_recv_tokens_per_expert=packed_recv_count,
            origin_topk_ids=topk_ids,
            origin_topk_weights=topk_weights,
            out_dtype=output_dtype,
        )

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        previous_event: Optional[torch.cuda.Event] = None,
    ):
        done_event: Optional[torch.cuda.Event] = None

        record = _should_record_expert_distribution()

        if self._comm_stream:
            compute_stream = torch.cuda.current_stream()
            comm_stream = self._comm_stream  # comm stream

            for t in (hidden_states, topk_weights, topk_ids):
                t.record_stream(comm_stream)
            if scale is not None:
                scale.record_stream(comm_stream)

            with torch.cuda.stream(comm_stream):
                # if (previous_event) stream_wait(comm_stream, previous_event)
                # else stream_wait(comm_stream, compute_stream)

                if previous_event is not None:
                    comm_stream.wait_event(previous_event)
                else:
                    comm_stream.wait_stream(compute_stream)

                dispatch_fn = (
                    self.mori_op.dispatch_send
                    if self.enable_sdma
                    else self.mori_op.dispatch
                )
                (
                    packed_recv_hidden,
                    recv_topk_weights,
                    recv_scales,
                    recv_topk_ids,
                    packed_recv_count,
                ) = dispatch_fn(
                    hidden_states,
                    topk_weights,
                    scale,
                    topk_ids,
                    call_local_expert_count=record,
                )
                if self.enable_sdma:
                    self.mori_op.dispatch_recv()

                if self.async_finish:
                    done_event = torch.cuda.Event(blocking=False, interprocess=False)
                    done_event.record(comm_stream)
                else:
                    compute_stream.wait_stream(comm_stream)

            for t in (
                packed_recv_hidden,
                recv_topk_weights,
                recv_scales,
                recv_topk_ids,
            ):
                if t is not None:
                    t.record_stream(comm_stream)
        else:

            (
                packed_recv_hidden,
                recv_topk_weights,
                recv_scales,
                recv_topk_ids,
                packed_recv_count,
            ) = self.mori_op.dispatch(
                hidden_states,
                topk_weights,
                scale,
                topk_ids,
                call_local_expert_count=record,
            )

        # mori local_expert_count is a GPU tensor; route it through the
        # low_latency hook only when the recorder is actually active.
        if record:
            get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(
                self.mori_op.local_expert_count
            )

        return (
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
            done_event,
        )

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        previous_event = self._capture_event_if_async() if self._comm_stream else None
        return hidden_states, topk_ids, topk_weights, previous_event

    def combine_b(self, hidden_states, topk_ids, topk_weights, previous_event):

        hidden_states, done_event = self._combine_core(
            hidden_states, topk_ids, topk_weights, previous_event
        )

        if self._comm_stream and self.async_finish and done_event is not None:
            torch.cuda.current_stream().wait_event(done_event)

        return hidden_states

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event: Optional[torch.cuda.Event],
    ):
        done_event: Optional[torch.cuda.Event] = None

        if self._comm_stream:
            compute_stream = torch.cuda.current_stream()
            comm_stream = self._comm_stream

            for t in (hidden_states, topk_ids, topk_weights):
                t.record_stream(comm_stream)

            with torch.cuda.stream(comm_stream):
                if previous_event is not None:
                    comm_stream.wait_event(previous_event)
                else:
                    comm_stream.wait_stream(compute_stream)

                combine_fn = (
                    self.mori_op.combine_send
                    if self.enable_sdma
                    else self.mori_op.combine
                )
                combine_kwargs = self._combine_kwargs(hidden_states)
                combined_hidden_states = combine_fn(
                    hidden_states, None, topk_ids, **combine_kwargs
                )[0]
                if self.enable_sdma:
                    self.mori_op.combine_recv()

                if self.async_finish:
                    done_event = torch.cuda.Event(blocking=False, interprocess=False)
                    done_event.record(comm_stream)
                else:
                    compute_stream.wait_stream(comm_stream)

            combined_hidden_states.record_stream(comm_stream)

        else:
            combine_kwargs = self._combine_kwargs(hidden_states)
            combined_hidden_states = self.mori_op.combine(
                hidden_states, None, topk_ids, **combine_kwargs
            )[0]

        return combined_hidden_states, done_event

    def set_quant_config(self, quant_config: dict):
        super().set_quant_config(quant_config)


class _MoriEPDispatcherImplLowLatency(_MoriEPDispatcherImplBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quant_config = {}
        self.fp8_quant_func = get_hip_quant(QuantType.per_1x128)
        self.fp4_quant_func = get_hip_quant(QuantType.per_1x32)

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        import mori

        assert (
            self.mori_op.config.kernel_type
            is mori.ops.EpDispatchCombineKernelType.AsyncLL
        ), "mori asyncll mismatch"

        num_tokens = hidden_states.shape[0]
        output_dtype = hidden_states.dtype
        scale = None

        if self.dispatch_dtype == DispatchDtype.fp8:
            # FP8 quant
            if num_tokens > 0:
                # NOTE: aiter is able to handle token=0 case in UT. But for some
                # reason it failed at e2e case. Root cause TBD.
                hidden_states, scale = self.fp8_quant_func(
                    hidden_states, quant_dtype=fp8_dtype
                )
            else:
                hidden_states = torch.empty(
                    hidden_states.shape, dtype=fp8_dtype, device=hidden_states.device
                )
                scale = torch.empty(
                    (0, self.hidden_size // FP8_BLOCK_SIZE),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )

        elif self.dispatch_dtype == DispatchDtype.fp4:
            # FP4 quant
            if num_tokens > 0:
                hidden_states, scale = self.fp4_quant_func(hidden_states, shuffle=False)
            else:
                hidden_states = torch.empty(
                    (0, self.hidden_size // 2),
                    dtype=torch.float4_e2m1fn_x2,
                    device=hidden_states.device,
                )
                scale = torch.empty(
                    (0, self.hidden_size // MXFP4_BLOCK_SIZE),
                    dtype=torch.float8_e8m0fnu,
                    device=hidden_states.device,
                )

        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

        (
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
        ) = self._dispatch_core(hidden_states, topk_weights, topk_ids, scale=scale)

        return (
            packed_recv_hidden,
            recv_topk_weights,
            recv_topk_ids,
            recv_scales,
            packed_recv_count,
            topk_weights,
            topk_ids,
            output_dtype,
        )

    def dispatch_b(
        self,
        hidden_states,
        recv_topk_weights,
        recv_topk_ids,
        recv_scales,
        packed_recv_count,
        topk_weights,
        topk_ids,
        output_dtype,
    ):

        ##TODO(billishyahao): add assertion here to check async
        import mori

        assert (
            self.mori_op.config.kernel_type
            is mori.ops.EpDispatchCombineKernelType.AsyncLL
        ), "mori asyncll mismatch"

        record = _should_record_expert_distribution()
        self.mori_op.dispatch_recv(call_local_expert_count=record)

        if record:
            get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(
                self.mori_op.local_expert_count
            )

        return MoriEPLLDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=recv_scales,
            topk_ids=recv_topk_ids,
            topk_weights=recv_topk_weights,
            num_recv_tokens_per_expert=packed_recv_count,
            origin_topk_ids=topk_ids,
            origin_topk_weights=topk_weights,
            out_dtype=output_dtype,
        )

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ):
        ##TODO(billishyahao): add assertion here to check async

        (
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
        ) = self.mori_op.dispatch_send(hidden_states, topk_weights, scale, topk_ids)

        return (
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
        )

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[CombineOverlapArgs] = None,
    ):
        hidden_states = self._combine_core(
            hidden_states,
            topk_ids,
            topk_weights,
            overlap_args=overlap_args,
        )
        return hidden_states, topk_ids, topk_weights, overlap_args

    def combine_b(self, hidden_states, topk_ids, topk_weights, previous_event):

        self.mori_op.combine_recv()

        return hidden_states[0]

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[CombineOverlapArgs] = None,
    ):
        combined_hidden_states = self.mori_op.combine_send(
            hidden_states, None, topk_ids
        )

        return combined_hidden_states

    def set_quant_config(self, quant_config: dict):
        super().set_quant_config(quant_config)


@dataclass
class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class MoriEPDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        instance_id: int = 0,
    ):
        super().__init__()

        self.deepep_mode = deepep_mode

        async_mode = self.deepep_mode.enable_low_latency()
        if get_bool_env_var("SGLANG_ROCM_USE_MULTI_STREAM") and not async_mode:
            logger.warning_once(
                "SGLANG_ROCM_USE_MULTI_STREAM=1 is set but Mori AsyncLL is "
                "not enabled (--deepep-mode=%s). The alt-stream overlap only "
                "frees up CUs when dispatch/combine runs on the AsyncLL "
                "copy-engine kernel; otherwise it stays on CUs and competes "
                "with the alt-stream work. Pass --deepep-mode low_latency "
                "(or auto) to enable the AsyncLL kernel.",
                self.deepep_mode.value,
            )

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
            instance_id=instance_id,
        )

        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _MoriEPDispatcherImplLowLatency(
                **common_kwargs,
            )

        if self.deepep_mode.enable_normal():
            self._normal_dispatcher = _MoriEPDispatcherImplNormal(
                async_finish=async_finish,
                **common_kwargs,
            )

        self._stage = _Stage.INITIAL
        self._deepep_dispatch_hooks = MoriEPPDispatchHooks()

        # Mori dispatch produces global topk_ids in [0, num_experts); mask out
        # experts that are not local to this rank.
        self.expert_mask_gpu = None
        if _use_aiter and num_experts is not None and num_local_experts is not None:
            ep_rank = get_parallel().moe_ep_rank
            expert_mask = torch.zeros(
                num_experts,
                device=torch.cuda.current_device(),
                dtype=torch.int32,
            )
            start = ep_rank * num_local_experts
            expert_mask[start : start + num_local_experts] = 1
            self.expert_mask_gpu = expert_mask

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self._num_tokens = hidden_states.shape[0]
        self.dispatch_a(hidden_states, topk_output)
        if self._deepep_dispatch_hooks is not None:
            self._deepep_dispatch_hooks(self)
        ret = self.dispatch_b()
        return ret

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
    ) -> Tuple:
        self.combine_a(combine_input)
        hidden_states = self.combine_b()
        return hidden_states[: self._num_tokens]

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def _get_impl(self) -> _MoriEPDispatcherImplBase:
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            return self._normal_dispatcher
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage

    def set_quant_config(self, quant_config: dict):
        super().set_quant_config(quant_config)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_quant_config(quant_config)
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_quant_config(quant_config)

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ):
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )

    def clear_overlap_args(self):
        super().clear_overlap_args()
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.clear_overlap_args()
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.clear_overlap_args()

    def register_deepep_dispatch_hook(self, hook):
        return self._deepep_dispatch_hooks.register_hook(hook)
