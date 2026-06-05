"""Real-GPU device repro for the mori decode dispatch-buffer truncation bug.

MOTIVATION_RULES rule 1 (Root-cause evidence) + rule 2 (Fix):
this drives the REAL shipping function
``sglang.srt.layers.moe.moe_runner.aiter._pre_permute_deepep_to_aiter`` on a
ROCm GPU with a mori-shaped (tail-padded) dispatch output, then performs the
exact row-indexing that ``aiter.fused_moe`` does on the returned
``hidden_states`` (gather rows ``[0, recv)`` where ``recv =
num_recv_tokens_per_expert.sum()``).

- Under the ORIGINAL local cap (``origin_topk_ids.shape[0] * ws``) the returned
  buffer is sliced to ``cap < recv`` on a heterogeneous-batch rank, so the
  gather indexes rows ``[cap, recv)`` that were sliced away -> GPU *Memory
  access fault* (HIP illegal address). Run in a subprocess; we assert it dies.
- Under the GLOBAL fix (``sum(get_dp_global_num_tokens())``) the buffer covers
  ``recv`` on every rank, so the same gather succeeds in-process with a
  non-negative margin.

The no-GPU model (test_mori_trunc_cap_underflow.py) proves the arithmetic; this
proves the same mechanism faults / is clean on real hardware, and measures the
real recv-vs-cap margin (the routing-imbalance robustness question).

Run (inside a torch+ROCm container, e.g. mingzhi-pd-prefill):
    PYTHONPATH=<worktree>/python python3 -m pytest \
        test/registered/amd/test_mori_trunc_device_oob.py -s
"""

import os
import subprocess
import sys

import pytest

try:
    import torch

    HAS_GPU = torch.cuda.is_available()
except Exception:  # pragma: no cover - import guard
    HAS_GPU = False

# Representative-but-small shapes. Real decode (buffer tier=512, ws=8) pads the
# recv buffer to ws*tier = 4096 rows; we shrink tier so the test is cheap but
# keep ws=8 (== topk for DeepSeek-R1) and the tail-padded layout identical.
WS = 8           # moe expert-parallel world size (== topk in this config)
TIER = 64        # SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK (shrunk from 512)
H = 256          # hidden dim (shrunk from 7168)
TOPK = 8
PAD_ROWS = WS * TIER  # mori recv buffer rows (tail-padded worst case)

# Heterogeneous decode batch: THIS rank has a tiny origin (e.g. capture/warmup
# straggler) while the other ranks are busy. recv is routed in from ALL ranks.
ORIGIN_LOCAL = 2                         # this rank's pre-dispatch token count
GLOBAL_NUM_TOKENS = [ORIGIN_LOCAL] + [32] * (WS - 1)  # per-DP-rank origin list
RECV = sum(GLOBAL_NUM_TOKENS)            # 2 + 32*7 = 226 rows actually received
LOCAL_CAP = ORIGIN_LOCAL * WS            # original buggy cap = 16
GLOBAL_CAP = sum(GLOBAL_NUM_TOKENS)      # fixed cap = 226


def _build_dispatch_output():
    """A minimal but faithful MoriEPLLDispatchOutput on the current device."""
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriEPLLDispatchOutput,
    )

    dev = "cuda"
    # bf16 activations + no dispatch scale -> a1_scale stays None, so the
    # upscale branches are skipped and _resolve_mori_quant_type -> NONE. This
    # keeps the test on the pure truncation path (no quant kernels needed).
    hidden_states = torch.randn(PAD_ROWS, H, dtype=torch.bfloat16, device=dev)
    topk_ids = torch.zeros(PAD_ROWS, TOPK, dtype=torch.int64, device=dev)
    topk_weights = torch.ones(PAD_ROWS, TOPK, dtype=torch.float32, device=dev)
    origin_topk_ids = torch.zeros(ORIGIN_LOCAL, TOPK, dtype=torch.int64, device=dev)
    origin_topk_weights = torch.ones(
        ORIGIN_LOCAL, TOPK, dtype=torch.float32, device=dev
    )
    return MoriEPLLDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_recv_tokens_per_expert=[RECV],  # recv this rank's experts process
        origin_topk_ids=origin_topk_ids,
        origin_topk_weights=origin_topk_weights,
        out_dtype=torch.bfloat16,
        expected_m=RECV // WS,
    )


def _run_pre_permute(global_num_tokens):
    """Drive the real _pre_permute_deepep_to_aiter with patched dp_attention.

    global_num_tokens=None forces the original local origin*ws cap branch;
    a per-rank list of length WS exercises the global-cap fix branch.
    Returns the sliced hidden_states tensor (cap = its row count).
    """
    import sglang.srt.layers.moe.moe_runner.aiter as aiter_mod
    from sglang.srt.layers.moe.moe_runner.aiter import (
        AiterMoeQuantInfo,
        AiterQuantType,
        _pre_permute_deepep_to_aiter,
    )

    # decode path (not extend); inject the chosen global token distribution.
    # ws is normally read from the (uninitialized here) EP process group, so
    # pin it to WS directly -- this is the value the real decode server uses.
    aiter_mod.get_is_extend_in_batch = lambda: False
    aiter_mod.get_dp_global_num_tokens = lambda: global_num_tokens
    aiter_mod.get_moe_expert_parallel_world_size = lambda: WS

    dummy = torch.empty(0, device="cuda")
    quant_info = AiterMoeQuantInfo(
        w13_weight=dummy, w2_weight=dummy, quant_type=AiterQuantType.NONE
    )
    dispatch_output = _build_dispatch_output()
    out = _pre_permute_deepep_to_aiter(
        dispatch_output, quant_info, runner_config=None, running_state={}
    )
    return out.hidden_states


def _gather_recv_rows(hidden_states):
    """The fused_moe-style access: read rows [0, RECV) of the capped buffer.

    If hidden_states.shape[0] < RECV this indexes past the slice -> on ROCm a
    HIP illegal address / 'Memory access fault by GPU'.
    """
    idx = torch.arange(RECV, device=hidden_states.device)
    gathered = hidden_states.index_select(0, idx)
    torch.cuda.synchronize()
    return gathered


@pytest.mark.skipif(not HAS_GPU, reason="needs a ROCm/CUDA GPU")
def test_fix_global_cap_covers_recv_no_fault_real_gpu():
    """FIX: global cap >= recv on this rank -> real-GPU gather succeeds."""
    hs = _run_pre_permute(GLOBAL_NUM_TOKENS)
    cap = hs.shape[0]
    assert cap == GLOBAL_CAP, f"expected global cap {GLOBAL_CAP}, got {cap}"
    margin = cap - RECV
    # gather the real recv rows on-device; must not fault.
    gathered = _gather_recv_rows(hs)
    assert gathered.shape[0] == RECV
    print(
        f"[device][fix] origin_local={ORIGIN_LOCAL} global_cap={cap} "
        f"recv={RECV} margin={margin} (>=0, no GPU fault) "
        f"buffer_rows={PAD_ROWS} (cap<<buffer keeps perf)"
    )
    assert margin >= 0


@pytest.mark.skipif(not HAS_GPU, reason="needs a ROCm/CUDA GPU")
def test_base_local_cap_faults_real_gpu_subprocess():
    """BASE: local origin*ws cap < recv -> real-GPU gather faults (subprocess).

    A GPU memory access fault aborts the whole process, so we run the buggy
    path in a child and assert it dies (non-zero exit) with a fault signature,
    while the parent stays alive to keep iterating.
    """
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, __file__, "--worker-base-oob"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out)
    # The child prints the cap underflow first, then dies in the gather.
    assert "local_cap=" in out, "worker did not reach the cap print"
    assert proc.returncode != 0, (
        "base local-cap path did NOT fault on real GPU "
        f"(returncode={proc.returncode}); expected a memory access fault"
    )
    sig = ("memory access fault" in out.lower()) or (
        "illegal" in out.lower()
    ) or ("hip" in out.lower() and "fault" in out.lower())
    assert sig or proc.returncode < 0, (
        "process died but without a recognizable GPU-fault signature; "
        f"returncode={proc.returncode}, output tail:\n{out[-1000:]}"
    )


def _worker_base_oob():
    """Child entrypoint: run the buggy local-cap path and trigger the fault."""
    hs = _run_pre_permute(None)  # None -> origin*ws local cap branch
    cap = hs.shape[0]
    # Print BEFORE the faulting op so the parent sees the underflow even though
    # the gather aborts the process.
    print(
        f"[device][base] origin_local={ORIGIN_LOCAL} local_cap={cap} "
        f"recv={RECV} overrun={RECV - cap} -> indexing past slice",
        flush=True,
    )
    assert cap == LOCAL_CAP, f"expected local cap {LOCAL_CAP}, got {cap}"
    _gather_recv_rows(hs)  # rows [cap, RECV) are OOB -> GPU memory access fault
    print("[device][base] UNEXPECTED: gather did not fault", flush=True)


if __name__ == "__main__":
    if "--worker-base-oob" in sys.argv:
        _worker_base_oob()
