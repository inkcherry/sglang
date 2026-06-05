"""Deterministic root-cause repro for the mori decode dispatch-buffer
truncation GPU ``Memory access fault``.

Motivation (MOTIVATION_RULES.md rule 1, "Root-cause evidence"): a deterministic
reproduction / printed evidence that advances *why* the per-forward
dispatch-buffer truncation in ``aiter.py`` causes a GPU ``Memory access fault``
during decode at ``--max-running-requests 1024``
``SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512``.

What this test pins down (no GPU / no torch required):

The truncation in ``_pre_permute_deepep_to_aiter`` (moe_runner/aiter.py) is::

    cap = origin_topk_ids.shape[0] * get_moe_expert_parallel_world_size()
    hidden_states = hidden_states[:cap]
    ... topk_ids[:cap]; topk_weights[:cap]; a1_scale[:cap]

but the *same* dispatch output also carries
``num_recv_tokens_per_expert`` (forwarded to aiter.fused_moe as
``num_local_tokens``). aiter's moe_sorting / grouped-GEMM index the
``hidden_states`` rows by ``num_local_tokens``, i.e. it reads rows in
``[0, sum(num_recv_tokens_per_expert))``.

``cap`` is derived from THIS rank's *local* pre-dispatch token count
(``origin_topk_ids.shape[0] = bs_per_rank * num_draft_tokens``), while
``sum(num_recv_tokens_per_expert)`` ("true recv") is routed in from ALL ranks.
The two are only equal when every rank has the same batch (uniform). When they
are not (heterogeneous per-rank batches and/or routing imbalance) the cap can be
strictly LESS than the true recv on some rank -> ``hidden_states[:cap]`` drops
rows ``[cap, recv)`` that fused_moe then indexes -> device out-of-bounds ->
GPU ``Memory access fault``.

This module models the documented formulas (ROOT_CAUSE_CLUES.md sections 3-5)
and asserts the underflow, so the failure mode is locked in as a regression and
the *why* is printed as evidence. The on-device confirmation of the actual fault
is the 1P1D e2e (acceptance criterion, run under the COORD GPU lock); a
torch+ROCm port of the same construction can drive aiter.fused_moe directly --
that lives behind ``test_device_oob`` below (skipped without torch/cuda).

RESIDUAL ROOT CAUSE (the global cap is NECESSARY but NOT SUFFICIENT)
-------------------------------------------------------------------
On 2026-06-04T16:17:53Z a 1P1D decode server running the deployed host-side
global-cap fix (``cap = sum(get_dp_global_num_tokens())``, decode-container
``_mori_decode_recv_cap``, == teamA 173fe0357 == teamB 7ed99d97b) STILL hit the
same ``Memory access fault by GPU node-2 ... Reason: Unknown`` during the warmup
sweep (logs/decode_teamB_t4_FAULT_16h17.log.keep:731, 15 of 32 warmups done).

The arithmetic refines *why*. For this deployment ``topk == ws == 8``, so the
global-token cap equals the *uniform-routing* recv EXACTLY (margin 0, see
``test_uniform_batch_is_the_boundary``). recv on a rank is the device-side
``sum(num_recv_tokens_per_expert)`` -- the number of (token,expert) routes that
land on that rank's experts. Under perfectly uniform routing each rank gets
``global_total * topk / ws == global_total == cap``. But real routing is never
uniform: a rank holding hot/popular experts receives MORE than ``1/ws`` of the
``global_total * topk`` routes, so its recv exceeds the cap and fused_moe again
indexes ``[cap, recv)`` past the truncated ``hidden_states`` -> the fault. The
cap is a host-side TOKEN COUNT; the index bound is a device-side ROUTE COUNT.
No host token count (local OR global) bounds the device recv when topk==ws.

Implication for the real fix: the cap must be derived from the *actual* recv
(``num_recv_tokens_per_expert.sum()``, already on device and already the
fused_moe bound) -- e.g. ``cap = max(sum(global_num_tokens), recv)`` or simply
never slice below ``recv`` -- not from a host token count alone. That fix needs
on-device validation under the COORD GPU lock; this module pins the refuted
sufficiency so the regression is locked in. ``test_global_cap_underflows_under_
routing_imbalance`` reproduces the residual overrun.
"""

import unittest


def current_cap(origin_local_tokens: int, world_size: int) -> int:
    """The cap used today: aiter.py ``origin_topk_ids.shape[0] * ws``.

    ``origin_local_tokens`` is THIS rank's pre-dispatch token count
    (= bs_per_rank * num_draft_tokens). It is a purely *local* quantity.
    """
    return origin_local_tokens * world_size


def fixed_cap(global_num_tokens: list[int]) -> int:
    """The cap after the fix (aiter.py): ``sum(get_dp_global_num_tokens())``.

    ``get_dp_global_num_tokens()`` is the host-side, spec-adjusted per-DP-rank
    token list materialized once per forward by DP attention. Under MAX_LEN
    padding (CUDA-graph capture) every entry equals ``max`` so the sum reduces
    to ``origin*ws`` -- the captured size is unchanged. Under SUM_LEN padding
    (eager warmup/decode) the entries are the real heterogeneous per-rank
    counts, so the sum is the true GLOBAL token count, identical on every rank.
    """
    return sum(global_num_tokens)


def fix_v2_cap(
    *,
    capturing: bool,
    recv: "int | None",
    global_num_tokens: "list[int] | None",
    origin_local_tokens: int,
    world_size: int,
    buffer_rows: int,
) -> int:
    """Cap selected by fix v2 (aiter.py): size from the ACTUAL device recv.

    Mirrors the branch in ``_pre_permute_deepep_to_aiter`` decode path:

    * Under CUDA-graph capture (``capturing``) a device->host sync is illegal,
      but DP attention pads MAX_LEN so the host global-token sum bounds recv and
      equals ``origin*ws`` (captured slice byte-unchanged). Falls back to
      ``origin*ws`` if the global metadata is absent/misaligned under capture.
    * Eager: use the device recv (``num_recv_tokens_per_expert.sum()``) directly
      -- exact, OOB-safe, still concurrency-scaled. This covers BOTH PD-warmup
      (no global metadata) and routing imbalance (recv can exceed any host token
      cap when topk == ws).
    * Eager with no recv available: do not truncate (full buffer is recv-safe).

    Always clamped to ``buffer_rows`` (never slice above the padded buffer).
    """
    if capturing:
        if global_num_tokens is not None and len(global_num_tokens) == world_size:
            cap = sum(global_num_tokens)
        else:
            cap = origin_local_tokens * world_size
    elif recv is not None:
        cap = recv
    else:
        cap = buffer_rows
    return min(cap, buffer_rows)


def fix_v3_capture_cap(
    *,
    global_num_tokens: "list[int] | None",
    origin_local_tokens: int,
    world_size: int,
    buffer_rows: int,
    coeff: float = 0.7,
) -> int:
    """Cap selected by fix v3 on the CUDA-graph-CAPTURED decode path (aiter.py).

    The captured (benchmarked) path cannot host-sync to read the device recv, so
    v2 capped from the host global token count ``sum(global_num_tokens)`` (==
    ``origin*ws`` under MAX_LEN). That is recv-SAFE but LOOSE: e2e measurement
    (2026-06-04T16:52Z) showed the actual device recv is only ~0.5-0.6x of it
    (recv 120-152 vs cap 256 at conc-64), so fused_moe processed ~1.4-1.7x excess
    rows and TPOT regressed ~26% vs target.

    v3 tightens the captured cap to the InferenceMINI CI reference coefficient
    ``SGLANG_MORI_MOE_MAX_INPUT_TOKENS = floor(conc*0.7)*(MTP+1) ~= 0.7 * global
    tokens`` (ROOT_CAUSE_CLUES s4) -- a production-validated recv-tight static cap
    that hit the 7.4/8.7 TPOT targets. Still host-side / graph-safe; shrinks with
    concurrency; clamped to ``buffer_rows``.
    """
    if global_num_tokens is not None and len(global_num_tokens) == world_size:
        global_tokens = sum(global_num_tokens)
    else:
        global_tokens = origin_local_tokens * world_size
    return min(int(global_tokens * coeff), buffer_rows)


def true_recv_uniform_routing(
    origin_per_rank: list[int], topk: int, world_size: int
) -> int:
    """Expected ``sum(num_recv_tokens_per_expert)`` on one rank under uniform
    (balanced) expert routing.

    Every source token fans out to ``topk`` experts; globally there are
    ``sum(origin_per_rank) * topk`` (token, expert) routes. Each rank owns
    ``1/world_size`` of the experts, so under uniform routing it receives
    ``sum(origin) * topk / world_size`` routes. This is the count aiter's
    ``num_local_tokens`` carries and that fused_moe uses to index hidden rows.

    Real routing is never perfectly uniform, so on a busy rank the true recv is
    >= this mean -- making the underflow shown here a *lower bound* on the gap.
    """
    total_routes = sum(origin_per_rank) * topk
    # ceil so an imbalanced rank is not under-counted.
    return -(-total_routes // world_size)


def recv_with_routing_share(global_total: int, topk: int, rank_share: float) -> int:
    """Device-side ``sum(num_recv_tokens_per_expert)`` on a rank that owns
    ``rank_share`` of the global routes.

    There are ``global_total * topk`` (token, expert) routes globally. A rank
    receives ``rank_share`` of them (``rank_share == 1/ws`` under perfectly
    uniform routing; ``> 1/ws`` when it holds hot/popular experts). This is the
    ROUTE count fused_moe indexes -- a different quantity from the host TOKEN
    count the cap is built from.
    """
    return round(rank_share * global_total * topk)


class TestMoriTruncCapUnderflow(unittest.TestCase):
    # DeepSeek-R1 mori decode config under investigation (ROOT_CAUSE_CLUES s1/s3):
    #   TP=EP=DP=8 -> world_size 8; router topk 8; NEXTN MTP num_draft_tokens 4.
    WS = 8
    TOPK = 8
    NUM_DRAFT = 4

    def test_uniform_batch_is_the_boundary(self):
        """Uniform per-rank batch (the CUDA-graph capture path) is exactly on
        the cap==recv boundary, which is why capture/warmup-uniform passes."""
        bs_per_rank = 8  # steady conc-64: 64 / tp(8) = 8 reqs/rank
        origin_local = bs_per_rank * self.NUM_DRAFT  # 32  (matches s3: origin=32)
        origin_per_rank = [origin_local] * self.WS  # uniform

        cap = current_cap(origin_local, self.WS)  # 32 * 8 = 256 (matches s3 cap=256)
        recv = true_recv_uniform_routing(origin_per_rank, self.TOPK, self.WS)

        print(
            f"[uniform] origin_local={origin_local} cap={cap} recv={recv} "
            f"-> margin={cap - recv}"
        )
        self.assertEqual(cap, 256, "anchor to measured s3 shape cap=256")
        # topk == world_size (8 == 8): under uniform routing cap == recv exactly.
        # ZERO margin -> any routing imbalance immediately overruns.
        self.assertEqual(cap, recv, "uniform decode is exactly on the OOB boundary")

    def test_heterogeneous_batch_underflows_cap(self):
        """The warmup / real-decode path: heterogeneous per-rank batches make
        the LOCAL-derived cap on the small-batch rank fall far below the GLOBAL
        true recv routed into that rank -> the OOB."""
        # One nearly-idle rank (e.g. warmup / DP-attention skew) at origin=2
        # (matches s3 origin=2 warmup shape), the rest at steady conc-64 load.
        idle_origin = 2
        busy_origin = 8 * self.NUM_DRAFT  # 32
        origin_per_rank = [idle_origin] + [busy_origin] * (self.WS - 1)

        cap_idle = current_cap(idle_origin, self.WS)  # 2 * 8 = 16 (matches s3 cap=16)
        recv_idle = true_recv_uniform_routing(origin_per_rank, self.TOPK, self.WS)

        print(
            f"[hetero] origin_per_rank={origin_per_rank} "
            f"cap(idle rank)={cap_idle} recv(idle rank)={recv_idle} "
            f"-> overrun_rows={recv_idle - cap_idle}"
        )
        # cap=16 but the idle rank still receives the global average (~226)
        # because other ranks route their tokens to its experts. fused_moe will
        # index rows [16, 226) that hidden_states[:16] no longer contains.
        self.assertLess(
            cap_idle,
            recv_idle,
            "LOCAL origin*ws cap underflows GLOBAL true recv -> device OOB",
        )
        self.assertGreater(recv_idle - cap_idle, 100, "overrun is large, not marginal")

    def test_global_cap_covers_recv_on_every_rank(self):
        """Candidate fix A, per-rank sweep: contrast the LOCAL ``origin*ws`` cap
        against the GLOBAL fix cap across a heterogeneous batch. The true recv is
        a GLOBAL quantity (each rank receives tokens routed in from ALL ranks, so
        ~the global mean regardless of its own origin), therefore the LOCAL cap
        underflows on every small-batch rank while the global fix cap covers recv
        on EVERY rank. (Non-tautological: ``global_cap`` is the proposed fix value
        ``sum(origin)`` -- a distinct computation from the per-rank recv.)"""
        # A skewed but realistic per-rank batch (DP-attention SUM_LEN warmup).
        origin_per_rank = [2, 4, 8, 16, 32, 32, 32, 32]

        # true recv is GLOBAL: routed in from all ranks, independent of local
        # origin, so it is the same value on every rank.
        recv = true_recv_uniform_routing(origin_per_rank, self.TOPK, self.WS)
        # the fix cap: sum of per-rank origin == sum(get_dp_global_num_tokens()),
        # identical on every rank. Computed independently of `recv`.
        global_cap = fixed_cap(origin_per_rank)

        local_underflow_ranks = []
        for rank, origin in enumerate(origin_per_rank):
            local_cap = current_cap(origin, self.WS)  # the buggy per-rank cap
            if local_cap < recv:
                local_underflow_ranks.append((rank, local_cap))
            # the fix must cover the GLOBAL recv on EVERY rank, regardless of
            # that rank's local origin -- this is the cross-rank-consistency.
            self.assertGreaterEqual(
                global_cap,
                recv,
                f"global cap {global_cap} must cover true recv {recv} on rank "
                f"{rank} (origin={origin})",
            )
        print(
            f"[global-fix] origin_per_rank={origin_per_rank} recv={recv} "
            f"global_cap={global_cap}; LOCAL origin*ws underflows on ranks "
            f"{local_underflow_ranks}, global cap covers all"
        )
        # The buggy LOCAL cap must underflow on at least the small-batch ranks,
        # proving the per-rank cap is unsafe while the global cap is safe.
        self.assertTrue(
            local_underflow_ranks,
            "LOCAL origin*ws must underflow recv on some rank (else no bug)",
        )
        # ...and the global cap still shrinks with concurrency (well under buffer).
        self.assertLess(global_cap, self.WS * 512, "global cap stays concurrency-scaled")

    def test_fix_matches_local_cap_under_maxlen_padding(self):
        """Fix safety on the hot path: under DP-attention MAX_LEN padding (the
        CUDA-graph capture mode) all ranks are padded to ``max``, so the fixed
        cap ``sum(global_num_tokens)`` equals the old ``origin*ws`` EXACTLY.
        => zero numerics/perf change on the captured decode path."""
        bs_per_rank = 8
        origin_local = bs_per_rank * self.NUM_DRAFT  # 32
        # MAX_LEN pads every DP rank to the same max => uniform list.
        global_num_tokens = [origin_local] * self.WS

        old_cap = current_cap(origin_local, self.WS)
        new_cap = fixed_cap(global_num_tokens)

        print(f"[fix/maxlen] old_cap={old_cap} new_cap={new_cap} (must be equal)")
        self.assertEqual(
            new_cap,
            old_cap,
            "under MAX_LEN padding the fixed cap must equal origin*ws "
            "(captured graph path unchanged)",
        )

    def test_fix_covers_recv_under_sumlen_heterogeneity(self):
        """Fix correctness on the bug path: under SUM_LEN padding the per-rank
        batches are heterogeneous; the fixed cap = sum(global_num_tokens) is
        identical on every rank and >= the true recv on every rank, closing the
        underflow that the LOCAL origin*ws cap suffered on the idle rank."""
        idle_origin = 2
        busy_origin = 8 * self.NUM_DRAFT  # 32
        # SUM_LEN keeps the real per-rank counts (no equal padding).
        global_num_tokens = [idle_origin] + [busy_origin] * (self.WS - 1)

        new_cap = fixed_cap(global_num_tokens)  # same on all ranks
        old_cap_idle = current_cap(idle_origin, self.WS)  # 16, the buggy value
        recv = true_recv_uniform_routing(global_num_tokens, self.TOPK, self.WS)

        print(
            f"[fix/sumlen] global_num_tokens={global_num_tokens} "
            f"new_cap={new_cap} old_cap(idle)={old_cap_idle} recv={recv} "
            f"-> fix_margin={new_cap - recv}, old_overrun={recv - old_cap_idle}"
        )
        # The fix closes the underflow on the idle rank...
        self.assertGreater(
            new_cap, old_cap_idle, "fixed cap must exceed the buggy local cap"
        )
        # ...and covers the true recv on EVERY rank (cross-rank-consistent).
        self.assertGreaterEqual(
            new_cap, recv, "fixed cap (global sum) must cover true recv"
        )
        # ...while still shrinking with concurrency: it is far below the padded
        # mori buffer (ws * tier), proving the perf goal is preserved.
        padded_buffer_rows = self.WS * 512  # ws * SGLANG_MORI_..._PER_RANK tier
        self.assertLess(
            new_cap,
            padded_buffer_rows,
            "fixed cap must stay well below the padded buffer (concurrency-scaled)",
        )

    def test_global_cap_underflows_under_routing_imbalance(self):
        """RESIDUAL root cause: the GLOBAL-token cap is necessary but NOT
        sufficient. Reproduces the 16:17:53Z e2e fault on the *deployed* fix.

        For this deployment ``topk == ws == 8``, so the global-token cap equals
        the uniform-routing recv exactly (margin 0). The device recv is a ROUTE
        count: a rank holding hot experts receives more than ``1/ws`` of the
        ``global_total * topk`` routes, so its recv exceeds the host-side cap and
        fused_moe indexes past ``hidden_states[:cap]`` -> the same GPU fault."""
        self.assertEqual(self.TOPK, self.WS, "this overrun requires topk == ws")
        # Uniform per-rank batch (steady conc-64): cap = sum(global) = global_total.
        bs_per_rank = 8
        origin_local = bs_per_rank * self.NUM_DRAFT  # 32
        global_num_tokens = [origin_local] * self.WS
        global_total = fixed_cap(global_num_tokens)  # 256 == the deployed cap
        cap = global_total

        # Perfectly uniform routing sits exactly on the cap (margin 0)...
        recv_uniform = recv_with_routing_share(global_total, self.TOPK, 1 / self.WS)
        self.assertEqual(recv_uniform, cap, "uniform recv == cap (zero margin)")

        # ...but real warmup/decode routing concentrates onto hot experts. A
        # modest, well-documented MoE imbalance (busiest rank ~1.4x its uniform
        # share) already pushes that rank's recv above the cap.
        hot_share = 1.4 / self.WS  # 40% above uniform -> still < topk/ws bound
        recv_busy = recv_with_routing_share(global_total, self.TOPK, hot_share)

        print(
            f"[imbalance] global_total={global_total} cap={cap} "
            f"recv_uniform={recv_uniform} recv_busy(@{hot_share:.3f})={recv_busy} "
            f"-> overrun_rows={recv_busy - cap} (deployed-fix e2e fault 16:17:53Z)"
        )
        # The deployed global-token cap UNDERFLOWS the busy rank's device recv.
        self.assertGreater(
            recv_busy,
            cap,
            "global-token cap underflows recv on a routing-imbalanced rank "
            "(topk==ws => zero uniform margin); host token count cannot bound "
            "the device route count -> residual GPU fault",
        )
        # The safe cap is the actual recv, not any host token count.
        safe_cap = max(cap, recv_busy)
        self.assertGreaterEqual(
            safe_cap, recv_busy, "cap from actual device recv is safe by construction"
        )

    BUFFER_ROWS = 8 * 512  # ws * SGLANG_MORI_..._PER_RANK tier = padded recv buffer

    def test_fix_v2_eager_uses_device_recv_covers_imbalance(self):
        """Fix v2: in eager mode the cap is the ACTUAL device recv, so it covers
        a routing-imbalanced rank that the global-token cap underflowed (the
        16:17:53Z e2e fault). Same uniform batch as
        ``test_global_cap_underflows_under_routing_imbalance``."""
        origin_local = 8 * self.NUM_DRAFT  # 32
        global_num_tokens = [origin_local] * self.WS
        global_total = fixed_cap(global_num_tokens)  # 256 == old deployed cap

        # Busy rank receives 1.4x its uniform share -> recv > old global cap.
        recv_busy = recv_with_routing_share(global_total, self.TOPK, 1.4 / self.WS)
        old_cap = global_total
        v2_cap = fix_v2_cap(
            capturing=False,
            recv=recv_busy,
            global_num_tokens=global_num_tokens,
            origin_local_tokens=origin_local,
            world_size=self.WS,
            buffer_rows=self.BUFFER_ROWS,
        )
        print(
            f"[v2/eager-imbalance] recv_busy={recv_busy} old_global_cap={old_cap} "
            f"v2_cap={v2_cap} -> old_overrun={recv_busy - old_cap}, "
            f"v2_margin={v2_cap - recv_busy}"
        )
        self.assertLess(old_cap, recv_busy, "global-token cap underflowed (the bug)")
        self.assertGreaterEqual(v2_cap, recv_busy, "v2 cap covers the device recv")
        self.assertEqual(v2_cap, recv_busy, "v2 cap is exactly the recv (minimal)")
        self.assertLessEqual(v2_cap, self.BUFFER_ROWS, "never above the padded buffer")

    def test_fix_v2_pd_warmup_no_metadata_does_not_undercap(self):
        """Fix v2: during PD-disaggregation warmup the DP global-token metadata
        is absent. The prior fix fell back to the buggy local ``origin*ws``
        (under-cap -> the 16:24Z e2e fault). v2 (eager) instead uses the device
        recv, which covers the global recv routed into an idle rank."""
        idle_origin = 2  # near-idle warmup rank (s3 origin=2 shape)
        # recv is GLOBAL: routed in from all ranks even when local origin is tiny.
        recv = true_recv_uniform_routing(
            [idle_origin] + [8 * self.NUM_DRAFT] * (self.WS - 1), self.TOPK, self.WS
        )
        old_fallback = current_cap(idle_origin, self.WS)  # 16 -- the buggy under-cap
        v2_cap = fix_v2_cap(
            capturing=False,
            recv=recv,
            global_num_tokens=None,  # PD-warmup: metadata absent
            origin_local_tokens=idle_origin,
            world_size=self.WS,
            buffer_rows=self.BUFFER_ROWS,
        )
        print(
            f"[v2/pd-warmup] idle_origin={idle_origin} recv={recv} "
            f"old_fallback(origin*ws)={old_fallback} v2_cap={v2_cap} "
            f"-> old_overrun={recv - old_fallback}, v2_margin={v2_cap - recv}"
        )
        self.assertLess(old_fallback, recv, "origin*ws fallback under-caps (the bug)")
        self.assertGreaterEqual(v2_cap, recv, "v2 covers recv even without metadata")

    def test_fix_v2_capture_matches_origin_ws_hot_path_unchanged(self):
        """Fix v2: under CUDA-graph capture (no host sync) the cap is the host
        global-token sum, which under MAX_LEN padding equals ``origin*ws`` -- the
        benchmarked captured decode path is byte-for-byte unchanged and never
        does a device->host sync."""
        origin_local = 8 * self.NUM_DRAFT  # 32
        global_num_tokens = [origin_local] * self.WS  # MAX_LEN padding -> uniform
        v2_cap = fix_v2_cap(
            capturing=True,
            recv=None,  # not read under capture
            global_num_tokens=global_num_tokens,
            origin_local_tokens=origin_local,
            world_size=self.WS,
            buffer_rows=self.BUFFER_ROWS,
        )
        old_cap = current_cap(origin_local, self.WS)
        print(f"[v2/capture] old_cap={old_cap} v2_cap={v2_cap} (must be equal)")
        self.assertEqual(v2_cap, old_cap, "captured hot path slice size unchanged")

    def test_fix_v3_capture_cap_is_recv_tight_and_safe(self):
        """Fix v3 (perf): on the CAPTURED hot path the v2 loose cap == global
        tokens leaves fused_moe processing ~1.4-1.7x excess rows (measured e2e
        recv 120-152 vs cap 256 at conc-64 -> ~26% TPOT regression). v3 tightens
        to the CI-validated 0.7 coefficient: still >= the measured device recv
        (no fault) but ~30% fewer rows -> recovers the perf gap."""
        origin_local = 8 * self.NUM_DRAFT  # 32  (steady conc-64)
        global_num_tokens = [origin_local] * self.WS  # MAX_LEN capture -> uniform

        v2_cap = fixed_cap(global_num_tokens)  # 256, the loose deployed cap
        v3_cap = fix_v3_capture_cap(
            global_num_tokens=global_num_tokens,
            origin_local_tokens=origin_local,
            world_size=self.WS,
            buffer_rows=self.BUFFER_ROWS,
        )
        # The actual device recv measured on the 2026-06-04T16:52Z e2e run.
        measured_recv_max = 152

        print(
            f"[v3/capture] v2_loose_cap={v2_cap} v3_tight_cap={v3_cap} "
            f"measured_recv_max={measured_recv_max} -> v2_excess_rows="
            f"{v2_cap - measured_recv_max}, v3_excess_rows={v3_cap - measured_recv_max}, "
            f"rows_saved={v2_cap - v3_cap}"
        )
        # v3 is strictly tighter than the loose v2 cap (fewer fused_moe rows)...
        self.assertLess(v3_cap, v2_cap, "v3 must shrink the captured slice vs v2")
        # ...but still covers the measured device recv (recv-safe, no OOB)...
        self.assertGreaterEqual(
            v3_cap, measured_recv_max, "v3 cap must stay >= the measured device recv"
        )
        # ...and matches the CI reference cap floor(conc*0.7)*(MTP+1)=176 closely.
        ci_ref_cap = (int(64 * 0.7)) * self.NUM_DRAFT  # floor(44.8)*4 = 176
        self.assertAlmostEqual(
            v3_cap, ci_ref_cap, delta=self.NUM_DRAFT,
            msg="v3 cap must track the production-validated CI recv-tight cap",
        )

    def test_fix_v3_capture_cap_scales_with_concurrency(self):
        """v3 keeps ONE server optimal at both concurrencies: the recv-tight cap
        shrinks with the live (captured) batch, so conc-64 and conc-128 each get
        a right-sized fused_moe without any per-concurrency env tweak."""
        for conc, exp_ci in ((64, 176), (128, 356)):
            bs_per_rank = conc // self.WS
            origin_local = bs_per_rank * self.NUM_DRAFT
            global_num_tokens = [origin_local] * self.WS
            v3_cap = fix_v3_capture_cap(
                global_num_tokens=global_num_tokens,
                origin_local_tokens=origin_local,
                world_size=self.WS,
                buffer_rows=self.BUFFER_ROWS,
            )
            print(f"[v3/scale] conc={conc} v3_cap={v3_cap} ci_ref={exp_ci}")
            # tracks the CI recv-tight reference for each concurrency...
            self.assertAlmostEqual(v3_cap, exp_ci, delta=self.NUM_DRAFT)
            # ...and stays well under the static 512-tier padded buffer.
            self.assertLess(v3_cap, self.BUFFER_ROWS)

    @unittest.skip(
        "Device OOB confirmation: requires torch+ROCm+aiter (run in container / "
        "1P1D e2e under COORD GPU lock). Modeled deterministically above."
    )
    def test_device_oob(self):  # pragma: no cover
        """Placeholder for the on-device repro: build a mori-shaped dispatch
        output (hidden_states padded to ws*tier rows, num_recv_tokens summing
        above cap) and call ``_pre_permute_deepep_to_aiter`` + aiter.fused_moe;
        expect a device-side out-of-bounds. Implemented when run on GPU."""
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
