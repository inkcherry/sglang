"""Runtime prefill<->decode role switching for PD disaggregation.

The token KV pool is role-independent and never reallocated; only the
role-specific disaggregation structures are torn down and rebuilt on a flip.
Kept out of scheduler.py to avoid growing it further.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING, Optional, Sequence

import msgspec

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import PdRoleSwitchReqInput, PdRoleSwitchReqOutput
from sglang.srt.runtime_context import get_context, get_schedule
from sglang.srt.session.session_controller import SessionController

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def handle_pd_role_switch(
    scheduler: Scheduler, recv_req: PdRoleSwitchReqInput
) -> PdRoleSwitchReqOutput:
    """Flip the scheduler's disaggregation role at runtime. The instance must be
    idle; rebuild failure is fatal to the instance (no in-place rollback)."""
    old_role = scheduler.disaggregation_mode.value
    new_role = (recv_req.new_role or "").lower()

    def _fail(msg: str) -> PdRoleSwitchReqOutput:
        logger.warning(
            "PD role switch rejected (%s -> %s): %s", old_role, new_role, msg
        )
        return PdRoleSwitchReqOutput(
            success=False, message=msg, old_role=old_role, new_role=new_role
        )

    reason = _reject_reason(scheduler, new_role)
    if reason is not None:
        return _fail(reason)
    if new_role == old_role:
        return PdRoleSwitchReqOutput(
            success=True,
            message="already in target role",
            old_role=old_role,
            new_role=new_role,
        )
    if not scheduler.is_fully_idle():
        return _fail("instance is not idle; drain all requests before switching")

    scheduler._pd_role_switch_in_progress = True
    try:
        try:
            if new_role == "decode":
                # Before the reconcile, not after: the a2a buffer is sized from
                # the padded decode batch, which is only knowable once the
                # runner has published its live captured-bucket ladder.
                _capture_decode_cuda_graphs(scheduler, recv_req.decode_cuda_graph_bs)
            reconcile_role_config(scheduler, new_role, recv_req)
        except Exception as e:
            if scheduler._pd_role_switch_unhealthy:
                return _fail(f"a2a group is dead; restart required: {e}")
            return _fail(f"role config reconcile failed, nothing applied: {e}")

        # Teardown + role flip + rebuild are one logical atomic step. If any of
        # them raises, the instance is left half-torn-down (old role released,
        # new role not up) and isn't safe to serve, so mark it unhealthy. There
        # is no in-place rollback.
        try:
            scheduler._teardown_disaggregation()
            scheduler.server_args.override(
                "role_switch.flip", disaggregation_mode=new_role
            )
            # The rebuild and the scheduling policy read the bag, not server_args.
            get_context().override("role_switch.flip", disaggregation_mode=new_role)
            scheduler.init_disaggregation()
        except Exception as e:
            scheduler._pd_role_switch_unhealthy = True
            logger.critical(
                "PD role switch (%s -> %s) failed during teardown/rebuild; "
                "instance unhealthy: %s",
                old_role,
                new_role,
                e,
            )
            return _fail(
                f"role switch failed; instance unhealthy, restart required: {e}"
            )

        # Break out of the old-role event loop so the supervisor re-dispatches.
        scheduler._event_loop_should_restart = True
        logger.info("PD role switch succeeded: %s -> %s", old_role, new_role)
        return PdRoleSwitchReqOutput(
            success=True, message="ok", old_role=old_role, new_role=new_role
        )
    except Exception as e:
        logger.exception("PD role switch failed")
        return _fail(f"role switch raised: {e}")
    finally:
        scheduler._pd_role_switch_in_progress = False


def _capture_decode_cuda_graphs(
    scheduler: Scheduler, capture_bs: Optional[Sequence[int]]
) -> None:
    """Capture the decode graphs for the target role, or decode eagerly.

    A prefill instance launched with the decode graph disabled is the normal PD
    launch, and its bucket ladder can filter down to empty -- which the capture
    treats as fatal. Refusing the flip there makes the feature unusable, so fall
    back to eager. The runner edits the decode phase config in place before it
    can fail, so restore a pre-call copy of the whole phase; its server_args
    stamp needs no undo, no live reader consults it.
    """
    cfg = scheduler.server_args.cuda_graph_config
    decode_cfg = None if cfg is None else dataclasses.replace(cfg.decode)
    try:
        scheduler.tp_worker.ensure_decode_cuda_graphs(capture_bs)
    except Exception as e:
        if cfg is not None:
            cfg.decode = decode_cfg
        logger.warning(
            "PD role switch: decode CUDA graph capture failed (%s); "
            "the flipped instance will decode eagerly",
            e,
        )


class RoleTargets(msgspec.Struct, frozen=True):
    max_running_requests: int
    chunked_prefill_size: Optional[int]
    # Per-rank a2a dispatch capacity; None means this rank could not derive one.
    dispatch_tokens: Optional[int]
    moe_max_input_tokens: Optional[str]
    disable_radix_cache: bool


def launch_prefill_cache_class(
    *, disaggregation_mode: str, disable_radix_cache: bool
) -> bool:
    """The prefix-cache class a flip to prefill restores, settled at launch.

    A decode launch has been forced to chunk cache before the scheduler reads
    the flag, so it holds no prefill answer -- and the operator could not have
    given one, the forcing ignores `--disable-radix-cache` as well. Replaying it
    would serve prefill with no prefix reuse at all, so use the prefill default.
    """
    return False if disaggregation_mode == "decode" else disable_radix_cache


def reconcile_role_config(
    scheduler: Scheduler, new_role: str, recv_req: PdRoleSwitchReqInput
) -> None:
    """Re-derive the role-dependent runtime config and rebuild the a2a buffers.

    The a2a buffer must be sized from the SETTLED cap and chunk size, or it
    gets sized for the role being left; putting every write after the resize
    is what makes a failed flip leave nothing half-applied. The prefix cache is
    rebuilt later, in teardown, from the class this settles here.
    """
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriA2AGroupDead,
        rebuild_mori_dispatch_buffers,
    )

    targets = _derive_targets(scheduler, new_role, recv_req)
    try:
        rebuild_mori_dispatch_buffers(targets.dispatch_tokens, new_role)
    except MoriA2AGroupDead:
        # Not a config write, so all-or-nothing still holds: the buffers are
        # gone group-wide and the instance cannot serve the old role either.
        scheduler._pd_role_switch_unhealthy = True
        raise
    _commit_targets(scheduler, targets)


def _derive_targets(
    scheduler: Scheduler, new_role: str, recv_req: PdRoleSwitchReqInput
) -> RoleTargets:
    sa = scheduler.server_args
    # Against the LAUNCH ceiling, not the current value: re-reading a cap the
    # previous flip already lowered turns the clamp into a one-way ratchet.
    cap = min(
        recv_req.max_running_requests or scheduler._pd_role_switch_launch_cap,
        scheduler._pd_role_switch_launch_cap,
    )
    chunk = recv_req.chunked_prefill_size or get_schedule().chunked_prefill_size
    if new_role == "prefill":
        # A dynamic chunker may predict a bigger chunk; size to its ceiling.
        dispatch_tokens = (
            chunk
            if chunk and chunk > 0 and not scheduler.enable_dynamic_chunking
            else sa.max_prefill_tokens
        )
    else:
        padded = _pad_to_captured_bucket(
            cap, scheduler.tp_worker.get_decode_cuda_graph_bs()
        )
        dispatch_tokens = (
            None if padded is None else padded * (sa.speculative_num_draft_tokens or 1)
        )
    return RoleTargets(
        max_running_requests=cap,
        chunked_prefill_size=chunk,
        dispatch_tokens=dispatch_tokens,
        moe_max_input_tokens=os.environ.get(_MOE_MAX_INPUT_TOKENS_BY_ROLE[new_role]),
        disable_radix_cache=(
            not sa.disaggregation_decode_enable_radix_cache
            if new_role == "decode"
            # A decode-launched instance had the operator's own flag overwritten
            # by the decode forcing; the launch value is what it always served.
            else scheduler._pd_role_switch_launch_disable_radix_cache
        ),
    )


def _pad_to_captured_bucket(
    batch_size: int, capture_bs: Sequence[int]
) -> Optional[int]:
    """The batch the a2a actually sees: a CUDA-graph replay pads up to the
    smallest captured bucket first. `capture_bs` must be the runner's LIVE
    list -- the declared ladder is filtered for alignment, and dropping a
    bucket promotes the pad to the next one UP. mori has no runtime bounds
    check, so guessing low is a silent out-of-bounds write.
    """
    buckets = [b for b in capture_bs if b >= batch_size]
    if not buckets:
        # No graphs captured (genuinely eager) or the cap exceeds every bucket.
        return batch_size if not capture_bs else None
    return min(buckets)


def _commit_targets(scheduler: Scheduler, targets: RoleTargets) -> None:
    scheduler.max_running_requests = targets.max_running_requests
    # Both capture the cap as a field instead of reading it off the scheduler,
    # so a flip must restamp them. The inquirer is frozen and nothing else holds
    # a reference, so rebind a copy. PrefillAdder is rebuilt per batch already.
    scheduler.load_inquirer = dataclasses.replace(
        scheduler.load_inquirer, max_running_requests=targets.max_running_requests
    )
    scheduler.kv_events_publisher.max_running_requests = targets.max_running_requests
    # The scheduler field is a snapshot of the bag, normalized as it is at startup.
    chunk = targets.chunked_prefill_size
    get_context().override("pd_role_switch.reconcile", chunked_prefill_size=chunk)
    scheduler.chunked_prefill_size = chunk if chunk and chunk > 0 else None
    if targets.moe_max_input_tokens is None:
        os.environ.pop(_MOE_MAX_INPUT_TOKENS, None)
    else:
        os.environ[_MOE_MAX_INPUT_TOKENS] = targets.moe_max_input_tokens
    # Read back by the cache rebuild in teardown, which runs after this. Bare
    # assignment raises: server_args is read-only once the config is resolved.
    scheduler.server_args.override(
        "pd_role_switch.reconcile", disable_radix_cache=targets.disable_radix_cache
    )


_MOE_MAX_INPUT_TOKENS = "SGLANG_MORI_MOE_MAX_INPUT_TOKENS"
# The launcher exports one per role; the reconcile picks the target role's.
_MOE_MAX_INPUT_TOKENS_BY_ROLE = {
    "prefill": "MORI_MOE_MAX_INPUT_TOKENS_PREFILL",
    "decode": "MORI_MOE_MAX_INPUT_TOKENS_DECODE",
}


def _reject_reason(scheduler: Scheduler, new_role: str) -> Optional[str]:
    """Why the switch must be rejected before draining, or None to proceed.

    Table-driven: the first failing precondition's message is returned.
    """
    sa = scheduler.server_args
    km = _current_kv_manager(scheduler)
    # (failed?, lazy message). Messages are callables so only the selected one
    # is built (avoids touching fields irrelevant to the failing check).
    checks = (
        (
            not sa.enable_pd_role_switch,
            lambda: "--enable-pd-role-switch is not set on this instance",
        ),
        (
            scheduler._pd_role_switch_unhealthy,
            lambda: "instance is unhealthy after a failed role switch; restart required",
        ),
        (
            scheduler._pd_role_switch_in_progress,
            lambda: "another role switch is already in progress",
        ),
        (
            new_role not in ("prefill", "decode"),
            lambda: f"invalid new_role={new_role!r}",
        ),
        (
            scheduler.disaggregation_mode == DisaggregationMode.NULL,
            lambda: "instance is not running in PD disaggregation mode",
        ),
        (
            km is not None and not km.supports_role_switch,
            lambda: f"transfer backend {sa.disaggregation_transfer_backend!r} "
            "does not support runtime role switch",
        ),
    )
    return next((msg() for failed, msg in checks if failed), None)


def _current_kv_manager(scheduler: Scheduler):
    """The KV manager of the current role's disaggregation queue, or None."""
    if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
        q = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
    elif scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        q = getattr(scheduler, "disagg_decode_prealloc_queue", None)
    else:
        q = None
    return getattr(q, "kv_manager", None) if q is not None else None


def teardown_disaggregation(scheduler: Scheduler) -> None:
    """Release the current role's disaggregation structures (queues, metadata
    buffers, KV transfer manager) so the other role can be rebuilt."""
    mode = scheduler.disaggregation_mode
    if mode == DisaggregationMode.PREFILL:
        q = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
        if q is not None:
            km = getattr(q, "kv_manager", None)
            if km is not None:
                km.teardown()
            scheduler.disagg_prefill_bootstrap_queue = None
        scheduler.disagg_prefill_inflight_queue = []
    elif mode == DisaggregationMode.DECODE:
        q = getattr(scheduler, "disagg_decode_prealloc_queue", None)
        if q is not None:
            km = getattr(q, "kv_manager", None)
            if km is not None:
                km.teardown()
            scheduler.disagg_decode_prealloc_queue = None
        scheduler.disagg_decode_transfer_queue = None
    scheduler.disagg_metadata_buffers = None
    scheduler.req_to_metadata_buffer_idx_allocator = None
    _rebuild_prefix_cache_for_role(scheduler)


def _rebuild_prefix_cache_for_role(scheduler: Scheduler) -> None:
    """Release the prefix cache and rebuild it in the new role's class.

    Release is not optional with radix (or hicache) on: finished prefixes keep
    their KV-pool slots *locked* even while idle, so a carried-over tree both
    matches the new role against stale prefixes and leaks those slots on every
    flip. Rebuilding rather than resetting is what makes the class role-correct
    -- radix vs chunk is decided per role. A longest-prefix policy is not valid
    against a chunk cache, so the schedule policy is re-derived too.
    """
    tree_cache = scheduler.tree_cache
    if tree_cache is not None and not scheduler.disable_radix_cache:
        clear_storage = getattr(tree_cache, "clear_storage_backend", None)
        if callable(clear_storage):
            try:
                clear_storage()
            except Exception:
                logger.exception("hicache storage release on role switch failed")
        tree_cache.reset()
    scheduler.req_to_token_pool.clear()
    scheduler.token_to_kv_pool_allocator.clear()
    scheduler.init_kv_cache_and_memory_pool()
    scheduler.init_schedule_policy()
    scheduler.session_controller = SessionController(scheduler.tree_cache)
