import argparse
import concurrent.futures
import dataclasses
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt import runtime_context  # noqa: E402
from sglang.srt.disaggregation import role_switch  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.layers.moe.token_dispatcher import moriep  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    PdRoleSwitchReqInput,
    PdRoleSwitchReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.kv_events_publisher import (  # noqa: E402
    SchedulerKvEventsPublisher,
)
from sglang.srt.managers.scheduler_components.load_inquirer import (  # noqa: E402
    SchedulerLoadInquirer,
)
from sglang.srt.model_executor.cuda_graph_config import (  # noqa: E402
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestPdRoleSwitchServerArg(unittest.TestCase):
    def test_cli_flag_parses(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        off = parser.parse_args(["--model-path", "dummy"])
        self.assertFalse(off.enable_pd_role_switch)

        on = parser.parse_args(["--model-path", "dummy", "--enable-pd-role-switch"])
        self.assertTrue(on.enable_pd_role_switch)


def setUpModule():
    # The flip writes resolved config onto the bags, which need a published context.
    runtime_context.get_context().set_server_args(ServerArgs(model_path="dummy"))


def tearDownModule():
    runtime_context.reset_context()


class _FrozenServerArgs(SimpleNamespace):
    """Enforces the real contract: once the config is resolved server_args is
    read-only and every write goes through override(). A permissive stand-in
    lets a bare assignment pass here and raise on the first live flip."""

    _frozen = False

    def __setattr__(self, name, value):
        if self._frozen:
            raise AttributeError(f"server_args.{name} assigned after resolution")
        super().__setattr__(name, value)

    def override(self, source, **fields):
        for name, value in fields.items():
            super().__setattr__(name, value)


def _make_scheduler(mode, *, enable=True, idle=True):
    s = Scheduler.__new__(Scheduler)
    s.disaggregation_mode = mode
    sa = _FrozenServerArgs(
        enable_pd_role_switch=enable,
        disaggregation_mode=mode.value,
    )
    s.server_args = sa
    s.is_fully_idle = MagicMock(return_value=idle)
    s._teardown_disaggregation = MagicMock()
    s.init_disaggregation = MagicMock()
    s._event_loop_should_restart = False
    s._pd_role_switch_in_progress = False
    s._pd_role_switch_unhealthy = False
    s.tp_worker = MagicMock()
    s.tp_worker.get_decode_cuda_graph_bs.return_value = [64, 128, 256]
    # As a prefill instance is launched: decode graphs off until the first flip.
    sa.cuda_graph_config = CudaGraphConfig(decode=PhaseConfig(backend=Backend.DISABLED))
    sa.max_prefill_tokens = 16384
    sa.speculative_num_draft_tokens = None
    s.max_running_requests = 128
    s._pd_role_switch_launch_cap = 128
    # Launched with radix on, as a prefill instance is.
    sa.disaggregation_decode_enable_radix_cache = False
    sa.disable_radix_cache = False
    s._pd_role_switch_prefill_disable_radix_cache = False
    s.chunked_prefill_size = 8192
    # The bag outlives a single test, so restate it rather than inherit it.
    runtime_context.get_context().override(
        "test", chunked_prefill_size=8192, disaggregation_mode=mode.value
    )
    sa._frozen = True
    s.enable_dynamic_chunking = False
    # Real classes, not stand-ins: the inquirer is frozen and the publisher is
    # not, and a commit that conflates them raises only against the real thing.
    for holder, cls in (
        ("load_inquirer", SchedulerLoadInquirer),
        ("kv_events_publisher", SchedulerKvEventsPublisher),
    ):
        fields = {f.name: MagicMock() for f in dataclasses.fields(cls) if f.init}
        fields["max_running_requests"] = 128
        if "kv_events_config" in fields:
            fields["kv_events_config"] = None  # keeps __post_init__ inert
        setattr(s, holder, cls(**fields))
    return s


class TestHandlePdRoleSwitch(unittest.TestCase):
    """Cover the control-plane contract of Scheduler.handle_pd_role_switch.

    Only the role-flip *decision* logic is exercised here (no GPU): the heavy
    teardown/rebuild is mocked, so this asserts the guard branches and the
    orchestration order without standing up a model.
    """

    _scheduler = staticmethod(_make_scheduler)

    def test_rejected_when_flag_disabled(self):
        s = self._scheduler(DisaggregationMode.PREFILL, enable=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertIsInstance(out, PdRoleSwitchReqOutput)
        self.assertFalse(out.success)
        self.assertIn("enable-pd-role-switch", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_rejected_on_invalid_role(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(s, PdRoleSwitchReqInput(new_role="both"))
        self.assertFalse(out.success)
        self.assertIn("invalid new_role", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_rejected_when_not_in_pd_mode(self):
        s = self._scheduler(DisaggregationMode.NULL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("not running in PD", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_same_role_is_noop(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill")
        )
        self.assertTrue(out.success)
        self.assertEqual(out.message, "already in target role")
        s._teardown_disaggregation.assert_not_called()
        s.init_disaggregation.assert_not_called()
        self.assertFalse(s._event_loop_should_restart)

    def test_rejected_when_not_idle(self):
        s = self._scheduler(DisaggregationMode.PREFILL, idle=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("not idle", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_successful_flip_orchestration(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        self.assertTrue(out.success)
        self.assertEqual(out.old_role, "prefill")
        self.assertEqual(out.new_role, "decode")
        # Orchestration: drain -> teardown -> flip server arg -> rebuild -> signal.
        s._teardown_disaggregation.assert_called_once()
        self.assertEqual(s.server_args.disaggregation_mode, "decode")
        # init_disaggregation re-reads the role off the bag, so a flip that only
        # writes server_args rebuilds the role it was supposed to leave.
        self.assertEqual(runtime_context.get_disagg().disaggregation_mode, "decode")
        s.init_disaggregation.assert_called_once()
        self.assertTrue(s._event_loop_should_restart)
        # Flip to decode ensures decode CUDA graphs exist (idempotent capture).
        s.tp_worker.ensure_decode_cuda_graphs.assert_called_once()
        # The in-progress guard is released after a successful flip.
        self.assertFalse(s._pd_role_switch_in_progress)

    def test_flip_to_prefill_skips_decode_graph_capture(self):
        s = self._scheduler(DisaggregationMode.DECODE)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill")
        )
        self.assertTrue(out.success)
        self.assertEqual(out.new_role, "prefill")
        s.init_disaggregation.assert_called_once()
        # Flipping to prefill must not capture decode graphs.
        s.tp_worker.ensure_decode_cuda_graphs.assert_not_called()
        self.assertTrue(s._event_loop_should_restart)

    def test_rejected_when_switch_in_progress(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s._pd_role_switch_in_progress = True
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("in progress", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_rejected_when_unhealthy(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s._pd_role_switch_unhealthy = True
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("unhealthy", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_rebuild_failure_marks_unhealthy_and_notifies(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        # Rebuild of the new role fails after the old role was torn down.
        s.init_disaggregation = MagicMock(side_effect=RuntimeError("boom"))

        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        # Fail loud (notify), mark unhealthy, no in-place rollback attempt.
        self.assertFalse(out.success)
        self.assertIn("unhealthy", out.message)
        self.assertIn("restart", out.message)
        self.assertTrue(s._pd_role_switch_unhealthy)
        self.assertFalse(s._event_loop_should_restart)
        self.assertFalse(s._pd_role_switch_in_progress)
        # Teardown + rebuild attempted exactly once (no rollback).
        self.assertEqual(s._teardown_disaggregation.call_count, 1)
        self.assertEqual(s.init_disaggregation.call_count, 1)
        # A subsequent switch is rejected because the instance is unhealthy.
        out2 = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill")
        )
        self.assertFalse(out2.success)
        self.assertIn("unhealthy", out2.message)

    def test_teardown_failure_marks_unhealthy(self):
        """Teardown, the role flip and rebuild are one atomic step: a failure
        during teardown (not only rebuild) must also mark the instance unhealthy
        and must not proceed to rebuild."""
        s = self._scheduler(DisaggregationMode.PREFILL)
        s._teardown_disaggregation = MagicMock(side_effect=RuntimeError("boom"))

        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        self.assertFalse(out.success)
        self.assertIn("unhealthy", out.message)
        self.assertIn("restart", out.message)
        self.assertTrue(s._pd_role_switch_unhealthy)
        self.assertFalse(s._event_loop_should_restart)
        self.assertFalse(s._pd_role_switch_in_progress)
        # Teardown raised, so rebuild is never attempted.
        self.assertEqual(s._teardown_disaggregation.call_count, 1)
        s.init_disaggregation.assert_not_called()


class TestRoleConfigReconcile(unittest.TestCase):
    """The reconcile that re-derives the role-dependent config and resizes the
    mori a2a buffer, driven through the real handler."""

    _scheduler = staticmethod(_make_scheduler)

    def test_a2a_sized_from_settled_chunk_not_the_current_one(self):
        """Constraint 2: settle the cap and chunk size first, then size the a2a
        buffer from the settled values. Sizing from the live scheduler fields
        builds the buffer for the role being LEFT."""
        s = self._scheduler(DisaggregationMode.DECODE)
        with patch.object(moriep, "rebuild_mori_dispatch_buffers") as resize:
            out = Scheduler.handle_pd_role_switch(
                s,
                PdRoleSwitchReqInput(new_role="prefill", chunked_prefill_size=2048),
            )
        self.assertTrue(out.success)
        resize.assert_called_once_with(2048, "prefill")
        self.assertEqual(s.chunked_prefill_size, 2048)
        self.assertEqual(runtime_context.get_schedule().chunked_prefill_size, 2048)
        # A flip that does not restate the chunk must not resurrect the launch value.
        s.disaggregation_mode = DisaggregationMode.PREFILL
        with patch.object(moriep, "rebuild_mori_dispatch_buffers"):
            Scheduler.handle_pd_role_switch(s, PdRoleSwitchReqInput(new_role="decode"))
        self.assertEqual(runtime_context.get_schedule().chunked_prefill_size, 2048)

    def test_decode_sizes_for_the_padded_batch_from_the_live_runner(self):
        """Constraint 4: a CUDA-graph replay pads the batch up to the smallest
        captured bucket before the a2a sees it, and the bucket ladder must come
        from the live runner (the declared one is filtered, which pads UP)."""
        s = self._scheduler(DisaggregationMode.PREFILL)
        s._pd_role_switch_launch_cap = 129
        with patch.object(moriep, "rebuild_mori_dispatch_buffers") as resize:
            Scheduler.handle_pd_role_switch(s, PdRoleSwitchReqInput(new_role="decode"))
        resize.assert_called_once_with(256, "decode")

    def test_failed_resize_applies_nothing(self):
        """Constraint 3: the reconcile is all-or-nothing. Every write lands
        after the only fallible step, so a refused resize leaves a prefill
        instance carrying prefill's cap, chunk size and MOE_MAX_INPUT_TOKENS."""
        s = self._scheduler(DisaggregationMode.PREFILL)
        env = {
            "MORI_MOE_MAX_INPUT_TOKENS_DECODE": "2703",
            "SGLANG_MORI_MOE_MAX_INPUT_TOKENS": "32768",
        }
        with patch.dict("os.environ", env), patch.object(
            moriep,
            "rebuild_mori_dispatch_buffers",
            side_effect=moriep.MoriA2AResizeError("nope"),
        ):
            out = Scheduler.handle_pd_role_switch(
                s, PdRoleSwitchReqInput(new_role="decode", max_running_requests=64)
            )
            self.assertEqual(os.environ["SGLANG_MORI_MOE_MAX_INPUT_TOKENS"], "32768")
        self.assertFalse(out.success)
        self.assertIn("nothing applied", out.message)
        self.assertEqual(s.max_running_requests, 128)
        self.assertEqual(s.load_inquirer.max_running_requests, 128)
        s._teardown_disaggregation.assert_not_called()

    def test_dead_a2a_group_is_not_reported_as_a_recoverable_refusal(self):
        """A dead group is the one resize failure the old role cannot survive:
        the instance must go unhealthy rather than resume serving."""
        s = self._scheduler(DisaggregationMode.PREFILL)
        with patch.object(
            moriep,
            "rebuild_mori_dispatch_buffers",
            side_effect=moriep.MoriA2AGroupDead("gone"),
        ):
            out = Scheduler.handle_pd_role_switch(
                s, PdRoleSwitchReqInput(new_role="decode")
            )
        self.assertFalse(out.success)
        self.assertIn("restart required", out.message)
        self.assertTrue(s._pd_role_switch_unhealthy)
        self.assertEqual(s.max_running_requests, 128)

    def test_cache_class_follows_the_role_and_the_flip_back_restores_it(self):
        """A decode server is forced to chunk cache, so the class the rebuild
        reads must be re-derived per role: settling it once at launch leaves a
        flipped instance serving the class it was launched with."""
        s = self._scheduler(DisaggregationMode.PREFILL)
        with patch.object(moriep, "rebuild_mori_dispatch_buffers"):
            Scheduler.handle_pd_role_switch(s, PdRoleSwitchReqInput(new_role="decode"))
            self.assertTrue(s.server_args.disable_radix_cache)
            s.disaggregation_mode = DisaggregationMode.DECODE
            Scheduler.handle_pd_role_switch(s, PdRoleSwitchReqInput(new_role="prefill"))
        self.assertFalse(s.server_args.disable_radix_cache)

    def test_a_decode_launch_does_not_carry_its_forcing_into_prefill(self):
        """The launch capture is what a flip to prefill restores. A decode
        launch has had the flag overwritten by the chunk-cache forcing, so
        replaying it would serve prefill with no prefix reuse at all."""
        self.assertFalse(
            role_switch.launch_prefill_cache_class(
                disaggregation_mode="decode", disable_radix_cache=True
            )
        )
        self.assertTrue(
            role_switch.launch_prefill_cache_class(
                disaggregation_mode="prefill", disable_radix_cache=True
            )
        )

    def test_cap_clamp_does_not_ratchet_across_flips(self):
        """The cap may only be lowered relative to the LAUNCH ceiling. Comparing
        against the live value instead makes each flip lower it again, so a
        flip-back refuses the pool size it was actually launched with."""
        s = self._scheduler(DisaggregationMode.PREFILL)
        with patch.object(moriep, "rebuild_mori_dispatch_buffers"):
            Scheduler.handle_pd_role_switch(
                s, PdRoleSwitchReqInput(new_role="decode", max_running_requests=32)
            )
            self.assertEqual(s.max_running_requests, 32)
            s.disaggregation_mode = DisaggregationMode.DECODE
            Scheduler.handle_pd_role_switch(s, PdRoleSwitchReqInput(new_role="prefill"))
        self.assertEqual(s.max_running_requests, 128)


class TestMoriA2AResize(unittest.TestCase):
    """The EP-group side of the a2a rebuild."""

    def setUp(self):
        self.op = MagicMock(spec=["reconfigure"])
        moriep._LIVE_OPS[:] = [
            moriep._LiveOp(op=self.op, group=MagicMock(), rank=0, capacity=4096)
        ]
        self.addCleanup(moriep._LIVE_OPS.clear)

    def _resize(self, target, peer_votes, **env):
        """Run a rebuild with the peers' votes folded into the reduction."""

        def all_reduce(t, op=None, group=None):
            votes = [int(t.item())] + list(peer_votes)
            pick = min if op is moriep.torch.distributed.ReduceOp.MIN else max
            t.fill_(pick(votes))

        with patch.dict("os.environ", env), patch.object(
            moriep.torch.distributed, "all_reduce", side_effect=all_reduce
        ) as reduce:
            return moriep.rebuild_mori_dispatch_buffers(target, "decode"), reduce

    def test_underivable_target_is_a_vote_not_an_early_return(self):
        """Constraint 1: a rank that derives no target must still enter the
        collective. Returning early leaves its peers in a reduction nobody
        joins, which hangs the group."""
        result, reduce = self._resize(None, [512])
        self.assertIsNone(result)
        self.assertTrue(reduce.called)
        self.op.reconfigure.assert_not_called()

    def test_disagreeing_ranks_are_refused_not_resized(self):
        with self.assertRaises(moriep.MoriA2AResizeError):
            self._resize(256, [512])

    def test_target_above_the_process_ceiling_is_refused(self):
        """Constraint 5: SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK is a
        per-process allocation ceiling, not a role's size — an instance launched
        below max(prefill, decode) can never flip out of its launch role."""
        env = {"SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512"}
        with self.assertRaisesRegex(moriep.MoriA2AResizeError, "flip-capable"):
            self._resize(4096, [4096], **env)

    def test_capacity_is_tracked_not_read_back_off_the_op(self):
        """Constraint 6: a failed resize may leave the op's buffer pointers
        freed, and reading them back through pybind kills the rank before the
        error can cross. `spec=["reconfigure"]` fails if anything else is read."""
        (old, new), _ = self._resize(512, [512])
        self.assertEqual((old, new), (4096, 512))
        # Must be the gloo group: the default (nccl) PG raises mid-resize.
        self.op.reconfigure.assert_called_once_with(
            512, group=moriep._LIVE_OPS[0].group.cpu_group
        )
        self.assertEqual(moriep._LIVE_OPS[0].capacity, 512)

    def test_fatal_resize_is_distinguished_from_a_clean_refusal(self):
        """mori's ResizeFatal means at least one rank holds no buffers, so the
        OLD role cannot dispatch either. Reported as an ordinary refusal, the
        instance would keep serving into a dead op."""
        self.op.reconfigure.side_effect = type("ResizeFatal", (RuntimeError,), {})()
        with self.assertRaises(moriep.MoriA2AGroupDead):
            self._resize(512, [512])
        self.op.reconfigure.side_effect = ValueError("rejected")
        with self.assertRaises(ValueError):
            self._resize(512, [512])

    def test_refusal_after_an_earlier_op_resized_is_group_dead(self):
        """A clean refusal is only recoverable while nothing has been applied.
        `init_mori_op` is cached per config, so a process can hold more than one
        op; reported as an ordinary refusal, the caller would resume the old
        role over the op that did resize."""
        second = MagicMock(spec=["reconfigure"])
        second.reconfigure.side_effect = ValueError("rejected")
        moriep._LIVE_OPS.append(
            moriep._LiveOp(op=second, group=MagicMock(), rank=0, capacity=4096)
        )
        with self.assertRaises(moriep.MoriA2AGroupDead):
            self._resize(512, [512])
        self.assertEqual(moriep._LIVE_OPS[0].capacity, 512)


class TestPdRoleSwitchReqSerialization(unittest.TestCase):
    """Guard the wire contract of the /pd_role_switch req/resp structs.

    These caught real breakages when upstream moved BaseReq to msgspec: the
    request must accept an optional decode_cuda_graph_bs body field, and the
    response must be encodable for the HTTP layer (msgspec_to_builtins).
    """

    def test_req_accepts_optional_decode_cuda_graph_bs(self):
        req = PdRoleSwitchReqInput(new_role="decode", decode_cuda_graph_bs=[1, 2, 4])
        self.assertEqual(req.new_role, "decode")
        self.assertEqual(req.decode_cuda_graph_bs, [1, 2, 4])
        # Field is optional and defaults to None.
        self.assertIsNone(PdRoleSwitchReqInput(new_role="prefill").decode_cuda_graph_bs)

    def test_resp_is_json_encodable(self):
        from sglang.srt.utils.msgspec_utils import msgspec_to_builtins

        out = PdRoleSwitchReqOutput(
            success=True, message="ok", old_role="prefill", new_role="decode"
        )
        d = msgspec_to_builtins(out)
        self.assertEqual(d["success"], True)
        self.assertEqual(d["old_role"], "prefill")
        self.assertEqual(d["new_role"], "decode")
        self.assertEqual(d["message"], "ok")


class TestPdRoleSwitchStartupValidation(unittest.TestCase):
    """The startup guard admits pure TP, and -- only under the experimental
    gate -- expert parallelism over the mori a2a. Every other per-role buffer
    is sized once at startup, so a flip with it on would silently deadlock.

    The matrix below is the acceptance oracle: each row is checked both with
    and without the gate flag. Sizes reflect the RESOLVED view, which is why
    the guard runs after the resolution pipeline (mori forces ep=tp, DWDP
    forces dp attention and dp_size).
    """

    # (name, resolved-config overrides, accepted-without-gate, accepted-with-gate)
    MATRIX = (
        ("pure TP", {}, True, True),
        ("EP, no a2a", dict(ep_size=4), False, False),
        ("EP + mori a2a", dict(ep_size=4, moe_a2a_backend="mori"), False, True),
        (
            "inter-node EP + mori a2a",
            dict(ep_size=16, moe_a2a_backend="mori"),
            False,
            False,
        ),
        ("DP attention", dict(enable_dp_attention=True, dp_size=2), False, False),
        ("system DP", dict(dp_size=2), False, False),
        ("PP", dict(pp_size=2), False, False),
        ("EP + deepep a2a", dict(ep_size=4, moe_a2a_backend="deepep"), False, False),
        (
            "decode KV offload",
            dict(disaggregation_decode_enable_offload_kvcache=True),
            False,
            False,
        ),
        (
            "EP + mori a2a + prefill delayer",
            dict(ep_size=4, moe_a2a_backend="mori", enable_prefill_delayer=True),
            False,
            False,
        ),
        (
            "DWDP",
            dict(enable_dp_attention=True, ep_size=4, dp_size=4),
            False,
            False,
        ),
        (
            "EP + mori a2a, no role switch",
            dict(ep_size=4, moe_a2a_backend="mori", enable_pd_role_switch=False),
            True,
            True,
        ),
    )

    def _sa(self, **kw):
        base = dict(
            disaggregation_mode="prefill",
            enable_pd_role_switch=True,
            enable_pd_role_switch_experimental_moe=False,
            enable_dp_attention=False,
            ep_size=1,
            moe_a2a_backend="none",
            pp_size=1,
            dp_size=1,
            disaggregation_decode_enable_offload_kvcache=False,
            enable_prefill_delayer=False,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def _run(self, sa):
        from sglang.srt.arg_groups.pd_disaggregation_hook import (
            check_pd_role_switch_support,
        )

        check_pd_role_switch_support(sa)

    def test_gate_matrix(self):
        for name, cfg, ok_plain, ok_gated in self.MATRIX:
            for gate, expected in ((False, ok_plain), (True, ok_gated)):
                with self.subTest(row=name, gate=gate):
                    sa = self._sa(enable_pd_role_switch_experimental_moe=gate, **cfg)
                    if expected:
                        self._run(sa)
                    else:
                        with self.assertRaises(ValueError):
                            self._run(sa)

    def test_reject_names_every_unsupported_feature(self):
        sa = self._sa(enable_dp_attention=True, ep_size=4, dp_size=4)
        with self.assertRaises(ValueError) as ctx:
            self._run(sa)
        for clause in ("DP attention", "expert parallelism", "data parallelism"):
            self.assertIn(clause, str(ctx.exception))

    def test_guard_runs_after_parallelism_is_resolved(self):
        """DWDP forces dp attention and dp_size long after the PD arg hook has
        run, so the guard must not live in that hook."""
        order = ServerArgs.__post_init__.__code__.co_names
        self.assertGreater(
            order.index("_check_pd_role_switch_support"), order.index("_handle_dwdp")
        )


import threading  # noqa: E402
import time  # noqa: E402

import zmq  # noqa: E402

try:
    from sglang.srt.disaggregation.common.utils import FastQueue  # noqa: E402
    from sglang.srt.disaggregation.mori.conn import MoriKVManager  # noqa: E402

    _HAS_MORI = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_MORI = False

try:
    from sglang.srt.disaggregation.common.utils import (  # noqa: E402,F811
        FastQueue as _FQ,
    )
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager  # noqa: E402

    _HAS_MOONCAKE = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_MOONCAKE = False

try:
    from sglang.srt.disaggregation.role_switch import (  # noqa: E402
        _rebuild_prefix_cache_for_role,
        teardown_disaggregation,
    )

    _HAS_ROLE_SWITCH = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_ROLE_SWITCH = False


@unittest.skipUnless(_HAS_MORI, "mori not importable in this environment")
class TestMoriTeardownNoThreadLeak(unittest.TestCase):
    """teardown() must stop+join the transfer workers it started, so a P->D->P
    flip loop does not leak _num_shards transfer threads per cycle."""

    def test_teardown_joins_transfer_workers(self):
        m = MoriKVManager.__new__(MoriKVManager)
        m.disaggregation_mode = DisaggregationMode.PREFILL
        m._stopped = False
        m._worker_threads = []
        m._transfer_queues = [FastQueue() for _ in range(3)]
        m.server_socket = MagicMock()
        m._zmq_ctx = MagicMock()
        m.engine = MagicMock()
        m.kv_mem_descs = m.aux_mem_descs = m.state_mem_descs = []
        for q in m._transfer_queues:
            t = threading.Thread(target=m._transfer_worker, args=(q,), daemon=True)
            t.start()
            m._worker_threads.append(t)
        started = list(m._worker_threads)
        time.sleep(0.05)  # let workers park in FastQueue.get()
        for t in started:
            self.assertTrue(t.is_alive())

        MoriKVManager.teardown(m)

        for t in started:
            self.assertFalse(t.is_alive(), "transfer worker survived teardown (leak)")
        self.assertEqual(m._worker_threads, [])
        self.assertEqual(m._transfer_queues, [])


@unittest.skipUnless(_HAS_MOONCAKE, "mooncake not importable in this environment")
class TestMooncakeTeardownNoThreadLeak(unittest.TestCase):
    """teardown() must stop+join the transfer workers it started, so a P->D->P
    flip loop does not leak transfer threads per cycle."""

    def test_teardown_joins_transfer_workers(self):
        m = MooncakeKVManager.__new__(MooncakeKVManager)
        m.disaggregation_mode = DisaggregationMode.PREFILL
        m._stopped = False
        m.enable_trace = False
        m._worker_threads = []
        m.transfer_queues = [_FQ() for _ in range(3)]
        m.executors = [concurrent.futures.ThreadPoolExecutor(1) for _ in range(3)]
        m.server_socket = MagicMock()
        m._zmq_ctx = MagicMock()
        m._socket_lock = threading.Lock()
        m._socket_cache = {}
        m._monitor_cache = {}
        m.engine = MagicMock()
        m.kv_args = SimpleNamespace(
            kv_data_ptrs=[], aux_data_ptrs=[], state_data_ptrs=[]
        )
        for i, (q, ex) in enumerate(zip(m.transfer_queues, m.executors)):
            t = threading.Thread(
                target=m.transfer_worker, args=(q, ex, None, i), daemon=True
            )
            t.start()
            m._worker_threads.append(t)
        started = list(m._worker_threads)
        time.sleep(0.05)  # let workers park in FastQueue.get()
        for t in started:
            self.assertTrue(t.is_alive())

        MooncakeKVManager.teardown(m)

        for t in started:
            self.assertFalse(t.is_alive(), "transfer worker survived teardown (leak)")
        self.assertEqual(m._worker_threads, [])
        self.assertEqual(m.transfer_queues, [])
        self.assertEqual(m.executors, [])


@unittest.skipUnless(_HAS_MOONCAKE, "mooncake not importable in this environment")
class TestMooncakeBootstrapThreadRobustness(unittest.TestCase):
    """The prefill bootstrap loop moved from a blocking recv_multipart() to a
    500ms poll + _stopped check (so teardown, i.e. a runtime role switch, can
    stop it). That loop runs on every mooncake PD instance, so pin the
    contract with real ZMQ traffic driven through the ABORT -> ABORT_ACK
    path: no message loss while idle or bursting, and prompt exit once
    _stopped is set. Unlike mori, the loop has no try/except around recv: a
    recv error terminates the thread (see test_recv_error_kills_thread).
    """

    class _FlakySocket(zmq.Socket):
        """Real PULL socket whose next recv can be forced to fail once,
        emulating a transient ZMQ error between poll() and recv()."""

        fail_next_recv = False

        def recv_multipart(self, *args, **kwargs):
            if type(self).fail_next_recv:
                type(self).fail_next_recv = False
                raise RuntimeError("transient recv failure")
            return super().recv_multipart(*args, **kwargs)

    def setUp(self):
        self._FlakySocket.fail_next_recv = False
        self._ctx = zmq.Context()
        sock = self._FlakySocket(self._ctx, zmq.PULL)
        port = sock.bind_to_random_port("tcp://127.0.0.1")
        m = MooncakeKVManager.__new__(MooncakeKVManager)
        m._stopped = False
        m._worker_threads = []
        m.server_socket = sock
        # The receive path is gated on this flag: role switch must be on for
        # the poll-with-timeout loop these tests exercise.
        m.server_args = SimpleNamespace(enable_pd_role_switch=True)
        # ABORT for an unknown room takes the "ignoring" branch and still
        # ACKs, giving a side-effect-free probe of the receive loop.
        m.request_status = {}
        m._connect = MagicMock()
        self.m = m
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.connect(f"tcp://127.0.0.1:{port}")

    def tearDown(self):
        self.m._stopped = True
        for t in self.m._worker_threads:
            t.join(timeout=3.0)
        self._push.close(linger=0)
        self.m.server_socket.close(linger=0)
        self._ctx.destroy(linger=0)

    def _start(self):
        MooncakeKVManager.start_prefill_thread(self.m)
        (thread,) = self.m._worker_threads
        return thread

    def _send_abort(self, room):
        self._push.send_multipart(
            [b"ABORT", str(room).encode("ascii"), b"127.0.0.1", b"9999"]
        )

    def _wait_acks(self, n, timeout=10.0):
        send = self.m._connect.return_value.send_multipart
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if send.call_count >= n:
                return
            time.sleep(0.02)
        self.fail(f"expected {n} ABORT_ACKs, got {send.call_count}")

    def test_messages_processed_across_idle_poll_timeouts(self):
        self._start()
        self._send_abort(1)
        self._wait_acks(1)
        # Idle past a full poll timeout, then traffic must still flow: the
        # empty-poll -> continue path must not disturb the socket.
        time.sleep(0.8)
        self._send_abort(2)
        self._wait_acks(2)

    def test_no_message_loss_under_burst(self):
        self._start()
        n = 200
        for i in range(n):
            self._send_abort(i)
        # Two-step poll+recv must consume every queued message exactly once.
        self._wait_acks(n)

    def test_recv_error_kills_thread(self):
        # No try/except guards recv() in the mooncake loop (unlike mori): a
        # recv error terminates the thread and the loop stops processing.
        # Pin that contract so adding error handling stays a deliberate,
        # reviewed change rather than a silent behavior shift.
        thread = self._start()
        self._FlakySocket.fail_next_recv = True
        self._send_abort(3)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive(), "bootstrap thread survived recv error")
        self.assertFalse(self._FlakySocket.fail_next_recv)  # fault consumed

    def test_exits_promptly_when_stopped_while_idle(self):
        thread = self._start()
        self.m._stopped = True
        # Poll timeout is 500ms, so the flag must be observed within ~1 cycle
        # (this is what keeps teardown / role switch from hanging).
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive(), "bootstrap thread leaked past stop")


def _radix_scheduler(disable_radix_cache):
    s = MagicMock()
    s.disable_radix_cache = disable_radix_cache
    tree = MagicMock()
    del tree.clear_storage_backend  # plain RadixCache has none
    s.tree_cache = tree
    s.req_to_token_pool = MagicMock()
    s.token_to_kv_pool_allocator = MagicMock()
    return s


@unittest.skipUnless(_HAS_ROLE_SWITCH, "role_switch not importable in this env")
class TestPrefixCacheRebuildOnRoleSwitch(unittest.TestCase):
    """Radix vs chunk is keyed on the role, so the tree cache is rebuilt in the
    target role's class, not reset in place -- and the prefixes that keep KV
    slots locked are released first, which only radix holds."""

    def test_rebuild_rebinds_the_holders_and_releases_radix_prefixes_only(self):
        for disable_radix_cache, resets in ((True, 0), (False, 1)):
            with self.subTest(disable_radix_cache=disable_radix_cache):
                s = _radix_scheduler(disable_radix_cache)
                _rebuild_prefix_cache_for_role(s)
                self.assertEqual(s.tree_cache.reset.call_count, resets)
                s.req_to_token_pool.clear.assert_called_once_with()
                s.token_to_kv_pool_allocator.clear.assert_called_once_with()
                s.init_kv_cache_and_memory_pool.assert_called_once_with()
                s.init_schedule_policy.assert_called_once_with()
                self.assertIs(s.session_controller.tree_cache, s.tree_cache)

    def test_teardown_invokes_the_rebuild(self):
        s = _radix_scheduler(disable_radix_cache=False)
        s.disaggregation_mode = DisaggregationMode.PREFILL
        s.disagg_prefill_bootstrap_queue = None  # no queue -> skip km.teardown()
        teardown_disaggregation(s)
        s.init_kv_cache_and_memory_pool.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
