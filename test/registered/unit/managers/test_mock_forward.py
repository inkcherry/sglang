"""Unit tests for SGLANG_MOCK_FORWARD scheduler-only testing mode.

Covers:
  - Env-var defaults and override semantics
    (SGLANG_MOCK_FORWARD, SGLANG_MOCK_FORWARD_OUTPUT_LEN)
  - scheduler._apply_mock_forward_overrides validation matrix:
      * no-op when mock is off
      * raises NotImplementedError on incompatible features
        (speculative_algorithm, disaggregation_mode, enable_pdmux)
      * collects all incompat reasons into one raise message
      * allows tp_size > 1 and dp_size > 1 (spike-validated)
      * force-disables cuda_graph and overlap_schedule with a warning
  - entrypoints.http_server._print_mock_forward_banner_if_enabled:
      * no-op when off, prints frame + anti-patterns when on
  - managers.mock_forward.mock_forward_batch_generation:
      * fake_logits shape/dtype, next_token_ids shape/dtype/value,
        can_run_cuda_graph = False
  - managers.schedule_batch.Req.update_finish_state mock branch:
      * finishes by output length only, respects max_new_tokens,
      * bypasses EOS/grammar/stop_str checks
"""

import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _server_args_stub(**overrides):
    """Lightweight SimpleNamespace stub of ServerArgs for override tests."""
    defaults = dict(
        speculative_algorithm=None,
        tp_size=1,
        dp_size=1,
        disaggregation_mode="null",
        enable_pdmux=False,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestMockForwardEnvVars(CustomTestCase):
    def setUp(self):
        envs.SGLANG_MOCK_FORWARD.clear()
        envs.SGLANG_MOCK_FORWARD_OUTPUT_LEN.clear()

    def tearDown(self):
        envs.SGLANG_MOCK_FORWARD.clear()
        envs.SGLANG_MOCK_FORWARD_OUTPUT_LEN.clear()

    def test_defaults(self):
        self.assertFalse(envs.SGLANG_MOCK_FORWARD.get())
        self.assertEqual(envs.SGLANG_MOCK_FORWARD_OUTPUT_LEN.get(), 32)

    def test_override(self):
        with envs.SGLANG_MOCK_FORWARD.override(True):
            self.assertTrue(envs.SGLANG_MOCK_FORWARD.get())
        # restored after context exit
        self.assertFalse(envs.SGLANG_MOCK_FORWARD.get())

    def test_output_len_override(self):
        with envs.SGLANG_MOCK_FORWARD_OUTPUT_LEN.override(64):
            self.assertEqual(envs.SGLANG_MOCK_FORWARD_OUTPUT_LEN.get(), 64)
        self.assertEqual(envs.SGLANG_MOCK_FORWARD_OUTPUT_LEN.get(), 32)


class TestApplyMockForwardOverrides(CustomTestCase):
    """scheduler._apply_mock_forward_overrides validation matrix."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.managers.scheduler import _apply_mock_forward_overrides

        cls.fn = staticmethod(_apply_mock_forward_overrides)

    def test_noop_when_mock_off(self):
        # mock OFF + every incompat field set, args must remain untouched
        sa = _server_args_stub(
            speculative_algorithm="EAGLE",
            tp_size=8,
            dp_size=4,
            disaggregation_mode="prefill",
            enable_pdmux=True,
            disable_cuda_graph=False,
            disable_overlap_schedule=False,
        )
        before = (
            sa.disable_cuda_graph,
            sa.disable_overlap_schedule,
            sa.speculative_algorithm,
            sa.tp_size,
            sa.dp_size,
        )
        self.fn(sa)
        after = (
            sa.disable_cuda_graph,
            sa.disable_overlap_schedule,
            sa.speculative_algorithm,
            sa.tp_size,
            sa.dp_size,
        )
        self.assertEqual(before, after)

    def test_raise_on_spec_algo(self):
        with envs.SGLANG_MOCK_FORWARD.override(True):
            sa = _server_args_stub(speculative_algorithm="EAGLE")
            with self.assertRaises(NotImplementedError) as cm:
                self.fn(sa)
            self.assertIn("speculative_algorithm=EAGLE", str(cm.exception))

    def test_raise_on_disagg(self):
        with envs.SGLANG_MOCK_FORWARD.override(True):
            sa = _server_args_stub(disaggregation_mode="prefill")
            with self.assertRaises(NotImplementedError) as cm:
                self.fn(sa)
            self.assertIn("disaggregation_mode=prefill", str(cm.exception))

    def test_raise_on_pdmux(self):
        with envs.SGLANG_MOCK_FORWARD.override(True):
            sa = _server_args_stub(enable_pdmux=True)
            with self.assertRaises(NotImplementedError) as cm:
                self.fn(sa)
            self.assertIn("enable_pdmux", str(cm.exception))

    def test_collect_multiple_incompat(self):
        with envs.SGLANG_MOCK_FORWARD.override(True):
            sa = _server_args_stub(
                speculative_algorithm="EAGLE",
                enable_pdmux=True,
                disaggregation_mode="decode",
            )
            with self.assertRaises(NotImplementedError) as cm:
                self.fn(sa)
            msg = str(cm.exception)
            self.assertIn("speculative_algorithm", msg)
            self.assertIn("disaggregation_mode", msg)
            self.assertIn("enable_pdmux", msg)

    def test_allow_tp_gt_1(self):
        # Spike-validated: tp_size > 1 must not raise.
        with envs.SGLANG_MOCK_FORWARD.override(True):
            sa = _server_args_stub(tp_size=8)
            try:
                self.fn(sa)
            except NotImplementedError:
                self.fail("tp_size > 1 must be allowed under mock forward")

    def test_allow_dp_gt_1(self):
        # Spike-validated: dp_size > 1 must not raise.
        with envs.SGLANG_MOCK_FORWARD.override(True):
            sa = _server_args_stub(dp_size=8)
            try:
                self.fn(sa)
            except NotImplementedError:
                self.fail("dp_size > 1 must be allowed under mock forward")

    def test_force_disable_cuda_graph(self):
        with envs.SGLANG_MOCK_FORWARD.override(True):
            sa = _server_args_stub(disable_cuda_graph=False)
            self.fn(sa)
            self.assertTrue(sa.disable_cuda_graph)

    def test_force_disable_overlap_schedule(self):
        with envs.SGLANG_MOCK_FORWARD.override(True):
            sa = _server_args_stub(disable_overlap_schedule=False)
            self.fn(sa)
            self.assertTrue(sa.disable_overlap_schedule)


class TestMockForwardBanner(CustomTestCase):
    """_print_mock_forward_banner_if_enabled stderr behavior."""

    def _capture(self, fn):
        captured = io.StringIO()
        real = sys.stderr
        sys.stderr = captured
        try:
            fn()
        finally:
            sys.stderr = real
        return captured.getvalue()

    def test_banner_noop_when_off(self):
        from sglang.srt.entrypoints.http_server import (
            _print_mock_forward_banner_if_enabled,
        )

        out = self._capture(_print_mock_forward_banner_if_enabled)
        self.assertEqual(out, "")

    def test_banner_prints_when_on(self):
        from sglang.srt.entrypoints.http_server import (
            _print_mock_forward_banner_if_enabled,
        )

        with envs.SGLANG_MOCK_FORWARD.override(True):
            out = self._capture(_print_mock_forward_banner_if_enabled)
        # frame, headline, scope, and at least one anti-pattern must appear
        self.assertIn("=" * 78, out)
        self.assertIn("SGLANG_MOCK_FORWARD ENABLED", out)
        self.assertIn("ALL GENERATED TOKENS ARE FAKE", out)
        self.assertIn("SCHEDULER-PATH TESTING ONLY", out)
        self.assertIn("PR performance comparisons", out)


class TestMockForwardBatchGeneration(CustomTestCase):
    """mock_forward_batch_generation output shape / dtype / values."""

    def _call(self, batch_size, vocab_size, dtype=torch.float32):
        from sglang.srt.managers.mock_forward import mock_forward_batch_generation

        batch = SimpleNamespace(reqs=[None] * batch_size)
        model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            model_config=SimpleNamespace(vocab_size=vocab_size),
            dtype=dtype,
        )
        return mock_forward_batch_generation(batch, model_runner)

    def test_logits_shape_and_zero(self):
        out = self._call(batch_size=3, vocab_size=100)
        logits = out.logits_output.next_token_logits
        self.assertEqual(tuple(logits.shape), (3, 100))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))

    def test_next_token_ids_shape_and_value(self):
        out = self._call(batch_size=5, vocab_size=200)
        tok = out.next_token_ids
        self.assertEqual(tuple(tok.shape), (5,))
        self.assertEqual(tok.dtype, torch.int64)
        # fake_token_id = max(1, vocab_size // 2) = 100
        expected = torch.full((5,), 100, dtype=torch.int64)
        self.assertTrue(torch.equal(tok, expected))

    def test_can_run_cuda_graph_is_false(self):
        out = self._call(batch_size=1, vocab_size=64)
        self.assertFalse(out.can_run_cuda_graph)

    def test_fake_token_avoids_zero_for_tiny_vocab(self):
        # vocab_size=1 -> max(1, 0) = 1, ensures token 0 is not picked
        out = self._call(batch_size=1, vocab_size=1)
        self.assertEqual(out.next_token_ids[0].item(), 1)


class TestUpdateFinishStateMockBranch(CustomTestCase):
    """Req.update_finish_state mock branch — finishes by output length."""

    def setUp(self):
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req

        # Bind as plain functions so calling self._call(req) does not
        # rebind `self` to the test instance.
        self._call = Req.update_finish_state
        self.FINISH_LENGTH = FINISH_LENGTH

    def _fake_req(self, output_ids_len, max_new_tokens=128):
        # Mock branch only reads .finished(), .output_ids,
        # .sampling_params.max_new_tokens; it sets .finished_reason and
        # .finished_len, then returns. Other Req attributes are untouched.
        req = MagicMock()
        req.finished = lambda: False
        req.output_ids = list(range(output_ids_len))
        req.sampling_params.max_new_tokens = max_new_tokens
        req.finished_reason = None
        return req

    def test_finishes_at_target_length(self):
        with envs.SGLANG_MOCK_FORWARD.override(True), envs.SGLANG_MOCK_FORWARD_OUTPUT_LEN.override(
            32
        ):
            req = self._fake_req(output_ids_len=32, max_new_tokens=128)
            self._call(req)
            self.assertIsInstance(req.finished_reason, self.FINISH_LENGTH)
            self.assertEqual(req.finished_reason.length, 32)
            self.assertEqual(req.finished_len, 32)

    def test_does_not_finish_below_target(self):
        with envs.SGLANG_MOCK_FORWARD.override(True), envs.SGLANG_MOCK_FORWARD_OUTPUT_LEN.override(
            32
        ):
            req = self._fake_req(output_ids_len=31, max_new_tokens=128)
            self._call(req)
            self.assertIsNone(req.finished_reason)

    def test_respects_max_new_tokens_when_smaller(self):
        # max_new_tokens=10 < mock_output_len=32, request must finish at 10
        with envs.SGLANG_MOCK_FORWARD.override(True), envs.SGLANG_MOCK_FORWARD_OUTPUT_LEN.override(
            32
        ):
            req = self._fake_req(output_ids_len=10, max_new_tokens=10)
            self._call(req)
            self.assertIsInstance(req.finished_reason, self.FINISH_LENGTH)
            self.assertEqual(req.finished_reason.length, 10)

    def test_bypasses_eos_path(self):
        # If output_ids_len < target, finished_reason must stay None even
        # though _check_token_based_finish would have hit a stop token if
        # the real code path had run. We verify the bypass indirectly:
        # finished_reason remains None and update_finish_state must NOT
        # reach into req.sampling_params.stop_token_ids (which would
        # otherwise raise on the MagicMock-shaped attribute access used
        # by the real path).
        with envs.SGLANG_MOCK_FORWARD.override(True), envs.SGLANG_MOCK_FORWARD_OUTPUT_LEN.override(
            64
        ):
            req = self._fake_req(output_ids_len=5, max_new_tokens=128)
            # Sentinel: if mock branch were not bypassing the EOS path,
            # downstream helpers would mutate `to_finish`, etc. Just call
            # it; assertion is that finished_reason stays None and no
            # exception is raised.
            self._call(req)
            self.assertIsNone(req.finished_reason)


if __name__ == "__main__":
    unittest.main()
