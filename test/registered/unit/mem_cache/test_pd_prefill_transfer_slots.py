import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GB = 1 << 30
STATE_BYTES = 64 << 20


def make_configurator(
    mode, *, max_running_requests=16, attn_dp_size=1, draft_tokens=None
):
    server_args = SimpleNamespace(
        disaggregation_mode=mode,
        max_running_requests=max_running_requests,
        max_mamba_cache_size=None,
        disable_radix_cache=True,
        speculative_num_draft_tokens=draft_tokens,
        mamba_full_memory_ratio=0.86,
    )
    server_args.override = lambda _source, **fields: [
        setattr(server_args, name, value) for name, value in fields.items()
    ]
    configurator = SimpleNamespace(
        server_args=server_args,
        ps=SimpleNamespace(attn_dp_size=attn_dp_size),
        mambaish_config=SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(mamba_cache_per_req=STATE_BYTES)
        ),
        spec_algorithm=SimpleNamespace(is_none=lambda: draft_tokens is None),
    )
    configurator._prefill_transfer_slots = lambda count: (
        KVCacheConfigurator._prefill_transfer_slots(configurator, count)
    )
    return configurator


class TestPrefillTransferSlots(unittest.TestCase):
    def test_only_prefill_reserves_transfer_slots(self):
        for mode, expected in (("prefill", 16), ("decode", 0), ("null", 0)):
            configurator = make_configurator(mode)
            self.assertEqual(
                KVCacheConfigurator._prefill_transfer_slots(configurator, 16),
                expected,
            )

    def test_state_pool_and_memory_include_prefill_transfers(self):
        configurator = make_configurator("prefill")
        remaining = KVCacheConfigurator._handle_max_mamba_cache(configurator, 100.0)
        self.assertEqual(configurator.server_args.max_mamba_cache_size, 32)
        self.assertAlmostEqual(remaining, 100.0 - 32 * STATE_BYTES / GB)

    def test_draft_memory_is_reserved_only_for_running_requests(self):
        configurator = make_configurator("prefill", draft_tokens=3)
        remaining = KVCacheConfigurator._handle_max_mamba_cache(configurator, 100.0)
        self.assertEqual(configurator.server_args.max_mamba_cache_size, 32)
        expected = (32 + 16 * 3) * STATE_BYTES / GB
        self.assertAlmostEqual(remaining, 100.0 - expected)

    def test_headroom_is_per_attention_worker(self):
        configurator = make_configurator("prefill", attn_dp_size=2)
        KVCacheConfigurator._handle_max_mamba_cache(configurator, 100.0)
        self.assertEqual(configurator.server_args.max_mamba_cache_size, 16)


if __name__ == "__main__":
    unittest.main()
