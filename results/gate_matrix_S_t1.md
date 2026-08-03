# S turn 1 — startup gate verification

Commit: e6d2606a82 (+ test commit). Base 9c31c6c756.

Command (inside container `mingzhi-lean`, CPU-only arg resolution, no model load):

```
docker exec -w /home/mingzliu/pdrs_lean_team/worktrees/teamS mingzhi-lean bash -c \
 'PYTHONPATH=python python - <<EOF
from sglang.srt.server_args import ServerArgs
M="/shared_inference/models_blog/DeepSeek-V3-5layer"
common=dict(model_path=M, tp_size=4, chunked_prefill_size=2048,
            disaggregation_mode="prefill", enable_pd_role_switch=True,
            trust_remote_code=True)
... construct each row, print ACCEPTED / the ValueError ...
EOF'
```

Rows measured (matching the oracle's expectations):

| row | no gate | with gate |
|---|---|---|
| pure TP | ACCEPTED | ACCEPTED |
| EP only, no a2a | rejected (ep_size 4) | rejected (ep_size 4) + gate hint |
| mori a2a only (mori forces ep=tp=4) | rejected (ep_size 4, mori) | **ACCEPTED** |
| DWDP `--dwdp-size 4` | rejected (DP attention, ep_size 4, dp_size 4) | rejected, same clauses |
| deepep a2a | rejected (ep_size 4, deepep) | rejected + gate hint |

DWDP was **ACCEPTED at base** (director's `base_nogate_rejects_9c31c6c756.txt`);
moving the guard from `_handle_pd_disaggregation` (which runs before
`_handle_dwdp`) to the end of the resolution pipeline fixes it. Same fix makes
`--moe-a2a-backend mori` alone report `ep_size 4`, as the oracle does.

Unit tests: `pytest test/registered/unit/disaggregation/test_pd_role_switch.py`
-> 26 passed, 18 subtests passed.

Note: `--moe-a2a-backend mori` with the default `chunked_prefill_size` trips an
unrelated pre-existing assertion (`SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK
(default 4096) must be >= chunked_prefill_size`); rows above pass 2048.
