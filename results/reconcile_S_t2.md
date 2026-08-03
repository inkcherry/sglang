# S turn 2 — reconcile + a2a resize: test run and mutation check

Container `mingzhi-lean`, CPU only, worktree `worktrees/teamS`.

## Suite

    docker exec -w <worktree> mingzhi-lean bash -lc \
      'PYTHONPATH=python python -m pytest \
       test/registered/unit/disaggregation/test_pd_role_switch.py -q'
    -> 34 passed, 18 subtests passed

## Mutation check — each constraint has exactly one failing test

Same command after injecting one defect at a time (all reverted afterwards).

| injected defect | constraint | test that failed |
|---|---|---|
| `padded = cap` (skip graph padding) | 4 | test_decode_sizes_for_the_padded_batch_from_the_live_runner |
| size prefill a2a from `sa.chunked_prefill_size` not the settled chunk | 2 | test_a2a_sized_from_settled_chunk_not_the_current_one |
| `return None` before the reduction when the target is None | 1 | test_underivable_target_is_a_vote_not_an_early_return |
| drop the `agreed > ceiling` check | 5 | test_target_above_the_process_ceiling_is_refused |
| `_commit_targets` before the resize | 3 | test_failed_resize_applies_nothing |
| read `op.config.max_num_inp_token_per_rank` for the old capacity | 6 | test_capacity_is_tracked_not_read_back_off_the_op |
| clamp against `scheduler.max_running_requests` not the launch cap | ratchet | test_cap_clamp_does_not_ratchet_across_flips |

No collateral failures except the padded-batch test also failing under the
ratchet mutation, which is expected: the cap feeds the pad lookup.
