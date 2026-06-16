"""Mock forward: short-circuit replacement for tp_worker.forward_batch_generation.

This module exists solely for scheduler / dispatch / request-lifecycle testing.
When enabled, it returns shape-valid fake outputs WITHOUT touching the GPU,
allowing the CPU-side scheduler path to be measured without GPU-time noise.

DO NOT USE FOR:
  - PR performance comparisons (hides GPU-side regressions)
  - Capacity planning (real GPU time is what matters)
  - Output validation (generated tokens are pre-determined fakes)

See docs/dev/mock_forward.md for the full anti-pattern list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


def mock_forward_batch_generation(
    batch: "ScheduleBatch",
    model_runner: "ModelRunner",
) -> GenerationBatchResult:
    """Drop-in fake replacement for tp_worker.forward_batch_generation.

    Returns a GenerationBatchResult whose tensors are shape-valid but
    semantically meaningless. The downstream scheduler / batch_result_processor
    / detokenizer pipeline runs unchanged on top of these fake values.

    Note: this function intentionally bypasses ForwardBatch.init_new(),
    model_runner.forward(), AND model_runner.sample(). It is invoked at the
    tp_worker boundary, so nothing below the worker layer executes.
    """
    bs = len(batch.reqs)
    device = model_runner.device
    vocab_size = model_runner.model_config.vocab_size
    dtype = model_runner.dtype

    # Fake logits: zeros are fine because we also bypass the real sampler.
    # Shape (bs, vocab_size) is what downstream readers expect.
    fake_logits = torch.zeros((bs, vocab_size), device=device, dtype=dtype)
    # Fake logprobs: populated so requests with basic output logprob
    # (return_logprob=True, top_logprobs_num=0, no explicit logprob_start_len)
    # do not crash on None.tolist() / None[i] in batch_result_processor /
    # logprob_result_processor. Values are meaningless under mock; only shape
    # and dtype matter.
    #
    # NOTE: top_logprobs and input logprobs are NOT populated here. Those
    # paths are rejected up front in scheduler.handle_generate_request
    # (_mock_forward_logprob_reject_reason), so they never reach this output.
    # Filling shape-valid placeholders for them would require replicating the
    # input-logprob length bookkeeping, which is not worth it for fake data.
    fake_logprobs = torch.zeros((bs,), device=device, dtype=torch.float32)
    logits_output = LogitsProcessorOutput(
        next_token_logits=fake_logits,
        next_token_logprobs=fake_logprobs,
    )

    # Fake next_token_ids: pick a deterministic mid-vocab token. We avoid 0
    # (often <pad>) and stay well clear of common EOS ranges. The actual
    # value does not matter for scheduler testing -- Req.update_finish_state's
    # mock branch finishes requests by output length, not by token id.
    fake_token_id = max(1, vocab_size // 2)
    next_token_ids = torch.full(
        (bs,),
        fake_token_id,
        dtype=torch.int64,
        device=device,
    )

    return GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=next_token_ids,
        can_run_cuda_graph=False,
    )
