# Mock Forward

> **Status: stub.** This document is a placeholder so runtime references
> (`NotImplementedError` messages, startup banner, HTTP middleware
> docstrings) resolve to a real page. A full anti-pattern guide with a
> concrete incident case study lands in a follow-up.

## What it is

A scheduler-only testing mode triggered by env var `SGLANG_MOCK_FORWARD=1`.
It short-circuits `tp_worker.forward_batch_generation` with shape-valid
fake output so the scheduler / KV allocator / ZMQ / tokenize / detokenize
CPU pipeline can be exercised without GPU forward time dominating the
wall clock.

## Quick start

```bash
SGLANG_MOCK_FORWARD=1 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 30000
```

Optional env vars:

| Name | Default | Meaning |
|---|---|---|
| `SGLANG_MOCK_FORWARD` | `0` | Master switch |
| `SGLANG_MOCK_FORWARD_OUTPUT_LEN` | `32` | Mock requests finish after this many output tokens |

Clients can detect mock mode via the response header
`SGLang-Mock-Forward: true`.

## Do NOT use for

- **PR performance comparisons** — mock hides GPU-side regressions; you'd
  conclude a scheduler is faster when in fact the GPU got slower.
- **Capacity planning** — real GPU time is what bounds throughput in
  production; mock-mode rates are not comparable.
- **Output validation** — generated tokens are pre-determined fakes
  (`vocab_size // 2`), they have no semantic meaning.
- **Mock-vs-real latency compare** — mock latency reflects only the
  CPU pipeline; real latency is dominated by GPU forward.

## Not supported in v1

Hard-raises `NotImplementedError` at startup when mock is on combined
with any of:

- `--speculative-algorithm *` (draft worker is outside the cut point)
- `--disaggregation-mode != null`
- `--enable-pdmux` (split-prefill uses a separate worker entrypoint)
- `--pp-size > 1` (PP non-last ranks need `pp_hidden_states_proxy_tensors`)
- `is_embedding=True` (embedding/reward models use a separate worker
  method `forward_batch_embedding`)

Auto-force-off (with a `WARNING` log line):

- `--disable-cuda-graph` (CUDA graph capture relies on a real forward)
- `--disable-overlap-schedule` (overlap exists to hide GPU forward time,
  which is zero under mock; also avoids a CUDA-only JIT kernel that
  crashes on ROCm)

## Trustable vs. untrustable signals under mock

| Signal | Trustable? | Why |
|---|---|---|
| Scheduler decisions, batching, preemption | ✅ | Real code path |
| KV allocator + radix-tree behavior | ✅ | Allocation happens before the cut point |
| Prefix-cache hit rate | ✅ | Token-id match, not KV content |
| Stream output (SSE chunks) | ✅ | Real detokenize / IPC |
| Token throughput, latency | ❌ | No GPU work; numbers are meaningless |
| Output text content | ❌ | Pre-determined fake token |
| `return_logprob` values | ❌ | Populated with zeros to avoid crashes |

## TODO (follow-up)

- Concrete anti-pattern incident case study (with measured numbers)
- Operational gotchas (zombie scheduler processes holding GPU memory
  after kill; `sglang.benchmark` editable-install packaging caveat)
- Future work (spec decoding mock, multimodal mock, NVIDIA validation)
