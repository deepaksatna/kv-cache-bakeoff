# Methodology — Pre-Registered

This document was written and committed before the second engine was pulled from NGC. It freezes the experimental design so that no number reported in the README could have been chosen post-hoc.

## Hypothesis

Given the same model, same hardware, and same KV precision (BF16), two different inference engines (TRT-LLM and vLLM) will produce measurably different latency / throughput / KV-cache-economics characteristics. We expect:

- TPOT differences driven by attention-kernel implementation choices.
- TTFT differences driven by prefill-batching strategy.
- KV pool size differences driven by engine state overhead (CUDA graphs, plugin tensors).

## Variables

**Independent variable:** inference engine, controlled by NIM model profile selection.
- Engine A: profile `vllm-bf16-tp8-pp1` (vLLM 0.10)
- Engine B: profile `tensorrt_llm-a100_sxm4_40gb-bf16-tp8-pp1-latency`

**Held constant:**
- Model: `nvidia/llama-3.3-nemotron-super-49b-v1.5`
- Hardware: 8× A100 SXM4 40GB on a single bare-metal node, NVLink intra-node
- KV precision: BF16
- Tensor parallelism: 8
- Pipeline parallelism: 1
- `NIM_GPU_MEMORY_UTILIZATION=0.90`
- Sampling: temperature 0.0, top_p 1.0
- Tokenizer: model-bundled
- Stream: true
- Max output tokens: 1024 (NIAH), 1024 (concurrency)
- Random seed for prompt generation: 42

## Workloads

### Workload 1 — NIAH (KV read pressure)

A 5-digit alphanumeric passcode (e.g., `93810-ALPHA`) is embedded inside synthetic filler text at one of three depths:
- 0.1 — near the start of context
- 0.5 — middle of context
- 0.9 — near the end of context

The model is asked to repeat just the passcode. **Scoring:** case-insensitive substring match of the passcode in the model's answer (after stripping any `<think>...</think>` block).

Context targets:
- 4096 tokens (~13 KB filler)
- 16384 tokens (~52 KB filler)
- 65536 tokens (~210 KB filler)

Samples: 3 per (context × depth) → 9 calls per (engine × context).

### Workload 2 — Concurrency (production load)

Same NIAH prompt template (depth 0.5, fixed at 16K context). N parallel requests fired simultaneously via `ThreadPoolExecutor(max_workers=N)`. Batch wall time = max wall across the N requests.

Concurrency levels: 1, 2, 4, 8.
Batches per level: 2.

## Metrics

| Metric | Definition | Captured how |
|---|---|---|
| Hit | passcode appears in extracted answer (case-insensitive substring) | client-side scoring |
| TTFT | time to first SSE chunk with non-empty `delta.content` | `time.perf_counter()` before request; subtract at first chunk |
| Wall | total time from request start to `[DONE]` | `time.perf_counter()` |
| TPOT | (wall − TTFT) / (out_tok − 1) | derived |
| in_tok | `usage.prompt_tokens` from final stream chunk | NIM-reported |
| out_tok | `usage.completion_tokens` from final stream chunk | NIM-reported |
| Aggregate throughput (concurrency only) | (sum input + output tokens across batch) / batch_wall_seconds | derived |

## Reasoning-model handling

Llama-3.3-Nemotron-Super-49B emits chain-of-thought wrapped in `<think>...</think>` before the answer. The client extracts text after `</think>` for scoring:

```python
def extract_answer(text):
    m = re.search(r"</think>\s*(.+)", text, re.DOTALL)
    return (m.group(1) if m else text).strip()
```

## Statistical aggregation

- p50 = `sorted_xs[len // 2]`
- p99 = `sorted_xs[max(0, int(len * 0.99) - 1)]`

Sample sizes (9 per NIAH cell, 2 batches × N requests for concurrency) are adequate for ratio reporting at the precision shown (3 significant figures). They are not adequate for formal hypothesis testing (we do not report confidence intervals).

## Run sequence

1. Pause production NIM (scale StatefulSet to 0).
2. Launch Engine A bench pod, wait for `/v1/models` to return 200.
3. Run NIAH workload (27 calls).
4. Run concurrency workload (8 batches).
5. Capture engine init metadata from NIM logs and 8-GPU `nvidia-smi`.
6. Tear down Engine A pod.
7. Repeat 2–6 for Engine B.
8. Restore production NIM.

The hostPath model cache survives pod teardown so Engine B doesn't pay NGC re-download cost (apart from engine-specific binaries it doesn't yet have).

## Hardware quiescence

The 8-GPU node was dedicated to one bench pod at a time. No other workloads ran during measurement. The other GPU node in the cluster was unrelated. Network traffic from the polling watcher and `nvidia-smi` snapshots was negligible compared to bench traffic.

## What we are NOT measuring

- Correctness on real downstream tasks (just NIAH recall, not MMLU / HumanEval / MATH)
- Quality at long context beyond 64K
- KV read vs write trade-off in mixed workloads
- Cold-start cost beyond the smoke test
- Multi-node scaling
- Different prompt templates / chat formats
- Speculative decoding overlay

See `LIMITATIONS.md` for the full list of caveats.
