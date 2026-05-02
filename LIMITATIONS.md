# Limitations

These are the things you should know before generalizing from these numbers.

## Hardware

- **Single hardware platform: A100 SXM4 40GB.**
  Nothing here generalizes to H100, H200, B200, MI300X, TPU, or Apple Silicon. If you have those, you also have access to FP8 / FP4 KV which changes the entire story.
- **Single node, NVLink intra-node only.** No multi-node scaling tested. NCCL bottlenecks at >8 GPUs (cross-node) would change throughput numbers significantly.
- **Single OCI shape (BM.GPU4.8).** Different vendors / form factors of A100 (e.g. PCIe vs SXM, 80GB vs 40GB) will differ.

## Model

- **Single model: `nvidia/llama-3.3-nemotron-super-49b-v1.5`.**
  - Reasoning model. Results may differ for non-reasoning models (Llama-3.3-70B-Instruct, Mixtral, Qwen, etc.).
  - 49B params. Smaller models (8B, 13B) and larger (70B, 405B) will have different prefill/generate ratios.
- **Single chat template.** The model's bundled template was used. Changing the template changes the input token count and may change behavior.

## KV cache strategy

- **BF16 KV only.** A100 cannot run FP8 or NVFP4 KV — the manifest in this NIM image confirms zero compatible profiles for Ampere. The "compression bake-off" we set out to do collapses to "engine bake-off" on this hardware.
- **Prefix caching enabled on both engines.** Disabling it would hurt both, probably not equally.

## Workload coverage

- **Only NIAH for the read-pressure measurement.** Other long-context benchmarks (RULER multi-needle, multi-hop, variable tracking; LongBench v2; NoCha) were planned but not run in the borrow window.
- **No reasoning-heavy / KV-write workload.** Original plan included a small-input / long-output (4K+ tokens) test to stress KV write. The bench pod was torn down before this could run. Reported TPOT is from output sequences ≤1024 tokens.
- **Context capped at 64K.** Did not measure 128K. Engine init showed both engines support 131,072 tokens, but we did not benchmark there.
- **Single depth pattern.** 0.1 / 0.5 / 0.9. Did not test mid-deep depths like 0.25, 0.75. NIAH literature suggests depth 0.5 (middle) is hardest for many models — we cover it.

## Statistical rigor

- **9 samples per NIAH cell, 2 batches per concurrency cell.** Adequate for the reported ratios at 3-significant-figure precision. **Not adequate for formal hypothesis testing.** We do not report confidence intervals or run paired tests.
- **Single seed (42).** Did not test seed sensitivity. With temperature=0 and no other entropy source, this should be irrelevant — but we did not verify.

## Engine internals

- **NIM as the deployment substrate.** Both engines were run via NIM (NVIDIA's containerized inference). Bare vLLM / TRT-LLM with hand-tuned configs may produce different numbers. NIM defaults are used.
- **`NIM_GPU_MEMORY_UTILIZATION=0.90` for both.** Trying 0.95 would give both engines a small KV pool boost; we did not test this axis.
- **vLLM 0.10 specifically.** Older or newer vLLM versions will differ. Version pinned in `results/*__engine_init.txt`.
- **TRT-LLM version is whatever NIM 1.14.0 ships.** Recorded in the engine init log.

## Cold-start observation

The "vLLM 16s cold start" finding comes from a 5-needle smoke test, not the full bench. It is documented in `results/*__smoke.json`. Production engineers should treat it as an anecdote that points at a real CUDA-graph capture cost — not as a calibrated benchmark.

## Output token divergence between engines

We observed that the same prompt at temperature=0 produced different output token counts on the two engines (vLLM 940 tok vs TRT-LLM 257 tok on a smoke prompt). Possible explanations:

1. Different stop-token / EOS handling.
2. Different sampling implementations (temperature=0 should be greedy, but tie-breaking can differ).
3. Numerical differences in attention computation cascading to different argmax choices.

We did not investigate this further. It is **flagged as an open follow-up** because it has implications for token-cost estimation and prompt-template work.

## What this benchmark does NOT prove

- It does **not** prove TRT-LLM is "better" than vLLM as a general statement. It shows a specific advantage on a specific model on specific hardware under specific workloads.
- It does **not** prove vLLM is bad. vLLM wins on aggregate throughput at conc=8 in this matrix and is far easier to deploy outside NIM.
- It does **not** generalize to other NVIDIA reasoning models (DeepSeek-R1-Distilled, Llama-Nemotron-Ultra-253B, etc.) without re-measurement.
- It does **not** generalize to non-NIM deployments. A bare-metal vLLM may be configured differently and perform differently from the NIM-packaged vLLM measured here.

## Honest recommendation

If you are picking an engine for **A100-based production deployment of a reasoning LLM** with **conc ≤ 4** and you care about **first-token latency**, this benchmark suggests TRT-LLM. If your workload is **conc ≥ 8** with **relaxed TTFT requirements**, vLLM is competitive on throughput and easier to operate.

For any other combination — different hardware, different model, different workload shape — re-run the bench. Scripts and manifests are in this repo for that reason.
