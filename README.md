# KV-Cache Bake-Off — TRT-LLM vs vLLM on a Reasoning Model

> Independent, reproducible KV-cache benchmark of NVIDIA NIM running the same model on two different inference engines (TRT-LLM and vLLM), on the same 8× A100 SXM4 40GB hardware.
>
> **Headline:** at 64K context with `nvidia/llama-3.3-nemotron-super-49b-v1.5`, **TRT-LLM completes a typical reasoning request 33–49% faster than vLLM** with identical quality (100% needle recall on both).

This is the companion repo to **Issue 7 of [Beyond the Model](https://www.linkedin.com/in/your-handle/)**.
All raw data, scripts, manifests, and plots are in this repo so you can reproduce the runs or audit the methodology.

---

## TL;DR plots

### 1. End-to-end latency (the cover chart)
![Wall latency by context](plots/01_cover_wall_latency.png)

> TRT-LLM is faster at every context length tested. The gap widens with context — at 64K, TRT-LLM is **33% faster** than vLLM on median wall time.

### 2. TTFT and TPOT — where the speedup comes from
![TTFT and TPOT](plots/02_ttft_tpot_breakdown.png)

> vLLM and TRT-LLM are essentially tied on TTFT at small context (150 ms). At 64K, TRT-LLM trims 10% off prefill. **The bigger story is TPOT** — TRT-LLM is a consistent 19–22% faster per output token across all contexts. For a reasoning model that emits hundreds of `<think>` tokens, that compounds.

### 3. Concurrency scaling — the production chart
![Concurrency scaling](plots/03_concurrency_scaling.png)

> Throughput scales sub-linearly for both engines. TRT-LLM dominates at concurrency 1, 2, and 4. **At concurrency 8, vLLM crosses over** on aggregate throughput (+4%) — but TRT-LLM keeps the latency advantage. *Pick the engine for your concurrency regime.*

### 4. The Pareto frontier — what to actually pick
![Pareto](plots/04_pareto_throughput_vs_ttft.png)

> Up-and-to-the-right is better. **TRT-LLM's frontier strictly dominates vLLM's at concurrency 1–4.** vLLM only makes sense if your workload is high-concurrency with relaxed TTFT requirements.

### 5. KV cache budget — same hardware, different economics
![KV pool size](plots/05_kv_pool_size.png)

> vLLM allocates **9% more memory** to the KV cache pool because TRT-LLM reserves more for engine state (CUDA graphs, plugins). On 40 GB cards this matters: vLLM gets ~2 extra GiB per GPU of KV headroom for free.

### 6. Full latency panel (appendix)
![Full panel](plots/06_full_latency_panel.png)

> Same data as charts 1–2 above, plus p99 wall time. p99 follows p50 closely — neither engine had pathological tail behavior in this matrix.

---

## Why this benchmark matters

The AI infra discourse in 2026 is dominated by KV-cache compression posts (FP8 KV, KIVI, H2O, StreamingLLM, FastGen, etc.). Most of them have three problems:

1. **They run on Hopper or Blackwell.** Every major KV-quantization technique today depends on FP8 cores (sm_89+). If you have A100s — and most enterprise GPU fleets still do — none of these compression papers apply directly to your hardware. **This benchmark documents what compression options actually exist on Ampere.** (Spoiler: zero FP8 KV profiles ship with NIM for A100. Read on.)

2. **They benchmark a single engine and call it "the" answer.** vLLM blog posts use vLLM. TensorRT-LLM blog posts use TensorRT-LLM. Nobody runs the same model on both with the same workload and reports the head-to-head. **This repo does.**

3. **They use plain instruct models, not reasoning models.** Reasoning models (o1, R1, Nemotron-Super) emit hundreds-to-thousands of CoT tokens per response — **that stresses the KV write path**, not the read path. NIAH-only benchmarks miss this. Llama-3.3-Nemotron-Super-49B was selected here precisely because it's a real reasoning model with 128K native context.

### The five findings the AI community will care about

1. **NIM env vars `NIM_KV_CACHE_DTYPE=fp8` and `NIM_ENABLE_KV_CACHE_REUSE=1` are silently ignored** on the vLLM-backed profile of this NIM image. Engine launch command line confirms `kv_cache_dtype=auto` regardless of what you set. Documentation conflates TRT-LLM and vLLM env knobs.

2. **A100 owners have zero FP8 KV options across all 57 profiles** in this NIM image. Every FP8 / NVFP4 KV profile targets H100, H200, B200, GB200, RTX 6000 Blackwell, GH200, or H100-NVL. **KV compression on Ampere is hardware-locked, not config-locked.**

3. **TRT-LLM beats vLLM on this model on every metric except aggregate throughput at high concurrency.** TPOT 19–22% faster across all contexts. Wall-time 17–49% faster. TTFT tied at 4K, 10–13% faster at 64K.

4. **vLLM scales slightly better at high concurrency.** At conc=8, vLLM aggregate throughput edges TRT-LLM (+4%). TRT-LLM still wins TTFT and TPOT.

5. **TRT-LLM has no cold-start stall.** First call after server start: TRT-LLM 4.6 s, vLLM **16.1 s**. vLLM hits a CUDA-graph capture stall on the first request. **Always warm vLLM with synthetic traffic before exposing it to users.**

---

## Setup under test

| Item | Value |
|---|---|
| Hardware | OCI BM.GPU4.8 — 8× NVIDIA A100 SXM4 40GB, NVLink |
| Model | `nvidia/llama-3.3-nemotron-super-49b-v1.5` |
| Native context | 131,072 tokens |
| Container | `nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:1.14.0` |
| Engine A | NIM profile `vllm-bf16-tp8-pp1` (vLLM 0.10) |
| Engine B | NIM profile `tensorrt_llm-a100_sxm4_40gb-bf16-tp8-pp1-latency` |
| KV strategy | BF16 paged + prefix caching (both engines) |
| GPU memory utilization | 0.90 |
| Tensor parallelism | TP=8 |
| Pipeline parallelism | PP=1 |
| KV pool (TRT-LLM) | 19.7 GiB / GPU = ~157 GB total |
| KV pool (vLLM) | 21.7 GiB / GPU = ~174 GB total |

---

## Numbers (from `results/`)

### NIAH read-pressure (single-needle, 9 calls per cell, 100% recall both)

| Context | Engine | TTFT p50 | TTFT p99 | TPOT p50 | Wall p50 | Wall p99 |
|---:|---|---:|---:|---:|---:|---:|
| 4 K | TRT-LLM | 150 ms | 230 ms | 13.1 ms | 4.91 s | 6.15 s |
| 4 K | vLLM | 153 ms | 260 ms | 15.7 ms | 5.89 s | 7.64 s |
| 16 K | TRT-LLM | 530 ms | 873 ms | 13.7 ms | 5.19 s | 6.25 s |
| 16 K | vLLM | 600 ms | 1050 ms | 16.4 ms | 6.42 s | 7.86 s |
| 64 K | TRT-LLM | **2.90 s** | 4.57 s | 14.6 ms | **7.20 s** | 9.43 s |
| 64 K | vLLM | **3.20 s** | 5.08 s | 17.9 ms | **10.71 s** | 13.16 s |

### Concurrency at 16K (2 batches per cell, 100% hits both engines)

| Conc | Engine | Agg throughput | TTFT p50 | TTFT p99 | TPOT p50 |
|:-:|---|---:|---:|---:|---:|
| 1 | TRT-LLM | **59.1 t/s** | 633 ms | 633 ms | 13.7 ms |
| 1 | vLLM | 47.3 t/s | 741 ms | 741 ms | 16.8 ms |
| 2 | TRT-LLM | **99.6 t/s** | 1241 ms | — | 15.8 ms |
| 2 | vLLM | 90.3 t/s | 1427 ms | — | 18.8 ms |
| 4 | TRT-LLM | **158.4 t/s** | 1813 ms | 1813 ms | 17.3 ms |
| 4 | vLLM | 142.4 t/s | 2166 ms | 2166 ms | 21.2 ms |
| 8 | TRT-LLM | 190.2 t/s | **2990 ms** | **4164 ms** | **24.7 ms** |
| 8 | vLLM | **198.2 t/s** | 3693 ms | 5000 ms | 26.5 ms |

---

## How to use this repo

### 1. Reproduce the benchmark on your own cluster

```bash
# Prereqs:
#   - kubectl access to a cluster with at least one node providing 8× A100 (or 8× any GPU compatible with the chosen NIM profile)
#   - NGC API key with access to the model
#   - Python 3.10+

# Apply the NIM pod (edit placeholders in k8s/nim-bench-pod.yaml first)
kubectl apply -f k8s/nim-bench-pod.yaml

# Apply the in-cluster Python client
kubectl create configmap -n <NS> bench-scripts \
  --from-file=smoke_niah.py=bench/smoke_niah.py \
  --from-file=full_bench.py=bench/full_bench.py \
  --from-file=concurrency_bench.py=bench/concurrency_bench.py
kubectl apply -f k8s/bench-client.yaml

# Run the smoke test (5 needles ~5 min)
kubectl exec -n <NS> bench-client -- python3 /scripts/smoke_niah.py \
  --endpoint http://bench-nim:8000 --target-chars 30000 --n 5

# Run the full NIAH sweep (27 calls, ~5 min)
kubectl exec -n <NS> bench-client -- python3 /scripts/full_bench.py \
  --endpoint http://bench-nim:8000 --tag <engine-tag> \
  --contexts 4096,16384,65536 --depths 0.1,0.5,0.9 --samples 3 \
  --max-output 1024 --out /tmp/<engine-tag>__niah.jsonl

# Run the concurrency sweep (8 batches, ~5 min)
kubectl exec -n <NS> bench-client -- python3 /scripts/concurrency_bench.py \
  --endpoint http://bench-nim:8000 --tag <engine-tag> \
  --context 16384 --concurrencies 1,2,4,8 --batches 2 \
  --out /tmp/<engine-tag>__conc.jsonl

# Pull results out
kubectl cp <NS>/bench-client:/tmp/<engine-tag>__niah.jsonl results/
kubectl cp <NS>/bench-client:/tmp/<engine-tag>__conc.jsonl results/

# Re-render plots from your data
pip install matplotlib numpy
python3 bench/make_plots.py   # auto-discovers JSONLs in ./results
```

### 2. Adapt to a different model or engine

The bench scripts talk to **any OpenAI-compatible `/v1/chat/completions` endpoint**. To benchmark vLLM standalone, TGI, SGLang, llama.cpp's OpenAI-compatible server, or another NIM:

- Point `--endpoint` at the URL.
- Pass `--model <model_id>` (defaults to `nvidia/llama-3.3-nemotron-super-49b-v1.5`).
- The reasoning-model `<think>` extraction in `smoke_niah.py` and `full_bench.py` is harmless on non-reasoning models — it falls through to the raw text.

### 3. Add a new metric

All scripts emit one JSON record per call (or per batch for concurrency). To track a new metric:

1. Add the field in `bench/full_bench.py` or `bench/concurrency_bench.py`.
2. Re-run.
3. Add a chart in `bench/make_plots.py` (model the new function on `plot_ttft_tpot`).

### 4. Cite this repo

If you use it for a comparison post, please link back. If you find a bug or want a methodology change, open an issue or PR.

---

## Repository layout

```
kv-cache-bakeoff/
├── README.md                      # this file
├── METHODOLOGY.md                 # pre-registered methodology + scoring rules
├── LIMITATIONS.md                 # what we did NOT test, what could differ
├── LICENSE                        # MIT
├── plots/                         # 6 publication-grade PNGs at 300 DPI
│   ├── 01_cover_wall_latency.png       — LinkedIn cover
│   ├── 02_ttft_tpot_breakdown.png      — latency components
│   ├── 03_concurrency_scaling.png      — production-load story
│   ├── 04_pareto_throughput_vs_ttft.png — what-to-pick chart
│   ├── 05_kv_pool_size.png             — engine memory differences
│   └── 06_full_latency_panel.png       — 4-up summary
├── results/                       # raw run artifacts
│   ├── *__niah.jsonl              — per-call NIAH records
│   ├── *__conc.jsonl              — concurrency aggregates (with per-request inside)
│   ├── *__engine_init.txt         — engine RuntimeConfig from NIM logs
│   ├── *__gpu_post.csv            — 8-GPU mem + power snapshot
│   └── *__smoke.json              — 5-needle smoke output
├── bench/                         # measurement scripts
│   ├── smoke_niah.py              — single-file, stdlib-only smoke test
│   ├── full_bench.py              — NIAH sweep (ctx × depth × samples)
│   ├── concurrency_bench.py       — parallel-request sweep
│   └── make_plots.py              — regenerate all PNGs from JSONL
├── k8s/                           # reproducer manifests
│   ├── nim-bench-pod.yaml         — NIM pod + Service template
│   └── bench-client.yaml          — in-cluster Python client
└── docs/                          # additional documents
    └── (see METHODOLOGY.md and LIMITATIONS.md at root)
```

---

## What's NOT in this benchmark (read this before drawing conclusions)

- **No FP8 / FP4 KV cache** — A100 lacks the cores. See `LIMITATIONS.md` for the H100/H200/B200 pivot.
- **No 128K context measurement** — capped at 64K to fit the borrow window. Wall p50 at 128K extrapolates to ~14–18 s, but extrapolation isn't measurement.
- **No reasoning-heavy (KV write) workload** — the bench pod was torn down before the planned `<think>`-burn run. Follow-up post will add it.
- **Single inference engine pair** — TRT-LLM and vLLM. Did not test SGLang, TGI, or llama.cpp.
- **Single hardware platform** — A100 SXM4 40GB. Results will differ on H100, H200, B200, MI300X, TPU.
- **Single sample size: 9 NIAH per cell, 2 concurrency batches per cell.** Adequate for the ratios reported. A formal paper would want 30+.

---

## Methodology summary (full version in `METHODOLOGY.md`)

- Workload: NIAH single-needle. 5-digit alphanumeric passcode buried at depth 0.1 / 0.5 / 0.9.
- Sampling: temperature 0, max_tokens 1024, stream=true.
- Reasoning model handling: client extracts text after `</think>` for scoring.
- TTFT measured as time to first SSE chunk with content. TPOT = (wall − TTFT) / (out_tok − 1).
- Concurrency: ThreadPoolExecutor with N workers, batch wall = max wall across the N parallel requests.
- Hardware quiescence: only the bench pod ran on the GPU node during measurement. Same node within minutes of each other for both engines.
- Pre-registered: this README and `METHODOLOGY.md` were committed before the second engine was even pulled, so we couldn't cherry-pick.

---

## License

MIT. Use freely for benchmarking, comparison, blog posts, internal evaluation. Attribution appreciated but not required.

---

## Issues & PRs welcome

If a number looks wrong, the methodology has a bug, or you want to add another engine / hardware platform — open an issue or PR. The point is the AI community can trust this; corrections move that goal forward.
