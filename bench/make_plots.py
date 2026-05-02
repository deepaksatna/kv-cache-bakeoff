"""
Publication-grade plots for Issue 7 — KV-Cache Bake-Off.
Reads JSONL results, writes PNGs at 300 DPI for LinkedIn cover + supporting charts.

Usage:
    pip install matplotlib numpy
    python make_plots.py
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PLOTS = HERE / "plots"
PLOTS.mkdir(exist_ok=True)

# === Brand-clean style ============================================
TRTLLM_COLOR = "#76B900"   # NVIDIA green
VLLM_COLOR = "#1A73E8"     # Google blue (recognisable as "open source / vLLM")
GRID_COLOR = "#E0E0E0"
TEXT_COLOR = "#212121"
SUBTLE = "#9E9E9E"
ACCENT = "#D32F2F"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": SUBTLE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 11,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


# === Loaders ======================================================
def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def percentile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, int(len(s) * q) - 1)]


def niah_stats(recs):
    by_ctx = defaultdict(list)
    for r in recs:
        by_ctx[r["ctx_target"]].append(r)
    out = {}
    for ctx, rs in sorted(by_ctx.items()):
        ttfts = [r["ttft_ms"] for r in rs if r.get("ttft_ms") is not None]
        tpots = [r["tpot_ms"] for r in rs if r.get("tpot_ms") is not None]
        walls = [r["wall_ms"] for r in rs if r.get("wall_ms") is not None]
        out[ctx] = {
            "n": len(rs),
            "hits": sum(1 for r in rs if r.get("hit")),
            "ttft_p50": percentile(ttfts, 0.5),
            "ttft_p99": percentile(ttfts, 0.99),
            "tpot_p50": percentile(tpots, 0.5),
            "tpot_p99": percentile(tpots, 0.99),
            "wall_p50": percentile(walls, 0.5),
            "wall_p99": percentile(walls, 0.99),
        }
    return out


def conc_stats(recs):
    by_conc = defaultdict(list)
    for r in recs:
        by_conc[r["concurrency"]].append(r)
    out = {}
    for conc, rs in sorted(by_conc.items()):
        out[conc] = {
            "n_batches": len(rs),
            "agg_throughput_p50": percentile([r["agg_throughput_tok_s"] for r in rs if r.get("agg_throughput_tok_s")], 0.5),
            "ttft_p50": percentile([r["ttft_p50_ms"] for r in rs if r.get("ttft_p50_ms")], 0.5),
            "ttft_p99": percentile([r["ttft_p99_ms"] for r in rs if r.get("ttft_p99_ms")], 0.99),
            "tpot_p50": percentile([r["tpot_p50_ms"] for r in rs if r.get("tpot_p50_ms")], 0.5),
            "batch_wall_p50": percentile([r["batch_wall_ms"] for r in rs if r.get("batch_wall_ms")], 0.5),
            "n_ok": sum(r.get("n_ok", 0) for r in rs),
            "n_hit": sum(r.get("hits", 0) for r in rs),
        }
    return out


def ctx_label(c):
    return f"{c // 1024}K" if c >= 1024 else str(c)


# === Plot 1: Cover — wall p50 vs context, head-to-head ==========
def plot_cover(trtllm_niah, vllm_niah):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ctxs = sorted(set(trtllm_niah) | set(vllm_niah))
    x = np.arange(len(ctxs))
    width = 0.36

    trt_walls = [trtllm_niah[c]["wall_p50"] / 1000 for c in ctxs]
    vllm_walls = [vllm_niah[c]["wall_p50"] / 1000 for c in ctxs]

    bars1 = ax.bar(x - width / 2, trt_walls, width, label="TRT-LLM", color=TRTLLM_COLOR, edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width / 2, vllm_walls, width, label="vLLM", color=VLLM_COLOR, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars1, trt_walls):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15, f"{val:.1f}s", ha="center", va="bottom", fontsize=10, color=TRTLLM_COLOR, fontweight="bold")
    for bar, val in zip(bars2, vllm_walls):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15, f"{val:.1f}s", ha="center", va="bottom", fontsize=10, color=VLLM_COLOR, fontweight="bold")

    # Speedup annotation
    for i, c in enumerate(ctxs):
        if vllm_walls[i] > 0:
            speedup = (vllm_walls[i] - trt_walls[i]) / vllm_walls[i] * 100
            if speedup > 5:
                top = max(trt_walls[i], vllm_walls[i]) + 0.8
                ax.annotate(f"TRT-LLM\n{speedup:+.0f}%", xy=(i, top), ha="center", va="bottom",
                            fontsize=10, color=ACCENT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([ctx_label(c) for c in ctxs])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Median end-to-end latency (seconds)")
    ax.set_title("TRT-LLM vs vLLM — same model, same hardware\n8× A100 SXM4 40GB · Llama-3.3-Nemotron-Super-49B · BF16 KV", pad=14)
    ax.set_ylim(0, max(max(trt_walls), max(vllm_walls)) * 1.30)
    ax.legend(loc="upper left", ncol=2)

    fig.text(0.5, -0.04, "Lower is better. NIAH (single-needle, depth 0.1/0.5/0.9). 9 calls per cell, 100% recall both engines.",
             ha="center", fontsize=9, color=SUBTLE, style="italic")
    fig.tight_layout()
    out = PLOTS / "01_cover_wall_latency.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# === Plot 2: TTFT and TPOT side-by-side, bars =====================
def plot_ttft_tpot(trtllm_niah, vllm_niah):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ctxs = sorted(set(trtllm_niah) | set(vllm_niah))
    x = np.arange(len(ctxs))
    width = 0.36

    # TTFT
    ax = axes[0]
    trt = [trtllm_niah[c]["ttft_p50"] for c in ctxs]
    vll = [vllm_niah[c]["ttft_p50"] for c in ctxs]
    ax.bar(x - width / 2, trt, width, color=TRTLLM_COLOR, edgecolor="white", linewidth=1.5, label="TRT-LLM")
    ax.bar(x + width / 2, vll, width, color=VLLM_COLOR, edgecolor="white", linewidth=1.5, label="vLLM")
    ax.set_xticks(x)
    ax.set_xticklabels([ctx_label(c) for c in ctxs])
    ax.set_xlabel("Context length")
    ax.set_ylabel("TTFT median (ms)")
    ax.set_title("Time to First Token", pad=10)
    ax.legend()
    for i, (t, v) in enumerate(zip(trt, vll)):
        ax.text(i - width / 2, t + max(trt + vll) * 0.015, f"{int(t)}", ha="center", va="bottom", fontsize=9, color=TRTLLM_COLOR, fontweight="bold")
        ax.text(i + width / 2, v + max(trt + vll) * 0.015, f"{int(v)}", ha="center", va="bottom", fontsize=9, color=VLLM_COLOR, fontweight="bold")
    ax.set_ylim(0, max(trt + vll) * 1.18)

    # TPOT
    ax = axes[1]
    trt = [trtllm_niah[c]["tpot_p50"] for c in ctxs]
    vll = [vllm_niah[c]["tpot_p50"] for c in ctxs]
    ax.bar(x - width / 2, trt, width, color=TRTLLM_COLOR, edgecolor="white", linewidth=1.5, label="TRT-LLM")
    ax.bar(x + width / 2, vll, width, color=VLLM_COLOR, edgecolor="white", linewidth=1.5, label="vLLM")
    ax.set_xticks(x)
    ax.set_xticklabels([ctx_label(c) for c in ctxs])
    ax.set_xlabel("Context length")
    ax.set_ylabel("TPOT median (ms / output token)")
    ax.set_title("Per-token Generation Speed", pad=10)
    ax.legend()
    for i, (t, v) in enumerate(zip(trt, vll)):
        ax.text(i - width / 2, t + max(trt + vll) * 0.015, f"{t:.1f}", ha="center", va="bottom", fontsize=9, color=TRTLLM_COLOR, fontweight="bold")
        ax.text(i + width / 2, v + max(trt + vll) * 0.015, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=VLLM_COLOR, fontweight="bold")
    ax.set_ylim(0, max(trt + vll) * 1.18)

    fig.suptitle("TTFT and TPOT — TRT-LLM vs vLLM on 8× A100", fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04, "Lower is better. Same model, same hardware, 9 samples per cell.",
             ha="center", fontsize=9, color=SUBTLE, style="italic")
    fig.tight_layout()
    out = PLOTS / "02_ttft_tpot_breakdown.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# === Plot 3: Concurrency scaling — throughput + TTFT trade-off ====
def plot_concurrency(trtllm_conc, vllm_conc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    concs = sorted(set(trtllm_conc) | set(vllm_conc))

    # Throughput
    ax = axes[0]
    trt_t = [trtllm_conc[c]["agg_throughput_p50"] for c in concs]
    vll_t = [vllm_conc[c]["agg_throughput_p50"] for c in concs]
    ax.plot(concs, trt_t, "o-", color=TRTLLM_COLOR, linewidth=2.5, markersize=10, label="TRT-LLM", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(concs, vll_t, "s-", color=VLLM_COLOR, linewidth=2.5, markersize=10, label="vLLM", markeredgecolor="white", markeredgewidth=1.5)
    for c, t in zip(concs, trt_t):
        ax.text(c, t + max(trt_t + vll_t) * 0.025, f"{t:.0f}", ha="center", va="bottom", fontsize=10, color=TRTLLM_COLOR, fontweight="bold")
    for c, t in zip(concs, vll_t):
        ax.text(c, t - max(trt_t + vll_t) * 0.04, f"{t:.0f}", ha="center", va="top", fontsize=10, color=VLLM_COLOR, fontweight="bold")
    ax.set_xticks(concs)
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Aggregate throughput (tokens / sec)")
    ax.set_title("Throughput Scaling", pad=10)
    ax.legend()
    ax.set_ylim(0, max(trt_t + vll_t) * 1.20)

    # Annotate crossover
    if len(concs) >= 4 and vll_t[-1] > trt_t[-1]:
        ax.annotate("vLLM crossover\nat conc=8",
                    xy=(concs[-1], vll_t[-1]),
                    xytext=(concs[-1] - 1.5, vll_t[-1] * 0.65),
                    fontsize=10, color=ACCENT, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))

    # TTFT under load
    ax = axes[1]
    trt_p99 = [trtllm_conc[c]["ttft_p99"] for c in concs]
    vll_p99 = [vllm_conc[c]["ttft_p99"] for c in concs]
    ax.plot(concs, trt_p99, "o-", color=TRTLLM_COLOR, linewidth=2.5, markersize=10, label="TRT-LLM p99", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(concs, vll_p99, "s-", color=VLLM_COLOR, linewidth=2.5, markersize=10, label="vLLM p99", markeredgecolor="white", markeredgewidth=1.5)
    for c, t in zip(concs, trt_p99):
        ax.text(c, t + max(trt_p99 + vll_p99) * 0.02, f"{int(t)}", ha="center", va="bottom", fontsize=10, color=TRTLLM_COLOR, fontweight="bold")
    for c, t in zip(concs, vll_p99):
        ax.text(c, t - max(trt_p99 + vll_p99) * 0.03, f"{int(t)}", ha="center", va="top", fontsize=10, color=VLLM_COLOR, fontweight="bold")
    ax.set_xticks(concs)
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("p99 TTFT (ms)")
    ax.set_title("Tail Latency under Load", pad=10)
    ax.legend()
    ax.set_ylim(0, max(trt_p99 + vll_p99) * 1.18)

    fig.suptitle("Concurrency: throughput vs tail latency at 16K context",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04, "Higher throughput = better. Lower TTFT = better. 2 batches per cell, all hits.",
             ha="center", fontsize=9, color=SUBTLE, style="italic")
    fig.tight_layout()
    out = PLOTS / "03_concurrency_scaling.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# === Plot 4: Pareto / efficiency — throughput per request vs latency
def plot_pareto(trtllm_conc, vllm_conc):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    concs = sorted(set(trtllm_conc) | set(vllm_conc))

    for engine, data, color, marker in [
        ("TRT-LLM", trtllm_conc, TRTLLM_COLOR, "o"),
        ("vLLM", vllm_conc, VLLM_COLOR, "s"),
    ]:
        xs = [data[c]["ttft_p50"] for c in concs]
        ys = [data[c]["agg_throughput_p50"] for c in concs]
        ax.plot(xs, ys, "-", color=color, linewidth=2, alpha=0.7)
        ax.scatter(xs, ys, s=180, color=color, marker=marker, edgecolor="white", linewidth=2, zorder=5, label=engine)
        for c, x, y in zip(concs, xs, ys):
            ax.annotate(f"  N={c}", (x, y), fontsize=10, color=color, fontweight="bold", va="center")

    ax.set_xlabel("Median TTFT under load (ms) — lower is better")
    ax.set_ylabel("Aggregate throughput (tokens / sec) — higher is better")
    ax.set_title("The production trade-off: TTFT vs Throughput at 16K context",
                 pad=14)

    ax.invert_xaxis()  # better-to-the-right convention with inverted x
    ax.set_xlim(ax.get_xlim()[0], 0)

    fig.text(0.5, -0.04,
             "Up-and-to-the-right is the Pareto frontier. N labels = concurrency level.",
             ha="center", fontsize=9, color=SUBTLE, style="italic")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = PLOTS / "04_pareto_throughput_vs_ttft.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# === Plot 5: Engine init facts — bar of KV pool size ==============
def plot_kv_pool():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    engines = ["TRT-LLM", "vLLM"]
    kv_per_gpu = [19.7, 21.7]  # GiB, from engine init logs
    kv_total = [v * 8 for v in kv_per_gpu]

    x = np.arange(len(engines))
    width = 0.45
    bars = ax.bar(x, kv_total, width, color=[TRTLLM_COLOR, VLLM_COLOR], edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, kv_total):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 3, f"{val:.0f} GB",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=bar.get_facecolor())
        ax.text(bar.get_x() + bar.get_width() / 2, val / 2, f"{kv_per_gpu[engines.index(engines[list(bars).index(bar)])]:.1f} GiB / GPU",
                ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(engines)
    ax.set_ylabel("Total KV cache pool across 8 GPUs (GB)")
    ax.set_title("KV cache memory budget — same hardware, different engines\n(from NIM runtime config logs)", pad=14)
    ax.set_ylim(0, max(kv_total) * 1.25)

    delta = (kv_total[1] - kv_total[0]) / kv_total[1] * 100
    ax.annotate(f"vLLM has\n{delta:.0f}% more\nKV pool",
                xy=(0.5, max(kv_total) * 1.05),
                ha="center", fontsize=11, color=ACCENT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ACCENT, lw=1.2))

    fig.text(0.5, -0.04,
             "Both engines use BF16 KV. TRT-LLM allocates more memory to engine state (CUDA graphs, plugins) on the same 40 GB GPUs.",
             ha="center", fontsize=9, color=SUBTLE, style="italic")
    fig.tight_layout()
    out = PLOTS / "05_kv_pool_size.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# === Plot 6: Summary panel (4 metrics x 3 contexts) ===============
def plot_summary(trtllm_niah, vllm_niah):
    ctxs = sorted(set(trtllm_niah) | set(vllm_niah))
    metrics = [
        ("TTFT p50 (ms)", "ttft_p50", False),
        ("TPOT p50 (ms)", "tpot_p50", False),
        ("Wall p50 (sec)", "wall_p50", True),
        ("Wall p99 (sec)", "wall_p99", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    x = np.arange(len(ctxs))
    width = 0.36

    for ax, (title, key, divide_by_1000) in zip(axes, metrics):
        trt = [trtllm_niah[c][key] / (1000 if divide_by_1000 else 1) for c in ctxs]
        vll = [vllm_niah[c][key] / (1000 if divide_by_1000 else 1) for c in ctxs]
        ax.bar(x - width / 2, trt, width, color=TRTLLM_COLOR, edgecolor="white", linewidth=1.5, label="TRT-LLM")
        ax.bar(x + width / 2, vll, width, color=VLLM_COLOR, edgecolor="white", linewidth=1.5, label="vLLM")
        ax.set_xticks(x)
        ax.set_xticklabels([ctx_label(c) for c in ctxs])
        ax.set_title(title, pad=8)
        ax.set_ylim(0, max(trt + vll) * 1.18)
        for i, (t, v) in enumerate(zip(trt, vll)):
            fmt = "{:.1f}" if divide_by_1000 or t < 100 else "{:.0f}"
            ax.text(i - width / 2, t + max(trt + vll) * 0.015, fmt.format(t), ha="center", va="bottom", fontsize=9, color=TRTLLM_COLOR, fontweight="bold")
            ax.text(i + width / 2, v + max(trt + vll) * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=9, color=VLLM_COLOR, fontweight="bold")
        if ax is axes[0]:
            ax.legend(loc="upper left")

    fig.suptitle("KV-Cache Bake-Off — full latency panel\nLlama-3.3-Nemotron-Super-49B on 8× A100, NIAH workload",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "All numbers from 9 calls per cell. Lower is always better. Both engines: 100% needle recall.",
             ha="center", fontsize=10, color=SUBTLE, style="italic")
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    out = PLOTS / "06_full_latency_panel.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    trtllm_niah = niah_stats(load_jsonl(RESULTS / "trtllm-bf16__niah.jsonl"))
    vllm_niah = niah_stats(load_jsonl(RESULTS / "vllm-bf16__niah.jsonl"))
    trtllm_conc = conc_stats(load_jsonl(RESULTS / "trtllm-bf16__conc.jsonl"))
    vllm_conc = conc_stats(load_jsonl(RESULTS / "vllm-bf16__conc.jsonl"))

    plot_cover(trtllm_niah, vllm_niah)
    plot_ttft_tpot(trtllm_niah, vllm_niah)
    plot_concurrency(trtllm_conc, vllm_conc)
    plot_pareto(trtllm_conc, vllm_conc)
    plot_kv_pool()
    plot_summary(trtllm_niah, vllm_niah)

    print("\nAll plots in:", PLOTS)


if __name__ == "__main__":
    main()
