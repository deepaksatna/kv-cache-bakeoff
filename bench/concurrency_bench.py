"""
Concurrency sweep: launch N parallel requests, measure aggregate behavior.
Fixed context (default 16K), vary N in {1, 2, 4, 8, 16}.
Per cell: 3 batches of N parallel requests, capture per-request TTFT/TPOT/wall + aggregate throughput.

Usage:
  python3 /scripts/concurrency_bench.py \
    --endpoint http://bench-nim-trtllm-bf16:8000 \
    --tag trtllm-bf16 \
    --context 16384 \
    --concurrencies 1,2,4,8,16 \
    --batches 3 \
    --out /results/trtllm-bf16__concurrency.jsonl
"""
import argparse, json, random, time
import urllib.request, re
from concurrent.futures import ThreadPoolExecutor, as_completed

FILLER = "The grass is green and the sky is blue. The sun rises in the east. " * 4
NEEDLE = "The secret passcode is {code}. Please remember it for later."
QUESTION = "What is the secret passcode? Reply with only the passcode value."

def make_prompt(target_chars, code, depth):
    needle = NEEDLE.format(code=code)
    filler = (FILLER * (target_chars // len(FILLER) + 1))[:max(0, target_chars - len(needle))]
    cut = int(len(filler) * depth)
    return f"<haystack>\n{filler[:cut]}\n{needle}\n{filler[cut:]}\n</haystack>\n\n{QUESTION}"

def extract_answer(text):
    m = re.search(r"</think>\s*(.+)", text, re.DOTALL)
    return (m.group(1) if m else text).strip()

def stream_call(endpoint, prompt, max_tokens=1024, timeout=900):
    body = json.dumps({
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    ttft = None
    out_tok = 0
    in_tok = 0
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks.append(content); out_tok += 1
                u = obj.get("usage")
                if u:
                    in_tok = u.get("prompt_tokens", in_tok)
                    out_tok = u.get("completion_tokens", out_tok)
        wall = time.perf_counter() - t0
        text = "".join(chunks)
        return {
            "wall_ms": round(wall * 1000, 1),
            "ttft_ms": round(ttft * 1000, 1) if ttft else None,
            "tpot_ms": round((wall - (ttft or 0)) * 1000 / max(out_tok - 1, 1), 2) if out_tok > 1 else None,
            "in_tok": in_tok, "out_tok": out_tok,
            "answer": extract_answer(text)[:120],
        }
    except Exception as e:
        return {"error": str(e)[:200], "wall_ms": (time.perf_counter() - t0) * 1000}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--context", type=int, default=16384)
    p.add_argument("--concurrencies", default="1,2,4,8,16")
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    target_chars = int(args.context * 3.2)
    out_f = open(args.out, "w")
    rng = random.Random(42)

    for conc in [int(c) for c in args.concurrencies.split(",")]:
        for batch_idx in range(args.batches):
            # Build N prompts (different needles, same depth=0.5 for fairness)
            prompts = []
            for i in range(conc):
                code = f"{rng.randint(10000,99999)}-{rng.choice(['ALPHA','BETA','GAMMA','DELTA','OMEGA'])}"
                prompts.append((code, make_prompt(target_chars, code, 0.5)))

            # Fire all in parallel, measure batch wall time
            print(f"[conc] N={conc} batch={batch_idx+1}/{args.batches} firing...")
            t0 = time.perf_counter()
            results = []
            with ThreadPoolExecutor(max_workers=conc) as ex:
                futs = {ex.submit(stream_call, args.endpoint, p, 1024): code for code, p in prompts}
                for fut in as_completed(futs):
                    code = futs[fut]
                    r = fut.result()
                    r["code"] = code
                    r["hit"] = code.lower() in r.get("answer", "").lower() if not r.get("error") else False
                    results.append(r)
            batch_wall = (time.perf_counter() - t0) * 1000

            # Aggregate
            ok = [r for r in results if not r.get("error")]
            if ok:
                ttfts = sorted([r["ttft_ms"] for r in ok if r.get("ttft_ms")])
                tpots = sorted([r["tpot_ms"] for r in ok if r.get("tpot_ms")])
                walls = sorted([r["wall_ms"] for r in ok])
                total_in = sum(r.get("in_tok", 0) for r in ok)
                total_out = sum(r.get("out_tok", 0) for r in ok)
                agg = {
                    "tag": args.tag, "context": args.context, "concurrency": conc, "batch": batch_idx,
                    "n_ok": len(ok), "n_err": len(results) - len(ok),
                    "hits": sum(1 for r in ok if r.get("hit")),
                    "batch_wall_ms": round(batch_wall, 1),
                    "agg_throughput_tok_s": round((total_in + total_out) / (batch_wall / 1000), 1),
                    "ttft_p50_ms": ttfts[len(ttfts) // 2] if ttfts else None,
                    "ttft_p99_ms": ttfts[max(0, int(len(ttfts) * 0.99) - 1)] if ttfts else None,
                    "tpot_p50_ms": tpots[len(tpots) // 2] if tpots else None,
                    "tpot_p99_ms": tpots[max(0, int(len(tpots) * 0.99) - 1)] if tpots else None,
                    "wall_p50_ms": walls[len(walls) // 2],
                    "wall_p99_ms": walls[max(0, int(len(walls) * 0.99) - 1)],
                    "per_request": results,
                }
            else:
                agg = {
                    "tag": args.tag, "context": args.context, "concurrency": conc, "batch": batch_idx,
                    "n_ok": 0, "n_err": len(results), "errors": [r.get("error") for r in results[:3]],
                }
            out_f.write(json.dumps(agg) + "\n"); out_f.flush()
            print(f"  -> ok={agg.get('n_ok')}/{conc} hits={agg.get('hits')} "
                  f"batch_wall={int(agg.get('batch_wall_ms', 0))}ms "
                  f"agg_tput={agg.get('agg_throughput_tok_s')}t/s "
                  f"ttft_p50={agg.get('ttft_p50_ms')}ms ttft_p99={agg.get('ttft_p99_ms')}ms")
    out_f.close()
    print(f"[conc] DONE -> {args.out}")

if __name__ == "__main__":
    main()
