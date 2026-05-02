"""
Full-proof KV-cache benchmark client.

For each context length × each needle depth × each sample:
  * Send NIAH prompt with stream=true
  * Capture TTFT (first chunk arrival), TPOT, total wall, input/output tokens
  * Score whether the needle was recalled

Output: JSONL, one line per call. Aggregate offline with Pareto/percentile math.

Usage:
  python3 /scripts/full_bench.py \
    --endpoint http://bench-nim-trtllm-bf16:8000 \
    --tag trtllm-bf16 \
    --contexts 4096,16384,65536,131072 \
    --depths 0.1,0.3,0.5,0.7,0.9 \
    --samples 3 \
    --out /results/trtllm-bf16__full.jsonl
"""
import argparse, json, random, sys, time
import urllib.request, urllib.error
import re

FILLER = (
    "The grass is green and the sky is blue. The sun rises in the east. "
    "Coffee is best served hot. Mountains are made of rock. Rivers flow downhill. "
    "Books are read in libraries. Cats sleep in patches of sunlight. "
    "The wind whistles through the leaves. Children laugh on playgrounds. "
)
NEEDLE = "The secret passcode is {code}. Please remember it for later."
QUESTION = "What is the secret passcode that was hidden in the document above? Reply with only the passcode value, nothing else."

def make_prompt(target_chars, code, depth):
    needle = NEEDLE.format(code=code)
    filler = (FILLER * (target_chars // len(FILLER) + 1))[:max(0, target_chars - len(needle))]
    cut = int(len(filler) * depth)
    return f"<haystack>\n{filler[:cut]}\n{needle}\n{filler[cut:]}\n</haystack>\n\n{QUESTION}"

def extract_answer(text):
    m = re.search(r"</think>\s*(.+)", text, re.DOTALL)
    return (m.group(1) if m else text).strip()

def stream_call(endpoint, prompt, max_tokens=2048, timeout=600):
    body = json.dumps({
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    out_tok = 0
    in_tok = 0
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
                chunks.append(content)
                out_tok += 1
            usage = obj.get("usage")
            if usage:
                in_tok = usage.get("prompt_tokens", in_tok)
                out_tok = usage.get("completion_tokens", out_tok)
    wall = time.perf_counter() - t0
    text = "".join(chunks)
    return {
        "text": text,
        "wall_ms": round(wall * 1000, 1),
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "tpot_ms": round((wall - (ttft or 0)) * 1000 / max(out_tok - 1, 1), 2) if out_tok > 1 else None,
        "in_tok": in_tok,
        "out_tok": out_tok,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--contexts", default="4096,16384,65536,131072")
    p.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--max-output", type=int, default=2048)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    contexts = [int(c) for c in args.contexts.split(",")]
    depths = [float(d) for d in args.depths.split(",")]

    rng = random.Random(args.seed)
    out_f = open(args.out, "w")
    total_cells = len(contexts) * len(depths) * args.samples
    done = 0
    started = time.time()
    print(f"[bench] {args.tag}: {len(contexts)} contexts × {len(depths)} depths × {args.samples} samples = {total_cells} calls")

    for ctx in contexts:
        target_chars = int(ctx * 3.2)  # ~3.2 chars/token rough
        for depth in depths:
            for sample in range(args.samples):
                code = f"{rng.randint(10000,99999)}-{rng.choice(['ALPHA','BETA','GAMMA','DELTA','OMEGA'])}"
                prompt = make_prompt(target_chars, code, depth)
                done += 1
                t0 = time.time()
                try:
                    r = stream_call(args.endpoint, prompt, max_tokens=args.max_output)
                    answer = extract_answer(r["text"])
                    hit = code.lower() in answer.lower()
                    rec = {
                        "tag": args.tag, "ctx_target": ctx, "depth": depth, "sample": sample,
                        "code": code, "hit": hit,
                        "answer_first_120": answer[:120],
                        "wall_ms": r["wall_ms"], "ttft_ms": r["ttft_ms"], "tpot_ms": r["tpot_ms"],
                        "in_tok": r["in_tok"], "out_tok": r["out_tok"],
                    }
                except Exception as e:
                    rec = {
                        "tag": args.tag, "ctx_target": ctx, "depth": depth, "sample": sample,
                        "code": code, "hit": False, "error": str(e)[:200],
                    }
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                elapsed = time.time() - started
                eta_s = (elapsed / done) * (total_cells - done)
                print(f"  [{done}/{total_cells}] ctx={ctx:>6} d={depth:.1f} s={sample}  "
                      f"hit={rec.get('hit')}  in={rec.get('in_tok')}t  ttft={rec.get('ttft_ms')}ms  "
                      f"tpot={rec.get('tpot_ms')}ms  wall={int(time.time()-t0)}s  ETA={int(eta_s)}s")

    out_f.close()
    print(f"[bench] DONE in {int(time.time()-started)}s. Wrote {args.out}")

if __name__ == "__main__":
    main()
