"""Minimal NIAH smoke test for any OpenAI-compatible endpoint (NIM, vLLM, TGI, etc.).
No external dependencies beyond stdlib. ~5 min for 10 needles at 8K context.

Usage:
    python3 smoke_niah.py --endpoint http://localhost:8000 --target-chars 30000 --n 10
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request

FILLER = (
    "The grass is green and the sky is blue. The sun rises in the east. "
    "Coffee is best served hot. Mountains are made of rock. Rivers flow downhill. "
)
NEEDLE_TEMPLATE = "The secret passcode is {code}. Remember it."
QUESTION = "What is the secret passcode? Answer with only the passcode value."


def make_prompt(target_chars: int, code: str, depth_pct: float) -> str:
    needle = NEEDLE_TEMPLATE.format(code=code)
    filler = (FILLER * (target_chars // len(FILLER) + 1))[:target_chars - len(needle)]
    cut = int(len(filler) * depth_pct)
    return f"<haystack>\n{filler[:cut]}\n{needle}\n{filler[cut:]}\n</haystack>\n\n{QUESTION}"


def extract_answer(text: str) -> str:
    """Reasoning models wrap CoT in <think>...</think>. Strip it for scoring."""
    m = re.search(r"</think>\s*(.+)", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def call(endpoint: str, model: str, prompt: str, max_tokens: int = 2048, timeout: int = 600) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read().decode())
    return {
        "text": j["choices"][0]["message"]["content"],
        "wall_ms": (time.perf_counter() - t0) * 1000,
        "in_tok": j.get("usage", {}).get("prompt_tokens", 0),
        "out_tok": j.get("usage", {}).get("completion_tokens", 0),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", default="nvidia/llama-3.3-nemotron-super-49b-v1.5")
    p.add_argument("--target-chars", type=int, default=30000)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    results = []
    for i in range(args.n):
        code = f"{random.randint(10000, 99999)}-{random.choice(['ALPHA', 'BETA', 'GAMMA'])}"
        depth = (i + 0.5) / args.n
        prompt = make_prompt(args.target_chars, code, depth)
        try:
            r = call(args.endpoint, args.model, prompt)
            answer = extract_answer(r["text"])
            hit = code.lower() in answer.lower()
            results.append({
                "i": i, "depth": round(depth, 2), "code": code,
                "hit": hit, "answer": answer[:120],
                "wall_ms": round(r["wall_ms"], 0),
                "in_tok": r["in_tok"], "out_tok": r["out_tok"],
            })
            print(f"[{i+1}/{args.n}] depth={depth:.2f} hit={hit} "
                  f"in={r['in_tok']}t out={r['out_tok']}t wall={r['wall_ms']:.0f}ms "
                  f"answer={answer[:60]!r}")
        except Exception as e:
            print(f"[{i+1}/{args.n}] ERROR: {e}", file=sys.stderr)
            results.append({"i": i, "error": str(e)[:200]})

    ok = sum(1 for r in results if r.get("hit"))
    print(f"\n=== SMOKE TEST: {ok}/{len(results)} hits ===")
    print(json.dumps({"summary": {"total": len(results), "hits": ok}, "results": results}, indent=2))


if __name__ == "__main__":
    main()
