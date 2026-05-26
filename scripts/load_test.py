"""Concurrent load test for the Bonsai Image backend.

Fires N requests against `/api/generate` with bounded concurrency, prints
per-request latency, and summarizes throughput + p50/p95. Stdlib only —
no `pip install` needed.

Useful for exercising the multi-GPU code path: with 4×L40S you should see
4 requests run concurrently and the rest queue at the asyncio.Semaphore
inside each replica (visible as `Pending: N` on the operator dashboard).

Usage:
    # Hit the HF Space at default concurrency 4, 20 requests, 1024×1024:
    python scripts/load_test.py

    # Stress: 16 concurrent, 64 total, smaller images for faster turn-around
    python scripts/load_test.py --concurrency 16 --requests 64 --size 512x512

    # Hit a local serve.sh on Mac:
    python scripts/load_test.py --url http://127.0.0.1:8000 --concurrency 2

    # Bypass the Next.js proxy entirely and hit the backend port directly
    # (useful on the local Mac when /api/generate timeouts confuse things):
    python scripts/load_test.py --path /generate --url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


SAMPLE_PROMPTS = [
    "a tiny bonsai tree in a ceramic pot",
    "a red fox curled in a snowy forest",
    "an astronaut riding a horse on the moon",
    "a cyberpunk city street at night with neon lights",
    "a watercolor painting of a sailboat at sunset",
    "a portrait of a wise old wizard with a long beard",
    "a futuristic robot fixing a vintage radio",
    "a peaceful zen garden with cherry blossoms",
    "an underwater coral reef teeming with fish",
    "a steampunk airship over Victorian London",
]


def fire(url: str, path: str, payload: dict, idx: int) -> dict:
    req = urllib.request.Request(
        f"{url}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read()
            return {
                "idx": idx,
                "ok": resp.status == 200,
                "status": resp.status,
                "elapsed": time.monotonic() - start,
                "size": len(body),
            }
    except urllib.error.HTTPError as exc:
        return {"idx": idx, "ok": False, "status": exc.code, "elapsed": time.monotonic() - start, "error": str(exc)}
    except Exception as exc:
        return {"idx": idx, "ok": False, "status": 0, "elapsed": time.monotonic() - start, "error": str(exc)}


def main(args: argparse.Namespace) -> None:
    width, height = map(int, args.size.split("x"))

    def make_payload(i: int) -> dict:
        return {
            "prompt": random.choice(SAMPLE_PROMPTS),
            "seed": random.randint(0, 2**31 - 1),
            "steps": args.steps,
            "width": width,
            "height": height,
        }

    print(f"# load test")
    print(f"#   target:      {args.url}{args.path}")
    print(f"#   requests:    {args.requests}")
    print(f"#   concurrency: {args.concurrency}")
    print(f"#   shape:       {width}x{height}  ({args.steps} steps)")
    print()

    results: list[dict] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [exe.submit(fire, args.url, args.path, make_payload(i), i) for i in range(args.requests)]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            mark = "ok" if r["ok"] else "FAIL"
            size_kb = (r.get("size") or 0) // 1024
            err = f"  {r.get('error', '')}" if not r["ok"] else ""
            print(f"  [{r['idx']:>3}] {mark:>4}  {r['elapsed']:6.1f}s  {size_kb:>5} KB{err}", flush=True)
    total = time.monotonic() - started

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    print()
    print(f"  total wall:  {total:>6.1f}s")
    print(f"  success:     {len(ok)}/{len(results)}")
    if fail:
        print(f"  failures:    {len(fail)}")
        for f in fail[:5]:
            print(f"    [{f['idx']}] status={f.get('status')}  err={f.get('error', '-')[:80]}")
    if ok:
        d = sorted(r["elapsed"] for r in ok)
        n = len(d)
        print(f"  latency p50: {d[n // 2]:>6.1f}s")
        print(f"  latency p95: {d[min(n - 1, int(n * 0.95))]:>6.1f}s")
        print(f"  latency min: {d[0]:>6.1f}s")
        print(f"  latency max: {d[-1]:>6.1f}s")
        print(f"  throughput:  {len(ok) / total:>6.2f} req/s")
        print(f"  per-req avg: {total / len(ok):>6.1f}s/req")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Load test for Bonsai Image backend")
    p.add_argument("--url", default="https://prism-ml-bonsai-image-demo.hf.space",
                   help="Base URL — HF Space or local serve.sh root")
    p.add_argument("--path", default="/api/generate",
                   help="Endpoint path. Use /generate to bypass Next.js and hit the backend directly")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--size", default="1024x1024")
    p.add_argument("--steps", type=int, default=4)
    args = p.parse_args()
    main(args)
