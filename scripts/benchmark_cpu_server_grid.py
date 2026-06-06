#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

from PIL import Image


def parse_size(raw: str) -> tuple[int, int]:
    width_text, height_text = raw.lower().split("x", 1)
    return int(width_text), int(height_text)


def bench_one(
    endpoint: str,
    prompt: str,
    output_path: Path,
    width: int,
    height: int,
    steps: int,
    seed: int,
    guidance: float,
    max_seq: int,
    timeout: int,
) -> dict[str, object]:
    payload = {
        "prompt": prompt,
        "output": str(output_path),
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "guidance": guidance,
        "max_seq": max_seq,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started_at = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    image = Image.open(output_path).convert("RGB")
    values = list(image.tobytes())
    result["mean"] = round(sum(values) / len(values), 1)
    result["std"] = round(statistics.pstdev(values), 1)
    result["wall_client_seconds"] = round(time.time() - started_at, 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a warm Bonsai CPU image server over a size grid.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8011/generate")
    parser.add_argument("--prompt", default="bonsai")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--max-seq", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=760000)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/telegram"),
        help="Directory for generated images.",
    )
    parser.add_argument(
        "sizes",
        nargs="+",
        help="One or more WIDTHxHEIGHT sizes, for example: 224x224 256x224 320x320",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, raw_size in enumerate(args.sizes):
        width, height = parse_size(raw_size)
        output_path = args.output_dir / f"server-bonsai-grid-{width}x{height}-{args.steps}step.png"
        result = bench_one(
            endpoint=args.endpoint,
            prompt=args.prompt,
            output_path=output_path,
            width=width,
            height=height,
            steps=args.steps,
            seed=args.seed_base + index,
            guidance=args.guidance,
            max_seq=args.max_seq,
            timeout=args.timeout,
        )
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
