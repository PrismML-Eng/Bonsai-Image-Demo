#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_cpu_experimental as gen


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"missing {label}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the GemLite CPU transformer into an unpacked diffusers directory."
    )
    parser.add_argument(
        "--model-root",
        default=str(REPO_ROOT / "models" / "bonsai-image-4B-ternary-gemlite"),
        help="Model root containing transformer-gemlite-int2, text_encoder-hqq-4bit, and vae.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "models" / "bonsai-image-4B-ternary-unpacked" / "transformer"),
        help="Destination directory for Flux2Transformer2DModel.save_pretrained().",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Dense transformer dtype to materialize before save_pretrained().",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    args = parser.parse_args()

    model_root = Path(args.model_root)
    output_dir = Path(args.output_dir)
    source_dir = model_root / "transformer-gemlite-int2"

    require_dir(model_root / "text_encoder-hqq-4bit", "text encoder")
    require_dir(model_root / "vae", "VAE")
    require_dir(source_dir, "GemLite transformer")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"output directory already exists and is not empty: {output_dir}\n"
            "pass --overwrite to refresh it"
        )

    gen.CPU_INFERENCE_DTYPE = gen.resolve_inference_dtype(args.dtype)
    start = time.time()
    gen.log(
        f"exporting dense unpacked transformer from {source_dir} to {output_dir} "
        f"dtype={gen.CPU_INFERENCE_DTYPE}"
    )
    model = gen.load_dense_transformer(model_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    gen.log(f"saved unpacked transformer to {output_dir} elapsed={time.time()-start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
