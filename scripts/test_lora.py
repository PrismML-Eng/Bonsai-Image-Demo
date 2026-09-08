"""Smoke-test whether a FLUX.2 Klein LoRA can attach to the local Bonsai model."""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = DEMO_DIR / "models"

MODELS = {
    "ternary-mlx": ("bonsai-ternary-mlx", "bonsai-image-4B-ternary-mlx"),
    "binary-mlx": ("bonsai-binary-mlx", "bonsai-image-4B-binary-mlx"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the local MLX Bonsai model and try applying one LoRA.",
    )
    parser.add_argument(
        "lora",
        help="LoRA path: local file, Hugging Face repo, or repo:filename.safetensors.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="ternary-mlx",
        help="Local MLX model to test against.",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="LoRA scale.")
    parser.add_argument(
        "--standard-transformer",
        action="store_true",
        help="Disable the optimized klein-fast transformer for comparison.",
    )
    return parser.parse_args()


def main() -> int:
    if sys.platform != "darwin":
        print("This smoke test targets the MLX path and must be run on macOS.")
        return 2

    args = parse_args()
    from backend.pipeline import PipelineConfig, _build_model
    from mflux.models.common.resolution.lora_resolution import LoraResolution
    from mflux.models.flux2.flux2_initializer import Flux2Initializer

    try:
        lora_path = Path(LoraResolution.resolve(args.lora)).expanduser()
    except Exception as exc:
        print(f"LoRA could not be resolved: {exc}")
        return 2
    if not lora_path.is_file():
        print(f"LoRA file not found: {lora_path}")
        return 2

    backend, model_dir = MODELS[args.model]
    config = PipelineConfig.from_env()
    model_path = str(MODELS_DIR / model_dir)

    print(f"Loading {args.model} from {model_path}")
    model = _build_model(backend=backend, model_path=model_path, config=config)
    model._lora_paths_arg = [str(lora_path)]
    model._lora_scales_arg = [args.scale]

    if args.standard_transformer:
        print("Reloading with standard transformer for comparison")
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        precision = "2bit" if args.model == "ternary-mlx" else "1bit"
        model = Flux2Klein(
            model_path=model_path,
            use_klein_fast_transformer=False,
            klein_fast_precision=precision,
            vae_variant="small",
            evict_text_encoder=config.evict_text_encoder,
            lazy_components=config.lazy_components,
            bucketed_seq_len=config.bucketed_seq_len,
            lora_paths=[str(lora_path)],
            lora_scales=[args.scale],
        )

    buffer = io.StringIO()
    print(f"Applying LoRA {lora_path} at scale {args.scale}")
    with contextlib.redirect_stdout(buffer):
        Flux2Initializer.load_transformer_and_vae(model)

    loader_output = buffer.getvalue()
    print(loader_output, end="")

    match = re.search(r"Applied to (\d+) layers", loader_output)
    applied_layers = int(match.group(1)) if match else 0
    if applied_layers <= 0:
        transformer = getattr(model, "transformer", None)
        print(f"Runtime transformer: {type(transformer).__module__}.{type(transformer).__name__}")
        print(f"Model lazy_components: {getattr(model, '_lazy_components', None)}")
        if transformer is not None:
            names = [
                name
                for name in dir(transformer)
                if not name.startswith("_") and "block" in name.lower()
            ]
            if names:
                print("Transformer block-like attributes:")
                for name in sorted(names):
                    print(f"  - {name}")
        print("RESULT: incompatible or not applied (0 target layers patched).")
        return 1

    print(f"RESULT: LoRA attached successfully ({applied_layers} target layers patched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
