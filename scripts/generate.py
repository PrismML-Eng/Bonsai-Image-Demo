"""Generate one image with Bonsai.

Two execution paths, picked by `sys.platform`. Both stay in-process.

- macOS Apple Silicon → imports prism-image-studio's `FluxPipeline`,
  runs the ternary MLX kernels directly.
- Linux GPU → imports prism-image-studio-backend-gpu's `GpuPipeline`,
  runs the gemlite kernels directly. No HTTP server, no subprocess.

The mlx / mflux dep tree isn't installed on Linux (and vice-versa for
gemlite/HQQ on Mac) — see pyproject.toml `sys_platform` markers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Log format matches backend_gpu/scripts/smoke_e2e.py so the wall-clock stages
# line up across both entry points.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("generate")

DEMO_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = DEMO_DIR / "models"

# Persist Triton's JIT cache and gemlite's autotune cache under outputs/ so
# subsequent runs skip per-shape kernel compile + autotune search. Linux-only
# — mflux on macOS uses precompiled Metal binaries and has neither cache.
# TRITON_CACHE_DIR must be set before torch/triton imports, hence the env-var
# dance here at module load.
TRITON_CACHE_DIR = DEMO_DIR / "outputs" / ".triton_cache"
GEMLITE_PERSIST_PATH = DEMO_DIR / "outputs" / ".gemlite_cache" / "autotune.json"
if sys.platform != "darwin":
    TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GEMLITE_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE_DIR))

# Model registry: short name → (image-studio backend id, dir under models/).
# Add a new entry here to expose another arm via --model; pair it with a
# matching case in scripts/download_model.sh.
MODELS = {
    "ternary-mlx":     ("bonsai-ternary-mlx",     "bonsai-image-4B-ternary-mlx"),
    "ternary-gemlite": ("bonsai-ternary-gemlite", "bonsai-image-4B-ternary-gemlite"),
    "binary-mlx":      ("bonsai-binary-mlx",      "bonsai-image-4B-binary-mlx"),
    "binary-gemlite":  ("bonsai-binary-gemlite",  "bonsai-image-4B-binary-gemlite"),
}


def default_model() -> str:
    return "ternary-mlx" if sys.platform == "darwin" else "ternary-gemlite"


def parse_size(s: str) -> tuple[int, int]:
    """Parse 'WxH' (e.g. '1024x1024') into (width, height)."""
    s = s.lower().replace("×", "x")
    try:
        w_str, h_str = s.split("x", 1)
        w, h = int(w_str), int(h_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--size must be 'WxH' (e.g. 1024x1024), got {s!r}"
        )
    for dim, name in ((w, "width"), (h, "height")):
        if not 256 <= dim <= 2048:
            raise argparse.ArgumentTypeError(
                f"--size {name} {dim} out of range — must be 256–2048"
            )
        if dim % 16:
            raise argparse.ArgumentTypeError(
                f"--size {name} {dim} must be a multiple of 16"
            )
    return w, h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate an image with Bonsai.",
        epilog=(
            "Recommended sizes (multiples of 16):\n"
            "  Aspect             Fast (~0.25MP)   Quality (~1MP)\n"
            "  Square    (1:1)    512×512          1024×1024\n"
            "  Landscape (3:2)    624×416          1248×832\n"
            "  Portrait  (2:3)    416×624          832×1248\n"
            "  Wide      (2:1)    704×352          1408×704\n"
            "  Tall      (1:2)    352×704          704×1408\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "-m", "--model",
        choices=sorted(MODELS),
        default=default_model(),
        help=f"Model (default: {default_model()} on this platform).",
    )
    p.add_argument("-p", "--prompt", required=True, help="Text prompt.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random integer seed; if unset, a fresh one is picked and printed.")
    p.add_argument("--steps", type=int, default=4, help="Inference steps (recommended: 4).")
    p.add_argument(
        "--lora-paths",
        nargs="*",
        default=None,
        help="Optional LoRA .safetensors path(s), MLX path only.",
    )
    p.add_argument(
        "--lora-scales",
        nargs="*",
        type=float,
        default=None,
        help="Optional LoRA scale(s), one per --lora-paths entry.",
    )
    p.add_argument(
        "--size", type=parse_size, default=(512, 512),
        help="Image size as WxH (default: 512x512). See recommended sizes below.",
    )
    p.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    p.add_argument(
        "--open",
        action="store_true",
        help="Open the generated image after saving (macOS only).",
    )
    p.add_argument(
        "--force-gpu-run",
        action="store_true",
        help=(
            "Linux/GPU only: run the pipeline in-process anyway. Without this "
            "flag, generate.sh on Linux exits and points at serve.sh, since "
            "each cold call pays ~25s of imports + model load + JIT for ~1s "
            "of diffusion. macOS ignores this flag (mflux cold-start is much "
            "cheaper)."
        ),
    )
    return p.parse_args()


def resolve_output(args: argparse.Namespace, seed: int) -> Path:
    if args.output is not None:
        out = args.output.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # outputs/{model}/image_{date}_seed{seed}.png — keeps models separate and
    # encodes the seed so a single look at the filename is enough to reproduce.
    out = DEMO_DIR / "outputs" / args.model / f"image_{ts}_seed{seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def append_metadata(
    args: argparse.Namespace,
    output: Path,
    seed: int,
    width: int,
    height: int,
    duration_seconds: float,
    stages: dict[str, float] | None = None,
) -> Path:
    """Append a record for this generation to outputs/{model}/generations.json."""
    meta_dir = DEMO_DIR / "outputs" / args.model
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "generations.json"
    try:
        rel_output = output.relative_to(DEMO_DIR)
    except ValueError:
        rel_output = output
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "prompt": args.prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": args.steps,
        "duration_seconds": round(duration_seconds, 3),
        "output": str(rel_output),
    }
    if stages is not None:
        record["stage_seconds"] = {k: round(v, 3) for k, v in stages.items()}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []
    existing.append(record)
    meta_path.write_text(json.dumps(existing, indent=2) + "\n")
    return meta_path


def require_model_dir(model: str) -> Path:
    _, subdir = MODELS[model]
    model_root = MODELS_DIR / subdir
    if not model_root.exists():
        sys.exit(
            f"Model directory not found: {model_root}\n"
            f"Run: ./scripts/download_model.sh --model {model}"
        )
    return model_root


# ─── macOS path: in-process MLX via prism-image-studio ─────────────────────

def generate_macos(
    args: argparse.Namespace, seed: int, width: int, height: int
) -> tuple[bytes, dict[str, float]]:
    """macOS Apple Silicon path. Returns (png_bytes, {setup_s, diffusion_s})."""
    log.info("[1/2] setup: imports + pipeline ...")
    setup_t0 = time.perf_counter()
    from backend.pipeline import FluxPipeline, PipelineConfig
    backend_id, _ = MODELS[args.model]
    model_root = require_model_dir(args.model)
    pipeline = FluxPipeline(PipelineConfig(
        backend=backend_id,
        baked_model_path=str(model_root),
        te_4bit=True,
        evict_text_encoder=True,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    ))
    setup_s = time.perf_counter() - setup_t0
    log.info("       setup done in %.1fs", setup_s)

    log.info("[2/2] diffusion (steps=%d size=%dx%d) ...", args.steps, width, height)
    diff_t0 = time.perf_counter()
    png_bytes = pipeline.generate_png(
        prompt=args.prompt,
        seed=seed,
        steps=args.steps,
        height=height,
        width=width,
    )
    diffusion_s = time.perf_counter() - diff_t0
    log.info("       diffusion done in %.2fs", diffusion_s)

    return png_bytes, {"setup_s": setup_s, "diffusion_s": diffusion_s}


# ─── Linux path: in-process gemlite via backend_gpu.GpuPipeline ────────────

def _find_subdir(root: Path, *hints: str) -> Path:
    """Return the child dir of `root` whose name contains any of the hints.

    The HF repo's subdir names aren't always identical across variants
    (`transformer/` vs `transformer-packed-gemlite/` vs `transformer-packed-mflux/`,
    etc.) — match by substring so the same code works across layout changes.
    """
    matches = [
        p for p in root.iterdir()
        if p.is_dir() and any(h in p.name for h in hints)
    ]
    if not matches:
        present = ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir())) or "(empty)"
        raise FileNotFoundError(
            f"No subdir matching {hints!r} under {root}. Present: {present}"
        )
    # Prefer the most specific match: longest name wins. Avoids picking
    # `tokenizer/` when looking for `text_encoder/` if the former contains the
    # latter as a substring.
    matches.sort(key=lambda p: len(p.name), reverse=True)
    return matches[0]


def _resolve_gemlite_transformer_dir(variant: str) -> Path:
    """Resolve a variant's gemlite transformer dir, with an absolute fallback."""
    model_root = MODELS_DIR / f"bonsai-image-4B-{variant}-gemlite"
    expected_suffix = {"ternary": "int2", "binary": "int1"}[variant]
    hits = sorted(model_root.glob("transformer-gemlite-*"), key=lambda p: len(p.name), reverse=True)
    if hits:
        return hits[0]
    return model_root / f"transformer-gemlite-{expected_suffix}"


def generate_linux(
    args: argparse.Namespace, seed: int, width: int, height: int
) -> tuple[bytes, dict[str, float]]:
    """Linux GPU path. Returns (png_bytes, {setup_s, diffusion_s})."""
    log.info("[1/2] setup: imports + pipeline + prewarm ...")
    setup_t0 = time.perf_counter()

    from backend_gpu.pipeline_gpu import GpuPipeline
    from gemlite.core import GemLiteLinearTriton

    backend_id, _ = MODELS[args.model]
    model_root = require_model_dir(args.model)
    text_encoder_dir = _find_subdir(model_root, "text_encoder")
    # GpuPipeline validates both gemlite transformer slots at init. Set
    # both kwargs so it never falls back to the package's /root/models/*
    # defaults on local runs.
    # HF repo layout nests the Qwen tokenizer inside the text-encoder dir
    # (matches DEFAULT_TOKENIZER_PATH in pipeline_gpu.py). Derive instead of
    # scanning model_root, where no top-level `tokenizer/` exists.
    pipeline = GpuPipeline(
        backend=backend_id,
        ternary_transformer_path=str(_resolve_gemlite_transformer_dir("ternary")),
        binary_transformer_path=str(_resolve_gemlite_transformer_dir("binary")),
        text_encoder_path=str(text_encoder_dir),
        vae_path=str(_find_subdir(model_root, "vae")),
        tokenizer_path=str(text_encoder_dir / "tokenizer"),
    )

    # Stack persisted autotune entries on top of the bundled cache that
    # prewarm() loads. Per-shape JIT short-circuits for shapes seen on a
    # prior boot.
    if GEMLITE_PERSIST_PATH.exists():
        GemLiteLinearTriton.load_config(str(GEMLITE_PERSIST_PATH), print_error=False)

    pipeline.prewarm()
    setup_s = time.perf_counter() - setup_t0
    log.info("       setup done in %.1fs", setup_s)

    log.info("[2/2] diffusion (steps=%d size=%dx%d) ...", args.steps, width, height)
    diff_t0 = time.perf_counter()
    png_bytes = pipeline.generate_png(
        prompt=args.prompt,
        seed=seed,
        steps=args.steps,
        height=height,
        width=width,
    )
    diffusion_s = time.perf_counter() - diff_t0
    log.info("       diffusion done in %.2fs (peak HBM %.1f MiB)",
             diffusion_s, pipeline.last_peak_memory_mb or 0.0)

    # Persist any new autotune configs discovered this run.
    GemLiteLinearTriton.cache_config(str(GEMLITE_PERSIST_PATH))

    return png_bytes, {"setup_s": setup_s, "diffusion_s": diffusion_s}


# ─── main ───────────────────────────────────────────────────────────────────

def previous_runs_at_shape(model: str, width: int, height: int) -> list[float]:
    """Return duration_seconds of past runs at this exact resolution for `model`.

    Source: outputs/{model}/generations.json. Used to differentiate the cold
    first-run estimate from a "kernels-already-warm" estimate when picking the
    upfront banner to show the user.
    """
    meta_path = DEMO_DIR / "outputs" / model / "generations.json"
    if not meta_path.exists():
        return []
    try:
        entries = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [
        float(e["duration_seconds"])
        for e in entries
        if isinstance(e, dict)
        and e.get("width") == width
        and e.get("height") == height
        and isinstance(e.get("duration_seconds"), (int, float))
    ]


def _print_serve_recommendation() -> int:
    """Print the 'use serve.sh' message and return an exit code.

    Why: each in-process call rebuilds the full pipeline. ~25s of imports +
    model load + first-shape JIT pays for ~1s of diffusion. A long-lived
    server keeps the pipeline resident and turns subsequent calls into ~1s.
    Surfacing that trade-off explicitly stops users from blaming the model
    for slowness that's actually per-process startup tax.
    """
    print(
        "\n"
        "  WARNING: generate.sh on GPU (Linux/CUDA) is NOT the fast path on its own.\n"
        "\n"
        "  Each call pays ~30 seconds of start time (initialize libraries, load\n"
        "  and convert model, compile kernels, etc) for a few seconds of actual\n"
        "  image generation.\n"
        "\n"
        "  ── Fast workflow ───────────────────────────────────────────────\n"
        "  Boot the daemon once with serve.sh, then either:\n"
        "    1. open http://localhost:3000  (Next.js studio)\n"
        "    2. POST directly to the backend:\n"
        "         curl -s -o image.png \\\n"
        "              -H 'Content-Type: application/json' \\\n"
        "              -d '{\"prompt\":\"your prompt\",\"backend\":\"bonsai-ternary-gemlite\",\n"
        "                   \"seed\":42,\"steps\":4,\"width\":512,\"height\":512}' \\\n"
        "              http://127.0.0.1:8000/generate\n"
        "\n"
        "  ── In-process anyway ───────────────────────────────────────────\n"
        "  If you really want to run it in-process, re-run with --force-gpu-run.\n"
    )
    return 2


def main() -> None:
    args = parse_args()
    # The gemlite path's per-process startup tax (~25s) only applies on Linux.
    # mflux/MLX on Apple Silicon has cheap cold-start since the kernels are
    # already-compiled Metal binaries, so the Mac path runs as before.
    if sys.platform != "darwin" and not args.force_gpu_run:
        sys.exit(_print_serve_recommendation())

    seed = args.seed if args.seed is not None else secrets.randbits(31)
    output = resolve_output(args, seed)
    width, height = args.size

    log.info("args: prompt=%r seed=%d steps=%d size=%dx%d model=%s output=%s",
             args.prompt, seed, args.steps, width, height, args.model, output)
    if sys.platform != "darwin":
        log.info("triton_cache=%s gemlite_persist=%s",
                 os.environ.get("TRITON_CACHE_DIR", "(default)"), GEMLITE_PERSIST_PATH)

    # Upfront banner: if generations.json has past runs at this exact shape, we
    # expect cached Triton kernels + gemlite configs to apply, so estimate
    # from history. Otherwise warn that the first call pays a cold compile.
    prior = previous_runs_at_shape(args.model, width, height)
    print()
    if prior:
        mean_s = sum(prior) / len(prior)
        best_s = min(prior)
        print(f"  ⚡ {len(prior)} prior run(s) at {width}×{height} — using warmed-up kernels.")
        print(f"     historical wall: mean {mean_s:.1f}s, best {best_s:.1f}s.")
    else:
        print(f"  ⏳ FIRST run at {width}×{height} for model {args.model!r}.")
        if sys.platform == "darwin":
            print(f"     Cold imports + ~4 GiB weight load + first-shape MLX kernel compile means a")
            print(f"     slow first call. Subsequent calls at this shape are faster once the MLX")
            print(f"     metallib cache fills in. Hardware-bound — this script learns from past runs")
            print(f"     and reports the historical mean on later calls.")
        else:
            print(f"     Cold imports + ~4 GiB weight load + first-shape Triton JIT + gemlite autotune")
            print(f"     means ~30-60s on this call. Subsequent calls at this shape land several seconds")
            print(f"     faster once the gemlite + triton caches under outputs/ fill in. Hardware-bound —")
            print(f"     this script learns from past runs and reports the historical mean on later calls.")
    print()

    wall_t0 = time.perf_counter()
    if sys.platform == "darwin":
        png_bytes, stages = generate_macos(args, seed, width, height)
    else:
        png_bytes, stages = generate_linux(args, seed, width, height)
    wall_s = time.perf_counter() - wall_t0

    output.write_bytes(png_bytes)
    meta_path = append_metadata(args, output, seed, width, height, wall_s, stages)

    print()
    print(f"=== generate ({args.model} | {width}x{height} | steps={args.steps} | seed={seed}) ===")
    print(f"  prompt:    {args.prompt!r}")
    print()
    if sys.platform == "darwin":
        _setup_desc = "imports + model → MLX (Apple Silicon)"
        _diff_desc = "sampling + VAE decode + PNG encode"
    else:
        _setup_desc = "imports + model → CUDA + autotune restore"
        _diff_desc = "forward + VAE + PNG encode"
    print(f"  setup     : {stages['setup_s']:7.2f} s   ({_setup_desc})")
    print(f"  diffusion : {stages['diffusion_s']:7.2f} s   ({_diff_desc})")
    print(f"  ─────────────────────")
    print(f"  wall      : {wall_s:7.2f} s")
    print()
    print(f"  💡 For much faster generation, run `./scripts/serve.sh` and use the studio.")
    print()
    print(f"  output : {output}")
    print(f"  meta   : {meta_path}")

    if args.open and sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", str(output)], check=False)


if __name__ == "__main__":
    main()
