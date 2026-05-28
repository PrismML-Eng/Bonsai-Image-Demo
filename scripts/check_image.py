"""Sanity-check a generated PNG for obvious pipeline failures.

Exits 0 on success, 1 on failure (with a human-readable reason).

Two failure modes are caught:

1. All-black / near-black output — the VAE decoder or PNG write failed and
   produced a blank canvas.  Mean pixel brightness across all channels must
   exceed MIN_MEAN.

2. Constant noise / single solid colour — the text encoder or scheduler
   failed to influence the latents, producing the "gray-brown noise"
   symptom reported in GitHub issue #<TBD>.  The population standard
   deviation of pixel values across all channels must exceed MIN_STD.

Usage:
    python scripts/check_image.py path/to/image.png
    python scripts/check_image.py path/to/image.png --min-mean 5 --min-std 15
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

# Defaults chosen so that:
#   - An all-black image (mean ~0) fails MIN_MEAN.
#   - The specific "constant gray-brown noise" bug (mean ~116, std ~8.5) that
#     was reported on MLX/Apple Silicon fails MIN_STD.
#   - A real generated image (std typically >20 for any interesting prompt)
#     passes both thresholds comfortably.
MIN_MEAN = 5.0   # pixel brightness: catches all-black outputs
MIN_STD = 15.0   # pixel std-dev:   catches constant-noise / solid-colour outputs


def load_pixel_values(path: Path) -> list[int]:
    """Return a flat list of uint8 channel values from *path* (any mode)."""
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError:
        sys.exit(
            "PIL (Pillow) is required for check_image.py — "
            "install it with: pip install Pillow"
        )
    img = Image.open(path).convert("RGB")
    # tobytes() returns raw R,G,B bytes in row-major order — no deprecated API.
    return list(img.tobytes())


def main() -> None:
    p = argparse.ArgumentParser(description="Sanity-check a generated PNG.")
    p.add_argument("image", type=Path, help="Path to the PNG file to check.")
    p.add_argument(
        "--min-mean",
        type=float,
        default=MIN_MEAN,
        metavar="N",
        help=f"Minimum mean pixel brightness (default: {MIN_MEAN}).",
    )
    p.add_argument(
        "--min-std",
        type=float,
        default=MIN_STD,
        metavar="N",
        help=(
            f"Minimum pixel standard deviation (default: {MIN_STD}). "
            "Values below this suggest constant noise or a solid colour."
        ),
    )
    args = p.parse_args()

    path: Path = args.image
    if not path.exists():
        sys.exit(f"FAIL: file not found: {path}")

    values = load_pixel_values(path)
    if not values:
        sys.exit(f"FAIL: image has no pixels: {path}")

    mean = sum(values) / len(values)
    std = statistics.pstdev(values)  # population std (faster, no sample correction)

    # Per-channel means for diagnostics (values are interleaved R,G,B).
    r_vals = values[0::3]
    g_vals = values[1::3]
    b_vals = values[2::3]
    r_mean = sum(r_vals) / len(r_vals)
    g_mean = sum(g_vals) / len(g_vals)
    b_mean = sum(b_vals) / len(b_vals)

    print(
        f"check_image: {path.name}  "
        f"mean={mean:.1f}  std={std:.1f}  "
        f"R={r_mean:.1f}  G={g_mean:.1f}  B={b_mean:.1f}"
    )

    failures: list[str] = []
    if mean < args.min_mean:
        failures.append(
            f"mean brightness {mean:.1f} < {args.min_mean} "
            "(image appears all-black or near-black)"
        )
    if std < args.min_std:
        failures.append(
            f"pixel std-dev {std:.1f} < {args.min_std} "
            "(image looks like constant noise or a solid colour — "
            "possible text-encoder / scheduler bug)"
        )

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)

    print("OK: image passes quality checks")


if __name__ == "__main__":
    main()
