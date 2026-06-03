#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

from PIL import Image, ImageDraw


PERMUTATIONS = list(permutations(range(4)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["permute-2x2-within-block", "contact-sheet-2x2-within-block"],
        default="contact-sheet-2x2-within-block",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--cell-size", type=int, default=8)
    parser.add_argument("--perm-index", type=int, default=0)
    return parser.parse_args()


def split_quadrants(block: Image.Image, cell_size: int) -> list[Image.Image]:
    boxes = [
        (0, 0, cell_size, cell_size),
        (cell_size, 0, 2 * cell_size, cell_size),
        (0, cell_size, cell_size, 2 * cell_size),
        (cell_size, cell_size, 2 * cell_size, 2 * cell_size),
    ]
    return [block.crop(box) for box in boxes]


def join_quadrants(quads: list[Image.Image], perm: tuple[int, int, int, int], cell_size: int) -> Image.Image:
    block = Image.new("RGB", (2 * cell_size, 2 * cell_size))
    dests = [(0, 0), (cell_size, 0), (0, cell_size), (cell_size, cell_size)]
    for dst_idx, src_idx in enumerate(perm):
        block.paste(quads[src_idx], dests[dst_idx])
    return block


def permute_image(img: Image.Image, block_size: int, cell_size: int, perm: tuple[int, int, int, int]) -> Image.Image:
    width, height = img.size
    if width % block_size != 0 or height % block_size != 0:
        raise SystemExit(f"image size {img.size} must be divisible by block-size {block_size}")
    if block_size != 2 * cell_size:
        raise SystemExit("block-size must equal 2 * cell-size for the 2x2 permutation mode")

    out = Image.new("RGB", img.size)
    for top in range(0, height, block_size):
        for left in range(0, width, block_size):
            block = img.crop((left, top, left + block_size, top + block_size))
            quads = split_quadrants(block, cell_size)
            out.paste(join_quadrants(quads, perm, cell_size), (left, top))
    return out


def contact_sheet(img: Image.Image, block_size: int, cell_size: int) -> Image.Image:
    tiles: list[tuple[str, Image.Image]] = []
    for idx, perm in enumerate(PERMUTATIONS):
        label = f"{idx}: {perm}"
        tiles.append((label, permute_image(img, block_size, cell_size, perm)))

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    label_h = 18
    sheet = Image.new("RGB", (cols * img.width, rows * (img.height + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, tile) in enumerate(tiles):
        x = (idx % cols) * img.width
        y = (idx // cols) * (img.height + label_h)
        sheet.paste(tile, (x, y + label_h))
        draw.text((x + 4, y + 2), label, fill="black")
    return sheet


def main() -> int:
    args = parse_args()
    img = Image.open(args.input).convert("RGB")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "permute-2x2-within-block":
        perm = PERMUTATIONS[args.perm_index]
        out = permute_image(img, args.block_size, args.cell_size, perm)
    else:
        out = contact_sheet(img, args.block_size, args.cell_size)

    out.save(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
