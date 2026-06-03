# CPU image generation works end to end on unpacked path at 128x128

## Summary

CPU image generation is working end to end on the unpacked transformer path at `128x128`.

The clearest validated outputs were:

- plain `ostrich`
- a clean 4-quadrant multi-color layout
- a large centered red circle
- a coherent ostrich silhouette

This is useful because it establishes that the CPU path can already do:

- globally coherent composition
- structured color layouts
- simple object-level prompts

## Key evidence

Validated `128x128`, `4-step` outputs:

- `ostrich`
- `a large red circle centered on a pure white background, filling most of the image, flat solid color, hard clean edge, no other objects, minimalist`
- `a large black silhouette of an ostrich centered on a pure white background, full body, hard clean edge, minimalist, no other objects`
- a 4-quadrant color layout prompt

## Practical CPU guidance

Two practical constraints stood out during bring-up:

1. `64x64` is a poor structure/composition validation target for this pipeline.
2. Final outputs are much more reliable than intermediate-step VAE decodes when judging correctness.

At this resolution regime:

- `64x64` corresponds to a `4x4` packed latent grid
- `128x128` corresponds to an `8x8` packed latent grid

Using `128x128` made the difference between misleading geometry tests and meaningful validation.

## Reproduction shape

Example command shape used for successful CPU runs:

```bash
python scripts/generate_cpu_experimental.py \
  --prompt 'ostrich' \
  --height 128 \
  --width 128 \
  --steps 4 \
  --seed 7 \
  --transformer-dir models/bonsai-image-4B-ternary-unpacked/transformer
```

## Suggested takeaway

The unpacked CPU path appears good enough to document as a real bring-up configuration for `bonsai-image`, at least for `128x128` validation and simple-to-moderate image composition.
