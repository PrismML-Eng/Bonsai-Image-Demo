# CPU image generation now demonstrated end to end

## Summary

This note documents a successful CPU demonstration for `bonsai-image` on the unpacked transformer path.

The headline result is straightforward: CPU image generation can produce coherent `128x128` outputs end to end.

The two clearest examples are:

- a coherent plain `ostrich`
- a clean 4-quadrant multi-color layout

Additional validated outputs include:

- a coherent ostrich silhouette
- a large centered red circle
- the same plain `ostrich` result in repeated successful runs

That establishes three concrete facts:

1. the unpacked CPU path can converge on globally coherent images
2. both geometric and object-level prompts can work on CPU
3. `128x128` is a practical resolution for validating CPU image generation behavior

This is not a full root-cause fix report for every failure mode. It is a status report showing that CPU image generation is real, reproducible, and already capable of meaningful outputs.

## Demonstrated capability

For this pipeline:

- `64x64` corresponds to a `4x4` packed latent grid
- `128x128` corresponds to an `8x8` packed latent grid

On the unpacked transformer CPU path, `128x128` was sufficient to demonstrate:

- global composition across many packed tokens
- successful convergence on simple object prompts
- successful convergence on large centered shapes
- successful convergence on structured color layouts

## Validated `128x128` outputs

On the CPU path using the unpacked transformer, these `128x128`, `4-step` prompts converged coherently:

- plain `ostrich`
- `a large red circle centered on a pure white background, filling most of the image, flat solid color, hard clean edge, no other objects, minimalist`
- `a large black silhouette of an ostrich centered on a pure white background, full body, hard clean edge, minimalist, no other objects`
- a 4-quadrant color layout prompt

## Suggested upstream guidance

It may help to document the following as the current practical CPU guidance:

1. Use `128x128+` for structure/composition validation.
2. Treat final outputs as the primary correctness signal.
3. Use the unpacked transformer CPU path as a reference-capable configuration for image bring-up.

## Runner-side settings that matched successful runs

Two runner-side settings were helpful in the successful CPU runs:

1. Increasing default prompt context from `64` to `512`
2. Using `128x128` instead of `64x64` for meaningful structure/composition validation

## Reproduction shape

Example command shape used for successful CPU runs:

```bash
python scripts/generate_cpu_experimental.py \
  --prompt 'ostrich' \
  --output outputs/cpu-ostrich.png \
  --height 128 \
  --width 128 \
  --steps 4 \
  --seed 7 \
  --transformer-dir models/bonsai-image-4B-ternary-unpacked/transformer
```

## Example output categories

- plain ostrich
- ostrich silhouette
- large centered red circle
- 4-quadrant color layout

## Notes

- This note intentionally avoids host-specific details, private paths, tokens, and unrelated local setup details.
- If a PR is not the right venue, the same content could be posted as an issue or discussion instead.
