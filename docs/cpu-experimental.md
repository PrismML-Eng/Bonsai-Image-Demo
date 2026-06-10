# Experimental CPU Bring-Up

This repo's documented fast path is:

- macOS Apple Silicon with MLX artifacts
- Linux/Windows with NVIDIA GPU and GemLite/HQQ artifacts

The CPU path is experimental. It works from the GemLite/HQQ assets, not from the default macOS MLX download.

## Current status

CPU image generation has been demonstrated end to end on the unpacked transformer
path at `128x128`.

Validated `128x128`, `4-step` outputs included:

- plain `ostrich`
- a large centered red circle
- a coherent ostrich silhouette
- a 4-quadrant color layout

That is enough to show that the unpacked CPU path can produce globally coherent
images for both simple geometry and object prompts.

## What to download

On macOS, `./scripts/download_model.sh` defaults to MLX weights. That is the wrong artifact set for `scripts/generate_cpu_experimental.py`.

Use the GemLite download explicitly:

```bash
./scripts/download_model.sh --model ternary-gemlite
```

The experimental CPU runner expects all of these to exist:

- `models/bonsai-image-4B-ternary-gemlite/text_encoder-hqq-4bit`
- `models/bonsai-image-4B-ternary-gemlite/vae`
- `models/bonsai-image-4B-ternary-gemlite/transformer-gemlite-int2`
- `models/bonsai-image-4B-ternary-unpacked/transformer`

## Setup

`hqq` is part of the CPU path because the text encoder is stored as HQQ weights. `./setup.sh` now installs it on macOS and Linux.

If you are on a machine without NVIDIA, let setup continue without claiming the GPU path will work:

```bash
BONSAI_ALLOW_UNSUPPORTED=1 SKIP_DOWNLOAD=1 ./setup.sh
./scripts/download_model.sh --model ternary-gemlite
```

Then export the unpacked transformer once:

```bash
./scripts/export_unpacked_cpu_transformer.sh
```

Run the preflight check before trying to render:

```bash
./scripts/preflight_cpu_experimental.sh
```

## Verified clean-room walkthrough

The sequence below was rerun in a fresh checkout on this ARM64 CPU-only host on
June 10, 2026.

Fresh checkout:

```bash
git clone --branch dv/cpu-image-bringup-report \
  git@github.com:DenisValeev/Bonsai-Image-Demo.git \
  /tmp/Bonsai-image-demo-cpu
cd /tmp/Bonsai-image-demo-cpu
```

Install dependencies and download the GemLite model:

```bash
BONSAI_ALLOW_UNSUPPORTED=1 ./setup.sh
```

Export the unpacked transformer and check prerequisites:

```bash
./scripts/export_unpacked_cpu_transformer.sh
./scripts/preflight_cpu_experimental.sh
```

Try the low-memory CPU ostrich render:

```bash
./scripts/generate_cpu_low_memory.sh \
  --prompt "ostrich" \
  --output outputs/cpu-ostrich.png \
  --height 128 \
  --width 128 \
  --steps 4 \
  --seed 7
```

### Important current limitation

That exact fresh-export path still produced a semantically wrong tiled/noisy
image, even though the render completed successfully.

The current working workaround is to replace the freshly exported unpacked
transformer with a known-good monolithic checkpoint, then rerun the same ostrich
command:

```bash
rm -rf models/bonsai-image-4B-ternary-unpacked/transformer
mkdir -p models/bonsai-image-4B-ternary-unpacked/transformer
cp /path/to/known-good/config.json \
  models/bonsai-image-4B-ternary-unpacked/transformer/
cp /path/to/known-good/diffusion_pytorch_model.safetensors \
  models/bonsai-image-4B-ternary-unpacked/transformer/

./scripts/generate_cpu_low_memory.sh \
  --prompt "ostrich" \
  --output outputs/cpu-ostrich.png \
  --height 128 \
  --width 128 \
  --steps 4 \
  --seed 7
```

With that validated monolithic checkpoint in place, the fresh checkout produced
a coherent `128x128` ostrich.

## Low-memory flow

For low-memory CPU runs, keep prompt encoding separate from VAE/transformer loading. This avoids overlapping text-encoder memory with the render process.

Use the wrapper:

```bash
./scripts/generate_cpu_low_memory.sh \
  --prompt "ostrich" \
  --output outputs/cpu-ostrich.png \
  --height 128 \
  --width 128 \
  --steps 4 \
  --seed 7
```

Or run the two-process flow directly:

```bash
python scripts/generate_cpu_experimental.py \
  --prompt "ostrich" \
  --output outputs/cpu-ostrich.png \
  --height 128 \
  --width 128 \
  --steps 4 \
  --seed 7 \
  --prompt-cache-dir outputs/prompt_cache_fp32_auto \
  --prompt-cache-only

python scripts/generate_cpu_experimental.py \
  --prompt "ostrich" \
  --output outputs/cpu-ostrich.png \
  --height 128 \
  --width 128 \
  --steps 4 \
  --seed 7 \
  --prompt-cache-dir outputs/prompt_cache_fp32_auto
```

## Smoke test target

Start with `128x128` and `4` steps.

- `128x128` is the practical CPU smoke-test shape used during bring-up.
- `64x64` is too small for reliable structure checks.
- `1024x1024` is not a reasonable first target on an 8 GB RAM machine.

At this resolution regime:

- `64x64` corresponds to a `4x4` packed latent grid
- `128x128` corresponds to an `8x8` packed latent grid

## Validation guidance

- Treat final outputs as the primary correctness signal.
- Do not over-index on intermediate-step VAE decodes; they can be misleading.
- Use the unpacked transformer CPU path as the reference bring-up configuration.

## Reproduction shape

Example command shape used for successful CPU validation:

```bash
python scripts/generate_cpu_experimental.py \
  --prompt "ostrich" \
  --output outputs/cpu-ostrich.png \
  --height 128 \
  --width 128 \
  --steps 4 \
  --seed 7 \
  --transformer-dir models/bonsai-image-4B-ternary-unpacked/transformer
```
