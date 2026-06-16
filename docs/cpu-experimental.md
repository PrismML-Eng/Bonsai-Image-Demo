# Experimental CPU Bring-Up

This repo's documented fast path is:

- macOS Apple Silicon with MLX artifacts
- Linux/Windows with NVIDIA GPU and GemLite/HQQ artifacts

The CPU path is experimental. It works from the GemLite/HQQ assets, not from the default macOS MLX download.

On Apple Silicon CPU-only runs, `--text-encoder-dtype auto` resolves to
`float16` to keep prompt-cache creation inside memory on 8 GB machines. The
main transformer/VAE dtype remains `auto`, which resolves to `float32` on this
host.

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

## Fresh clone quickstart

From a new checkout of this branch on a CPU-only Apple Silicon Mac:

```bash
git clone --branch dv/cpu-image-bringup-report \
  https://github.com/DenisValeev/Bonsai-Image-Demo.git \
  /tmp/Bonsai-image-demo-cpu
cd /tmp/Bonsai-image-demo-cpu

BONSAI_ALLOW_UNSUPPORTED=1 SKIP_DOWNLOAD=1 ./setup.sh
./scripts/download_model.sh --model ternary-gemlite
./scripts/export_unpacked_cpu_transformer.sh
./scripts/preflight_cpu_experimental.sh
./scripts/generate_cpu_low_memory.sh \
  --prompt "ostrich" \
  --output outputs/cpu-ostrich-128.png \
  --height 128 \
  --width 128 \
  --steps 4 \
  --seed 7
```

Notes:

- Run `preflight_cpu_experimental.sh` after `export_unpacked_cpu_transformer.sh`.
  Before export, a fresh checkout is expected to be missing
  `models/bonsai-image-4B-ternary-unpacked/transformer`.
- If you are reusing a checkout that already has an unpacked transformer from
  an older CPU export, refresh it with:

```bash
./scripts/export_unpacked_cpu_transformer.sh --overwrite
```

## Setup

`hqq` is part of the CPU path because the text encoder is stored as HQQ weights. The CPU export/render scripts also import `diffusers`, `safetensors`, and `transformers` directly, so `./setup.sh` needs to leave those in the venv on macOS as well as Linux.

If you are on a machine without NVIDIA, let setup continue without claiming the GPU path will work:

```bash
BONSAI_ALLOW_UNSUPPORTED=1 SKIP_DOWNLOAD=1 ./setup.sh
./scripts/download_model.sh --model ternary-gemlite
```

If you already ran setup before this dependency fix landed, refresh the venv once:

```bash
uv sync
```

Then export the unpacked transformer once. The exporter writes the Diffusers
safetensors payload directly from the GemLite state one tensor at a time; it
does not materialize the full dense transformer in memory.

```bash
./scripts/export_unpacked_cpu_transformer.sh
```

Run the preflight check before trying to render:

```bash
./scripts/preflight_cpu_experimental.sh
```

## Verified clean-room walkthrough

The sequence below was rerun in a fresh checkout on this ARM64 CPU-only host on
June 10, 2026, and it produced a semantically correct final ostrich PNG.

Fresh checkout:

```bash
git clone --branch dv/cpu-image-bringup-report \
  git@github.com:DenisValeev/Bonsai-Image-Demo.git \
  /tmp/Bonsai-image-demo-cpu
cd /tmp/Bonsai-image-demo-cpu
```

Install dependencies and download the GemLite model:

```bash
BONSAI_ALLOW_UNSUPPORTED=1 SKIP_DOWNLOAD=1 ./setup.sh
./scripts/download_model.sh --model ternary-gemlite
```

Export the unpacked transformer and check prerequisites:

```bash
./scripts/export_unpacked_cpu_transformer.sh
./scripts/preflight_cpu_experimental.sh
```

Render a final-only `128x128` ostrich:

```bash
./scripts/generate_cpu_low_memory.sh \
  --prompt "ostrich" \
  --output outputs/cpu-ostrich.png \
  --height 128 \
  --width 128 \
  --steps 4 \
  --seed 7
```

Expected result:

- `outputs/cpu-ostrich.png` is a coherent `128x128` ostrich, not tiled/noisy
  garbage.

The export script writes a monolithic safetensors payload. That matches the
loader path used for the validated CPU bring-up flow.

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
- `512x512` also completed on this 8 GB Apple Silicon host after the GemLite
  layout fix: about 255 seconds wall time, 5.47 GB peak memory footprint.
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
