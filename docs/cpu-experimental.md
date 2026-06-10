# Experimental CPU Bring-Up

This repo's documented fast path is:

- macOS Apple Silicon with MLX artifacts
- Linux/Windows with NVIDIA GPU and GemLite/HQQ artifacts

The CPU path is experimental. It works from the GemLite/HQQ assets, not from the default macOS MLX download.

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
