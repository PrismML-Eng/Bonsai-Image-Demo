# M5 (Apple GPU gen ≥ 17) gray-noise bug — root cause & fix

> **Most users don't need to read this.** `setup.sh` detects the miscompile
> automatically and either applies the fix (ternary) or prints the stopgap
> (binary). This doc is background + reference for when the automatic path
> doesn't cover you: the binary (1-bit) model, modifying `.metal` kernels, or
> rebasing the fork.

## Symptom
On M5 / M5 Pro / M5 Max, every generation is the same gray-brown noise texture
(`mean≈116, std≈8.7`) regardless of prompt, seed, size, or steps. M4 and earlier
are unaffected. Tracked in issue #6.

## Root cause
**Xcode 26.5 / Metal toolchain `metalfe-32023.883` miscompiles MLX's M5-only
NAX (Neural Accelerator) GEMM Metal shaders.** It is *not* an mlx source bug and
*not* a hardware bug:

- bf16/fp16 matmul on M5 returns decorrelated, ~½-magnitude garbage (`A@B` GPU vs
  CPU relative error ≈ **1.1**); fp32 is fine.
- This corrupts every transformer matmul → the text conditioning is washed out →
  the model collapses to a fixed noise attractor (prompt has no effect). The VAE
  and text encoder are fine.
- M4 never runs the NAX path, so the miscompiled kernels are never executed there.

Evidence that it is the *compiled metallib*, not the source:

| mlx build | metallib compiled by | bf16 `A@B` GPU-vs-CPU | result |
|---|---|---|---|
| fork `b9effaf6`, local | Xcode 26.5 | ~1.1 | ❌ noise |
| upstream `v0.31.2`, local | Xcode 26.5 | ~1.1 | ❌ noise |
| upstream `a6222f53d`/`2414e5df6`, local | Xcode 26.5 | ~1.1 | ❌ noise |
| prebuilt PyPI `mlx-metal` 0.31.2 | mlx CI (SDK 26.4) | ~1e-3 | ✅ correct |

Swapping *only* the `mlx.metallib` (26.4-compiled → into the otherwise-broken
local build) flips it from broken to correct. Same `.so`, same source, same M5.
Upstream: https://github.com/ml-explore/mlx/issues/3586

## Ternary vs binary — they need different fixes
This matters because the mlx fork's 1-bit support **adds custom Metal kernels**
(`quantized.metal`, `quantized_nax.metal`), and upstream mlx has **no 1-bit
support at all** (v0.31.2 `ops.cpp` rejects `bits < 2`).

- **Ternary (2-bit):** uses the *standard* upstream quant/NAX kernels, which are
  present in the prebuilt `mlx-metal` wheel. → Fix A (prebuilt metallib) works,
  full NAX speed. The fork isn't even required for ternary.
- **Binary (1-bit):** needs the fork's custom 1-bit kernels, which are **not** in
  the stock wheel. Dropping in the prebuilt metallib would remove them. → on M5,
  build the fork metallib with Xcode 26.4 (Fix B), run NAX-off (`g16s`), or get
  1-bit support upstreamed (see below).

## Fix A — ship the prebuilt `mlx-metal` metallib (ternary; default in `setup.sh`)
`setup.sh` step 5b, when the variant is **ternary**, builds the fork's mlx *core*
from source, overwrites the locally-compiled `mlx.metallib` with the matching
prebuilt `mlx-metal` wheel, and verifies the GPU matmul is correct. For the
**binary** variant it skips the swap (to preserve 1-bit kernels) and prints the
`g16s` / Xcode-26.4 guidance.

This keeps NAX **enabled** → correct **and** fastest (it also picks up newer NAX
tuning than the fork's pinned base):

| config (512² / 1024², warm) | correct | speed |
|---|---|---|
| fork native NAX (old, broken) | ❌ | 2.79 s / 10.46 s |
| `MLX_METAL_GPU_ARCH=g16s` (NAX off) | ✅ | 6.92 s / 24.07 s |
| **rebased core + prebuilt 0.31.2 metallib** | ✅ | **2.27 s / 8.65 s** |

**Requirement:** the built mlx version must match a released `mlx-metal` wheel,
so the fork must be pinned to a tagged release (e.g. **0.31.2**), not an arbitrary
dev commit. See "Rebasing the fork" below.

## Rebasing the fork onto a tagged mlx (required for Fix A)
The fork's only change is one line in `mlx/backend/metal/quantized.cpp`:

```diff
-  if ((K == 128 || K == 64) && is_power_of_2(bits)) {
+  if ((K == 128 || (K == 64 && bits >= 2)) && is_power_of_2(bits)) {
```

Reapply it on top of `v0.31.2` (latest 0.31.x; satisfies mflux's `<0.32.0`):

```sh
git clone https://github.com/ml-explore/mlx.git && cd mlx
git checkout -b m5-fix-0.31.2 v0.31.2
# apply the one-line guard above to mlx/backend/metal/quantized.cpp
git commit -am "Guard fast-path Metal kernel dispatch for 1-bit quantization"
git push <PrismML-Eng/mlx remote> m5-fix-0.31.2
```

Then point the build at the new rev in `vendor/image-studio/pyproject.toml`:

```toml
[tool.uv.sources]
mlx = { git = "https://github.com/PrismML-Eng/mlx.git", rev = "<new sha>" }
```

(Verified: `v0.31.2 + this patch` core, built locally, + prebuilt 0.31.2
metallib → matmul correct, images clean.)

## Fix B — compile the metallib locally with the Xcode 26.4 toolchain (belt & suspenders)
If you must build the metallib from source (e.g. you modify `.metal` kernels),
use the **Xcode 26.4** Metal toolchain instead of 26.5:

```sh
xcodes install 26.4            # ~10 GB, Apple-ID gated
sudo xcode-select --switch /Applications/Xcode-26.4.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain   # pulls the 26.4 toolchain
# then rebuild mlx from source as usual
```

Correct, but does **not** include newer NAX tuning unless you also rebase the
fork — so Fix A is preferred.

## Best long-term — upstream the 1-bit support, then drop the fork
Upstream mlx rejects `bits < 2`, so the fork exists almost entirely for the
binary (1-bit) model. If the fork's 1-bit affine-quant kernels are upstreamed
(PR to ml-explore/mlx), then a released `mlx` + prebuilt `mlx-metal` would cover
**both** ternary and binary — no fork, no local metallib compile, so the Xcode
26.5 miscompile never enters the picture and both models run correct + full NAX
speed on M5. The 1-bit dispatch guard / qmv_fast fix would go in the same PR.
Until then: ternary via Fix A, binary via Fix B or the stopgap below.

## Stopgap — disable NAX (no rebuild)
Immediate, toolchain-independent, but ~2.5–3× slower (drops to the M4 kernel
path):

```sh
export MLX_METAL_GPU_ARCH=applegpu_g16s
```
