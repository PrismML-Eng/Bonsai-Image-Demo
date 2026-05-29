# M5 (Apple GPU gen ≥ 17) gray-noise bug

> **You probably don't need this.** `setup.sh` detects the bug and fixes it
> automatically for the ternary model. Read on only if the automatic fix
> doesn't apply: the binary (1-bit) model, editing `.metal` kernels, or
> bumping the mlx fork.

## What you'd see
On M5 / M5 Pro / M5 Max, every generation is the same gray-brown noise
(`mean≈116, std≈8.7`) regardless of prompt/seed/size. M4 and earlier are fine.
Tracked in issue #6.

## Why
Xcode 26.5's Metal toolchain (`metalfe-32023.883`) miscompiles MLX's M5-only
NAX GEMM shaders, so bf16 matmul on the GPU returns ~½-magnitude garbage
(`A@B` GPU-vs-CPU error ≈ 1.1; fp32 is fine). That washes out the transformer's
text conditioning, collapsing every prompt to the same noise. It's the *locally
compiled* `mlx.metallib`, not the mlx source: dropping in a CI-compiled metallib
(same `.so`, same source) flips it from broken to correct.
Upstream: https://github.com/ml-explore/mlx/issues/3586

## What `setup.sh` does (ternary, automatic)
Runs a bf16 GPU-vs-CPU matmul check; if the local build is miscompiled, it
overwrites the source-built `mlx.metallib` with the matching prebuilt
`mlx-metal` wheel and re-verifies. Keeps NAX on → correct and full speed.

## Fallbacks (when the automatic swap doesn't apply)

**Quickest stopgap — disable NAX (any variant, no rebuild):** correct but
~2.5–3× slower (drops to the M4 kernel path).
```sh
export MLX_METAL_GPU_ARCH=applegpu_g16s
```

**Binary (1-bit) model:** the swap is skipped — the prebuilt wheel lacks the
fork's custom 1-bit kernels, so it can't replace the metallib. Use the `g16s`
stopgap above, or build the metallib with the **Xcode 26.4** toolchain:
```sh
xcodes install 26.4
sudo xcode-select --switch /Applications/Xcode-26.4.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
# then rebuild mlx from source
```

**Bumping the mlx fork:** the prebuilt swap needs the built mlx version to match
a released `mlx-metal` wheel, so pin the fork to a tagged release (e.g. 0.31.2),
not a dev commit. The fork's only change is one line in
`mlx/backend/metal/quantized.cpp` (guard the 1-bit fast path):
```diff
-  if ((K == 128 || K == 64) && is_power_of_2(bits)) {
+  if ((K == 128 || (K == 64 && bits >= 2)) && is_power_of_2(bits)) {
```
Reapply it on `v0.31.2`, push, and point `vendor/image-studio/pyproject.toml`'s
`[tool.uv.sources]` mlx `rev` at the new commit.

## Long-term
Upstream mlx rejects `bits < 2`, so the fork exists almost entirely for the
binary model. Upstreaming the 1-bit kernels would let a released `mlx` +
prebuilt `mlx-metal` cover both variants — no fork, no local metallib compile,
so the miscompile never applies.
