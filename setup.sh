#!/bin/sh
# Bonsai Image Demo — One-command setup for macOS and Linux.
# Installs uv, Python venv, and mflux. Does NOT download models.
#
# Usage:
#   ./setup.sh
set -e

# ── Resolve paths ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
. "$SCRIPT_DIR/scripts/common.sh"

VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"
PYTHON_VERSION="3.11"
: "${BONSAI_ALLOW_UNSUPPORTED:=0}"

# Semver comparison: returns 0 if $1 >= $2
_version_ge() {
    _a=$1 _b=$2
    while [ -n "$_a" ] || [ -n "$_b" ]; do
        _ap=${_a%%.*} _bp=${_b%%.*}
        [ "$_a" = "$_ap" ] && _a="" || _a=${_a#*.}
        [ "$_b" = "$_bp" ] && _b="" || _b=${_b#*.}
        [ -z "$_ap" ] && _ap=0; [ -z "$_bp" ] && _bp=0
        [ "$_ap" -gt "$_bp" ] 2>/dev/null && return 0
        [ "$_ap" -lt "$_bp" ] 2>/dev/null && return 1
    done
    return 0
}

# Smart apt install (try without sudo, escalate if needed)
_smart_apt_install() {
    _pkgs="$*"

    apt-get update -y </dev/null >/dev/null || true
    apt-get install -y $_pkgs </dev/null >/dev/null || true

    _still_missing=""
    for _p in $_pkgs; do
        case "$_p" in
            build-essential) command -v gcc >/dev/null 2>&1 || _still_missing="$_still_missing $_p" ;;
            *) command -v "$_p" >/dev/null 2>&1 || _still_missing="$_still_missing $_p" ;;
        esac
    done
    _still_missing=$(echo "$_still_missing" | sed 's/^ *//')
    [ -z "$_still_missing" ] && return 0

    if command -v sudo >/dev/null 2>&1; then
        echo ""
        warn "Need elevated permissions to install: $_still_missing"
        printf "  Allow sudo? [Y/n] "
        if [ -r /dev/tty ]; then read -r _yn </dev/tty; else read -r _yn; fi
        case "$_yn" in
            [nN]*)
                echo "  Please install manually: sudo apt-get install -y $_still_missing"
                exit 1 ;;
            *)
                sudo apt-get update -y </dev/null
                sudo apt-get install -y $_still_missing </dev/null ;;
        esac
    else
        err "sudo not available. Install as root: apt-get install -y $_still_missing"
        exit 1
    fi
}

echo ""
echo "========================================="
echo "   Bonsai Image Demo Setup"
echo "========================================="
echo ""

# ────────────────────────────────────────────────────
#  1. Detect platform
# ────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
step "Detected platform: $OS ($ARCH)"

case "$OS" in
    Darwin)
        if [ "$ARCH" != "arm64" ]; then
            err "mflux requires Apple Silicon (arm64). Intel Macs are not supported."
            exit 1
        fi
        step "Checking Xcode Command Line Tools ..."
        if ! xcode-select -p >/dev/null 2>&1; then
            warn "Xcode CLT not installed. Installing now (a system dialog will appear) ..."
            xcode-select --install </dev/null || true
            echo ""
            echo "  After the Xcode CLT installation completes, please re-run:"
            echo "    ./setup.sh"
            exit 1
        fi
        info "Xcode CLT found at $(xcode-select -p)"

        # mlx compiles Metal shaders at build time, which needs the full Xcode
        # app *and* the Metal Toolchain component — not just the CLT. Two
        # failure modes are surfaced separately:
        #   1. `metal` binary missing entirely (CLT-only setup).
        #   2. `metal` present but can't execute (toolchain component absent /
        #      Xcode first-launch not completed).
        step "Checking Metal toolchain (required by mlx) ..."
        if ! xcrun metal --version >/dev/null 2>&1; then
            if ! xcrun --find metal >/dev/null 2>&1; then
                err "The 'metal' shader compiler isn't available."
                echo ""
                echo "  mlx needs the full Xcode app (not just CLT)."
                echo "  1. Install Xcode from the App Store:"
                echo "       https://developer.apple.com/xcode/"
                echo "  2. Point xcode-select at it:"
                echo "       sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
                echo "  3. Accept the license and complete first-launch setup:"
                echo "       sudo xcodebuild -license accept"
                echo "       xcodebuild -runFirstLaunch"
                echo "  4. Download the Metal Toolchain component:"
                echo "       xcodebuild -downloadComponent MetalToolchain"
                echo "  5. Re-run ./setup.sh"
            else
                err "The 'metal' compiler is present but can't execute."
                echo ""
                echo "  The Metal Toolchain component is probably not installed."
                echo "  Run, then re-run ./setup.sh:"
                echo "    sudo xcodebuild -license accept"
                echo "    xcodebuild -runFirstLaunch"
                echo "    xcodebuild -downloadComponent MetalToolchain"
            fi
            exit 1
        fi
        info "Metal toolchain ok: $(xcrun metal --version 2>&1 | head -n 1)"
        ;;

    Linux)
        step "Checking system packages ..."
        _missing=""
        command -v git  >/dev/null 2>&1 || _missing="$_missing git"
        command -v gcc  >/dev/null 2>&1 || _missing="$_missing build-essential"
        command -v curl >/dev/null 2>&1 && true || {
            command -v wget >/dev/null 2>&1 || _missing="$_missing curl"
        }
        _missing=$(echo "$_missing" | sed 's/^ *//')

        if [ -n "$_missing" ]; then
            warn "Missing packages: $_missing"
            if command -v apt-get >/dev/null 2>&1; then
                _smart_apt_install $_missing
            else
                err "apt-get not found. Please install: $_missing"
                exit 1
            fi
        fi
        info "System packages OK."

        # GPU check (non-fatal). Linux path skips mlx/mflux entirely and runs
        # gemlite + HQQ kernels via image-studio's backend_gpu.
        if has_nvidia_gpu; then
            info "NVIDIA GPU detected — using gemlite/HQQ via backend_gpu (mlx/mflux skipped)."
        else
            if [ "$BONSAI_ALLOW_UNSUPPORTED" = "1" ]; then
                warn "No NVIDIA GPU detected. backend_gpu needs CUDA — generation will fail."
                echo "       BONSAI_ALLOW_UNSUPPORTED=1 is set, so setup will continue anyway."
            else
                err "No NVIDIA GPU detected. This Linux path requires NVIDIA/CUDA and would only fail later during generation."
                echo ""
                echo "  Supported local targets:"
                echo "    - Apple Silicon macOS (MLX)"
                echo "    - Linux with NVIDIA GPU + driver/CUDA"
                echo ""
                echo "  Experimental CPU bring-up is documented in:"
                echo "    docs/cpu-experimental.md"
                echo ""
                echo "  If you only want the files/models installed for inspection, rerun with:"
                echo "    BONSAI_ALLOW_UNSUPPORTED=1 ./setup.sh"
                exit 1
            fi
        fi
        ;;

    *)
        err "Unsupported OS: $OS. mflux supports macOS (Apple Silicon) and Linux (NVIDIA CUDA)."
        exit 1
        ;;
esac

force_native_macos_build_arch "$OS" "$ARCH"

# ────────────────────────────────────────────────────
#  2. Install uv
# ────────────────────────────────────────────────────
UV_MIN="0.7.0"

_uv_ok() {
    command -v uv >/dev/null 2>&1 || return 1
    _ver=$(uv --version 2>/dev/null | awk '{print $2}')
    [ -n "$_ver" ] && _version_ge "$_ver" "$UV_MIN"
}

step "Checking uv ..."
if _uv_ok; then
    info "uv $(uv --version 2>/dev/null | awk '{print $2}') found."
else
    step "Installing uv ..."
    _tmp=$(mktemp)
    download "https://astral.sh/uv/install.sh" "$_tmp"
    sh "$_tmp" </dev/null
    rm -f "$_tmp"
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    export PATH="$HOME/.local/bin:$PATH"
    if ! _uv_ok; then
        err "uv installation failed. Install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
    info "uv installed."
fi

# ────────────────────────────────────────────────────
#  3. Create Python venv
# ────────────────────────────────────────────────────
step "Setting up Python environment ..."
if [ -x "$VENV_PY" ]; then
    info "Existing venv found at $VENV_DIR"
else
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    info "Created venv with Python $PYTHON_VERSION"
fi

# ────────────────────────────────────────────────────
#  4. Clone private deps into vendor/
# ────────────────────────────────────────────────────
# Private repos are pulled via HTTPS so users don't need an SSH key configured.
# When these repos go public, the [tool.uv.sources] entries in pyproject.toml
# can be swapped for plain version specs and this section becomes obsolete.
VENDOR_DIR="$SCRIPT_DIR/vendor"
mkdir -p "$VENDOR_DIR"

_clone_vendor() {
    _name="$1" _url="$2" _branch="${3:-}"
    if [ -d "$VENDOR_DIR/$_name/.git" ]; then
        info "vendor/$_name already cloned — skipping (git pull manually to update)."
    else
        step "Cloning $_name${_branch:+ ($_branch)} into vendor/ ..."
        if [ -n "$_branch" ]; then
            git clone --branch "$_branch" "$_url" "$VENDOR_DIR/$_name"
        else
            git clone "$_url" "$VENDOR_DIR/$_name"
        fi
    fi
}

# image-studio is now tracking main — the PR `fix/dynamic-backend-and-presets`
# (dynamic /backends, 32-aligned presets, backend_gpu packaging) was merged
# 2026-05-14, so the build-system + resolutions.ts patches further below are
# no longer needed at clone time.
_clone_vendor image-studio https://github.com/PrismML-Eng/image-studio.git
_clone_vendor mflux-prism  https://github.com/PrismML-Eng/mflux-prism.git

# image-studio's upstream still pins mflux to a git rev, which conflicts with
# our vendor path source (uv treats path vs git as different URLs even when
# they resolve to the same commit). Rewrite it to the sibling path that's now
# present at vendor/mflux-prism. Idempotent: only touches the file if the git
# source is still there. Remove this once image-studio's upstream is updated.
_studio_pp="$VENDOR_DIR/image-studio/pyproject.toml"
if grep -q '^mflux = { git = ' "$_studio_pp"; then
    step "Patching vendor/image-studio mflux source to match vendor layout ..."
    sed -i.bak 's|^mflux = { git = .*$|mflux = { path = "../mflux-prism", editable = true }|' "$_studio_pp"
    rm -f "$_studio_pp.bak"
fi

# --- DISABLED while pinned to fix/dynamic-backend-and-presets ----------------
# Both patches below are on the PR branch. Uncomment if you switch
# _clone_vendor back to plain main and want to keep things working in the
# interim.
#
# # image-studio's backend_gpu/pyproject.toml ships without a [build-system] or
# # [tool.setuptools] section, so uv falls back to setuptools' legacy build and
# # its flat-layout autodiscovery aborts when it sees diffusion_klein.py +
# # pipeline_gpu.py + server.py as competing top-level modules. Map the package
# # name onto `.` so the editable install works. Idempotent — skipped once
# # upstream image-studio adds these sections.
# _gpu_pp="$VENDOR_DIR/image-studio/backend_gpu/pyproject.toml"
# if [ -f "$_gpu_pp" ] && ! grep -q '^\[build-system\]' "$_gpu_pp"; then
#     step "Patching vendor/image-studio/backend_gpu/pyproject.toml (build-system + setuptools layout) ..."
#     _gpu_pp_tmp="$_gpu_pp.tmp.$$"
#     {
#         printf '[build-system]\nrequires = ["setuptools>=68"]\nbuild-backend = "setuptools.build_meta"\n\n'
#         cat "$_gpu_pp"
#         printf '\n[tool.setuptools]\npackages = ["backend_gpu"]\npackage-dir = {"backend_gpu" = "."}\n'
#     } > "$_gpu_pp_tmp"
#     mv "$_gpu_pp_tmp" "$_gpu_pp"
# fi
#
# # image-studio's frontend picker lists 3:2 fast-tier sizes 624×416 / 416×624,
# # but diffusion_klein.py rejects anything not a multiple of 32. Rewrite them
# # to 576×384 / 384×576 (same exact 3:2 aspect, ~0.22 MP, both mult-of-32).
# # Idempotent — once upstream is updated, the grep below stops matching.
# _res_ts="$VENDOR_DIR/image-studio/frontend/lib/resolutions.ts"
# if [ -f "$_res_ts" ] && grep -q 'width: 624, height: 416' "$_res_ts"; then
#     step "Patching vendor/image-studio/frontend/lib/resolutions.ts (3:2 fast sizes → mult-of-32) ..."
#     sed -i.bak \
#         -e 's|3:2 — 624 × 416", aspect: "3:2", width: 624, height: 416|3:2 — 576 × 384", aspect: "3:2", width: 576, height: 384|' \
#         -e 's|2:3 — 416 × 624", aspect: "2:3", width: 416, height: 624|2:3 — 384 × 576", aspect: "2:3", width: 384, height: 576|' \
#         "$_res_ts"
#     rm -f "$_res_ts.bak"
# fi
# ----------------------------------------------------------------------------

# ────────────────────────────────────────────────────
#  5. Install mflux
# ────────────────────────────────────────────────────
step "Installing mflux ..."
_pkg_cutoff="$(package_cutoff_date)"

# MLX's build bakes preprocessed Metal headers into the wheel as "compiled
# preambles" for runtime JIT. If `xcrun metal` was unavailable when the build
# ran (e.g. Metal Toolchain not yet installed), the preprocessor produces
# error text that gets captured as the preamble and persists in the cached
# wheel — subsequent builds reuse it without regenerating. If `uv sync` fails
# we wipe uv's git+build caches for mlx and retry once.
_uv_sync_with_retry() {
    if uv sync --exclude-newer "$_pkg_cutoff"; then
        return 0
    fi
    warn "uv sync failed. Wiping cached mlx git checkout / build dir and retrying once."
    # uv's cache layout varies slightly across versions; rm both well-known
    # locations and any mlx-bearing entry under them.
    for _glob in \
        "$HOME/.cache/uv/git-v0/checkouts/"*mlx*/ \
        "$HOME/.cache/uv/builds-v0/"*mlx*/ \
        "$HOME/.cache/uv/sdists-v0/"*mlx*/ ; do
        for _path in $_glob; do
            [ -e "$_path" ] && rm -rf "$_path"
        done
    done 2>/dev/null
    uv sync --exclude-newer "$_pkg_cutoff" --reinstall-package mlx
}
_uv_sync_with_retry
info "mflux installed (versions <= $_pkg_cutoff)."

# ────────────────────────────────────────────────────
#  5b. M5 NAX metallib fix (Metal-compiler miscompile workaround)
# ────────────────────────────────────────────────────
# Xcode 26.5 / metalfe-32023.883 miscompiles MLX's M5-only NAX GEMM shaders,
# so the locally compiled mlx.metallib produces garbage matmuls on M5+ (Apple
# GPU gen >= 17) — the "gray-brown noise" bug. The shaders themselves are fine:
# the prebuilt `mlx-metal` wheel (compiled with a correct SDK) works.
# See PrismML/Bonsai-Image-Demo#6 and ml-explore/mlx#3586.
#
# Gate-first: we detect the miscompile with a bf16 GPU-vs-CPU matmul check and
# ONLY swap the metallib when it's actually broken. So M4 (NAX path unused) and
# any machine with a fixed Xcode/Metal toolchain keep their native build
# untouched — no needless 150 MB download, and the binary variant keeps the
# fork's custom kernels unless the bug is genuinely present.
#
# IMPORTANT — ternary vs binary (only matters when the bug IS detected):
#   * ternary (2-bit) uses standard upstream kernels, present in the prebuilt
#     `mlx-metal` wheel, so we can drop the prebuilt metallib in over the
#     source-built one. Correct + full NAX speed.
#   * binary (1-bit) needs the fork's CUSTOM 1-bit kernels (quantized*.metal),
#     which are NOT in the stock wheel — swapping would remove them. So for
#     binary on M5 we keep the fork metallib and fall back to NAX-off (g16s),
#     or you must compile the fork metallib with the Xcode 26.4 toolchain.
# See docs/m5-metal.md.
#
# The ternary swap is destructive for binary (it strips the fork's 1-bit
# kernels), and the variant is normally chosen per-invocation at generation
# time, not via BONSAI_VARIANT at setup time. So besides the env var, we refuse
# to swap whenever a 1-bit model is already present under models/. Binary users
# who run setup before downloading a model should export BONSAI_VARIANT=binary.
_variant="${BONSAI_VARIANT:-ternary}"
if [ -z "${BONSAI_VARIANT:-}" ]; then
    for _bin_model in "$SCRIPT_DIR"/models/bonsai-image-4B-binary-*; do
        [ -d "$_bin_model" ] && _variant="binary" && break
    done
fi

# bf16 GPU-vs-CPU matmul check: exit 0 if the GPU kernel is correct (reldiff
# < 0.1), 1 if it's miscompiled (the M5 NAX bug gives reldiff ~1.1).
#
# The probe must be large enough to select the miscompiled M5 NAX GEMM kernel
# on EVERY M5 variant. The NAX dispatch threshold rises as GPU core count drops,
# so a probe sized for the 20-core M5 Pro (g17s) can stay below the 10-core M5
# Air's (g17g) threshold and read clean on a broken build — a false negative
# that skips the swap and leaves output gray (issue #6). History: 64x512 missed
# the Pro; 1024x2048 @ 2048 still missed the Air. So probe at the model's own
# matmul size (K=3072, N=12288) — then NAX engages exactly when the model's
# matmuls do, regardless of core count.
_mlx_matmul_ok() {
    "$VENV_PY" - <<'PYV'
import sys, mlx.core as mx
a = mx.random.normal((64, 3072)).astype(mx.bfloat16)
b = mx.random.normal((3072, 12288)).astype(mx.bfloat16); mx.eval(a, b)
with mx.stream(mx.gpu): g = (a @ b); mx.eval(g)
with mx.stream(mx.cpu): c = (a @ b); mx.eval(c)
g = g.astype(mx.float32); c = c.astype(mx.float32); mx.eval(g, c)
rd = float(((g - c) ** 2).sum() ** 0.5) / (float((c ** 2).sum() ** 0.5) + 1e-9)
sys.exit(0 if rd < 0.1 else 1)
PYV
}

if [ "$OS" = "Darwin" ]; then
    step "Checking GPU matmul kernel (M5 NAX miscompile detection) ..."
    if _mlx_matmul_ok; then
        info "GPU matmul correct — native mlx build is good, no metallib swap needed."
    elif [ "$_variant" = "binary" ]; then
        warn "M5 NAX miscompile detected, but binary (1-bit) relies on the fork's custom"
        warn "kernels that the prebuilt metallib lacks — cannot swap. Run with:"
        warn "  MLX_METAL_GPU_ARCH=applegpu_g16s   (correct, ~2.5-3x slower)"
        warn "or build the fork metallib with the Xcode 26.4 toolchain (docs/m5-metal.md)."
    else
        step "M5 NAX miscompile detected — applying prebuilt mlx-metal metallib fix (ternary) ..."
        _mlx_ver="$("$VENV_PY" -c 'import re,mlx.core as mx; m=re.match(r"\d+\.\d+\.\d+", mx.__version__); print(m.group() if m else "")' 2>/dev/null)"
        _mlx_lib="$("$VENV_PY" -c 'import os,mlx.core as mx; print(os.path.join(os.path.dirname(mx.__file__), "lib", "mlx.metallib"))' 2>/dev/null)"
        if [ -n "$_mlx_ver" ] && [ -f "$_mlx_lib" ]; then
            _tmpd="$(mktemp -d)"
            if uv pip install --target "$_tmpd" "mlx-metal==$_mlx_ver" >/dev/null 2>&1; then
                _good="$(find "$_tmpd" -name mlx.metallib 2>/dev/null | head -1)"
                if [ -n "$_good" ]; then
                    cp "$_good" "$_mlx_lib"
                    info "Replaced source-built metallib with prebuilt mlx-metal $_mlx_ver."
                    # Re-verify only after an actual swap, so the result message
                    # reflects the swap rather than a path where nothing changed.
                    if _mlx_matmul_ok; then
                        info "M5 NAX matmul kernel verified correct after metallib swap."
                    else
                        warn "GPU matmul still incorrect after swap. Set MLX_METAL_GPU_ARCH=applegpu_g16s,"
                        warn "or build mlx's metallib with the Xcode 26.4 toolchain (docs/m5-metal.md)."
                    fi
                else
                    warn "mlx-metal $_mlx_ver had no metallib; skipping the swap."
                    warn "M5 users: export MLX_METAL_GPU_ARCH=applegpu_g16s as a fallback (correct, ~2.5-3x slower)."
                fi
            else
                warn "No prebuilt mlx-metal==$_mlx_ver (mlx must be pinned to a released version, e.g. 0.31.2)."
                warn "M5 users: export MLX_METAL_GPU_ARCH=applegpu_g16s as a fallback (correct, ~2.5-3x slower)."
            fi
            rm -rf "$_tmpd"
        fi
    fi
fi

# ────────────────────────────────────────────────────
#  6. Wire bundled Node.js into .venv/bin
# ────────────────────────────────────────────────────
# `nodejs-wheel-binaries` drops node + npm under
# site-packages/nodejs_wheel/{bin,lib} but (a) doesn't add them to PATH and
# (b) the `bin/npm` shim was packaged as a file copy rather than a symlink,
# so its `require('../lib/cli.js')` resolves to a nonexistent path. We
# re-symlink the npm/npx shims to the real cli.js inside the package, then
# expose node + npm + npx + corepack from .venv/bin so callers can just do
# `.venv/bin/npm install`.
step "Wiring bundled Node.js into .venv/bin ..."
NODEJS_WHEEL_DIR=$("$VENV_PY" -c \
    'import os, nodejs_wheel; print(os.path.dirname(nodejs_wheel.__file__))' \
    2>/dev/null || true)
if [ -z "$NODEJS_WHEEL_DIR" ]; then
    err "nodejs_wheel package not importable; uv sync may have failed."
    exit 1
fi

# Replace the broken bin/npm + bin/npx with symlinks to the real cli.js so
# their internal `require('../lib/cli.js')` resolves correctly.
ln -sf "../lib/node_modules/npm/bin/npm-cli.js" "$NODEJS_WHEEL_DIR/bin/npm"
ln -sf "../lib/node_modules/npm/bin/npx-cli.js" "$NODEJS_WHEEL_DIR/bin/npx"

for _b in node npm npx corepack; do
    if [ -e "$NODEJS_WHEEL_DIR/bin/$_b" ]; then
        ln -sf "$NODEJS_WHEEL_DIR/bin/$_b" "$VENV_DIR/bin/$_b"
    fi
done
info "node $("$VENV_DIR/bin/node" --version) + npm $("$VENV_DIR/bin/npm" --version) ready in .venv/bin"

chmod +x "$SCRIPT_DIR"/scripts/*.sh 2>/dev/null || true

# ────────────────────────────────────────────────────
#  7. Download the default model (skippable)
# ────────────────────────────────────────────────────
#
# Calls scripts/download_model.sh for the active model (defaults to ternary).
# Override with BONSAI_VARIANT=binary, or skip entirely with SKIP_DOWNLOAD=1.
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    _variant="${BONSAI_VARIANT:-ternary}"
    step "Downloading default model: ${_variant} (BONSAI_VARIANT=binary to switch, SKIP_DOWNLOAD=1 to skip)..."
    if "$SCRIPT_DIR/scripts/download_model.sh" "$_variant"; then
        info "${_variant} model present."
    else
        warn "Model download failed. Retry with:"
        echo "    ./scripts/download_model.sh ${_variant}"
    fi
fi

# ────────────────────────────────────────────────────
#  Done!
# ────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "   Setup complete!"
echo "========================================="
echo ""
echo "  Run the full studio (backend + frontend) — recommended:"
echo "    ./scripts/serve.sh"
echo ""
echo "  Once serve.sh is up, send a prompt from the terminal:"
echo "    ./scripts/send_request.sh --prompt \"a tiny bonsai tree in a ceramic pot\""
echo ""
echo "  Or, for a one-shot run without a server (pays cold-start every call):"
echo "    ./scripts/generate.sh --prompt \"a tiny bonsai tree in a ceramic pot\""
echo ""
