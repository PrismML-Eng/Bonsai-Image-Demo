#!/bin/sh
# Shared helpers for Bonsai Image demo scripts.
# Source this file: . "$(dirname "$0")/common.sh"

# ── Models ──
# Two model variants of Bonsai-Image-4B (1.58-bit ternary), one per platform:
#   ternary-mlx       macOS Apple Silicon — mlx packed 2-bit
#   ternary-gemlite   Linux CUDA          — gemlite packed 2-bit

# Pick the default model for the current platform.
#   default_model              → ternary on this platform
#   default_model ternary      → ternary on this platform
#   default_model binary       → binary on this platform
# Returns "" if the platform isn't supported.
default_model() {
    _variant="${1:-ternary}"
    case "$(uname -s):$(uname -m)" in
        Darwin:arm64) echo "${_variant}-mlx" ;;
        Linux:*)      echo "${_variant}-gemlite" ;;
        *)            echo "" ;;
    esac
}

# ── Package age cutoff ──
# Minimum age (in days) for any package version we install via uv or npm.
# A package compromised today is likely still on the registry and could be
# pulled by a fresh resolve; requiring N days lets the ecosystem yank bad
# releases before we touch them. Set BONSAI_PACKAGE_MIN_AGE_DAYS=0 to disable.
BONSAI_PACKAGE_MIN_AGE_DAYS="${BONSAI_PACKAGE_MIN_AGE_DAYS:-7}"

# Emit YYYY-MM-DD that's N days ago in UTC. BSD (macOS) and GNU (Linux) `date`
# disagree on flags, so we try BSD first and fall back to GNU.
package_cutoff_date() {
    date -u -v"-${BONSAI_PACKAGE_MIN_AGE_DAYS}d" +%Y-%m-%d 2>/dev/null \
        || date -u -d "${BONSAI_PACKAGE_MIN_AGE_DAYS} days ago" +%Y-%m-%d
}

# ── Colors ──
if [ -t 1 ]; then
    _CLR_GREEN="\033[32m"
    _CLR_YELLOW="\033[33m"
    _CLR_RED="\033[31m"
    _CLR_CYAN="\033[36m"
    _CLR_RESET="\033[0m"
else
    _CLR_GREEN="" _CLR_YELLOW="" _CLR_RED="" _CLR_CYAN="" _CLR_RESET=""
fi

info()  { printf "${_CLR_GREEN}[OK]${_CLR_RESET}   %s\n" "$*"; }
warn()  { printf "${_CLR_YELLOW}[WARN]${_CLR_RESET} %s\n" "$*"; }
err()   { printf "${_CLR_RED}[ERR]${_CLR_RESET}  %s\n" "$*" >&2; }
step()  { printf "${_CLR_CYAN}==>    %s${_CLR_RESET}\n" "$*"; }

# ── download(url, dest) ──
download() {
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$2" "$1"
    else
        err "Neither curl nor wget found. Install one and re-run."
        exit 1
    fi
}

# ── Resolve DEMO_DIR (parent of scripts/) ──
resolve_demo_dir() {
    _script_dir="$(cd "$(dirname "$0")" && pwd)"
    echo "$(cd "$_script_dir/.." && pwd)"
}

# ── Ensure .venv is active ──
ensure_venv() {
    _demo="$1"
    if [ -z "$VIRTUAL_ENV" ] && [ -f "$_demo/.venv/bin/activate" ]; then
        . "$_demo/.venv/bin/activate"
    fi
    if [ -z "$VIRTUAL_ENV" ]; then
        err "Python venv not found. Run ./setup.sh first."
        exit 1
    fi
}

# ── Platform checks ──
is_apple_silicon() {
    [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]
}

is_linux() {
    [ "$(uname -s)" = "Linux" ]
}

has_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || command -v nvcc >/dev/null 2>&1
}

require_linux_nvidia_gpu() {
    if ! has_nvidia_gpu; then
        err "Linux generation requires an NVIDIA GPU/driver/CUDA on this repo's gemlite path."
        echo "       This host does not expose nvidia-smi or nvcc."
        echo "       If you only need the files installed, use BONSAI_ALLOW_UNSUPPORTED=1 during setup."
        exit 1
    fi
}

force_native_macos_build_arch() {
    _os="${1:-$(uname -s)}"
    _arch="${2:-$(uname -m)}"
    if [ "$_os" = "Darwin" ] && [ "$_arch" = "arm64" ]; then
        if [ "${ARCHFLAGS:-}" != "-arch arm64" ]; then
            [ -n "${ARCHFLAGS:-}" ] && warn "Overriding ARCHFLAGS=$ARCHFLAGS for Apple Silicon builds."
            export ARCHFLAGS="-arch arm64"
        fi
        if [ "${CMAKE_OSX_ARCHITECTURES:-}" != "arm64" ]; then
            export CMAKE_OSX_ARCHITECTURES="arm64"
        fi
    fi
}
