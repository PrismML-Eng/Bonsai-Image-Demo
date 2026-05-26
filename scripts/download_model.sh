#!/bin/sh
# Download a model for the Bonsai Image demo.
#
# Usage:
#   ./scripts/download_model.sh                       # ternary (default), platform-aware
#   ./scripts/download_model.sh ternary               # same — explicit
#   ./scripts/download_model.sh binary                # binary 1-bit, platform-aware
#   ./scripts/download_model.sh --model binary-gemlite  # full form, override backend
#   BONSAI_TOKEN=hf_... ./scripts/download_model.sh   # private until public launch
#
# Short form (ternary | binary) picks mlx on macOS and gemlite on Linux.
# Long form (ternary-mlx | binary-gemlite | …) overrides the backend choice.
#
# BONSAI_TOKEN is optional if you've already done `huggingface-cli login`
# (or `setup.sh` cached a token).
set -e

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
. "$DEMO_DIR/scripts/common.sh"
ensure_venv "$DEMO_DIR"

_usage() {
    cat <<EOF
Usage: $0 [<variant>] [-m|--model <full-name>]

  Short form (picks backend automatically per platform):
    ternary        → ternary-mlx on macOS, ternary-gemlite on Linux
    binary         → binary-mlx  on macOS, binary-gemlite  on Linux

  Long form (explicit backend):
    ternary-mlx      prism-ml/bonsai-image-ternary-4B-mlx-2bit
    ternary-gemlite  prism-ml/bonsai-image-ternary-4B-gemlite-2bit
    binary-mlx       prism-ml/bonsai-image-binary-4B-mlx-1bit
    binary-gemlite   prism-ml/bonsai-image-binary-4B-gemlite-1bit

If nothing is passed, ternary for this platform is picked.
EOF
}

# ── parse args ──
# Both `--model X` and bare positional `X` are accepted. Short form
# (ternary | binary) gets expanded to the platform-specific full name
# (ternary-mlx, binary-gemlite, …) further down.
_model=""
while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model) _model="$2"; shift 2 ;;
        -h|--help)  _usage; exit 0 ;;
        ternary|binary|ternary-mlx|ternary-gemlite|binary-mlx|binary-gemlite)
            _model="$1"; shift ;;
        *) err "Unknown argument: $1"; _usage; exit 1 ;;
    esac
done

if [ -z "$_model" ]; then
    _model="$(default_model)"   # ternary on this platform
fi

# Short form → expand to <variant>-<backend> for this platform.
case "$_model" in
    ternary|binary)
        _full="$(default_model "$_model")"
        if [ -z "$_full" ]; then
            err "Couldn't pick a backend for this platform ($(uname -s) $(uname -m))."
            err "Pass --model ${_model}-mlx or --model ${_model}-gemlite explicitly."
            exit 1
        fi
        _model="$_full"
        ;;
esac

if [ -z "$_model" ]; then
    err "Couldn't pick a default model for this platform ($(uname -s) $(uname -m))."
    _usage
    exit 1
fi

case "$_model" in
    ternary-mlx|binary-mlx|ternary-gemlite|binary-gemlite)
        # `<variant>-<backend>` → derive bits + HF repo path + local dir.
        # ternary → 2 bits, binary → 1 bit.
        _variant="${_model%-*}"        # ternary | binary
        _backend="${_model#*-}"         # mlx | gemlite
        _bits=2
        [ "$_variant" = "binary" ] && _bits=1
        _saved_dir="$DEMO_DIR/models/bonsai-image-4B-${_variant}-${_backend}"
        _display="Bonsai-Image-4B (${_variant} ${_backend}-${_bits}bit)"
        _hf_repo="prism-ml/bonsai-image-${_variant}-4B-${_backend}-${_bits}bit"
        ;;
    *)
        err "Invalid --model: $_model (must be ternary-mlx | ternary-gemlite | binary-mlx | binary-gemlite)"
        _usage
        exit 1
        ;;
esac

# Always call snapshot_download: with `local_dir` set, HF Hub HEADs each file
# and compares etags against what's on disk. Unchanged files are skipped;
# changed/added files redownload. This means pushing new weights upstream
# auto-propagates on the next call — no force-flag, no manual cache wipe.
# Cost when fully cached: ~10-30s of metadata HEADs depending on file count.
if [ -d "$_saved_dir" ]; then
    step "Syncing ${_display} (existing in ${_saved_dir}; etag-checking) ..."
    echo "  (HF Hub HEADs each file; only changed files redownload)"
else
    step "Downloading ${_display} to ${_saved_dir} ..."
    echo "  (This downloads from HuggingFace — may take a few minutes)"
fi

# snapshot_download with local_dir writes files straight under models/<…>/ in
# the standard HF layout (transformer/, vae/, text_encoder/, …), bypassing
# ~/.cache/huggingface/hub. Keeps the demo directory self-contained.
#
# HF_HUB_ENABLE_HF_TRANSFER=1 switches the per-file downloader to hf_transfer
# (Rust, parallel HTTP range requests). Bumps a stalled-link 10-20 MB/s
# baseline up to saturating residential gigabit on typical files. Set
# BONSAI_DISABLE_HF_TRANSFER=1 to fall back to the python requests backend.
_hf_transfer_env=""
if [ "${BONSAI_DISABLE_HF_TRANSFER:-0}" != "1" ]; then
    _hf_transfer_env="HF_HUB_ENABLE_HF_TRANSFER=1"
fi

env $_hf_transfer_env "$DEMO_DIR/.venv/bin/python" -c "
from huggingface_hub import snapshot_download, login
token = '$BONSAI_TOKEN' or None
if token:
    login(token=token, add_to_git_credential=False)
snapshot_download(
    repo_id='$_hf_repo',
    local_dir='$_saved_dir',
    max_workers=16,
)
"

info "Model saved to $_saved_dir"
