#!/bin/sh
set -e

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
. "$DEMO_DIR/scripts/common.sh"
ensure_venv "$DEMO_DIR"

MODEL_ROOT="$DEMO_DIR/models/bonsai-image-4B-ternary-gemlite"
UNPACKED_ROOT="$DEMO_DIR/models/bonsai-image-4B-ternary-unpacked"

while [ $# -gt 0 ]; do
    case "$1" in
        --model-root)
            MODEL_ROOT="$2"
            shift 2
            ;;
        --unpacked-root)
            UNPACKED_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--model-root PATH] [--unpacked-root PATH]

Checks the experimental CPU bring-up prerequisites:
  - <model-root>/text_encoder-hqq-4bit
  - <model-root>/vae
  - <model-root>/transformer-gemlite-int2
  - <unpacked-root>/transformer
  - importable Python packages: hqq, diffusers, safetensors, transformers
EOF
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            exit 1
            ;;
    esac
done

missing=0

check_dir() {
    _path="$1"
    _label="$2"
    if [ -d "$_path" ]; then
        info "Found $_label: $_path"
    else
        err "Missing $_label: $_path"
        missing=1
    fi
}

step "Checking experimental CPU prerequisites ..."
check_dir "$MODEL_ROOT/text_encoder-hqq-4bit" "text encoder"
check_dir "$MODEL_ROOT/vae" "VAE"
check_dir "$MODEL_ROOT/transformer-gemlite-int2" "GemLite transformer"
check_dir "$UNPACKED_ROOT/transformer" "unpacked transformer"

step "Checking unpacked transformer files ..."
if [ -s "$UNPACKED_ROOT/transformer/diffusion_pytorch_model.safetensors" ] || \
   [ -s "$UNPACKED_ROOT/transformer/diffusion_pytorch_model.safetensors.index.json" ] || \
   find "$UNPACKED_ROOT/transformer" -maxdepth 1 -type f -name 'diffusion_pytorch_model-*.safetensors' -size +0c 2>/dev/null | grep -q .; then
    info "Found non-empty unpacked transformer weights"
else
    err "Unpacked transformer directory exists but does not contain non-empty saved weights"
    missing=1
fi

step "Checking CPU export layout marker ..."
if [ -s "$UNPACKED_ROOT/transformer/cpu_decode_layout_version.json" ]; then
    if "$DEMO_DIR/.venv/bin/python" - "$UNPACKED_ROOT/transformer/cpu_decode_layout_version.json" <<'PY'
import json
import sys
from pathlib import Path

marker = json.loads(Path(sys.argv[1]).read_text())
if marker.get("gemlite_decode_layout") != "over-k-transposed":
    raise SystemExit(1)
PY
    then
        info "Found corrected GemLite CPU decode layout marker"
    else
        err "Unpacked transformer marker does not match the corrected GemLite CPU layout"
        missing=1
    fi
else
    err "Missing CPU decode layout marker: $UNPACKED_ROOT/transformer/cpu_decode_layout_version.json"
    echo "       Refresh the unpacked transformer with:"
    echo "       ./scripts/export_unpacked_cpu_transformer.sh --overwrite"
    missing=1
fi

step "Checking Python CPU dependencies ..."
if "$DEMO_DIR/.venv/bin/python" -c 'import diffusers, hqq, safetensors, transformers' >/dev/null 2>&1; then
    info "Python packages available: diffusers, hqq, safetensors, transformers"
else
    err "Missing one or more Python packages: diffusers, hqq, safetensors, transformers"
    missing=1
fi

if [ "$missing" -ne 0 ]; then
    echo ""
    warn "CPU bring-up is incomplete."
    echo "  Download the GemLite model artifacts explicitly:"
    echo "    ./scripts/download_model.sh --model ternary-gemlite"
    echo ""
    echo "  If export fails with ModuleNotFoundError on a fresh macOS setup, refresh the venv:"
    echo "    uv sync"
    echo ""
    echo "  Export the unpacked transformer once:"
    echo "    ./scripts/export_unpacked_cpu_transformer.sh"
    exit 1
fi

echo ""
info "Experimental CPU bring-up looks complete."
