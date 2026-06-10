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
  - importable Python package: hqq
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

step "Checking Python CPU dependency: hqq ..."
if "$DEMO_DIR/.venv/bin/python" -c 'import hqq' >/dev/null 2>&1; then
    info "Python package available: hqq"
else
    err "Python package missing: hqq"
    missing=1
fi

if [ "$missing" -ne 0 ]; then
    echo ""
    warn "CPU bring-up is incomplete."
    echo "  Download the GemLite model artifacts explicitly:"
    echo "    ./scripts/download_model.sh --model ternary-gemlite"
    echo ""
    echo "  Export the unpacked transformer once:"
    echo "    ./scripts/export_unpacked_cpu_transformer.sh"
    exit 1
fi

echo ""
info "Experimental CPU bring-up looks complete."
