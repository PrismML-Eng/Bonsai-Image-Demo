#!/bin/sh
# Thin wrapper around scripts/generate.py. The generation logic lives in Python.
#
# Usage:
#   ./scripts/generate.sh -p "a tiny bonsai tree"
#   ./scripts/generate.sh -p "a red cube" --steps 8 --seed 42
set -e

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
. "$DEMO_DIR/scripts/common.sh"
ensure_venv "$DEMO_DIR"

if is_linux; then
    require_linux_nvidia_gpu
fi

exec "$DEMO_DIR/.venv/bin/python" "$DEMO_DIR/scripts/generate.py" "$@"
