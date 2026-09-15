#!/bin/sh
set -e

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
. "$DEMO_DIR/scripts/common.sh"
ensure_venv "$DEMO_DIR"

exec "$DEMO_DIR/.venv/bin/python" "$DEMO_DIR/scripts/export_unpacked_cpu_transformer.py" "$@"
