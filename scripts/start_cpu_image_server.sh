#!/bin/sh
set -e

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
. "$DEMO_DIR/scripts/common.sh"
ensure_venv "$DEMO_DIR"

: "${BONSAI_CPU_SERVER_HOST:=127.0.0.1}"
: "${BONSAI_CPU_SERVER_PORT:=8011}"

exec "$DEMO_DIR/.venv/bin/python" "$DEMO_DIR/scripts/cpu_image_server.py" \
  --host "$BONSAI_CPU_SERVER_HOST" \
  --port "$BONSAI_CPU_SERVER_PORT" \
  "$@"
