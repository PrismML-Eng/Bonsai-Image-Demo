#!/bin/sh
set -e

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
. "$DEMO_DIR/scripts/common.sh"
ensure_venv "$DEMO_DIR"

: "${BONSAI_CPU_SERVER_HOST:=127.0.0.1}"
: "${BONSAI_CPU_SERVER_PORT:=8011}"
CPU_THREADS="${BONSAI_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$CPU_THREADS}"

exec "$DEMO_DIR/.venv/bin/python" "$DEMO_DIR/scripts/cpu_image_server.py" \
  --host "$BONSAI_CPU_SERVER_HOST" \
  --port "$BONSAI_CPU_SERVER_PORT" \
  "$@"
