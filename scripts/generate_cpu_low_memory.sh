#!/bin/sh
set -e

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
. "$DEMO_DIR/scripts/common.sh"
ensure_venv "$DEMO_DIR"

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            exec "$DEMO_DIR/.venv/bin/python" "$DEMO_DIR/scripts/generate_cpu_experimental.py" --help
            ;;
    esac
done

PROMPT_CACHE_DIR="${BONSAI_PROMPT_CACHE_DIR:-$DEMO_DIR/outputs/prompt_cache_fp32_auto}"
PYTHON_BIN="$DEMO_DIR/.venv/bin/python"
SCRIPT="$DEMO_DIR/scripts/generate_cpu_experimental.py"
CPU_THREADS="${BONSAI_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 1)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$CPU_THREADS}"

step "Stage 1/2: prompt cache in a fresh process ..."
"$PYTHON_BIN" "$SCRIPT" --prompt-cache-dir "$PROMPT_CACHE_DIR" --prompt-cache-only "$@"

step "Stage 2/2: render from cached prompt embeds in a second fresh process ..."
exec "$PYTHON_BIN" "$SCRIPT" --prompt-cache-dir "$PROMPT_CACHE_DIR" "$@"
