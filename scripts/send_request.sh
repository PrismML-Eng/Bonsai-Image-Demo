#!/bin/sh
# Send a /generate request to an already-running studio (started by serve.sh).
# Same CLI surface as generate.sh, but each call is a thin HTTP POST — the
# backend keeps weights resident so subsequent renders are several times
# faster than the cold-start CLI path.
#
# Usage:
#   ./scripts/send_request.sh -p "a tiny bonsai tree"
#   ./scripts/send_request.sh -p "..." --size 1024x1024 --seed 42 --steps 8
#   BACKEND_PORT=8800 ./scripts/send_request.sh -p "..."     # custom port
set -e

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
. "$DEMO_DIR/scripts/common.sh"
ensure_venv "$DEMO_DIR"

: "${BACKEND_PORT:=8000}"
: "${BACKEND_HOST:=127.0.0.1}"
HOST="http://${BACKEND_HOST}:${BACKEND_PORT}"

_usage() {
    cat <<EOF
Usage: $0 -p "<prompt>" [--seed N] [--steps N] [--size WxH] [--output PATH] [--open]

Sends a POST /generate to a running serve.sh studio. Same args as generate.sh.
Override the target with BACKEND_PORT (default 8000) or BACKEND_HOST (127.0.0.1).
EOF
}

# ── parse args ──────────────────────────────────────────────────────────
_prompt=""
_seed=""
_steps=4
_size="512x512"
_output=""
_open=0
while [ $# -gt 0 ]; do
    case "$1" in
        -p|--prompt) _prompt="$2"; shift 2 ;;
        --seed)      _seed="$2"; shift 2 ;;
        --steps)     _steps="$2"; shift 2 ;;
        --size)      _size="$2"; shift 2 ;;
        --output)    _output="$2"; shift 2 ;;
        --open)      _open=1; shift ;;
        -h|--help)   _usage; exit 0 ;;
        *)           err "Unknown argument: $1"; _usage; exit 1 ;;
    esac
done

[ -n "$_prompt" ] || { err "Required: -p/--prompt"; _usage; exit 1; }

_width="${_size%x*}"
_height="${_size##*x}"
case "$_width" in (''|*[!0-9]*) err "Invalid --size: $_size"; exit 1 ;; esac
case "$_height" in (''|*[!0-9]*) err "Invalid --size: $_size"; exit 1 ;; esac

# ── pick a positive 31-bit seed if the user didn't supply one ───────────
if [ -z "$_seed" ]; then
    _seed=$("$DEMO_DIR/.venv/bin/python" -c 'import secrets; print(secrets.randbits(31))')
fi

# ── default output: outputs/{model}/image_{ts}_seed{seed}.png ──────────
# Probe /backends to find what arm the server is actually running so output
# lands under the right dir AND so we can pin the request's `backend` field
# (FastAPI defaults it to DEFAULT_GPU_BACKEND = bonsai-binary-gemlite, which
# triggers a swap into a transformer the server didn't preload). The Linux
# shim returns {kind, default_family}; the Mac router returns {default}.
_active_backend=$(curl -fsS --max-time 5 "$HOST/backends" 2>/dev/null \
    | "$DEMO_DIR/.venv/bin/python" -c '
import json, sys
d = json.load(sys.stdin)
if "default" in d:
    print(d["default"])
else:
    print(f"{d[\"default_family\"]}-{d[\"kind\"]}")' \
    2>/dev/null) || true
case "$_active_backend" in
    bonsai-ternary-mlx)     _model_label="ternary-mlx" ;;
    bonsai-binary-mlx)      _model_label="binary-mlx" ;;
    bonsai-ternary-gemlite) _model_label="ternary-gemlite" ;;
    bonsai-binary-gemlite)  _model_label="binary-gemlite" ;;
    *)                      _model_label="$(default_model)" ;;
esac

if [ -z "$_output" ]; then
    _ts=$(date -u +%Y%m%d_%H%M%S)
    _output="$DEMO_DIR/outputs/${_model_label}/image_${_ts}_seed${_seed}.png"
fi
mkdir -p "$(dirname "$_output")"

# ── build JSON payload (delegate string escaping to Python) ────────────
# Include `backend` so the server doesn't fall back to its FastAPI default
# (DEFAULT_GPU_BACKEND = bonsai-binary-gemlite). Empty string when the probe
# failed → server picks its own default, same as before.
_payload=$("$DEMO_DIR/.venv/bin/python" -c '
import json, sys
prompt, seed, steps, height, width, backend = sys.argv[1:]
payload = {
    "prompt": prompt,
    "seed": int(seed),
    "steps": int(steps),
    "height": int(height),
    "width": int(width),
}
if backend:
    payload["backend"] = backend
print(json.dumps(payload))' "$_prompt" "$_seed" "$_steps" "$_height" "$_width" "$_active_backend")

# ── POST and save the PNG bytes ─────────────────────────────────────────
step "POST $HOST/generate  ($_model_label, ${_width}×${_height}, seed=$_seed)"
_t0=$(date +%s)
_http=$(curl -sS -o "$_output" -w "%{http_code}" \
    -X POST "$HOST/generate" \
    -H "Content-Type: application/json" \
    --max-time 600 \
    --data-binary "$_payload") || {
    err "request failed — is serve.sh running on $HOST?"
    exit 1
}
_wall=$(( $(date +%s) - _t0 ))

if [ "$_http" != "200" ]; then
    err "HTTP $_http from $HOST/generate"
    echo "  body (first 500 bytes):"
    head -c 500 "$_output"; echo
    rm -f "$_output"
    exit 1
fi

echo
echo "  prompt: $_prompt"
echo "  seed:   $_seed"
echo "  size:   ${_width}x${_height}"
echo "  wall:   ${_wall}s"
echo "  path:   $_output"

if [ "$_open" = 1 ] && [ "$(uname -s)" = "Darwin" ]; then
    open "$_output" 2>/dev/null || true
fi
