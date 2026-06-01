#!/bin/sh
# Launch the Bonsai Image studio: image-studio's FastAPI backend + Next.js frontend,
# wired up against this repo's models/ tree.
#
# Backend listens on :8000, frontend on :3000.
# Both run in the foreground; Ctrl+C tears them down together.
#
# Platform detection:
#   - macOS arm64  → mflux on MLX (Apple Silicon). Boots backend.server:app.
#   - Linux + CUDA → image-studio backend_gpu (gemlite + HQQ kernels). mlx/mflux
#                    not used. Boots scripts.local_backend:app (auth-stripped
#                    wrapper with a 512×512 warmup baked into the lifespan).
#   - Windows      → unsupported.
#
# Env knobs:
#   BACKEND_PORT           Override 8000.
#   FRONTEND_PORT          Override 3000.
#   STUDIO_DIR             Path to image-studio checkout (default: vendor/image-studio).
#   BACKEND_READY_TIMEOUT  Seconds to wait for backend's /backends to answer
#                          (default 180). Bump on slow GPUs — T4 cold JIT for
#                          one 512² shape alone is ~3 min; warming 1024² too
#                          easily exceeds 600s. Try 1800 on Colab T4.
set -e

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
. "$DEMO_DIR/scripts/common.sh"
ensure_venv "$DEMO_DIR"

: "${BACKEND_PORT:=8000}"
: "${FRONTEND_PORT:=3000}"
: "${STUDIO_DIR:=$DEMO_DIR/vendor/image-studio}"

# ── platform check ──
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Darwin)
        if [ "$ARCH" != "arm64" ]; then
            err "mflux requires Apple Silicon on macOS (got $ARCH)."
            exit 1
        fi
        info "Platform: macOS $ARCH — MLX on Apple Silicon"
        ;;
    Linux)
        if has_nvidia_gpu; then
            info "Platform: Linux + NVIDIA GPU — backend_gpu (gemlite/HQQ on CUDA)"
        else
            warn "Platform: Linux without NVIDIA GPU — backend_gpu needs CUDA, generation will fail."
            echo "       Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
            echo "       Continuing anyway."
        fi
        ;;
    *)
        err "Unsupported OS: $OS (supported: macOS arm64, Linux + CUDA)."
        exit 1
        ;;
esac

# ── resolve image-studio + frontend ──
if [ ! -d "$STUDIO_DIR" ]; then
    err "image-studio not found at $STUDIO_DIR"
    echo "       Run ./setup.sh to clone it into vendor/, or set \$STUDIO_DIR."
    exit 1
fi
FRONTEND_DIR="$STUDIO_DIR/frontend"
if [ ! -d "$FRONTEND_DIR" ]; then
    err "frontend not found at $FRONTEND_DIR"
    exit 1
fi

# Variant: "ternary" (default, 1.58-bit) or "binary" (1-bit). Override per
# launch: BONSAI_VARIANT=binary ./scripts/serve.sh
: "${BONSAI_VARIANT:=ternary}"
case "$BONSAI_VARIANT" in
    ternary|binary) ;;
    *) err "BONSAI_VARIANT must be 'ternary' or 'binary' (got $BONSAI_VARIANT)"; exit 1 ;;
esac

case "$OS" in
    Darwin)
        _default_backend="bonsai-${BONSAI_VARIANT}-mlx"
        # scripts/local_backend_mac wraps image-studio's backend.server with
        # a demo-shaped /backends route (only the two Bonsai variants — no
        # bfl-klein-bf16, which would otherwise trigger a 24 GB HF download
        # on first dropdown pick). Mirrors the Linux scripts.local_backend
        # shim pattern.
        _backend_module="scripts.local_backend_mac:app"
        _model_dir="$DEMO_DIR/models/bonsai-image-4B-${BONSAI_VARIANT}-mlx"
        ;;
    Linux)
        # The Linux path swaps to image-studio's GPU backend (gemlite kernels)
        # via scripts/local_backend.py, which strips the bearer-auth gate so
        # the frontend can hit it directly over loopback.
        _default_backend="bonsai-${BONSAI_VARIANT}-gemlite"
        _backend_module="scripts.local_backend:app"
        _model_dir="$DEMO_DIR/models/bonsai-image-4B-${BONSAI_VARIANT}-gemlite"

        # backend_gpu/pipeline_gpu.py reads SEPARATE env vars per variant
        # (TERNARY_TRANSFORMER_PATH vs BINARY_TRANSFORMER_PATH) and falls
        # back to /root/models/{bonsai-binary,bonsai-ternary}/ — paths that
        # don't exist locally. Glob for whichever transformer-gemlite-*
        # subdir each model dir actually ships (binary → -int1, ternary →
        # -int2), and set BOTH env vars when both model dirs are present,
        # so a runtime backend swap (frontend dropdown, or the FastAPI
        # request schema's DEFAULT_GPU_BACKEND="bonsai-binary-gemlite"
        # default) can't trip the upstream defaults.
        _ternary_dir="$DEMO_DIR/models/bonsai-image-4B-ternary-gemlite"
        _binary_dir="$DEMO_DIR/models/bonsai-image-4B-binary-gemlite"
        _ternary_transformer=$(ls -d "$_ternary_dir"/transformer-gemlite-* 2>/dev/null | head -1)
        _binary_transformer=$(ls -d "$_binary_dir"/transformer-gemlite-* 2>/dev/null | head -1)
        # Always populate BOTH env vars. If one variant isn't downloaded,
        # point at where it would live under models/ so backend_gpu reports
        # a local actionable path instead of treating it as "unset".
        _ternary_transformer_path="${_ternary_transformer:-$_ternary_dir/transformer-gemlite-int2}"
        _binary_transformer_path="${_binary_transformer:-$_binary_dir/transformer-gemlite-int1}"
        # The variant being launched must be present; the other is optional.
        if [ "$BONSAI_VARIANT" = "binary" ] && [ -z "$_binary_transformer" ]; then
            err "no transformer-gemlite-* subdir found under $_binary_dir"
            err "download the model first: ./scripts/download_model.sh binary"
            exit 1
        fi
        if [ "$BONSAI_VARIANT" = "ternary" ] && [ -z "$_ternary_transformer" ]; then
            err "no transformer-gemlite-* subdir found under $_ternary_dir"
            err "download the model first: ./scripts/download_model.sh ternary"
            exit 1
        fi
        _supported_families=""
        [ -n "$_ternary_transformer" ] && _supported_families="bonsai-ternary"
        if [ -n "$_binary_transformer" ]; then
            if [ -n "$_supported_families" ]; then
                _supported_families="$_supported_families,bonsai-binary"
            else
                _supported_families="bonsai-binary"
            fi
        fi
        ;;
esac

# ── port-in-use check ──
if command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:"$BACKEND_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        err "Port $BACKEND_PORT already in use (backend)."
        exit 1
    fi
    if lsof -nP -iTCP:"$FRONTEND_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        err "Port $FRONTEND_PORT already in use (frontend)."
        exit 1
    fi
fi

# ── frontend deps ──
# Prefer the node + npm shipped by `nodejs-wheel-binaries` in .venv/bin over
# whatever's on the user's system. Putting .venv/bin first on PATH means any
# child process spawned by npm (next-server, etc.) also resolves to the
# bundled node — keeps the toolchain self-contained.
VENV_BIN="$DEMO_DIR/.venv/bin"
if [ ! -x "$VENV_BIN/npm" ]; then
    err "Bundled npm not found at $VENV_BIN/npm — did setup.sh run successfully?"
    echo "       Try: uv sync"
    exit 1
fi

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    _pkg_cutoff="$(package_cutoff_date)"
    step "Installing frontend dependencies (first run, versions <= $_pkg_cutoff)..."
    (cd "$FRONTEND_DIR" \
        && PATH="$VENV_BIN:$PATH" \
           npm install --no-audit --no-fund --before "$_pkg_cutoff")
fi

# ── start both processes ──
LOG_DIR="$DEMO_DIR/.serve-logs"
mkdir -p "$LOG_DIR"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

step "Starting backend on :$BACKEND_PORT (default arm: $_default_backend)"
echo "       logs: $BACKEND_LOG"
if [ "$OS" = "Linux" ]; then
    # Run from $DEMO_DIR so `scripts.local_backend:app` resolves on the
    # default sys.path. GPU pipeline reads each artifact path from env.
    # Both transformer paths are set when the corresponding model dir is
    # present so the frontend backend dropdown (and the FastAPI request
    # schema's binary default) can swap without hitting the upstream
    # /root/models/ defaults. The TE/VAE/tokenizer point at the launched
    # variant — they're identical bytes across variants and survive the
    # swap (ensure_backend only reloads the transformer).
    (cd "$DEMO_DIR" \
        && env MFLUX_STUDIO_GPU_DEFAULT_BACKEND="$_default_backend" \
               BONSAI_SUPPORTED_FAMILIES="$_supported_families" \
               MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH="$_ternary_transformer_path" \
               MFLUX_STUDIO_GPU_BINARY_TRANSFORMER_PATH="$_binary_transformer_path" \
               MFLUX_STUDIO_GPU_TEXT_ENCODER_PATH="$_model_dir/text_encoder-hqq-4bit" \
               MFLUX_STUDIO_GPU_VAE_PATH="$_model_dir/vae" \
               MFLUX_STUDIO_GPU_TOKENIZER_PATH="$_model_dir/text_encoder-hqq-4bit/tokenizer" \
               "$DEMO_DIR/.venv/bin/uvicorn" "$_backend_module" \
                   --port "$BACKEND_PORT" \
                   > "$BACKEND_LOG" 2>&1) &
    BACKEND_PID=$!
else
    # FORCE_DISABLE_GPU keeps `/backends` from advertising the gemlite arms
    # to the frontend — without it the picker can offer bonsai-ternary-gemlite,
    # and selecting it tries to load it via FluxPipeline (remote-only, errors).
    #
    # image-studio's pipeline.py uses SEPARATE config fields per variant:
    #   bonsai-ternary-mlx → config.baked_model_path        (MFLUX_STUDIO_BAKED_MODEL_PATH)
    #   bonsai-binary-mlx  → config.baked_binary_model_path (MFLUX_STUDIO_BAKED_BINARY_MODEL_PATH)
    # If only the first is set and the user picks binary in the dropdown,
    # model_path arrives at Flux2Klein as None → mflux falls back to its
    # generic Klein/Qwen3 HF download instead of using the bundled local
    # text_encoder-mlx-4bit/. Export BOTH unconditionally so any runtime
    # dropdown swap can find its local weights. Image-studio only validates
    # `os.path.isabs` on these — a missing dir only fails when the backend
    # actually loads it.
    # bfl-klein-bf16 is hidden from the picker entirely via the
    # scripts.local_backend_mac shim (which overrides /backends).
    # No env flag needed for that — see _backend_module above.
    MFLUX_STUDIO_DEFAULT_BACKEND="$_default_backend" \
    MFLUX_STUDIO_BAKED_MODEL_PATH="$DEMO_DIR/models/bonsai-image-4B-ternary-mlx" \
    MFLUX_STUDIO_BAKED_BINARY_MODEL_PATH="$DEMO_DIR/models/bonsai-image-4B-binary-mlx" \
    MFLUX_STUDIO_TE_4BIT=true \
    MFLUX_STUDIO_FORCE_DISABLE_GPU=true \
        "$DEMO_DIR/.venv/bin/uvicorn" "$_backend_module" \
            --port "$BACKEND_PORT" \
            > "$BACKEND_LOG" 2>&1 &
    BACKEND_PID=$!
fi

# BONSAI_FRONTEND_PROD=1 swaps `next dev` (HMR + WebSocket dev client) for a
# production build + `next start`. Required when the frontend sits behind a
# proxy that doesn't tunnel WebSockets (e.g. Google Colab's session proxy) —
# without it, the HMR WS client retries forever and crashes hydration before
# onClick handlers attach, leaving buttons inert.
if [ "${BONSAI_FRONTEND_PROD:-0}" = "1" ]; then
    if [ ! -d "$FRONTEND_DIR/.next" ]; then
        step "Building frontend (production, BONSAI_FRONTEND_PROD=1) — first run only ..."
        (cd "$FRONTEND_DIR" \
            && PATH="$VENV_BIN:$PATH" \
               NEXT_PUBLIC_BACKEND_URL="http://127.0.0.1:$BACKEND_PORT" \
               npm run build) || {
            err "frontend build failed — check $FRONTEND_LOG"
            exit 1
        }
    else
        info "frontend already built (.next/ present) — skipping rebuild"
    fi
    _frontend_cmd="npm start"
else
    _frontend_cmd="npm run dev"
fi

step "Starting frontend on :$FRONTEND_PORT ($_frontend_cmd)"
echo "       logs: $FRONTEND_LOG"
# NEXT_PUBLIC_BACKEND_URL flows into the frontend's API route handlers
# (app/api/*/route.ts) so they hit the right port when BACKEND_PORT is
# overridden — defaults baked in those files target :8000.
(cd "$FRONTEND_DIR" \
    && PATH="$VENV_BIN:$PATH" \
       PORT="$FRONTEND_PORT" \
       NEXT_PUBLIC_BACKEND_URL="http://127.0.0.1:$BACKEND_PORT" \
       $_frontend_cmd) > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

# Kill children too — `npm run dev` spawns `next-server` as a grandchild and a
# plain kill to the npm wrapper leaves the node process bound to :3000.
_kill_tree() {
    _pid="$1"
    [ -n "$_pid" ] || return
    # Direct children
    _kids=$(pgrep -P "$_pid" 2>/dev/null || true)
    for _kid in $_kids; do _kill_tree "$_kid"; done
    kill "$_pid" 2>/dev/null || true
}

_kill_port() {
    _port="$1" _sig="${2:-TERM}"
    _pids=$(lsof -nP -iTCP:"$_port" -sTCP:LISTEN -t 2>/dev/null)
    [ -n "$_pids" ] && kill -"$_sig" $_pids 2>/dev/null || true
}

cleanup() {
    set +e
    info "Stopping (backend=$BACKEND_PID, frontend=$FRONTEND_PID)..."
    # Tree walk for direct children we know about.
    _kill_tree "$BACKEND_PID"
    _kill_tree "$FRONTEND_PID"
    # Belt-and-braces: `next dev` reparents `next-server` to launchd, so the
    # tree walk above misses it. Sweep anything still bound to our ports.
    sleep 1
    _kill_port "$BACKEND_PORT" TERM
    _kill_port "$FRONTEND_PORT" TERM
    sleep 1
    _kill_port "$BACKEND_PORT" KILL
    _kill_port "$FRONTEND_PORT" KILL
    # Drop the EXIT trap before exiting so we don't recurse, and don't wait —
    # reparented children won't show up in the shell's job table anyway.
    trap - EXIT
    exit 0
}
trap cleanup INT TERM EXIT

# ── readiness wait + URL print ──
# Backend's lifespan blocks the port bind until the GPU pipeline prewarm is
# done — so this probe runs for ~25-30s on Linux on the first call. We use
# curl rather than lsof since lsof has been spotty here picking up listening
# next-server sockets; an actual HTTP response is unambiguous.
#
# Returns:
#   0 — port responded
#   1 — process died before the port came up (caller should fail fast)
#   2 — timeout (process still alive, just slow — non-fatal)
_wait_for_port() {
    _port="$1" _name="$2" _pid="$3" _max="${4:-180}" _i=0
    # Fancy spinner only when stdout is a TTY. When the script is piped
    # (CI, tee, log capture) print a plain heartbeat line every 10s instead
    # so progress is still visible in non-interactive transcripts.
    _is_tty=0
    [ -t 1 ] && _is_tty=1
    [ "$_is_tty" = "1" ] && printf "       waiting for %s on :%s ...\n" "$_name" "$_port"
    while [ "$_i" -lt "$_max" ]; do
        if curl -fsS -m 1 -o /dev/null "http://127.0.0.1:$_port/" 2>/dev/null \
           || curl -sS -m 1 -o /dev/null "http://127.0.0.1:$_port/" 2>/dev/null; then
            [ "$_is_tty" = "1" ] && printf "\r\033[K"
            info "$_name ready on :$_port (took ${_i}s)"
            return 0
        fi
        # Bail immediately if the child process is gone — otherwise we'd
        # idle through the full 180s while a dead backend gives no signal.
        if ! kill -0 "$_pid" 2>/dev/null; then
            [ "$_is_tty" = "1" ] && printf "\r\033[K"
            return 1
        fi
        if [ "$_is_tty" = "1" ]; then
            case $((_i % 4)) in
                0) _frame="|" ;;
                1) _frame="/" ;;
                2) _frame="-" ;;
                3) _frame="\\" ;;
            esac
            printf "\r       %s %s booting ... %ss" "$_frame" "$_name" "$_i"
        elif [ "$_i" -gt 0 ] && [ $((_i % 10)) -eq 0 ]; then
            echo "       still waiting for $_name on :$_port ... ${_i}s"
        fi
        sleep 1
        _i=$((_i + 1))
    done
    [ "$_is_tty" = "1" ] && printf "\r\033[K"
    warn "$_name didn't respond on :$_port after ${_max}s — check $LOG_DIR/"
    return 2
}

# Capture return code under `set -e` — a bare non-zero from the function
# would otherwise exit the script before reaching the `case` dispatch.
_backend_status=0
_wait_for_port "$BACKEND_PORT" "Backend" "$BACKEND_PID" "${BACKEND_READY_TIMEOUT:-180}" || _backend_status=$?
case $_backend_status in
    0) ;;
    1)
        err "Backend exited before port $BACKEND_PORT came up. Last lines of $BACKEND_LOG:"
        echo "------------------------------------------------------------------------"
        tail -40 "$BACKEND_LOG" >&2
        echo "------------------------------------------------------------------------"
        exit 1
        ;;
esac
# Frontend timeout is non-fatal: dev-server can be slow to first-paint even
# when it's fine on follow-up reloads.
_wait_for_port "$FRONTEND_PORT" "Frontend" "$FRONTEND_PID" 60 || true

echo ""
echo "========================================================================"
echo ""
echo "  Frontend (open in your browser):"
echo "    http://localhost:$FRONTEND_PORT/"
echo ""
echo "  Backend API:"
echo "    http://localhost:$BACKEND_PORT/             root"
echo "    http://localhost:$BACKEND_PORT/backends     available arms + GPU probe"
echo "    http://localhost:$BACKEND_PORT/docs         OpenAPI UI"
echo ""
echo "  Logs: $LOG_DIR/"
echo ""
echo "========================================================================"
echo "  Ctrl+C to stop both."
echo ""

wait
