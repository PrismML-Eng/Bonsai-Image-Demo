"""Local Linux variant of image-studio's GPU backend.

Wraps `backend_gpu.server.app` (the FastAPI built by upstream image-studio)
with three demo-specific deltas:

1. Strip the Bearer-auth gate — single-box demo on loopback. Upstream
   backend_gpu is designed to sit behind the Mac `backend/` server over
   public HTTPS, which is why the auth gate is mandatory there.

2. Pin Triton's JIT compile cache to `outputs/.triton_cache/` and gemlite's
   autotune cache to `outputs/.gemlite_cache/autotune.json` so kernel
   binaries + per-shape configs survive process restarts. First boot pays
   the full compile/autotune cost; later boots replay from disk.

3. Tack a multi-shape warmup onto the lifespan — the server doesn't report
   "ready" until each declared shape has had one transformer forward. That
   step is enough to trigger gemlite autotune + Triton JIT for every kernel
   at that shape; VAE/text-encoder are not gemlite-bound and need no
   per-shape warmup. Default warms only 512×512; set
   BONSAI_WARMUP_SHAPES="512x512,1024x1024,..." to cover more.

Run via:
    uvicorn scripts.local_backend:app --port 8000
with the model paths exported as MFLUX_STUDIO_GPU_* env vars (see serve.sh).

Env knobs:
    BONSAI_WARMUP_SHAPES    Comma-separated WxH list. Default "512x512".
    BONSAI_WARMUP_STEPS     Diffusion steps per warmup call. Default 1.
    BONSAI_SKIP_WARMUP=1    Skip the warmup loop entirely.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent.parent
TRITON_CACHE_DIR = DEMO_DIR / "outputs" / ".triton_cache"
GEMLITE_PERSIST_PATH = DEMO_DIR / "outputs" / ".gemlite_cache" / "autotune.json"
# Tracks which (backend, shape) pairs have been warmed at least once on
# this GPU compute capability. Lives next to the gemlite autotune cache so
# wiping the cache dir invalidates the sentinel in lockstep. The parent
# dir itself is sm-namespaced via the symlinks entrypoint.sh sets up
# (gemlite-smXX), so the sentinel is implicitly GPU-tier specific too.
WARMUP_SENTINEL_PATH = GEMLITE_PERSIST_PATH.parent / "warmup-done.json"
TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
GEMLITE_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)

# Set both env vars BEFORE importing torch/triton via backend_gpu — Triton
# reads TRITON_CACHE_DIR exactly once at module load.
os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE_DIR))
# backend_gpu.server's lifespan refuses to start if MFLUX_STUDIO_GPU_TOKEN is
# unset. Populate a sentinel; auth is bypassed via dependency override below.
os.environ.setdefault("MFLUX_STUDIO_GPU_TOKEN", "local-demo-unused")

from backend_gpu import server as _gpu_server  # noqa: E402
from backend_gpu import pipeline_gpu as _pipeline_gpu  # noqa: E402
from gemlite.core import GemLiteLinearTriton  # noqa: E402

log = logging.getLogger(__name__)

app = _gpu_server.app
app.dependency_overrides[_gpu_server._verify_bearer] = lambda: None


# ── /backends shim ───────────────────────────────────────────────────────
# `backend_gpu/server.py` doesn't expose /backends — the upstream design
# expects the Mac `backend/` router to own that endpoint. On Linux there is
# no Mac router, so the frontend's `useBackends()` falls back to its baked-in
# default ("bonsai-binary-mlx", which doesn't exist here). Surface the one
# arm we actually serve so the studio picks it up.
@app.get("/backends")
def _backends() -> dict:
    """Backend-side response for /backends.

    Returns the (kind / supported_families / default_family) schema that
    image-studio's frontend useBackends() consumes. The arm string is
    canonically `<family>-<kind>` (e.g. `bonsai-ternary-gemlite`), so we
    split off the trailing `-mlx` or `-gemlite` for the kind and treat
    the prefix as the model family.

    Always advertises both Bonsai variants (ternary + binary) so the
    picker is symmetric with the Mac backend, which reports the same
    list. backend_gpu only loads ONE pipeline at runtime (the one named
    by MFLUX_STUDIO_GPU_DEFAULT_BACKEND); switching the dropdown to the
    other family doesn't reload weights, and /generate will still serve
    from the loaded pipeline. To actually swap which variant is loaded,
    restart the backend with a different BONSAI_VARIANT.
    """
    arm = os.environ.get("MFLUX_STUDIO_GPU_DEFAULT_BACKEND", "bonsai-ternary-gemlite")
    if arm.endswith("-gemlite"):
        default_family, kind = arm[: -len("-gemlite")], "gemlite"
    elif arm.endswith("-mlx"):
        default_family, kind = arm[: -len("-mlx")], "mlx"
    else:
        default_family, kind = arm, "gemlite"
    return {
        "kind": kind,
        "supported_families": ["bonsai-ternary", "bonsai-binary"],
        "default_family": default_family,
        "healthy": True,
        "reason": None,
    }


# ── Low-CPU-memory gemlite transformer loader ────────────────────────────
# Upstream `_load_gemlite_transformer` allocates fp32 random-init weights via
# `Flux2Transformer2DModel.from_config(cfg)` (~16 GB CPU for a 4 B-param
# model), casts to bf16 (~8 GB), holds the loaded state_dict (~5 GB), then
# casts the result to fp16 — peak CPU ≥20 GB, which OOM-kills the process on
# Colab free tier (12 GB cgroup).
#
# This replacement builds the model on the meta device (no real tensors
# allocated), casts state_dict entries to fp16 in place before assignment,
# and uses `assign=True` so load_state_dict swaps the meta tensors for the
# real ones directly. Peak CPU drops to roughly the state_dict size (~5 GB),
# fitting the free-tier cap.
#
# Functionally identical to upstream: same gemlite layer swap (via the
# shared `_load_gemlite_layers_from_state`), same fp16 inference stream,
# same `.weight = None` patch on gemlite modules.
def _low_mem_load_gemlite_transformer(path, *, device: str = _pipeline_gpu.DEFAULT_DEVICE):
    import json
    import torch
    from accelerate import init_empty_weights
    from diffusers import Flux2Transformer2DModel
    from gemlite.core import DType, GemLiteLinearTriton, set_packing_bitwidth

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Gemlite transformer artifact not found at {path}")
    state_path = path / "state_dict.pt"
    config_path = path / "config.json"
    qcfg_path = path / "quantization_config.json"
    autotune_path = path / "gemlite_autotune.json"
    for f in (state_path, config_path, qcfg_path, autotune_path):
        if not f.is_file():
            raise FileNotFoundError(f"Gemlite transformer missing {f.name} at {f}")

    with config_path.open() as fh:
        cfg = json.load(fh)
    with qcfg_path.open() as fh:
        qcfg = json.load(fh)
    bits = int(qcfg.get("bits", 1))
    group_size = int(qcfg.get("group_size", 128))
    packing_bw = int(qcfg.get("packing_bitwidth", 8))

    set_packing_bitwidth(packing_bw)
    GemLiteLinearTriton.load_config(str(autotune_path))

    log.info("loading gemlite transformer (LOW-MEM path): bits=%d gs=%d bw=%d",
             bits, group_size, packing_bw)

    # Meta device: nn.Parameters are placeholders, no real allocation. Module
    # __init__ logic still runs (so `in_features` / `out_features` are set on
    # nn.Linear children), but the underlying weight tensors don't exist yet.
    with init_empty_weights():
        model = Flux2Transformer2DModel.from_config(cfg)

    state = torch.load(str(state_path), map_location="cpu")

    # Cast floating-point state_dict entries to fp16 in place. Drops the
    # `model.to(fp16)` step at the end, which would have doubled memory.
    # Int tensors (gemlite-packed W_q, metadata, etc.) are left alone.
    for k in list(state.keys()):
        v = state[k]
        if torch.is_tensor(v) and v.is_floating_point() and v.dtype != torch.float16:
            state[k] = v.to(torch.float16)
            del v

    _, remainder = _pipeline_gpu._load_gemlite_layers_from_state(
        model, state,
        bits=bits, group_size=group_size, device=device,
        DType=DType, GemLiteLinearTriton=GemLiteLinearTriton,
    )
    del state  # let any unreferenced tensors free

    # assign=True (PyTorch ≥2.1): replaces meta placeholders with the loaded
    # tensors directly instead of copy_ing into them (which would fail on
    # meta). Side effect: the model ends up holding refs to whatever tensors
    # `remainder` held, with their real dtype (fp16 from the cast loop above).
    missing, unexpected = model.load_state_dict(remainder, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"unexpected non-gemlite state_dict keys: {unexpected[:8]}")
    if missing:
        raise RuntimeError(f"missing non-gemlite state_dict keys: {missing[:8]}")
    del remainder

    _pipeline_gpu._null_gemlite_weights(model, GemLiteLinearTriton)
    return model.to(device).eval()


# Swap the upstream loader for the low-mem variant. Done at module load,
# before uvicorn drives the lifespan that calls `pipeline.prewarm()` — which
# in turn calls `_load_gemlite_transformer` via module-level name lookup.
# Set BONSAI_DISABLE_LOWMEM_LOADER=1 to fall back to upstream (use this if
# you have plenty of RAM and want to compare).
if os.environ.get("BONSAI_DISABLE_LOWMEM_LOADER", "").lower() not in {"1", "true", "yes"}:
    _pipeline_gpu._load_gemlite_transformer = _low_mem_load_gemlite_transformer
    log.info("monkeypatched backend_gpu.pipeline_gpu._load_gemlite_transformer (low-mem variant)")

# Default: empty → no boot-time warmup. The persisted gemlite + Triton caches
# under outputs/.{gemlite,triton}_cache/ accumulate organically on real
# requests, so the per-shape JIT cost is paid at most once across reboots.
# Set BONSAI_WARMUP_SHAPES="512x512,1024x1024,..." to pre-pay it upfront if
# you want a hot first request.
_WARMUP_SHAPES = os.environ.get("BONSAI_WARMUP_SHAPES", "")
_WARMUP_STEPS = int(os.environ.get("BONSAI_WARMUP_STEPS", "1"))
_SKIP_WARMUP = os.environ.get("BONSAI_SKIP_WARMUP", "").lower() in {"1", "true", "yes"}
# Extra backends to warm at boot after the primary. Comma-separated list of
# image-studio backend ids (e.g. "bonsai-binary-gemlite"). Each entry triggers
# a transformer swap (~3.5 GB load) + per-shape JIT/autotune. Cached state
# persists via GEMLITE_PERSIST_PATH + TRITON_CACHE_DIR, so subsequent boots
# replay in seconds. Default empty → only the primary backend warms (current
# behavior).
_WARMUP_EXTRA_BACKENDS = os.environ.get("BONSAI_WARMUP_EXTRA_BACKENDS", "")


def _parse_shapes(spec: str) -> list[tuple[int, int]]:
    """Parse "WxH,WxH,..." → list of (w, h). Invalid tokens are logged and skipped."""
    out: list[tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        try:
            w, h = (int(x) for x in token.split("x", 1))
            out.append((w, h))
        except ValueError:
            log.warning("BONSAI_WARMUP_SHAPES: ignoring invalid token %r", token)
    return out


def _load_warmup_sentinel() -> dict:
    """Read the {backend: [shapes...]} sentinel. Empty dict on miss / parse error."""
    if not WARMUP_SENTINEL_PATH.exists():
        return {}
    try:
        with WARMUP_SENTINEL_PATH.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("warmup sentinel %s unreadable (%s); treating as empty",
                    WARMUP_SENTINEL_PATH, exc)
        return {}
    # Defensive: validate the shape — must be {str: list[str]}.
    if not isinstance(data, dict):
        return {}
    return {
        k: list(v) for k, v in data.items()
        if isinstance(k, str) and isinstance(v, list)
    }


def _save_warmup_sentinel(data: dict) -> None:
    """Persist atomically. Failures are logged but non-fatal — worst case
    we re-warm a shape next boot that was actually already warm. No data
    loss, just wasted seconds."""
    try:
        WARMUP_SENTINEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(WARMUP_SENTINEL_PATH) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, WARMUP_SENTINEL_PATH)
    except OSError as exc:
        log.warning("couldn't write warmup sentinel %s: %s", WARMUP_SENTINEL_PATH, exc)


@asynccontextmanager
async def _lifespan_with_warmup(fastapi_app):
    # Drive backend_gpu's original lifespan (loads 5 artifacts onto cuda:0 +
    # restores bundled gemlite_autotune.json from the model dir).
    async with _gpu_server.lifespan(fastapi_app):
        # Stack the persisted autotune cache on top of the bundled one —
        # always, regardless of whether we run a warmup pass. Any shape hit
        # on a previous boot short-circuits per-shape JIT/autotune.
        if GEMLITE_PERSIST_PATH.exists():
            log.info("loading persisted gemlite autotune: %s", GEMLITE_PERSIST_PATH)
            GemLiteLinearTriton.load_config(str(GEMLITE_PERSIST_PATH), print_error=False)

        shapes = _parse_shapes(_WARMUP_SHAPES) if not _SKIP_WARMUP else []
        if not shapes:
            log.info("warmup: skipped (BONSAI_WARMUP_SHAPES=%r, BONSAI_SKIP_WARMUP=%s)",
                     _WARMUP_SHAPES, _SKIP_WARMUP)
        else:
            pipeline = fastapi_app.state.pipeline
            primary_backend = pipeline.backend
            extras = [b.strip() for b in _WARMUP_EXTRA_BACKENDS.split(",") if b.strip()]
            extras = [b for b in extras if b != primary_backend]
            sentinel = _load_warmup_sentinel()
            log.info("warmup: %d shape(s) × %d step(s) on %s (shapes=%s; extras=%s; cached=%s)",
                     len(shapes), _WARMUP_STEPS, primary_backend, _WARMUP_SHAPES, extras or "none",
                     {k: len(v) for k, v in sentinel.items()} or "none")
            total_t0 = time.perf_counter()

            def _warm_shapes(label: str) -> None:
                # Skip any shape this backend has been warmed at before on
                # this GPU (parent dir is sm-namespaced). The kernel + autotune
                # entries are already in /data/cache/{triton,gemlite}-smXX/ so
                # /generate at this shape will hit cache, not JIT.
                already_warm = set(sentinel.get(label, []))
                to_warm = [(w, h) for (w, h) in shapes if f"{w}x{h}" not in already_warm]
                skipped = [s for s in (f"{w}x{h}" for w, h in shapes) if s in already_warm]
                if skipped:
                    log.info("  warmup %s: SKIP %d shape(s) already cached (%s)",
                             label, len(skipped), ",".join(skipped))
                if not to_warm:
                    return
                for w, h in to_warm:
                    shape_key = f"{w}x{h}"
                    t0 = time.perf_counter()
                    try:
                        pipeline.generate_png(
                            prompt="warmup", seed=0, steps=_WARMUP_STEPS, height=h, width=w,
                        )
                        log.info("  warmup %s %s in %.1fs", label, shape_key, time.perf_counter() - t0)
                        # Mark this (backend, shape) as warmed. Persist after
                        # each shape so partial completion progresses cumulatively
                        # — a Space restart mid-warmup picks up where we left off.
                        sentinel.setdefault(label, [])
                        if shape_key not in sentinel[label]:
                            sentinel[label].append(shape_key)
                            sentinel[label] = sorted(set(sentinel[label]))
                            _save_warmup_sentinel(sentinel)
                    except Exception as exc:
                        # Don't let a single failing shape sink the whole boot —
                        # log and keep going. The next /generate at this shape
                        # will retry (paying full JIT) but the rest of the boot
                        # is unblocked.
                        log.warning("warmup %s %s failed: %s", label, shape_key, exc)

            # Primary first — kernels for the resident backend land in the
            # gemlite + Triton caches on disk so subsequent boots hit them
            # before any user traffic.
            _warm_shapes(primary_backend)
            _save_gemlite_cache(f"after primary {primary_backend} warmup")

            # Extras: swap the resident transformer to each named backend
            # and warm the same shape set. Costs ~3.5 GB transformer load
            # per swap + per-shape JIT/autotune (cold) for that bit-width.
            # The first cold boot pays this fully; once cached, subsequent
            # boots replay both backends in ~seconds.
            for extra in extras:
                log.info("warmup: swapping resident transformer → %s", extra)
                try:
                    pipeline.ensure_backend(backend=extra, model_path=None)
                except Exception as exc:
                    log.warning("warmup: couldn't swap to %s (%s); skipping that arm", extra, exc)
                    continue
                _warm_shapes(extra)
                _save_gemlite_cache(f"after {extra} warmup")

            # Restore the primary as the resident backend — otherwise /backends
            # default + /generate without an explicit `backend` field would
            # land on whichever extra was warmed last.
            if pipeline.backend != primary_backend:
                log.info("warmup: restoring %s as resident backend", primary_backend)
                try:
                    pipeline.ensure_backend(backend=primary_backend, model_path=None)
                except Exception as exc:
                    log.error("warmup: failed to restore %s after extras (%s); "
                              "resident backend is now %s", primary_backend, exc, pipeline.backend)

            log.info("warmup total: %.1fs", time.perf_counter() - total_t0)
        yield
        # Capture shapes added during normal serving.
        _save_gemlite_cache("on shutdown")


def _save_gemlite_cache(when: str) -> None:
    try:
        GemLiteLinearTriton.cache_config(str(GEMLITE_PERSIST_PATH))
        log.info("persisted gemlite autotune cache (%s)", when)
    except Exception as exc:
        log.warning("failed to persist gemlite cache (%s): %s", when, exc)


# Starlette reads router.lifespan_context at ASGI startup — swapping it after
# app construction is picked up before uvicorn drives the first lifespan event.
app.router.lifespan_context = _lifespan_with_warmup

__all__ = ["app"]
