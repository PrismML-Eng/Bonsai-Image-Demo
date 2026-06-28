"""Mac shim: wrap image-studio's `backend.server:app` with a demo-shaped
`/backends` response so the picker shows only the Bonsai variants.

The upstream `/backends` route computes `supported_families` from
`BACKENDS` × `MODEL_FAMILIES`, which for the MLX kind includes
`bfl-klein-bf16`. That family isn't part of the Bonsai demo — and
selecting it triggers mflux to download the full ~24 GB BFL Klein
weights from HF on first use. Mirror what `scripts/local_backend.py`
does on Linux: strip the existing route and serve a hand-shaped reply
that advertises only `bonsai-ternary` + `bonsai-binary`.

Run with: uvicorn scripts.local_backend_mac:app
"""
from __future__ import annotations

import os

# Importing the upstream module registers all its routes (including the
# real `/generate`, `/generate/compare`, etc.) on `app`. We keep those —
# only `/backends` is replaced.
from backend.server import app  # noqa: F401  (re-exported)

# Strip image-studio's `/backends` route before registering ours.
# FastAPI's router iterates routes in registration order, so re-adding
# `@app.get("/backends")` without removing the original would be a no-op
# (the original still wins). Filter on `path` so we touch only that one.
app.router.routes = [
    r for r in app.router.routes if getattr(r, "path", "") != "/backends"
]


@app.get("/backends")
async def _backends(force_disable: bool = False) -> dict:
    """Demo-shaped `/backends` reply.

    Schema matches image-studio's `useBackends()` hook
    (kind / supported_families / default_family / healthy / reason).
    The arm we boot with is read from MFLUX_STUDIO_DEFAULT_BACKEND
    (parallel to MFLUX_STUDIO_GPU_DEFAULT_BACKEND on Linux); fall back
    to ternary if unset.

    `force_disable` arrives via query string on the upstream route's
    signature; harmless on Mac (no GPU probe to disable) — accepted
    only so the route signature matches the frontend's expectations.

    By default advertises both Bonsai variants (ternary + binary). When
    serve.sh sets BONSAI_SUPPORTED_FAMILIES, advertise only those
    downloaded families so the picker can't offer a variant whose
    weights aren't on disk — selecting an absent arm routes /generate to
    a non-existent model dir and 500s. Mirrors scripts/local_backend.py.
    """
    arm = os.environ.get("MFLUX_STUDIO_DEFAULT_BACKEND", "bonsai-ternary-mlx")
    if arm.endswith("-mlx"):
        default_family = arm[: -len("-mlx")]
    elif arm.endswith("-gemlite"):
        default_family = arm[: -len("-gemlite")]
    else:
        default_family = arm
    allowed_families = {"bonsai-ternary", "bonsai-binary"}
    configured_families = [
        family.strip()
        for family in os.environ.get("BONSAI_SUPPORTED_FAMILIES", "").split(",")
        if family.strip() in allowed_families
    ]
    supported_families = configured_families or ["bonsai-ternary", "bonsai-binary"]
    if default_family not in supported_families:
        default_family = supported_families[0]
    return {
        "kind": "mlx",
        "supported_families": supported_families,
        "default_family": default_family,
        "healthy": True,
        "reason": None,
    }


__all__ = ["app"]


@app.get("/")
async def _root() -> dict:
    """Friendly backend root for people who open the API port directly."""
    frontend_port = os.environ.get("FRONTEND_PORT", "3000")
    return {
        "service": "Bonsai Image backend",
        "studio_url": f"http://localhost:{frontend_port}/",
        "docs_url": "/docs",
        "health_url": "/backends",
    }
