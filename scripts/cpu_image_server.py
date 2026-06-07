#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_cpu_experimental as gen


class GenerateRequest(BaseModel):
    prompt: str
    output: str
    width: int
    height: int
    steps: int
    seed: int
    guidance: float = 1.0
    max_seq: int = 64


class ServerState:
    def __init__(self) -> None:
        self.model_root = Path(
            os.environ.get("BONSAI_MODEL_ROOT", str(REPO_ROOT / "models" / "bonsai-image-4B-ternary-gemlite"))
        )
        self.transformer_dir = Path(
            os.environ.get(
                "BONSAI_TRANSFORMER_DIR",
                str(REPO_ROOT / "models" / "bonsai-image-4B-ternary-unpacked" / "transformer"),
            )
        )
        self.prompt_cache_dir = Path(
            os.environ.get("BONSAI_PROMPT_CACHE_DIR", str(REPO_ROOT / "outputs" / "prompt_cache_fp32_auto"))
        )
        self.dtype_name = os.environ.get("BONSAI_DTYPE", "float32")
        self.text_encoder_dtype_name = os.environ.get("BONSAI_TEXT_ENCODER_DTYPE", "auto")
        self.threads = int(os.environ.get("BONSAI_THREADS", "4"))
        self.interop_threads = int(os.environ.get("BONSAI_INTEROP_THREADS", "4"))
        self.use_gemlite_dense = os.environ.get("BONSAI_GEMLITE_DENSE", "").lower() in {"1", "true", "yes"}
        self.compile_transformer = os.environ.get("BONSAI_COMPILE_TRANSFORMER", "").lower() in {"1", "true", "yes"}
        self.compile_mode = os.environ.get("BONSAI_COMPILE_MODE", "reduce-overhead")
        self.dynamic_int8 = os.environ.get("BONSAI_DYNAMIC_INT8", "").lower() in {"1", "true", "yes"}
        self.dynamic_int8_engine = os.environ.get("BONSAI_DYNAMIC_INT8_ENGINE", "qnnpack")
        self.lock = Lock()
        self.ready = False
        self.started_at = time.time()
        self.vae = None
        self.transformer = None


STATE = ServerState()


def configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    gen.torch.set_num_threads(STATE.threads)
    gen.torch.set_num_interop_threads(STATE.interop_threads)
    gen.CPU_INFERENCE_DTYPE = gen.resolve_inference_dtype(STATE.dtype_name)
    gen.TEXT_ENCODER_DTYPE = gen.resolve_text_encoder_dtype(STATE.text_encoder_dtype_name)


def load_models() -> None:
    configure_runtime()
    STATE.prompt_cache_dir.mkdir(parents=True, exist_ok=True)
    STATE.vae = gen.AutoencoderKLFlux2.from_pretrained(
        str(STATE.model_root / "vae"),
        torch_dtype=gen.CPU_INFERENCE_DTYPE,
    ).to("cpu").eval()
    STATE.transformer = (
        gen.load_dense_transformer(STATE.model_root)
        if STATE.use_gemlite_dense
        else gen.load_unpacked_transformer(STATE.transformer_dir)
    )
    if STATE.dynamic_int8:
        gen.torch.backends.quantized.engine = STATE.dynamic_int8_engine
        STATE.transformer = gen.torch.quantization.quantize_dynamic(
            STATE.transformer,
            {nn.Linear},
            dtype=gen.torch.qint8,
        )
    if STATE.compile_transformer:
        STATE.transformer = gen.torch.compile(STATE.transformer, mode=STATE.compile_mode)
    STATE.ready = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "ready": STATE.ready,
        "uptime_seconds": round(time.time() - STATE.started_at, 1),
        "dtype": STATE.dtype_name,
        "text_encoder_dtype": STATE.text_encoder_dtype_name,
        "threads": STATE.threads,
        "interop_threads": STATE.interop_threads,
        "gemlite_dense": STATE.use_gemlite_dense,
        "compile_transformer": STATE.compile_transformer,
        "compile_mode": STATE.compile_mode,
        "dynamic_int8": STATE.dynamic_int8,
        "dynamic_int8_engine": STATE.dynamic_int8_engine,
        "prompt_cache_dir": str(STATE.prompt_cache_dir),
    }


@app.post("/generate")
def generate(req: GenerateRequest) -> dict[str, object]:
    if not STATE.ready or STATE.vae is None or STATE.transformer is None:
        raise HTTPException(status_code=503, detail="server not ready")
    if req.width % 32 != 0 or req.height % 32 != 0:
        raise HTTPException(status_code=400, detail="height and width must be multiples of 32")

    output_path = Path(req.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_key = gen.prompt_cache_key(
        req.prompt,
        STATE.model_root,
        req.max_seq,
        include_inference_dtype=False,
    )
    cache_path = STATE.prompt_cache_dir / f"{cache_key}.pt"
    if not cache_path.exists():
        raise HTTPException(
            status_code=409,
            detail="prompt cache miss; uncached prompts are disabled on this host for the warm server route",
        )

    with STATE.lock:
        start = time.time()
        prompt_embeds = gen.torch.load(cache_path, map_location="cpu").to(gen.CPU_INFERENCE_DTYPE)
        prompt_seconds = time.time() - start
        image = gen.run_diffusion(
            STATE.transformer,
            STATE.vae,
            prompt_embeds,
            height=req.height,
            width=req.width,
            num_steps=req.steps,
            seed=req.seed,
            guidance=req.guidance,
        )
        image.save(output_path)
        total_seconds = time.time() - start

    return {
        "prompt": req.prompt,
        "output_path": str(output_path),
        "width": req.width,
        "height": req.height,
        "steps": req.steps,
        "seed": req.seed,
        "prompt_seconds": round(prompt_seconds, 1),
        "total_seconds": round(total_seconds, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm CPU Bonsai image server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--model-root", default=str(REPO_ROOT / "models" / "bonsai-image-4B-ternary-gemlite"))
    parser.add_argument(
        "--transformer-dir",
        default=str(REPO_ROOT / "models" / "bonsai-image-4B-ternary-unpacked" / "transformer"),
    )
    parser.add_argument("--prompt-cache-dir", default=str(REPO_ROOT / "outputs" / "prompt_cache_fp32_auto"))
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--text-encoder-dtype", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=4)
    parser.add_argument("--gemlite-dense", action="store_true")
    parser.add_argument("--compile-transformer", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--dynamic-int8", action="store_true")
    parser.add_argument("--dynamic-int8-engine", default="qnnpack")
    args = parser.parse_args()

    os.environ["BONSAI_MODEL_ROOT"] = args.model_root
    os.environ["BONSAI_TRANSFORMER_DIR"] = args.transformer_dir
    os.environ["BONSAI_PROMPT_CACHE_DIR"] = args.prompt_cache_dir
    os.environ["BONSAI_DTYPE"] = args.dtype
    os.environ["BONSAI_TEXT_ENCODER_DTYPE"] = args.text_encoder_dtype
    os.environ["BONSAI_THREADS"] = str(args.threads)
    os.environ["BONSAI_INTEROP_THREADS"] = str(args.interop_threads)
    os.environ["BONSAI_GEMLITE_DENSE"] = "1" if args.gemlite_dense else "0"
    os.environ["BONSAI_COMPILE_TRANSFORMER"] = "1" if args.compile_transformer else "0"
    os.environ["BONSAI_COMPILE_MODE"] = args.compile_mode
    os.environ["BONSAI_DYNAMIC_INT8"] = "1" if args.dynamic_int8 else "0"
    os.environ["BONSAI_DYNAMIC_INT8_ENGINE"] = args.dynamic_int8_engine

    global STATE
    STATE = ServerState()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
