#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKLFlux2, Flux2Pipeline, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps
from hqq.core.quantize import HQQLinear
from hqq.models.hf.base import AutoHQQHFModel
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_STUDIO_VENDOR = (REPO_ROOT / "vendor" / "image-studio").resolve()
if IMAGE_STUDIO_VENDOR.exists():
    sys.path.insert(0, str(IMAGE_STUDIO_VENDOR))

from backend_gpu.diffusion_klein import _mflux_empirical_mu  # noqa: E402


def log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def mem_gib() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        return 0.0
    return 0.0


TEXT_ENCODER_DTYPE = torch.bfloat16
CPU_INFERENCE_DTYPE = torch.float16


def cpu_supports_bf16() -> bool:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return " bf16" in f" {cpuinfo.lower()} "


def resolve_inference_dtype(name: str) -> torch.dtype:
    if name == "auto":
        # On CPUs without native bf16 support, prefer float32 for the unpacked
        # transformer path. This matches the historically successful red-circle
        # and ostrich validations on this host.
        return torch.bfloat16 if cpu_supports_bf16() else torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name}")


def resolve_text_encoder_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if cpu_supports_bf16() else torch.float32
    return resolve_inference_dtype(name)


def prompt_cache_key(
    prompt: str,
    model_root: Path,
    max_sequence_length: int,
    *,
    include_inference_dtype: bool = True,
) -> str:
    payload = {
        "cache_format": "trimmed-prompt-v1",
        "prompt": prompt,
        "model_root": str(model_root.resolve()),
        "max_sequence_length": int(max_sequence_length),
        "text_encoder_dtype": str(TEXT_ENCODER_DTYPE),
    }
    if include_inference_dtype:
        payload["dtype"] = str(CPU_INFERENCE_DTYPE)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def load_quantized_text_encoder(path: Path, dtype: torch.dtype) -> nn.Module:
    log("loading quantized text encoder")
    model = AutoHQQHFModel.from_quantized(str(path), device="cpu")
    return model.to(dtype)


def dequantize_text_encoder(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    count = 0
    start = time.time()
    for name, mod in list(model.named_modules()):
        if isinstance(mod, HQQLinear):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            weight = mod.dequantize().to(dtype)
            dense = nn.Linear(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
                dtype=dtype,
            )
            dense.weight = nn.Parameter(weight, requires_grad=False)
            if mod.bias is not None:
                dense.bias = nn.Parameter(mod.bias.to(dtype), requires_grad=False)
            setattr(parent, child_name, dense)
            count += 1
            if count % 20 == 0:
                gc.collect()
                log(f"text encoder dense layers: {count} rss={mem_gib():.2f} GiB elapsed={time.time()-start:.1f}s")
    log(f"text encoder dequantized: {count} dense layers rss={mem_gib():.2f} GiB elapsed={time.time()-start:.1f}s")
    return model


@torch.no_grad()
def encode_prompt(
    prompt: str,
    model_root: Path,
    *,
    max_sequence_length: int,
    cache_dir: Path | None = None,
    text_encoder: nn.Module | None = None,
    tokenizer: AutoTokenizer | None = None,
) -> torch.Tensor:
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{prompt_cache_key(prompt, model_root, max_sequence_length)}.pt"
        if cache_path.exists():
            log(f"loading cached prompt embeds from {cache_path}")
            return torch.load(cache_path, map_location="cpu")
        compat_cache_path = cache_dir / (
            f"{prompt_cache_key(prompt, model_root, max_sequence_length, include_inference_dtype=False)}.pt"
        )
        if compat_cache_path.exists():
            log(f"loading compatible cached prompt embeds from {compat_cache_path}")
            return torch.load(compat_cache_path, map_location="cpu").to(CPU_INFERENCE_DTYPE)

    owns_text_encoder = text_encoder is None
    owns_tokenizer = tokenizer is None
    if text_encoder is None or tokenizer is None:
        text_path = model_root / "text_encoder-hqq-4bit"
        tok_path = text_path / "tokenizer"
        if text_encoder is None:
            text_encoder = dequantize_text_encoder(
                load_quantized_text_encoder(text_path, TEXT_ENCODER_DTYPE),
                TEXT_ENCODER_DTYPE,
            )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(str(tok_path))
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_inputs = tokenizer(text, return_tensors="pt")
    actual_token_count = int(full_inputs["input_ids"].shape[1])
    effective_max_length = min(max_sequence_length, actual_token_count)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=effective_max_length,
    )
    used_tokens = int(inputs["attention_mask"][0].sum().item())
    if actual_token_count > max_sequence_length:
        log(
            f"prompt truncated from {actual_token_count} to {max_sequence_length} tokens; "
            "increase --max-seq for faithful prompt encoding"
        )
    else:
        log(
            f"prompt tokenized to {used_tokens} tokens; "
            f"effective_seq={effective_max_length} within max_seq={max_sequence_length}"
        )
    log("encoding prompt")
    start = time.time()
    output = text_encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    prompt_embeds = torch.stack([output.hidden_states[k] for k in (9, 18, 27)], dim=1)
    batch_size, num_channels, seq_len, hidden_dim = prompt_embeds.shape
    prompt_embeds = (
        prompt_embeds.permute(0, 2, 1, 3)
        .reshape(batch_size, seq_len, num_channels * hidden_dim)
        .to(CPU_INFERENCE_DTYPE)
        .cpu()
        .contiguous()
    )
    log(f"prompt encoded shape={tuple(prompt_embeds.shape)} rss={mem_gib():.2f} GiB elapsed={time.time()-start:.1f}s")
    if cache_dir is not None:
        assert cache_path is not None
        torch.save(prompt_embeds, cache_path)
        log(f"saved prompt embeds cache {cache_path}")
        compat_cache_path = cache_dir / (
            f"{prompt_cache_key(prompt, model_root, max_sequence_length, include_inference_dtype=False)}.pt"
        )
        if compat_cache_path != cache_path:
            torch.save(prompt_embeds, compat_cache_path)
            log(f"saved compatible prompt embeds cache {compat_cache_path}")
    del output, inputs
    if owns_text_encoder:
        del text_encoder
    if owns_tokenizer:
        del tokenizer
    gc.collect()
    return prompt_embeds


def unpack_rows(w_q: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    return torch.stack(
        [(w_q >> 0) & 0x3, (w_q >> 2) & 0x3, (w_q >> 4) & 0x3, (w_q >> 6) & 0x3],
        dim=1,
    ).reshape(rows, cols)


def unpack_cols(w_q: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    return torch.stack(
        [(w_q >> 0) & 0x3, (w_q >> 2) & 0x3, (w_q >> 4) & 0x3, (w_q >> 6) & 0x3],
        dim=2,
    ).reshape(rows, cols)


def decode_gemlite_weight(
    layer_state: dict[str, torch.Tensor],
    target_shape: tuple[int, int],
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    w_q = layer_state["W_q"]
    scales = layer_state["scales"].to(torch.float32)
    rows, cols = target_shape
    if w_q.shape[0] * 4 == rows and w_q.shape[1] == cols:
        chunks = unpack_rows(w_q, rows, cols)
        scale_full = scales.repeat_interleave(group_size, dim=0)[:rows, :]
        return ((chunks.to(torch.float32) - 1.0) * scale_full).to(dtype)
    if w_q.shape[0] == rows and w_q.shape[1] * 4 == cols:
        chunks = unpack_cols(w_q, rows, cols)
        scale_full = scales.repeat_interleave(group_size, dim=0)[:cols, :].T
        return ((chunks.to(torch.float32) - 1.0) * scale_full).to(dtype)
    if w_q.shape[0] * 4 == cols and w_q.shape[1] == rows:
        chunks = unpack_rows(w_q, cols, rows)
        scale_full = scales.repeat_interleave(group_size, dim=0)[:cols, :]
        return ((chunks.to(torch.float32) - 1.0) * scale_full).T.to(dtype)
    if w_q.shape[0] == cols and w_q.shape[1] * 4 == rows:
        chunks = unpack_cols(w_q, cols, rows)
        scale_full = scales.repeat_interleave(group_size, dim=0)[:rows, :].T
        return ((chunks.to(torch.float32) - 1.0) * scale_full).T.to(dtype)
    raise RuntimeError(
        f"Unhandled gemlite packing W_q={tuple(w_q.shape)} target={target_shape} scales={tuple(scales.shape)}"
    )


def load_dense_transformer(model_root: Path) -> nn.Module:
    path = model_root / "transformer-gemlite-int2"
    with (path / "config.json").open() as fh:
        cfg = json.load(fh)
    with (path / "quantization_config.json").open() as fh:
        qcfg = json.load(fh)
    group_size = int(qcfg.get("group_size", 128))

    log("building transformer shell")
    # The gemlite GPU backend requires fp16 activations, but this CPU fallback
    # is a dense PyTorch model. Use bf16 only on CPUs that can execute it
    # efficiently; otherwise stay in fp32.
    model = Flux2Transformer2DModel.from_config(cfg).to(CPU_INFERENCE_DTYPE)
    state = torch.load(str(path / "state_dict.pt"), map_location="cpu")
    buckets: dict[str, dict[str, torch.Tensor]] = {}
    remainder: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        fqn, _, leaf = key.rpartition(".")
        if leaf in {"W_q", "bias", "scales", "zeros", "metadata", "orig_shape"} and fqn:
            buckets.setdefault(fqn, {})[leaf] = value
        else:
            remainder[key] = value
    missing, unexpected = model.load_state_dict(remainder, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected transformer keys: {unexpected[:8]}")
    if len(missing) != len(buckets):
        raise RuntimeError(f"unexpected transformer missing keys count: {len(missing)} vs {len(buckets)}")
    del remainder, state
    gc.collect()

    start = time.time()
    items = sorted(buckets.items())
    for idx, (fqn, layer_state) in enumerate(items, 1):
        parent_fqn, _, child_name = fqn.rpartition(".")
        parent = model.get_submodule(parent_fqn) if parent_fqn else model
        child = getattr(parent, child_name)
        weight = decode_gemlite_weight(
            layer_state,
            tuple(child.weight.shape),
            group_size,
            CPU_INFERENCE_DTYPE,
        ).to(CPU_INFERENCE_DTYPE)
        child.weight = nn.Parameter(weight, requires_grad=False)
        if "bias" in layer_state and layer_state["bias"] is not None:
            child.bias = nn.Parameter(layer_state["bias"].to(CPU_INFERENCE_DTYPE), requires_grad=False)
        if idx % 10 == 0 or idx == len(items):
            gc.collect()
            log(f"transformer dense layers: {idx}/{len(items)} rss={mem_gib():.2f} GiB elapsed={time.time()-start:.1f}s")
    model._inference_dtype = CPU_INFERENCE_DTYPE  # type: ignore[attr-defined]
    return model.eval()


def load_unpacked_transformer(transformer_dir: Path) -> nn.Module:
    log(f"loading unpacked transformer from {transformer_dir}")
    monolith_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    sharded_index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if monolith_path.is_file() and sharded_index_path.is_file():
        # This repo currently has both an older monolithic safetensors payload
        # and a newer sharded index in the same unpacked directory. Prefer the
        # monolith because it is the last semantically validated CPU payload.
        log(f"loading monolithic unpacked transformer weights from {monolith_path}")
        config = Flux2Transformer2DModel.load_config(str(transformer_dir))
        model = Flux2Transformer2DModel.from_config(config)
        state_dict = load_safetensors_file(str(monolith_path), device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "monolithic transformer state_dict mismatch: "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        model = model.to(device="cpu", dtype=CPU_INFERENCE_DTYPE)
        model._inference_dtype = CPU_INFERENCE_DTYPE  # type: ignore[attr-defined]
        return model.eval()
    model = Flux2Transformer2DModel.from_pretrained(
        str(transformer_dir),
        torch_dtype=CPU_INFERENCE_DTYPE,
        local_files_only=True,
    ).to("cpu")
    model._inference_dtype = CPU_INFERENCE_DTYPE  # type: ignore[attr-defined]
    return model.eval()


def resolve_transformer_dir(model_root: Path, explicit_transformer_dir: str | None) -> Path | None:
    if explicit_transformer_dir:
        return Path(explicit_transformer_dir)

    direct = model_root / "transformer"
    if direct.is_dir():
        log(f"using unpacked transformer at {direct}")
        return direct

    sibling_name = model_root.name.replace("gemlite", "unpacked")
    sibling = model_root.parent / sibling_name / "transformer"
    if sibling.is_dir():
        log(
            "using sibling unpacked transformer instead of GemLite dense reconstruction: "
            f"{sibling}"
        )
        return sibling

    return None


def build_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=4096,
    )


@torch.no_grad()
def decode_latents_to_image(
    latents: torch.Tensor,
    latent_ids: torch.Tensor,
    vae: nn.Module,
    *,
    log_prefix: str,
) -> Image.Image:
    vae_device = next(vae.parameters()).device
    decode_start = time.time()
    log(f"{log_prefix} decode start rss={mem_gib():.2f} GiB")
    latents = Flux2Pipeline._unpack_latents_with_ids(latents, latent_ids)
    log(f"{log_prefix} latents unpacked rss={mem_gib():.2f} GiB elapsed={time.time()-decode_start:.1f}s")
    latents = latents.to(device=vae_device, dtype=CPU_INFERENCE_DTYPE)
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(latents.device, latents.dtype)
    latents = latents * bn_std + bn_mean
    latents = Flux2Pipeline._unpatchify_latents(latents)
    log(f"{log_prefix} vae decode start rss={mem_gib():.2f} GiB elapsed={time.time()-decode_start:.1f}s")
    image = vae.decode(latents, return_dict=False)[0]
    log(f"{log_prefix} vae decode done rss={mem_gib():.2f} GiB elapsed={time.time()-decode_start:.1f}s")
    img = image[0].clamp(-1.0, 1.0).float()
    img = (img + 1.0) * 127.5
    img = img.clamp(0.0, 255.0).round().to(torch.uint8)
    img = img.permute(1, 2, 0).cpu().numpy()
    log(f"{log_prefix} decode done rss={mem_gib():.2f} GiB elapsed={time.time()-decode_start:.1f}s")
    return Image.fromarray(img, mode="RGB")


@torch.no_grad()
def run_diffusion(
    transformer: nn.Module,
    vae: nn.Module,
    prompt_embeds: torch.Tensor,
    *,
    height: int,
    width: int,
    num_steps: int,
    seed: int,
    guidance: float,
    step_output_dir: Path | None = None,
    step_output_stem: str = "step",
) -> Image.Image:
    transformer_device = next(transformer.parameters()).device
    transformer_dtype = next(transformer.parameters()).dtype
    scheduler = build_scheduler()
    prompt_embeds = prompt_embeds.to(device=transformer_device, dtype=transformer_dtype)
    text_ids = Flux2Pipeline._prepare_text_ids(prompt_embeds).to(transformer_device)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    h_lat = 2 * (int(height) // (vae_scale_factor * 2))
    w_lat = 2 * (int(width) // (vae_scale_factor * 2))
    in_channels_latents = transformer.config.in_channels // 4

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    noise_shape = (1, in_channels_latents * 4, h_lat // 2, w_lat // 2)
    latents_4d = torch.randn(noise_shape, generator=gen, dtype=torch.float32).to(
        device=transformer_device, dtype=transformer_dtype
    )
    latent_ids = Flux2Pipeline._prepare_latent_ids(latents_4d).to(transformer_device)
    latents = Flux2Pipeline._pack_latents(latents_4d)
    image_seq_len = latents.shape[1]

    mu = _mflux_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
    sigmas = None if getattr(scheduler.config, "use_flow_sigmas", False) else torch.linspace(1.0, 1.0 / num_steps, num_steps).numpy()
    timesteps, _ = retrieve_timesteps(scheduler, num_steps, transformer_device, sigmas=sigmas, mu=mu)
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(0)
    guidance_t = torch.full([1], guidance, device=transformer_device, dtype=torch.float32).expand(latents.shape[0])

    start = time.time()
    for i, t in enumerate(timesteps, 1):
        step_start = time.time()
        log(f"diffusion step {i}/{len(timesteps)} start rss={mem_gib():.2f} GiB")
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance_t,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]
        latents_dtype = latents.dtype
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)
        log(
            f"diffusion step {i}/{len(timesteps)} done "
            f"rss={mem_gib():.2f} GiB elapsed={time.time()-step_start:.1f}s"
        )
        if step_output_dir is not None:
            step_image = decode_latents_to_image(
                latents.clone(),
                latent_ids,
                vae,
                log_prefix=f"step {i}/{len(timesteps)}",
            )
            step_path = step_output_dir / f"{step_output_stem}_step{i:02d}.png"
            step_image.save(step_path)
            log(f"saved {step_path}")
    log(f"diffusion complete rss={mem_gib():.2f} GiB elapsed={time.time()-start:.1f}s")

    return decode_latents_to_image(latents, latent_ids, vae, log_prefix="final")


def main() -> int:
    global CPU_INFERENCE_DTYPE
    global TEXT_ENCODER_DTYPE

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--max-seq", type=int, default=512)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Main transformer/VAE/latent dtype. auto prefers bf16 only with native support, otherwise float32 on CPU.",
    )
    parser.add_argument(
        "--text-encoder-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Text encoder dtype. auto prefers bf16 only with native support, otherwise float32 for faster CPU prompt encoding.",
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument("--interop-threads", type=int)
    parser.add_argument("--prompt-cache-dir")
    parser.add_argument("--model-root", default=str(REPO_ROOT / "models" / "bonsai-image-4B-ternary-gemlite"))
    parser.add_argument("--transformer-dir")
    parser.add_argument(
        "--gemlite-dense",
        action="store_true",
        help="force the experimental GemLite-to-dense CPU transformer path",
    )
    parser.add_argument(
        "--allow-sub128",
        action="store_true",
        help="allow exploratory renders below 128x128; useful for performance sweeps, but less reliable semantically",
    )
    parser.add_argument("--step-output-dir")
    parser.add_argument(
        "--prompt-cache-only",
        action="store_true",
        help="encode and cache the prompt embeds, then exit without loading VAE/transformer or rendering",
    )
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.steps <= 0:
        raise SystemExit("--steps must be a positive integer")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    if args.interop_threads is not None:
        torch.set_num_interop_threads(args.interop_threads)
    CPU_INFERENCE_DTYPE = resolve_inference_dtype(args.dtype)
    TEXT_ENCODER_DTYPE = resolve_text_encoder_dtype(args.text_encoder_dtype)
    model_root = Path(args.model_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    step_output_dir = Path(args.step_output_dir) if args.step_output_dir else None
    prompt_cache_dir = Path(args.prompt_cache_dir) if args.prompt_cache_dir else None
    if step_output_dir is not None:
        step_output_dir.mkdir(parents=True, exist_ok=True)

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise SystemExit("height and width must be multiples of 32")

    # Each packed latent token expands to a fixed 16x16 image patch after the
    # unpack + VAE decode path. 64x64 therefore gives only a 4x4 packed grid:
    # useful for coarse sanity checks, but too small for reliable geometry
    # tests like quadrants or thin lines. Use 128x128+ for structure debugging.
    min_dim = 96 if args.allow_sub128 else 128
    if args.height < min_dim or args.width < min_dim:
        raise SystemExit(
            f"height and width must be at least {min_dim} "
            f"for {'exploratory' if args.allow_sub128 else 'meaningful'} CPU renders"
        )

    total_start = time.time()
    log(
        f"cpu={platform.machine()} threads={torch.get_num_threads()} "
        f"interop={torch.get_num_interop_threads()} "
        f"text_dtype={TEXT_ENCODER_DTYPE} dtype={CPU_INFERENCE_DTYPE}"
    )
    prompt_embeds = encode_prompt(
        args.prompt,
        model_root,
        max_sequence_length=args.max_seq,
        cache_dir=prompt_cache_dir,
    )
    log(f"after prompt encode rss={mem_gib():.2f} GiB")
    if args.prompt_cache_only:
        log("prompt-cache-only requested; skipping model load and render")
        log(f"total elapsed={time.time()-total_start:.1f}s")
        return 0

    log("loading VAE")
    vae = AutoencoderKLFlux2.from_pretrained(
        str(model_root / "vae"),
        torch_dtype=CPU_INFERENCE_DTYPE,
    ).to("cpu").eval()
    log(f"vae ready rss={mem_gib():.2f} GiB")

    transformer_dir = None if args.gemlite_dense else resolve_transformer_dir(model_root, args.transformer_dir)
    transformer = (
        load_unpacked_transformer(transformer_dir)
        if transformer_dir is not None
        else load_dense_transformer(model_root)
    )
    log(f"transformer ready rss={mem_gib():.2f} GiB")
    image = run_diffusion(
        transformer,
        vae,
        prompt_embeds,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        seed=args.seed,
        guidance=args.guidance,
        step_output_dir=step_output_dir,
        step_output_stem=output_path.stem,
    )
    image.save(output_path)
    log(f"saved {output_path}")
    log(f"total elapsed={time.time()-total_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
