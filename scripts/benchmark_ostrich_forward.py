#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import Flux2Pipeline
from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_cpu_experimental as gen


class StopAfterBlocks(RuntimeError):
    pass


def build_inputs(
    *,
    transformer,
    prompt_embeds: torch.Tensor,
    width: int,
    height: int,
    num_steps: int,
    seed: int,
):
    transformer_device = next(transformer.parameters()).device
    transformer_dtype = next(transformer.parameters()).dtype
    prompt_embeds = prompt_embeds.to(device=transformer_device, dtype=transformer_dtype)
    text_ids = Flux2Pipeline._prepare_text_ids(prompt_embeds).to(transformer_device)

    in_channels_latents = transformer.config.in_channels // 4
    h_lat = height // 8
    w_lat = width // 8
    noise_shape = (1, in_channels_latents * 4, h_lat // 2, w_lat // 2)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents_4d = torch.randn(noise_shape, generator=generator, dtype=torch.float32).to(
        device=transformer_device, dtype=transformer_dtype
    )
    latent_ids = Flux2Pipeline._prepare_latent_ids(latents_4d).to(transformer_device)
    latents = Flux2Pipeline._pack_latents(latents_4d)

    scheduler = gen.build_scheduler()
    image_seq_len = latents.shape[1]
    mu = gen._mflux_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
    sigmas = None if getattr(scheduler.config, "use_flow_sigmas", False) else torch.linspace(
        1.0, 1.0 / num_steps, num_steps
    ).numpy()
    timesteps, _ = retrieve_timesteps(
        scheduler, num_steps, transformer_device, sigmas=sigmas, mu=mu
    )
    timestep = timesteps[0].expand(latents.shape[0]).to(latents.dtype) / 1000
    guidance = torch.full([1], 1.0, device=transformer_device, dtype=torch.float32).expand(
        latents.shape[0]
    )
    return {
        "prompt_embeds": prompt_embeds,
        "text_ids": text_ids,
        "latents": latents,
        "latent_ids": latent_ids,
        "timestep": timestep,
        "guidance": guidance,
        "seq_len": int(image_seq_len),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark one ostrich diffusion forward pass.")
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--transformer-dir", required=True)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--trace-blocks", action="store_true")
    parser.add_argument("--max-blocks", type=int)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    gen.CPU_INFERENCE_DTYPE = gen.resolve_inference_dtype(args.dtype)

    prompt_cache = Path(args.prompt_cache)
    transformer_dir = Path(args.transformer_dir)

    load_start = time.time()
    prompt_embeds = torch.load(prompt_cache, map_location="cpu").to(gen.CPU_INFERENCE_DTYPE)
    transformer = gen.load_unpacked_transformer(transformer_dir)
    load_seconds = time.time() - load_start
    print(json.dumps({"phase": "load", "seconds": round(load_seconds, 1)}), flush=True)

    inputs = build_inputs(
        transformer=transformer,
        prompt_embeds=prompt_embeds,
        width=args.width,
        height=args.height,
        num_steps=1,
        seed=args.seed,
    )

    block_starts: dict[int, float] = {}
    hooks = []
    if args.trace_blocks and hasattr(transformer, "transformer_blocks"):
        for idx, block in enumerate(transformer.transformer_blocks):
            def pre_hook(_module, _inputs, *, block_idx=idx):
                block_starts[block_idx] = time.time()

            def post_hook(_module, _inputs, _output, *, block_idx=idx):
                start = block_starts.pop(block_idx, None)
                if start is None:
                    return
                elapsed = time.time() - start
                print(
                    json.dumps(
                        {
                            "phase": "block",
                            "block": block_idx,
                            "seconds": round(elapsed, 3),
                        }
                    ),
                    flush=True,
                )
                if args.max_blocks is not None and (block_idx + 1) >= args.max_blocks:
                    raise StopAfterBlocks()

            hooks.append(block.register_forward_pre_hook(pre_hook))
            hooks.append(block.register_forward_hook(post_hook))

    for repeat in range(1, args.repeats + 1):
        start = time.time()
        stopped_early = False
        try:
            _ = transformer(
                hidden_states=inputs["latents"],
                timestep=inputs["timestep"],
                guidance=inputs["guidance"],
                encoder_hidden_states=inputs["prompt_embeds"],
                txt_ids=inputs["text_ids"],
                img_ids=inputs["latent_ids"],
                return_dict=False,
            )[0]
        except StopAfterBlocks:
            stopped_early = True
        elapsed = time.time() - start
        print(
            json.dumps(
                {
                    "phase": "forward",
                    "repeat": repeat,
                    "threads": args.threads,
                    "interop_threads": args.interop_threads,
                    "dtype": args.dtype,
                    "width": args.width,
                    "height": args.height,
                    "seq_len": inputs["seq_len"],
                    "seconds": round(elapsed, 1),
                    "stopped_early": stopped_early,
                }
            ),
            flush=True,
        )
    for hook in hooks:
        hook.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
