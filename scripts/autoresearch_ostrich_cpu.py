#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_cpu_experimental as gen


@dataclass(frozen=True)
class Candidate:
    name: str
    width: int
    height: int
    steps: int
    allow_sub128: bool
    guidance: float
    max_seq: int
    dtype: str
    text_encoder_dtype: str
    threads: int
    interop_threads: int


def candidate_grid() -> list[Candidate]:
    shared = dict(guidance=1.0, threads=4)
    layouts = [
        dict(width=128, height=128, allow_sub128=False),
        dict(width=96, height=96, allow_sub128=True),
    ]
    families = [
        dict(
            tag="fp16_auto_i4_s64",
            dtype="float16",
            text_encoder_dtype="auto",
            interop_threads=4,
            max_seq=64,
        ),
        dict(
            tag="fp16_auto_i4_s16",
            dtype="float16",
            text_encoder_dtype="auto",
            interop_threads=4,
            max_seq=16,
        ),
        dict(
            tag="fp16_auto_i1_s64",
            dtype="float16",
            text_encoder_dtype="auto",
            interop_threads=1,
            max_seq=64,
        ),
        dict(
            tag="fp16_auto_i1_s16",
            dtype="float16",
            text_encoder_dtype="auto",
            interop_threads=1,
            max_seq=16,
        ),
        dict(
            tag="fp32_auto_i4_s16",
            dtype="float32",
            text_encoder_dtype="auto",
            interop_threads=4,
            max_seq=16,
        ),
        dict(
            tag="fp32_auto_i4_s64",
            dtype="float32",
            text_encoder_dtype="auto",
            interop_threads=4,
            max_seq=64,
        ),
    ]

    candidates: list[Candidate] = []
    for layout in layouts:
        size_tag = f"{layout['width']}"
        for family in families:
            family_args = {key: value for key, value in family.items() if key != "tag"}
            for steps in (4, 3, 2, 1):
                candidates.append(
                    Candidate(
                        name=f"{size_tag}_{steps}step_{family['tag']}",
                        steps=steps,
                        **shared,
                        **layout,
                        **family_args,
                    )
                )
    return candidates


def runtime_key(candidate: Candidate) -> tuple[str, str, int, int]:
    return (
        candidate.dtype,
        candidate.text_encoder_dtype,
        candidate.threads,
        candidate.interop_threads,
    )


def check_image_stats(path: Path) -> tuple[int, str, str]:
    img = Image.open(path).convert("RGB")
    values = list(img.tobytes())
    mean = sum(values) / len(values)
    std = statistics.pstdev(values)
    r_vals = values[0::3]
    g_vals = values[1::3]
    b_vals = values[2::3]
    summary = (
        f"check_image: {path.name} mean={mean:.1f} std={std:.1f} "
        f"R={sum(r_vals)/len(r_vals):.1f} G={sum(g_vals)/len(g_vals):.1f} B={sum(b_vals)/len(b_vals):.1f}"
    )
    if mean < 5.0:
        return 1, summary, f"mean brightness {mean:.1f} < 5.0"
    if std < 15.0:
        return 1, summary, f"pixel std-dev {std:.1f} < 15.0"
    return 0, summary, ""


def configure_runtime(candidate: Candidate) -> None:
    torch.set_num_threads(candidate.threads)
    torch.set_num_interop_threads(candidate.interop_threads)
    gen.CPU_INFERENCE_DTYPE = gen.resolve_inference_dtype(candidate.dtype)
    gen.TEXT_ENCODER_DTYPE = gen.resolve_text_encoder_dtype(candidate.text_encoder_dtype)


def load_session(
    *,
    candidate: Candidate,
    prompt: str,
    model_root: Path,
    transformer_dir: Path | None,
    prompt_cache_dir: Path,
) -> tuple[torch.Tensor, torch.nn.Module, torch.nn.Module, float]:
    configure_runtime(candidate)
    started = time.time()
    print("  setup: encode prompt", flush=True)
    prompt_embeds = gen.encode_prompt(
        prompt,
        model_root,
        max_sequence_length=candidate.max_seq,
        cache_dir=prompt_cache_dir,
    )
    print(f"  setup: prompt ready elapsed={time.time()-started:.1f}s", flush=True)
    print("  setup: load VAE", flush=True)
    vae = gen.AutoencoderKLFlux2.from_pretrained(
        str(model_root / "vae"),
        torch_dtype=gen.CPU_INFERENCE_DTYPE,
    ).to("cpu").eval()
    print(f"  setup: VAE ready elapsed={time.time()-started:.1f}s", flush=True)
    print("  setup: load transformer", flush=True)
    transformer = (
        gen.load_unpacked_transformer(transformer_dir)
        if transformer_dir is not None
        else gen.load_dense_transformer(model_root)
    )
    print(f"  setup: transformer ready elapsed={time.time()-started:.1f}s", flush=True)
    return prompt_embeds, vae, transformer, time.time() - started


def run_candidate_warm(
    *,
    candidate: Candidate,
    prompt_embeds: torch.Tensor,
    vae: torch.nn.Module,
    transformer: torch.nn.Module,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    output_path = output_dir / f"{candidate.name}.png"
    started = time.time()
    image = gen.run_diffusion(
        transformer,
        vae,
        prompt_embeds,
        height=candidate.height,
        width=candidate.width,
        num_steps=candidate.steps,
        seed=seed,
        guidance=candidate.guidance,
    )
    image.save(output_path)
    wall = time.time() - started
    quality_rc, quality_stdout, quality_stderr = check_image_stats(output_path)
    return {
        "candidate": asdict(candidate),
        "returncode": 0,
        "wall_seconds": round(wall, 1),
        "quality_returncode": quality_rc,
        "quality_stdout": quality_stdout,
        "quality_stderr": quality_stderr,
        "output_path": str(output_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Warm autoresearch-style ostrich CPU sweeper.")
    p.add_argument("--prompt", default="ostrich")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int)
    p.add_argument("--match")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "autoresearch"))
    p.add_argument("--prompt-cache-dir", default=str(REPO_ROOT / "outputs" / "prompt_cache"))
    p.add_argument("--results-jsonl", default=str(REPO_ROOT / "outputs" / "autoresearch" / "results.jsonl"))
    p.add_argument("--model-root", default=str(REPO_ROOT / "models" / "bonsai-image-4B-ternary-gemlite"))
    p.add_argument("--transformer-dir", default=str(REPO_ROOT / "models" / "bonsai-image-4B-ternary-unpacked" / "transformer"))
    p.add_argument("--gemlite-dense", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--setup-only", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    prompt_cache_dir = Path(args.prompt_cache_dir)
    results_jsonl = Path(args.results_jsonl)
    model_root = Path(args.model_root)
    transformer_dir = None if args.gemlite_dense else (Path(args.transformer_dir) if args.transformer_dir else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl.parent.mkdir(parents=True, exist_ok=True)

    candidates = candidate_grid()
    if args.match:
        candidates = [c for c in candidates if args.match in c.name]
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if args.dry_run:
        for cand in candidates:
            print(json.dumps(asdict(cand), sort_keys=True))
        return 0

    grouped: dict[tuple[str, str, int, int], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[runtime_key(candidate)].append(candidate)

    best: dict[str, object] | None = None
    for _, group in grouped.items():
        session_ref = group[0]
        print(
            f"warming session dtype={session_ref.dtype} text_dtype={session_ref.text_encoder_dtype} "
            f"threads={session_ref.threads} interop={session_ref.interop_threads}",
            flush=True,
        )
        prompt_embeds, vae, transformer, setup_seconds = load_session(
            candidate=session_ref,
            prompt=args.prompt,
            model_root=model_root,
            transformer_dir=transformer_dir,
            prompt_cache_dir=prompt_cache_dir,
        )
        print(f"  session setup {setup_seconds:.1f}s", flush=True)
        if args.setup_only:
            print(
                json.dumps(
                    {
                        "candidate": asdict(session_ref),
                        "setup_only": True,
                        "session_setup_seconds": round(setup_seconds, 1),
                    },
                    sort_keys=True,
                )
            )
            del prompt_embeds, vae, transformer
            gc.collect()
            continue

        for index, candidate in enumerate(group, 1):
            print(f"[{index}/{len(group)}] {candidate.name}", flush=True)
            result = run_candidate_warm(
                candidate=candidate,
                prompt_embeds=prompt_embeds,
                vae=vae,
                transformer=transformer,
                seed=args.seed,
                output_dir=output_dir,
            )
            result["session_setup_seconds"] = round(setup_seconds, 1)
            with results_jsonl.open("a") as fh:
                fh.write(json.dumps(result) + "\n")

            passed = result["quality_returncode"] == 0
            if passed and (best is None or result["wall_seconds"] < best["wall_seconds"]):
                best = result
                print(
                    f"  new best: {candidate.name} warm_wall={result['wall_seconds']}s "
                    f"setup={setup_seconds:.1f}s path={result['output_path']}",
                    flush=True,
                )
            else:
                print(
                    f"  keep-best unchanged: quality={result['quality_returncode']} "
                    f"warm_wall={result['wall_seconds']}s",
                    flush=True,
                )

        del prompt_embeds, vae, transformer
        gc.collect()

    if best is not None:
        print(json.dumps(best, indent=2, sort_keys=True))
        return 0

    print("no passing candidates")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
