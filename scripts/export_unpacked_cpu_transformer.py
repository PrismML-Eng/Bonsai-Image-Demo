#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_cpu_experimental as gen


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"missing {label}: {path}")


DTYPE_INFO = {
    torch.float32: ("F32", 4),
    torch.float16: ("F16", 2),
    torch.bfloat16: ("BF16", 2),
    torch.int64: ("I64", 8),
    torch.int32: ("I32", 4),
    torch.uint8: ("U8", 1),
}


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().tobytes()
    return tensor.numpy().tobytes()


def tensor_nbytes(tensor: torch.Tensor) -> int:
    try:
        _, itemsize = DTYPE_INFO[tensor.dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported safetensors dtype: {tensor.dtype}") from exc
    return tensor.numel() * itemsize


def layer_state_for(state: dict[str, torch.Tensor], fqn: str) -> dict[str, torch.Tensor]:
    prefix = f"{fqn}."
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def build_tensor_plan(
    state: dict[str, torch.Tensor],
    quantized_fqns: list[str],
    group_size: int,
    dtype: torch.dtype,
) -> list[tuple[str, torch.Tensor | tuple[str, tuple[int, int], dict[str, torch.Tensor]]]]:
    quantized = set(quantized_fqns)
    plan: list[tuple[str, torch.Tensor | tuple[str, tuple[int, int], dict[str, torch.Tensor]]]] = []

    for key, value in sorted(state.items()):
        fqn, _, leaf = key.rpartition(".")
        if fqn in quantized and leaf in {"W_q", "bias", "scales", "zeros", "metadata", "orig_shape"}:
            continue
        plan.append((key, value))

    for fqn in sorted(quantized_fqns):
        layer_state = layer_state_for(state, fqn)
        if "orig_shape" not in layer_state:
            raise RuntimeError(f"missing orig_shape for quantized layer {fqn}")
        target_shape = tuple(int(v) for v in layer_state["orig_shape"].tolist())
        if len(target_shape) != 2:
            raise RuntimeError(f"unexpected orig_shape for {fqn}: {target_shape}")
        plan.append((f"{fqn}.weight", ("gemlite_weight", target_shape, layer_state)))
        if "bias" in layer_state and layer_state["bias"] is not None:
            plan.append((f"{fqn}.bias", layer_state["bias"].to(dtype)))

    return sorted(plan, key=lambda item: item[0])


def materialize_plan_item(
    item: torch.Tensor | tuple[str, tuple[int, int], dict[str, torch.Tensor]],
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(item, torch.Tensor):
        return item
    kind, target_shape, layer_state = item
    if kind != "gemlite_weight":
        raise RuntimeError(f"unknown tensor plan item: {kind}")
    return gen.decode_gemlite_weight(layer_state, target_shape, group_size, dtype).contiguous()


def save_safetensors_streaming(
    output_path: Path,
    plan: list[tuple[str, torch.Tensor | tuple[str, tuple[int, int], dict[str, torch.Tensor]]]],
    *,
    group_size: int,
    dtype: torch.dtype,
) -> None:
    header: dict[str, dict[str, object]] = {}
    offset = 0
    for key, item in plan:
        tensor = materialize_plan_item(item, group_size, dtype)
        dtype_name, _ = DTYPE_INFO[tensor.dtype]
        nbytes = tensor_nbytes(tensor)
        header[key] = {
            "dtype": dtype_name,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        del tensor

    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - (len(header_bytes) % 8)) % 8)

    with output_path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for idx, (key, item) in enumerate(plan, 1):
            tensor = materialize_plan_item(item, group_size, dtype)
            fh.write(tensor_bytes(tensor))
            if idx % 10 == 0 or idx == len(plan):
                gen.log(f"wrote tensors: {idx}/{len(plan)} last={key}")
            del tensor


def clean_output_dir(output_dir: Path) -> None:
    for path in output_dir.glob("diffusion_pytorch_model*.safetensors"):
        path.unlink()
    index_path = output_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        index_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the GemLite CPU transformer into an unpacked diffusers directory."
    )
    parser.add_argument(
        "--model-root",
        default=str(REPO_ROOT / "models" / "bonsai-image-4B-ternary-gemlite"),
        help="Model root containing transformer-gemlite-int2, text_encoder-hqq-4bit, and vae.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "models" / "bonsai-image-4B-ternary-unpacked" / "transformer"),
        help="Destination directory for Flux2Transformer2DModel.save_pretrained().",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Dense transformer dtype to materialize before save_pretrained().",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="20GB",
        help="Deprecated; kept for CLI compatibility. The CPU exporter now writes one monolithic safetensors file directly.",
    )
    args = parser.parse_args()

    model_root = Path(args.model_root)
    output_dir = Path(args.output_dir)
    source_dir = model_root / "transformer-gemlite-int2"

    require_dir(model_root / "text_encoder-hqq-4bit", "text encoder")
    require_dir(model_root / "vae", "VAE")
    require_dir(source_dir, "GemLite transformer")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"output directory already exists and is not empty: {output_dir}\n"
            "pass --overwrite to refresh it"
        )

    gen.CPU_INFERENCE_DTYPE = gen.resolve_inference_dtype(args.dtype)
    start = time.time()
    gen.log(
        f"streaming dense unpacked transformer from {source_dir} to {output_dir} "
        f"dtype={gen.CPU_INFERENCE_DTYPE}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(output_dir)

    with (source_dir / "quantization_config.json").open() as fh:
        qcfg = json.load(fh)
    group_size = int(qcfg.get("group_size", 128))
    quantized_fqns = list(qcfg.get("quantized_fqns", []))
    if not quantized_fqns:
        raise RuntimeError(f"quantization_config.json has no quantized_fqns: {source_dir}")

    shutil.copy2(source_dir / "config.json", output_dir / "config.json")
    state = torch.load(str(source_dir / "state_dict.pt"), map_location="cpu")
    plan = build_tensor_plan(
        state,
        quantized_fqns,
        group_size,
        gen.CPU_INFERENCE_DTYPE,
    )
    save_safetensors_streaming(
        output_dir / "diffusion_pytorch_model.safetensors",
        plan,
        group_size=group_size,
        dtype=gen.CPU_INFERENCE_DTYPE,
    )
    (output_dir / "cpu_decode_layout_version.json").write_text(
        json.dumps(
            {
                "gemlite_decode_layout": "over-k-transposed",
                "rewritten_quantized_weight_count": len(quantized_fqns),
                "source": str(source_dir / "state_dict.pt"),
                "export": "streaming-safetensors",
                "reason": "GemLite uint8-packed int2 weights are stored over K with transpose.",
            },
            indent=2,
        )
        + "\n"
    )
    gen.log(f"saved unpacked transformer to {output_dir} elapsed={time.time()-start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
