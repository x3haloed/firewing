#!/usr/bin/env python3
"""Generate real-checkpoint Qwen top-10 router fixtures without weight bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import transformers
from safetensors import safe_open

if __package__:
    from tools.generate_ngram_address_fixture import (
        checkpoint_revision,
        load_model_lock,
        locked_file,
        sha256_file,
    )
else:
    from generate_ngram_address_fixture import (  # type: ignore[no-redef]
        checkpoint_revision,
        load_model_lock,
        locked_file,
        sha256_file,
    )


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_real_top10_router"


def tensor_bytes(value: torch.Tensor) -> bytes:
    if value.dtype != torch.bfloat16 or not value.is_contiguous():
        raise ValueError("expected contiguous BF16 tensor")
    return value.view(torch.uint16).numpy().tobytes()


def make_hidden(size: int, spec: dict[str, int]) -> torch.Tensor:
    index = torch.arange(size, dtype=torch.int64)
    value = ((index * spec["multiplier"] + spec["add"]) % spec["modulus"] - spec["center"])
    hidden = value.to(torch.float32).div(spec["divisor"])
    if spec["sparse_stride"] > 1:
        hidden = torch.where(index % spec["sparse_stride"] == 0, hidden, 0.0)
    return hidden.to(torch.bfloat16).contiguous()


def build_fixture(checkpoint_dir: Path, model_lock_path: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")
    config_path = checkpoint_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    if (
        raw_config["hidden_size"] != 2560
        or raw_config["num_experts"] != 512
        or raw_config["num_experts_per_tok"] != 10
        or raw_config.get("norm_topk_prob", True) is not True
    ):
        raise ValueError("unsupported Qwen router configuration")
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    case_specs = [
        (0, {"multiplier": 37, "add": 11, "modulus": 257, "center": 128, "divisor": 128, "sparse_stride": 1}),
        (1, {"multiplier": 71, "add": 19, "modulus": 251, "center": 125, "divisor": 128, "sparse_stride": 1}),
        (47, {"multiplier": 97, "add": 7, "modulus": 241, "center": 120, "divisor": 64, "sparse_stride": 17}),
    ]
    cases = []
    for layer, input_spec in case_specs:
        tensor = f"model.language_model.layers.{layer}.mlp.gate.weight"
        shard = weight_map[tensor]
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(tensor).contiguous()
        if weight.dtype != torch.bfloat16 or list(weight.shape) != [512, 2560]:
            raise ValueError(f"unsupported router tensor {tensor}")
        hidden = make_hidden(2560, input_spec)
        logits = torch.nn.functional.linear(hidden, weight).contiguous()
        probabilities = torch.softmax(logits, dtype=torch.float32, dim=-1)
        top_values, top_indices = torch.topk(probabilities, 10, dim=-1)
        top_values = (top_values / top_values.sum()).to(logits.dtype).contiguous()
        locked = locked_file(lock, shard)
        cases.append(
            {
                "name": f"layer_{layer}_affine_mod",
                "layer": layer,
                "tensor": tensor,
                "shard": shard,
                "shard_bytes": locked["size"],
                "shard_sha256": locked["lfs_sha256"],
                "weight_payload_sha256": hashlib.sha256(tensor_bytes(weight)).hexdigest(),
                "input_spec": input_spec,
                "input_bf16_sha256": hashlib.sha256(tensor_bytes(hidden)).hexdigest(),
                "selected_experts": top_indices.tolist(),
                "selected_logits_bf16": logits[top_indices].tolist(),
                "normalized_scores_bf16": top_values.tolist(),
            }
        )
    return {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextTopKRouter",
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
        },
        "configuration": {
            "hidden_size": 2560,
            "num_experts": 512,
            "top_k": 10,
            "norm_topk_prob": True,
            "input_dtype": "BF16",
            "weight_dtype": "BF16",
            "router_logits_dtype": "BF16",
            "softmax_dtype": "F32",
        },
        "cases": cases,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.checkpoint_dir, args.model_lock)
    write_json(args.output, fixture)
    print(json.dumps({"output": os.fspath(args.output), "cases": len(fixture["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
