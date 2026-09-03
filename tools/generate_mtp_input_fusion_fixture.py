#!/usr/bin/env python3
"""Generate the real Qwen4-Exp MTP input-fusion fixture without payload bytes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

if __package__:
    from tools.generate_expert_fixture import capture_hash
    from tools.generate_ngram_address_fixture import (
        checkpoint_revision,
        load_model_lock,
        locked_file,
        sha256_file,
    )
else:
    from generate_expert_fixture import capture_hash  # type: ignore[no-redef]
    from generate_ngram_address_fixture import (  # type: ignore[no-redef]
        checkpoint_revision,
        load_model_lock,
        locked_file,
        sha256_file,
    )

MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_real_mtp_input_fusion"
SGLANG_COMMIT = "78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385"
HIDDEN = 2560
HC_COUNT = 4
HC_HIDDEN = HIDDEN * HC_COUNT
INPUT_SPECS = {
    "embedding": {"multiplier": 29, "add": 7, "modulus": 251, "center": 125, "divisor": 128},
    "target_hidden": {"multiplier": 43, "add": 17, "modulus": 263, "center": 131, "divisor": 128},
}
SEQUENCE_INPUT_SPECS = {
    "embedding": {"multiplier": 37, "add": 19, "modulus": 269, "center": 134, "divisor": 128},
    "target_hidden": {"multiplier": 47, "add": 23, "modulus": 271, "center": 135, "divisor": 128},
}
TENSORS = {
    "pre_fc_norm_embedding": ("mtp.pre_fc_norm_embedding.weight", [HIDDEN]),
    "pre_fc_norm_hidden": ("mtp.pre_fc_norm_hidden.weight", [HC_HIDDEN]),
    "fc_embedding": ("mtp.fc_embedding.weight", [HIDDEN, HIDDEN]),
    "fc_hidden": ("mtp.fc_hidden.weight", [HIDDEN, HIDDEN]),
}


def make_input(size: int, spec: dict[str, int]) -> torch.Tensor:
    index = torch.arange(size, dtype=torch.int64)
    value = ((index * spec["multiplier"] + spec["add"]) % spec["modulus"] - spec["center"])
    return value.to(torch.float32).div(spec["divisor"]).to(torch.bfloat16).contiguous()


def zero_centered_rms_norm(value: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    value_f32 = value.float()
    variance = value_f32.pow(2).mean(dim=-1, keepdim=True)
    return (value_f32 * torch.rsqrt(variance + epsilon) * (1.0 + weight.float())).to(torch.bfloat16).contiguous()


def build_fixture(checkpoint_dir: Path, model_lock_path: Path, source_lock_path: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")

    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if source_lock.get("commit") != SGLANG_COMMIT:
        raise ValueError("MTP source lock does not identify the supported SGLang commit")
    source_files = {item["path"]: item for item in source_lock.get("files", [])}
    mtp_source = source_files.get("python/sglang/srt/models/qwen4_exp_mtp.py")
    if mtp_source is None or mtp_source.get("sha256") != "2b2ec09230875279a75ae651a1d9e1d88999bc89748e9d0cb6b4a768ffc0e54e":
        raise ValueError("MTP source lock has an unsupported Qwen4-Exp implementation")

    config_path = checkpoint_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    if (
        raw_config.get("hidden_size") != HIDDEN
        or raw_config.get("hc_count") != HC_COUNT
        or raw_config.get("rms_norm_eps") != 1e-6
        or raw_config.get("mtp_num_hidden_layers") != 1
        or raw_config.get("mtp_use_dedicated_embeddings") is not False
        or raw_config.get("mtp", {}).get("layer_types") != ["full_attention"]
    ):
        raise ValueError("unsupported Qwen4-Exp MTP configuration")

    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    values: dict[str, torch.Tensor] = {}
    records = {}
    for key, (tensor_name, shape) in TENSORS.items():
        shard = weight_map.get(tensor_name)
        if not isinstance(shard, str):
            raise ValueError(f"tensor index is missing {tensor_name}")
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as source:
            value = source.get_tensor(tensor_name).contiguous()
        if value.dtype != torch.bfloat16 or list(value.shape) != shape:
            raise ValueError(f"unsupported tensor {tensor_name}")
        values[key] = value
        locked = locked_file(lock, shard)
        records[key] = {
            "tensor": tensor_name,
            "shape": shape,
            "shard": shard,
            "shard_bytes": locked["size"],
            "shard_sha256": locked["lfs_sha256"],
            "payload_sha256": capture_hash(value),
        }

    embedding = make_input(HIDDEN, INPUT_SPECS["embedding"])
    target_hidden = make_input(HC_HIDDEN, INPUT_SPECS["target_hidden"])
    embedding_normed = zero_centered_rms_norm(embedding, values["pre_fc_norm_embedding"], 1e-6)
    target_hidden_normed = zero_centered_rms_norm(target_hidden, values["pre_fc_norm_hidden"], 1e-6)
    embedding_projected = torch.nn.functional.linear(embedding_normed, values["fc_embedding"]).contiguous()
    target_hidden_view = target_hidden_normed.view(HC_COUNT, HIDDEN)
    target_hidden_projected = torch.nn.functional.linear(target_hidden_view, values["fc_hidden"]).contiguous()
    fused = (embedding_projected.unsqueeze(0) + target_hidden_projected).contiguous().view(HC_HIDDEN)
    captures = {
        "embedding": embedding,
        "target_hidden": target_hidden,
        "embedding_normed": embedding_normed,
        "target_hidden_normed": target_hidden_normed,
        "embedding_projected": embedding_projected,
        "target_hidden_projected": target_hidden_projected,
        "fused_hidden": fused,
    }
    if any(value.dtype != torch.bfloat16 or not value.is_contiguous() for value in captures.values()):
        raise ValueError("MTP input fusion did not preserve BF16 boundaries")

    sequence_embedding = make_input(HIDDEN, SEQUENCE_INPUT_SPECS["embedding"])
    sequence_hidden = make_input(HC_HIDDEN, SEQUENCE_INPUT_SPECS["target_hidden"])
    sequence_embedding_normed = zero_centered_rms_norm(
        sequence_embedding, values["pre_fc_norm_embedding"], 1e-6
    )
    sequence_hidden_normed = zero_centered_rms_norm(
        sequence_hidden, values["pre_fc_norm_hidden"], 1e-6
    )
    sequence_embedding_projected = torch.nn.functional.linear(
        sequence_embedding_normed, values["fc_embedding"]
    ).contiguous()
    sequence_hidden_projected = torch.nn.functional.linear(
        sequence_hidden_normed.view(HC_COUNT, HIDDEN), values["fc_hidden"]
    ).contiguous()
    sequence_fused = (
        sequence_embedding_projected.unsqueeze(0) + sequence_hidden_projected
    ).contiguous().view(HC_HIDDEN)
    sequence_captures = {
        "embedding": sequence_embedding,
        "target_hidden": sequence_hidden,
        "embedding_normed": sequence_embedding_normed,
        "target_hidden_normed": sequence_hidden_normed,
        "embedding_projected": sequence_embedding_projected,
        "target_hidden_projected": sequence_hidden_projected,
        "fused_hidden": sequence_fused,
    }

    return {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "sglang_qwen4_exp_mtp_source_derived",
            "commit": SGLANG_COMMIT,
            "source": "python/sglang/srt/models/qwen4_exp_mtp.py:Qwen4ExpForCausalLMMTP._fuse_residual_linear_shared",
            "source_sha256": mtp_source["sha256"],
            "source_lock_sha256": sha256_file(source_lock_path),
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
        },
        "configuration": {
            "hidden_size": HIDDEN,
            "hc_count": HC_COUNT,
            "target_hidden_size": HC_HIDDEN,
            "rms_norm_eps": 1e-6,
            "boundary_dtype": "BF16",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "mtp_layer_types": ["full_attention"],
        },
        "case": {
            "name": "real_mtp_four_stream_input_fusion",
            "input_specs": INPUT_SPECS,
            "tensors": records,
            "expected_bf16_sha256": {name: capture_hash(value) for name, value in captures.items()},
        },
        "sequence_case": {
            "name": "real_mtp_second_position_input_fusion",
            "input_specs": SEQUENCE_INPUT_SPECS,
            "expected_bf16_sha256": {
                name: capture_hash(value) for name, value in sequence_captures.items()
            },
        },
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
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.checkpoint_dir, args.model_lock, args.source_lock)
    write_json(args.output, fixture)
    print(json.dumps({"output": os.fspath(args.output), "captures": len(fixture["case"]["expected_bf16_sha256"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
