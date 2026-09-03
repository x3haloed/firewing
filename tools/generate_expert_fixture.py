#!/usr/bin/env python3
"""Generate a real-checkpoint Qwen routed-expert fixture without weight bytes."""

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
    from tools.generate_router_fixture import make_hidden, tensor_bytes
else:
    from generate_ngram_address_fixture import (  # type: ignore[no-redef]
        checkpoint_revision,
        load_model_lock,
        locked_file,
        sha256_file,
    )
    from generate_router_fixture import make_hidden, tensor_bytes  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_real_routed_expert"


def capture_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(value.contiguous())).hexdigest()


def expert_forward(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    route_weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    gate_up_output = torch.nn.functional.linear(hidden, gate_up).contiguous()
    gate, up = gate_up_output.chunk(2, dim=-1)
    gate = gate.contiguous()
    up = up.contiguous()
    swiglu = (torch.nn.functional.silu(gate) * up).contiguous()
    down_output = torch.nn.functional.linear(swiglu, down).contiguous()
    weighted_down = (down_output * route_weight).contiguous()
    outputs = {
        "gate_up": gate_up_output,
        "gate": gate,
        "up": up,
        "swiglu": swiglu,
        "down": down_output,
        "weighted_down": weighted_down,
    }
    if any(value.dtype != torch.bfloat16 for value in outputs.values()):
        raise ValueError("official expert path did not preserve BF16 boundaries")
    return outputs


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
        or raw_config["moe_intermediate_size"] != 640
        or raw_config["num_experts"] != 512
        or raw_config["num_experts_per_tok"] != 10
        or raw_config["hidden_act"] != "silu"
        or raw_config.get("norm_topk_prob", True) is not True
    ):
        raise ValueError("unsupported Qwen expert configuration")

    router_fixture_path = Path(__file__).parents[1] / "fixtures/router/qwen3_8_flash_next_real.json"
    router_fixture = json.loads(router_fixture_path.read_text(encoding="utf-8"))
    router_case = router_fixture["cases"][0]
    if router_fixture.get("revision") != revision or router_case.get("layer") != 0:
        raise ValueError("router fixture is not the pinned layer-0 authority")
    expert = router_case["selected_experts"][0]
    route_weight = torch.tensor(
        router_case["normalized_scores_bf16"][0], dtype=torch.bfloat16
    )
    hidden = make_hidden(2560, router_case["input_spec"])

    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    prefix = "model.language_model.layers.0.mlp.experts"
    gate_up_name = f"{prefix}.gate_up_proj"
    down_name = f"{prefix}.down_proj"
    gate_up_shard = weight_map[gate_up_name]
    down_shard = weight_map[down_name]
    with safe_open(checkpoint_dir / gate_up_shard, framework="pt", device="cpu") as handle:
        gate_up = handle.get_slice(gate_up_name)[expert].contiguous()
    with safe_open(checkpoint_dir / down_shard, framework="pt", device="cpu") as handle:
        down = handle.get_slice(down_name)[expert].contiguous()
    if gate_up.dtype != torch.bfloat16 or list(gate_up.shape) != [1280, 2560]:
        raise ValueError("unsupported gate/up expert slice")
    if down.dtype != torch.bfloat16 or list(down.shape) != [2560, 640]:
        raise ValueError("unsupported down expert slice")

    outputs = expert_forward(hidden, gate_up, down, route_weight)
    gate_up_lock = locked_file(lock, gate_up_shard)
    down_lock = locked_file(lock, down_shard)
    return {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextExperts.forward",
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
            "router_fixture_sha256": sha256_file(router_fixture_path),
        },
        "configuration": {
            "hidden_size": 2560,
            "intermediate_size": 640,
            "num_experts": 512,
            "top_k": 10,
            "activation": "silu",
            "input_dtype": "BF16",
            "weight_dtype": "BF16",
            "boundary_dtype": "BF16",
        },
        "case": {
            "name": "layer_0_top_1_expert",
            "layer": 0,
            "expert": expert,
            "route_weight_bf16": route_weight.item(),
            "input_spec": router_case["input_spec"],
            "input_bf16_sha256": capture_hash(hidden),
            "gate_up": {
                "tensor": gate_up_name,
                "shard": gate_up_shard,
                "shard_bytes": gate_up_lock["size"],
                "shard_sha256": gate_up_lock["lfs_sha256"],
                "expert_payload_sha256": capture_hash(gate_up),
            },
            "down": {
                "tensor": down_name,
                "shard": down_shard,
                "shard_bytes": down_lock["size"],
                "shard_sha256": down_lock["lfs_sha256"],
                "expert_payload_sha256": capture_hash(down),
            },
            "expected_bf16_sha256": {
                name: capture_hash(value) for name, value in outputs.items()
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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.checkpoint_dir, args.model_lock)
    write_json(args.output, fixture)
    print(json.dumps({"output": os.fspath(args.output), "expert": fixture["case"]["expert"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
