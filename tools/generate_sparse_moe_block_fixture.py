#!/usr/bin/env python3
"""Generate a real Qwen sparse-MoE block fixture without payload bytes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import transformers
from safetensors import safe_open

if __package__:
    from tools.generate_expert_fixture import capture_hash
    from tools.generate_mixture_fixture import execute_mixture
    from tools.generate_ngram_address_fixture import (
        checkpoint_revision,
        load_model_lock,
        locked_file,
        sha256_file,
    )
    from tools.generate_router_fixture import make_hidden
else:
    from generate_expert_fixture import capture_hash  # type: ignore[no-redef]
    from generate_mixture_fixture import execute_mixture  # type: ignore[no-redef]
    from generate_ngram_address_fixture import (  # type: ignore[no-redef]
        checkpoint_revision,
        load_model_lock,
        locked_file,
        sha256_file,
    )
    from generate_router_fixture import make_hidden  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_real_sparse_moe_block"


def shared_expert_forward(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    routed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    gate = torch.nn.functional.linear(hidden, gate_weight).contiguous()
    up = torch.nn.functional.linear(hidden, up_weight).contiguous()
    swiglu = (torch.nn.functional.silu(gate) * up).contiguous()
    down = torch.nn.functional.linear(swiglu, down_weight).contiguous()
    shared_gate_logit = torch.nn.functional.linear(hidden, shared_gate_weight).contiguous()
    shared_gate_sigmoid = torch.nn.functional.sigmoid(shared_gate_logit).contiguous()
    gated_shared = (shared_gate_sigmoid * down).contiguous()
    combined = (routed + gated_shared).contiguous()
    outputs = {
        "shared_gate": gate,
        "shared_up": up,
        "shared_swiglu": swiglu,
        "shared_down": down,
        "shared_gate_logit": shared_gate_logit,
        "shared_gate_sigmoid": shared_gate_sigmoid,
        "gated_shared": gated_shared,
        "routed_mixture": routed,
        "combined": combined,
    }
    if any(value.dtype != torch.bfloat16 for value in outputs.values()):
        raise ValueError("sparse-MoE block did not preserve BF16 boundaries")
    return outputs


def build_fixture(checkpoint_dir: Path, model_lock_path: Path) -> dict[str, Any]:
    root = Path(__file__).parents[1]
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")
    config_path = checkpoint_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    if (
        config["hidden_size"] != 2560
        or config["moe_intermediate_size"] != 640
        or config["shared_expert_intermediate_size"] != 640
        or config["num_experts"] != 512
        or config["num_experts_per_tok"] != 10
        or config["hidden_act"] != "silu"
    ):
        raise ValueError("unsupported Qwen sparse-MoE block configuration")

    router_path = root / "fixtures/router/qwen3_8_flash_next_real.json"
    expert_path = root / "fixtures/expert/qwen3_8_flash_next_real.json"
    mixture_path = root / "fixtures/mixture/qwen3_8_flash_next_real.json"
    router = json.loads(router_path.read_text(encoding="utf-8"))
    router_case = router["cases"][0]
    mixture_authority = json.loads(mixture_path.read_text(encoding="utf-8"))
    if (
        router.get("revision") != revision
        or router_case.get("layer") != 0
        or mixture_authority.get("revision") != revision
    ):
        raise ValueError("prior MoE fixtures do not match the pinned revision")
    hidden = make_hidden(2560, router_case["input_spec"])

    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    routed_prefix = "model.language_model.layers.0.mlp.experts"
    gate_up_name = f"{routed_prefix}.gate_up_proj"
    routed_down_name = f"{routed_prefix}.down_proj"
    shared_prefix = "model.language_model.layers.0.mlp.shared_expert"
    names = {
        "shared_gate_weight": f"{shared_prefix}.gate_proj.weight",
        "shared_up_weight": f"{shared_prefix}.up_proj.weight",
        "shared_down_weight": f"{shared_prefix}.down_proj.weight",
        "shared_expert_gate_weight": "model.language_model.layers.0.mlp.shared_expert_gate.weight",
    }
    gate_up_shard = weight_map[gate_up_name]
    common_shards = {weight_map[routed_down_name], *(weight_map[name] for name in names.values())}
    if len(common_shards) != 1:
        raise ValueError("unsupported layer-0 shared tensor sharding")
    common_shard = common_shards.pop()
    common_lock = locked_file(lock, common_shard)

    with safe_open(checkpoint_dir / gate_up_shard, framework="pt", device="cpu") as gate_up_file:
        with safe_open(checkpoint_dir / common_shard, framework="pt", device="cpu") as common_file:
            _, routed = execute_mixture(
                hidden,
                router_case["selected_experts"],
                router_case["normalized_scores_bf16"],
                gate_up_file,
                common_file,
                gate_up_name,
                routed_down_name,
            )
            tensors = {key: common_file.get_tensor(name).contiguous() for key, name in names.items()}

    expected_shapes = {
        "shared_gate_weight": [640, 2560],
        "shared_up_weight": [640, 2560],
        "shared_down_weight": [2560, 640],
        "shared_expert_gate_weight": [1, 2560],
    }
    if any(
        value.dtype != torch.bfloat16 or list(value.shape) != expected_shapes[key]
        for key, value in tensors.items()
    ):
        raise ValueError("unsupported shared expert tensor shape or dtype")
    outputs = shared_expert_forward(
        hidden,
        tensors["shared_gate_weight"],
        tensors["shared_up_weight"],
        tensors["shared_down_weight"],
        tensors["shared_expert_gate_weight"],
        routed,
    )
    if capture_hash(routed) != mixture_authority["case"]["mixture_bf16_sha256"]:
        raise ValueError("recomputed routed mixture disagrees with FW-0011")

    tensor_records = {
        key: {
            "tensor": names[key],
            "shape": expected_shapes[key],
            "payload_sha256": capture_hash(value),
        }
        for key, value in tensors.items()
    }
    return {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextSparseMoeBlock.forward",
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
            "router_fixture_sha256": sha256_file(router_path),
            "expert_fixture_sha256": sha256_file(expert_path),
            "mixture_fixture_sha256": sha256_file(mixture_path),
        },
        "configuration": {
            "hidden_size": 2560,
            "intermediate_size": 640,
            "shared_intermediate_size": 640,
            "num_experts": 512,
            "top_k": 10,
            "activation": "silu",
            "boundary_dtype": "BF16",
        },
        "case": {
            "name": "layer_0_affine_mod_sparse_moe_block",
            "layer": 0,
            "input_spec": router_case["input_spec"],
            "input_bf16_sha256": capture_hash(hidden),
            "common_shard": common_shard,
            "common_shard_bytes": common_lock["size"],
            "common_shard_sha256": common_lock["lfs_sha256"],
            "tensors": tensor_records,
            "expected_bf16_sha256": {
                key: capture_hash(value) for key, value in outputs.items()
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
    print(json.dumps({"output": os.fspath(args.output), "captures": len(fixture["case"]["expected_bf16_sha256"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
