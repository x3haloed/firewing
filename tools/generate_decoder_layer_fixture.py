#!/usr/bin/env python3
"""Generate a two-step complete real layer-0 decoder fixture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import transformers
from safetensors import safe_open
from transformers import DynamicCache
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextGatedResidual,
)

if __package__:
    from tools.generate_attention_residual_fixture import (
        DELTANET_SHAPES,
        HYPER_LOCAL_TENSORS,
        HYPER_SHAPES,
        INPUT_SPECS,
        capture,
        explicit_step,
        load_tensor,
        make_hyper_input,
    )
    from tools.generate_mixture_fixture import execute_mixture
    from tools.generate_ngram_address_fixture import (
        checkpoint_revision,
        load_model_lock,
        locked_file,
        sha256_file,
    )
    from tools.generate_sparse_moe_block_fixture import shared_expert_forward
else:
    from generate_attention_residual_fixture import (  # type: ignore[no-redef]
        DELTANET_SHAPES,
        HYPER_LOCAL_TENSORS,
        HYPER_SHAPES,
        INPUT_SPECS,
        capture,
        explicit_step,
        load_tensor,
        make_hyper_input,
    )
    from generate_mixture_fixture import execute_mixture  # type: ignore[no-redef]
    from generate_ngram_address_fixture import (  # type: ignore[no-redef]
        checkpoint_revision,
        load_model_lock,
        locked_file,
        sha256_file,
    )
    from generate_sparse_moe_block_fixture import shared_expert_forward  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_layer0_complete_cached_decoder"
HIDDEN = 2560
HC_COUNT = 4


def load_dense_tensors(
    checkpoint_dir: Path,
    lock: dict[str, Any],
    weight_map: dict[str, str],
    names: dict[str, tuple[str, list[int]]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    values: dict[str, torch.Tensor] = {}
    records: dict[str, Any] = {}
    for key, (name, shape) in names.items():
        value, record = load_tensor(checkpoint_dir, lock, weight_map, name, shape)
        values[key] = value
        records[key] = record
    return values, records


def build_fixture(
    checkpoint_dir: Path,
    model_lock_path: Path,
    attention_fixture_path: Path,
    sparse_moe_fixture_path: Path,
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")
    config_path = checkpoint_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    if (
        raw_config["hidden_size"] != HIDDEN
        or raw_config["hc_count"] != HC_COUNT
        or raw_config["layer_types"][0] != "linear_attention"
        or 1 in raw_config["ple_layer_ids"]
        or raw_config["moe_intermediate_size"] != 640
        or raw_config["shared_expert_intermediate_size"] != 640
        or raw_config["num_experts"] != 512
        or raw_config["num_experts_per_tok"] != 10
        or raw_config["hidden_act"] != "silu"
        or raw_config.get("norm_topk_prob", True) is not True
    ):
        raise ValueError("unsupported layer-0 decoder configuration")
    config = Qwen4ExpTextConfig(**raw_config)
    attention_hyper = Qwen4ExpTextGatedResidual(config).to(torch.bfloat16).eval()
    mlp_hyper = Qwen4ExpTextGatedResidual(config).to(torch.bfloat16).eval()
    deltanet = Qwen4ExpTextGatedDeltaNet(config, 0).to(torch.bfloat16).eval()
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    attention_hyper_state: dict[str, torch.Tensor] = {}
    for key, local_name in HYPER_LOCAL_TENSORS.items():
        name = f"model.language_model.layers.0.attn_hyper_connection.{local_name}"
        value, _ = load_tensor(checkpoint_dir, lock, weight_map, name, HYPER_SHAPES[key])
        attention_hyper_state[local_name] = value
    attention_hyper.load_state_dict(attention_hyper_state, strict=True)

    deltanet_state: dict[str, torch.Tensor] = {}
    for local_name, shape in DELTANET_SHAPES.items():
        name = f"model.language_model.layers.0.linear_attn.{local_name}"
        value, _ = load_tensor(checkpoint_dir, lock, weight_map, name, shape)
        deltanet_state[local_name] = value
    deltanet.load_state_dict(deltanet_state, strict=True)

    mlp_names = {
        f"mlp_hyper_connection.{key}": (
            f"model.language_model.layers.0.mlp_hyper_connection.{local_name}",
            HYPER_SHAPES[key],
        )
        for key, local_name in HYPER_LOCAL_TENSORS.items()
    }
    shared_prefix = "model.language_model.layers.0.mlp.shared_expert"
    dense_names = {
        **mlp_names,
        "router": ("model.language_model.layers.0.mlp.gate.weight", [512, HIDDEN]),
        "shared_gate_weight": (f"{shared_prefix}.gate_proj.weight", [640, HIDDEN]),
        "shared_up_weight": (f"{shared_prefix}.up_proj.weight", [640, HIDDEN]),
        "shared_down_weight": (f"{shared_prefix}.down_proj.weight", [HIDDEN, 640]),
        "shared_expert_gate_weight": (
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            [1, HIDDEN],
        ),
    }
    dense, tensor_records = load_dense_tensors(
        checkpoint_dir, lock, weight_map, dense_names
    )
    mlp_hyper.load_state_dict(
        {
            local_name: dense[f"mlp_hyper_connection.{key}"]
            for key, local_name in HYPER_LOCAL_TENSORS.items()
        },
        strict=True,
    )

    expert_prefix = "model.language_model.layers.0.mlp.experts"
    gate_up_name = f"{expert_prefix}.gate_up_proj"
    down_name = f"{expert_prefix}.down_proj"
    gate_up_shard = weight_map[gate_up_name]
    down_shard = weight_map[down_name]
    expert_banks = {}
    for key, name, shard, shape in (
        ("gate_up", gate_up_name, gate_up_shard, [512, 1280, HIDDEN]),
        ("down", down_name, down_shard, [512, HIDDEN, 640]),
    ):
        record = locked_file(lock, shard)
        expert_banks[key] = {
            "tensor": name,
            "dtype": "BF16",
            "shape": shape,
            "shard": shard,
            "shard_bytes": record["size"],
            "shard_sha256": record["lfs_sha256"],
        }

    explicit_cache = DynamicCache(config=config)
    official_cache = DynamicCache(config=config)
    steps = []
    with safe_open(checkpoint_dir / gate_up_shard, framework="pt", device="cpu") as gate_up_file:
        with safe_open(checkpoint_dir / down_shard, framework="pt", device="cpu") as down_file:
            for ordinal, spec in enumerate(INPUT_SPECS):
                hyper_input = make_hyper_input(spec)
                with torch.no_grad():
                    mixed_attention, preserved, attention_injection = attention_hyper(hyper_input)
                    attention_output, delta_captures = explicit_step(
                        deltanet, mixed_attention, explicit_cache, cached=ordinal > 0
                    )
                    official_attention = deltanet(mixed_attention, cache_params=official_cache)
                    post_attention = (
                        preserved
                        + (attention_output.unsqueeze(-2) * attention_injection.unsqueeze(-1)).flatten(-2)
                    ).contiguous()
                    mlp_input, mlp_preserved, mlp_injection = mlp_hyper(post_attention)
                    router_logits = torch.nn.functional.linear(mlp_input, dense["router"]).contiguous()
                    probabilities = torch.softmax(router_logits, dtype=torch.float32, dim=-1)
                    top_values, top_indices = torch.topk(probabilities, 10, dim=-1)
                    top_values = (top_values / top_values.sum(dim=-1, keepdim=True)).to(
                        router_logits.dtype
                    ).contiguous()
                if (
                    not torch.equal(preserved, hyper_input)
                    or not torch.equal(mlp_preserved, post_attention)
                    or not torch.equal(attention_output, official_attention)
                    or not torch.equal(
                        delta_captures["convolution_state"], official_cache.layers[0].conv_states[0]
                    )
                    or not torch.equal(
                        delta_captures["recurrent_state"], official_cache.layers[0].recurrent_states[0]
                    )
                ):
                    raise ValueError(f"official component disagreement at step {ordinal}")

                selection = top_indices.reshape(-1).tolist()
                scores = top_values.reshape(-1).tolist()
                experts, routed = execute_mixture(
                    mlp_input[0, 0], selection, scores, gate_up_file, down_file, gate_up_name, down_name
                )
                shared = shared_expert_forward(
                    mlp_input[0, 0],
                    dense["shared_gate_weight"],
                    dense["shared_up_weight"],
                    dense["shared_down_weight"],
                    dense["shared_expert_gate_weight"],
                    routed,
                )
                moe_output = shared["combined"].reshape(1, 1, HIDDEN).contiguous()
                injection_products = (
                    moe_output.unsqueeze(-2) * mlp_injection.unsqueeze(-1)
                ).contiguous()
                layer_output = (post_attention + injection_products.flatten(-2)).contiguous()
                captures = {
                    "post_attention": post_attention,
                    "mlp_input": mlp_input,
                    "mlp_injection_weights": mlp_injection,
                    "router_logits": router_logits,
                    "selected_scores": top_values,
                    "routed_mixture": routed,
                    "shared_gate": shared["shared_gate"],
                    "shared_up": shared["shared_up"],
                    "shared_swiglu": shared["shared_swiglu"],
                    "shared_down": shared["shared_down"],
                    "shared_gate_logit": shared["shared_gate_logit"],
                    "shared_gate_sigmoid": shared["shared_gate_sigmoid"],
                    "gated_shared": shared["gated_shared"],
                    "moe_output": moe_output,
                    "mlp_injection_products": injection_products,
                    "layer_output": layer_output,
                }
                steps.append(
                    {
                        "ordinal": ordinal,
                        "mode": "initial_chunk" if ordinal == 0 else "cached_recurrent",
                        "selected_experts": selection,
                        "expert_execution_order": sorted(selection),
                        "experts": experts,
                        "captures": {name: capture(value) for name, value in captures.items()},
                    }
                )

    return {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "source_derived_huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward",
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
            "attention_fixture_sha256": sha256_file(attention_fixture_path),
            "sparse_moe_fixture_sha256": sha256_file(sparse_moe_fixture_path),
        },
        "configuration": {
            "layer": 0,
            "layer_type": "linear_attention",
            "ple_applied": False,
            "hidden_size": HIDDEN,
            "hc_count": HC_COUNT,
            "num_experts": 512,
            "top_k": 10,
            "intermediate_size": 640,
            "shared_intermediate_size": 640,
            "boundary_dtype": "BF16",
            "router_softmax_dtype": "F32",
        },
        "case": {
            "name": "layer_0_two_token_complete_decoder",
            "tensors": tensor_records,
            "expert_banks": expert_banks,
            "steps": steps,
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
    parser.add_argument("--attention-fixture", required=True, type=Path)
    parser.add_argument("--sparse-moe-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(
        args.checkpoint_dir,
        args.model_lock,
        args.attention_fixture,
        args.sparse_moe_fixture,
    )
    write_json(args.output, fixture)
    print(
        json.dumps(
            {
                "output": os.fspath(args.output),
                "dense_tensors": len(fixture["case"]["tensors"]),
                "steps": len(fixture["case"]["steps"]),
                "captures_per_step": len(fixture["case"]["steps"][0]["captures"]),
                "selected_experts": [step["selected_experts"] for step in fixture["case"]["steps"]],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
