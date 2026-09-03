#!/usr/bin/env python3
"""Generate a complete real layer-3 full-attention decoder fixture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import torch
import transformers
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

if __package__:
    from tools.generate_attention_residual_fixture import HYPER_LOCAL_TENSORS, HYPER_SHAPES, capture, load_tensor
    from tools.generate_decoder_layer_fixture import load_dense_tensors
    from tools.generate_full_attention_residual_fixture import build_fixture as build_attention_residual
    from tools.generate_mixture_fixture import execute_mixture
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, locked_file, sha256_file
    from tools.generate_sparse_moe_block_fixture import shared_expert_forward
else:
    from generate_attention_residual_fixture import HYPER_LOCAL_TENSORS, HYPER_SHAPES, capture, load_tensor  # type: ignore[no-redef]
    from generate_decoder_layer_fixture import load_dense_tensors  # type: ignore[no-redef]
    from generate_full_attention_residual_fixture import build_fixture as build_attention_residual  # type: ignore[no-redef]
    from generate_mixture_fixture import execute_mixture  # type: ignore[no-redef]
    from generate_ngram_address_fixture import checkpoint_revision, load_model_lock, locked_file, sha256_file  # type: ignore[no-redef]
    from generate_sparse_moe_block_fixture import shared_expert_forward  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_layer3_complete_decoder"
LAYER = 3
HIDDEN = 2560
HC_COUNT = 4
EXPERTS = 512
TOP_K = 10
INTERMEDIATE = 640


def build_fixture(
    checkpoint_dir: Path,
    model_lock_path: Path,
    full_attention_fixture_path: Path,
    attention_residual_fixture_path: Path,
    *,
    _parent_execution: tuple[dict[str, Any], list[torch.Tensor]] | None = None,
    _parent_semantic: str = "qwen3_8_flash_next_layer3_full_attention_residual",
    _layer: int = LAYER,
    _layer_type: str = "full_attention",
    _semantic: str = SEMANTIC,
    _reference_hashes: dict[str, str] | None = None,
    _modes: tuple[str, ...] = ("initial", "active_qsa_pruning"),
    _require_committed_parent: bool = True,
    _layer_prefix: str | None = None,
    _mtp_config: bool = False,
    _return_outputs: bool = False,
    _mixture_observer: Callable[..., None] | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], list[torch.Tensor]]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    parent = (
        json.loads(attention_residual_fixture_path.read_text(encoding="utf-8"))
        if _require_committed_parent
        else None
    )
    if _parent_execution is None:
        generated, post_attention = build_attention_residual(
            checkpoint_dir,
            model_lock_path,
            full_attention_fixture_path,
            _return_outputs=True,
        )
    else:
        generated, post_attention = _parent_execution
    authority = parent if _require_committed_parent else generated
    if (
        revision != lock["revision"]
        or (_require_committed_parent and parent != generated)
        or authority is None
        or authority.get("revision") != revision
        or authority.get("semantic") != _parent_semantic
        or (_mtp_config and _layer != 0)
        or len(post_attention) != len(_modes)
    ):
        raise ValueError(f"layer-{_layer} decoder parent authority mismatch")
    config_path = checkpoint_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    effective_config = dict(raw_config)
    if _mtp_config:
        effective_config["num_hidden_layers"] = 1
        effective_config["layer_types"] = ["full_attention"]
        effective_config["full_attention_interval"] = 1
        effective_config["ple_layer_ids"] = []
    if (
        effective_config["layer_types"][_layer] != _layer_type
        or raw_config["hidden_size"] != HIDDEN
        or raw_config["hc_count"] != HC_COUNT
        or raw_config["num_experts"] != EXPERTS
        or raw_config["num_experts_per_tok"] != TOP_K
        or raw_config["moe_intermediate_size"] != INTERMEDIATE
        or raw_config["shared_expert_intermediate_size"] != INTERMEDIATE
        or raw_config["hidden_act"] != "silu"
        or raw_config.get("norm_topk_prob", True) is not True
    ):
        raise ValueError(f"unsupported layer-{_layer} decoder configuration")
    config = Qwen4ExpTextConfig(**effective_config)
    mlp_hyper = Qwen4ExpTextGatedResidual(config).to(torch.bfloat16).eval()
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    layer_prefix = _layer_prefix or f"model.language_model.layers.{_layer}"
    mlp_names = {
        f"mlp_hyper_connection.{key}": (
            f"{layer_prefix}.mlp_hyper_connection.{local_name}",
            HYPER_SHAPES[key],
        )
        for key, local_name in HYPER_LOCAL_TENSORS.items()
    }
    shared_prefix = f"{layer_prefix}.mlp.shared_expert"
    dense_names = {
        **mlp_names,
        "router": (f"{layer_prefix}.mlp.gate.weight", [EXPERTS, HIDDEN]),
        "shared_gate_weight": (f"{shared_prefix}.gate_proj.weight", [INTERMEDIATE, HIDDEN]),
        "shared_up_weight": (f"{shared_prefix}.up_proj.weight", [INTERMEDIATE, HIDDEN]),
        "shared_down_weight": (f"{shared_prefix}.down_proj.weight", [HIDDEN, INTERMEDIATE]),
        "shared_expert_gate_weight": (
            f"{layer_prefix}.mlp.shared_expert_gate.weight",
            [1, HIDDEN],
        ),
    }
    dense, tensor_records = load_dense_tensors(checkpoint_dir, lock, weight_map, dense_names)
    mlp_hyper.load_state_dict(
        {
            local_name: dense[f"mlp_hyper_connection.{key}"]
            for key, local_name in HYPER_LOCAL_TENSORS.items()
        },
        strict=True,
    )

    expert_prefix = f"{layer_prefix}.mlp.experts"
    gate_up_name = f"{expert_prefix}.gate_up_proj"
    down_name = f"{expert_prefix}.down_proj"
    expert_banks = {}
    for key, name, shape in (
        ("gate_up", gate_up_name, [EXPERTS, INTERMEDIATE * 2, HIDDEN]),
        ("down", down_name, [EXPERTS, HIDDEN, INTERMEDIATE]),
    ):
        shard = weight_map[name]
        record = locked_file(lock, shard)
        expert_banks[key] = {
            "tensor": name,
            "dtype": "BF16",
            "shape": shape,
            "shard": shard,
            "shard_bytes": record["size"],
            "shard_sha256": record["lfs_sha256"],
        }

    steps = []
    layer_outputs = []
    with safe_open(checkpoint_dir / weight_map[gate_up_name], framework="pt", device="cpu") as gate_up_file:
        with safe_open(checkpoint_dir / weight_map[down_name], framework="pt", device="cpu") as down_file:
            for ordinal, post in enumerate(post_attention):
                with torch.no_grad():
                    mlp_input, preserved, injection_weights = mlp_hyper(post)
                    router_logits = torch.nn.functional.linear(mlp_input, dense["router"]).contiguous()
                    probabilities = torch.softmax(router_logits, dtype=torch.float32, dim=-1)
                    top_values, top_indices = torch.topk(probabilities, TOP_K, dim=-1)
                    top_values = (top_values / top_values.sum(dim=-1, keepdim=True)).to(
                        router_logits.dtype
                    ).contiguous()
                if not torch.equal(preserved, post):
                    raise ValueError(f"layer-{_layer} MLP hyper connection mutated input at case {ordinal}")
                selection = top_indices.reshape(-1).tolist()
                scores = top_values.reshape(-1).tolist()
                experts, routed = execute_mixture(
                    mlp_input[0, 0], selection, scores, gate_up_file, down_file, gate_up_name, down_name
                )
                if _mixture_observer is not None:
                    _mixture_observer(
                        layer=_layer,
                        ordinal=ordinal,
                        hidden=mlp_input[0, 0],
                        selection=selection,
                        scores=scores,
                        gate_up_file=gate_up_file,
                        down_file=down_file,
                        gate_up_name=gate_up_name,
                        down_name=down_name,
                        reference_mixture=routed,
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
                    moe_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
                ).contiguous()
                layer_output = (preserved + injection_products.flatten(-2)).contiguous()
                layer_outputs.append(layer_output)
                captures = {
                    "post_attention": post,
                    "mlp_input": mlp_input,
                    "mlp_injection_weights": injection_weights,
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
                        "mode": _modes[ordinal],
                        "selected_experts": selection,
                        "expert_execution_order": sorted(selection),
                        "experts": experts,
                        "captures": {name: capture(value) for name, value in captures.items()},
                    }
                )

    fixture = {
        "schema_version": 1,
        "semantic": _semantic,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "source_derived_huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward",
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
            **(
                _reference_hashes
                if _reference_hashes is not None
                else {
                    "full_attention_fixture_sha256": sha256_file(full_attention_fixture_path),
                    "attention_residual_fixture_sha256": sha256_file(attention_residual_fixture_path),
                }
            ),
        },
        "configuration": {
            "layer": _layer,
            "layer_type": _layer_type,
            "hidden_size": HIDDEN,
            "hc_count": HC_COUNT,
            "num_experts": EXPERTS,
            "top_k": TOP_K,
            "intermediate_size": INTERMEDIATE,
            "shared_intermediate_size": INTERMEDIATE,
            "boundary_dtype": "BF16",
            "router_softmax_dtype": "F32",
        },
        "tensors": tensor_records,
        "expert_banks": expert_banks,
        "steps": steps,
    }
    if _return_outputs:
        return fixture, layer_outputs
    return fixture


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
    parser.add_argument("--full-attention-fixture", required=True, type=Path)
    parser.add_argument("--attention-residual-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(
        args.checkpoint_dir,
        args.model_lock,
        args.full_attention_fixture,
        args.attention_residual_fixture,
    )
    write_json(args.output, fixture)
    print(json.dumps({"output": os.fspath(args.output), "tensors": len(fixture["tensors"]), "steps": len(fixture["steps"]), "selected_experts": [step["selected_experts"] for step in fixture["steps"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
