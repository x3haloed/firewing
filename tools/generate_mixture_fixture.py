#!/usr/bin/env python3
"""Generate a real Qwen top-10 expert-mixture fixture without payload bytes."""

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
    from tools.generate_expert_fixture import capture_hash, expert_forward
    from tools.generate_ngram_address_fixture import (
        checkpoint_revision,
        load_model_lock,
        locked_file,
        sha256_file,
    )
    from tools.generate_router_fixture import make_hidden, tensor_bytes
else:
    from generate_expert_fixture import capture_hash, expert_forward  # type: ignore[no-redef]
    from generate_ngram_address_fixture import (  # type: ignore[no-redef]
        checkpoint_revision,
        load_model_lock,
        locked_file,
        sha256_file,
    )
    from generate_router_fixture import make_hidden, tensor_bytes  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_real_top10_expert_mixture"


def accumulate_bf16_in_expert_order(weighted: list[torch.Tensor]) -> torch.Tensor:
    if not weighted or any(value.dtype != torch.bfloat16 for value in weighted):
        raise ValueError("mixture contributions must be nonempty BF16 tensors")
    output = torch.zeros((1, weighted[0].numel()), dtype=torch.bfloat16)
    token_index = torch.tensor([0], dtype=torch.int64)
    for value in weighted:
        output.index_add_(0, token_index, value.reshape(1, -1))
    return output[0].contiguous()


def execute_mixture(
    hidden: torch.Tensor,
    selection: list[int],
    scores: list[float],
    gate_up_file: Any,
    down_file: Any,
    gate_up_name: str,
    down_name: str,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    score_by_expert = dict(zip(selection, scores, strict=True))
    entries = []
    contributions = []
    for expert in sorted(selection):
        gate_up = gate_up_file.get_slice(gate_up_name)[expert].contiguous()
        down = down_file.get_slice(down_name)[expert].contiguous()
        route_weight = torch.tensor(score_by_expert[expert], dtype=torch.bfloat16)
        outputs = expert_forward(hidden, gate_up, down, route_weight)
        contributions.append(outputs["weighted_down"])
        entries.append(
            {
                "expert": expert,
                "route_weight_bf16": route_weight.item(),
                "gate_up_payload_sha256": hashlib.sha256(tensor_bytes(gate_up)).hexdigest(),
                "down_payload_sha256": hashlib.sha256(tensor_bytes(down)).hexdigest(),
                "weighted_down_bf16_sha256": capture_hash(outputs["weighted_down"]),
            }
        )
    mixture = accumulate_bf16_in_expert_order(contributions)
    manual = torch.zeros_like(mixture)
    for contribution in contributions:
        manual = (manual + contribution).to(torch.bfloat16)
    if not torch.equal(mixture, manual):
        raise ValueError("index_add and explicit BF16 accumulation disagree")
    return entries, mixture


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
        or config["num_experts"] != 512
        or config["num_experts_per_tok"] != 10
        or config["hidden_act"] != "silu"
    ):
        raise ValueError("unsupported Qwen mixture configuration")

    router_path = root / "fixtures/router/qwen3_8_flash_next_real.json"
    expert_path = root / "fixtures/expert/qwen3_8_flash_next_real.json"
    router = json.loads(router_path.read_text(encoding="utf-8"))
    router_case = router["cases"][0]
    if router.get("revision") != revision or router_case.get("layer") != 0:
        raise ValueError("router fixture is not the pinned layer-0 authority")
    selection = router_case["selected_experts"]
    scores = router_case["normalized_scores_bf16"]
    if len(selection) != 10 or len(set(selection)) != 10 or len(scores) != 10:
        raise ValueError("router fixture top-10 shape mismatch")
    execution_order = sorted(selection)
    hidden = make_hidden(2560, router_case["input_spec"])

    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    prefix = "model.language_model.layers.0.mlp.experts"
    gate_up_name = f"{prefix}.gate_up_proj"
    down_name = f"{prefix}.down_proj"
    gate_up_shard = weight_map[gate_up_name]
    down_shard = weight_map[down_name]
    gate_up_lock = locked_file(lock, gate_up_shard)
    down_lock = locked_file(lock, down_shard)

    with safe_open(checkpoint_dir / gate_up_shard, framework="pt", device="cpu") as gate_up_file:
        with safe_open(checkpoint_dir / down_shard, framework="pt", device="cpu") as down_file:
            entries, mixture = execute_mixture(
                hidden, selection, scores, gate_up_file, down_file, gate_up_name, down_name
            )

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
            "router_fixture_sha256": sha256_file(router_path),
            "expert_fixture_sha256": sha256_file(expert_path),
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
            "mixture_accumulator_dtype": "BF16",
        },
        "case": {
            "name": "layer_0_affine_mod_top10",
            "layer": 0,
            "input_spec": router_case["input_spec"],
            "input_bf16_sha256": capture_hash(hidden),
            "top_k_selection_order": selection,
            "expert_execution_order": execution_order,
            "gate_up": {
                "tensor": gate_up_name,
                "shard": gate_up_shard,
                "shard_bytes": gate_up_lock["size"],
                "shard_sha256": gate_up_lock["lfs_sha256"],
            },
            "down": {
                "tensor": down_name,
                "shard": down_shard,
                "shard_bytes": down_lock["size"],
                "shard_sha256": down_lock["lfs_sha256"],
            },
            "experts": entries,
            "mixture_bf16_sha256": capture_hash(mixture),
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
    print(json.dumps({"output": os.fspath(args.output), "experts": len(fixture["case"]["experts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
