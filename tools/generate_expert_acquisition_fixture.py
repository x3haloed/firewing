#!/usr/bin/env python3
"""Generate an all-layer selected-expert acquisition fixture without payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

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
SEMANTIC = "qwen3_8_flash_next_all_layer_selected_expert_acquisition"
LAYERS = 48
EXPERTS = 512
TOP_K = 10
HIDDEN = 2560
INTERMEDIATE = 640
GATE_UP_BYTES = 2 * INTERMEDIATE * HIDDEN * 2
DOWN_BYTES = HIDDEN * INTERMEDIATE * 2
TRACE_BYTES = LAYERS * TOP_K * (GATE_UP_BYTES + DOWN_BYTES)


def input_spec(layer: int) -> dict[str, int]:
    if not 0 <= layer < LAYERS:
        raise ValueError("layer is out of range")
    return {
        "multiplier": 37 + 2 * layer,
        "add": 11 + 13 * layer,
        "modulus": 257,
        "center": 128,
        "divisor": 128,
        "sparse_stride": 1,
    }


def tensor_layout(path: Path, tensor: str) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors prefix")
        header_bytes = int.from_bytes(raw, "little")
        if not 0 < header_bytes <= 16 * 1024 * 1024:
            raise ValueError(f"{path}: invalid safetensors header length")
        header = json.loads(handle.read(header_bytes))
    item = header.get(tensor)
    if not isinstance(item, dict) or item.get("dtype") != "BF16":
        raise ValueError(f"{tensor}: missing or unsupported tensor")
    offsets = item.get("data_offsets")
    shape = item.get("shape")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) for value in offsets)
        or not isinstance(shape, list)
        or not all(isinstance(value, int) for value in shape)
    ):
        raise ValueError(f"{tensor}: malformed metadata")
    return {
        "shape": shape,
        "data_offsets": offsets,
        "absolute_offset": 8 + header_bytes + offsets[0],
        "payload_bytes": offsets[1] - offsets[0],
    }


def sha256_range(handle: BinaryIO, offset: int, count: int) -> str:
    if offset < 0 or count <= 0:
        raise ValueError("invalid hash range")
    handle.seek(offset)
    digest = hashlib.sha256()
    remaining = count
    while remaining:
        chunk = handle.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError("selected expert range reached EOF")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def build_fixture(checkpoint_dir: Path, model_lock_path: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")
    config_path = checkpoint_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    if (
        config["hidden_size"] != HIDDEN
        or config["moe_intermediate_size"] != INTERMEDIATE
        or config["num_experts"] != EXPERTS
        or config["num_experts_per_tok"] != TOP_K
        or len(config["layer_types"]) != LAYERS
    ):
        raise ValueError("unsupported Qwen acquisition configuration")
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    shard_names: set[str] = set()
    tensor_records: dict[str, tuple[str, dict[str, Any]]] = {}
    for layer in range(LAYERS):
        for suffix in ("gate.weight", "experts.gate_up_proj", "experts.down_proj"):
            name = f"model.language_model.layers.{layer}.mlp.{suffix}"
            shard = weight_map[name]
            shard_names.add(shard)
            tensor_records[name] = (shard, tensor_layout(checkpoint_dir / shard, name))
    shards = {}
    for shard in sorted(shard_names):
        record = locked_file(lock, shard)
        shards[shard] = {"bytes": record["size"], "sha256": record["lfs_sha256"]}

    handles = {shard: (checkpoint_dir / shard).open("rb") for shard in shard_names}
    layers = []
    try:
        for layer in range(LAYERS):
            spec = input_spec(layer)
            hidden = make_hidden(HIDDEN, spec)
            router_name = f"model.language_model.layers.{layer}.mlp.gate.weight"
            router_shard, router_layout = tensor_records[router_name]
            if router_layout["shape"] != [EXPERTS, HIDDEN]:
                raise ValueError(f"layer {layer}: router shape mismatch")
            with safe_open(checkpoint_dir / router_shard, framework="pt", device="cpu") as source:
                router_weight = source.get_tensor(router_name).contiguous()
            logits = torch.nn.functional.linear(hidden, router_weight)
            probabilities = torch.softmax(logits, dtype=torch.float32, dim=-1)
            _, selected = torch.topk(probabilities, TOP_K, dim=-1)
            selection_order = selected.tolist()
            execution_order = sorted(selection_order)

            gate_up_name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
            down_name = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
            gate_up_shard, gate_up_layout = tensor_records[gate_up_name]
            down_shard, down_layout = tensor_records[down_name]
            if (
                gate_up_layout["shape"] != [EXPERTS, 2 * INTERMEDIATE, HIDDEN]
                or gate_up_layout["payload_bytes"] != EXPERTS * GATE_UP_BYTES
                or down_layout["shape"] != [EXPERTS, HIDDEN, INTERMEDIATE]
                or down_layout["payload_bytes"] != EXPERTS * DOWN_BYTES
            ):
                raise ValueError(f"layer {layer}: expert layout mismatch")
            entries = []
            for expert in execution_order:
                gate_up_offset = gate_up_layout["absolute_offset"] + expert * GATE_UP_BYTES
                down_offset = down_layout["absolute_offset"] + expert * DOWN_BYTES
                entries.append(
                    {
                        "expert": expert,
                        "gate_up": {
                            "tensor": gate_up_name,
                            "shard": gate_up_shard,
                            "absolute_offset": gate_up_offset,
                            "logical_bytes": GATE_UP_BYTES,
                            "sha256": sha256_range(
                                handles[gate_up_shard], gate_up_offset, GATE_UP_BYTES
                            ),
                        },
                        "down": {
                            "tensor": down_name,
                            "shard": down_shard,
                            "absolute_offset": down_offset,
                            "logical_bytes": DOWN_BYTES,
                            "sha256": sha256_range(handles[down_shard], down_offset, DOWN_BYTES),
                        },
                    }
                )
            layers.append(
                {
                    "layer": layer,
                    "input_spec": spec,
                    "input_bf16_sha256": hashlib.sha256(tensor_bytes(hidden)).hexdigest(),
                    "router_tensor": router_name,
                    "router_shard": router_shard,
                    "top_k_selection_order": selection_order,
                    "expert_execution_order": execution_order,
                    "experts": entries,
                }
            )
    finally:
        for handle in handles.values():
            handle.close()

    logical_bytes = sum(
        entry[projection]["logical_bytes"]
        for layer in layers
        for entry in layer["experts"]
        for projection in ("gate_up", "down")
    )
    if logical_bytes != TRACE_BYTES:
        raise ValueError("all-layer acquisition byte ledger mismatch")
    return {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextTopKRouter.forward",
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
        },
        "configuration": {
            "layers": LAYERS,
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_experts": EXPERTS,
            "top_k": TOP_K,
            "input_dtype": "BF16",
            "weight_dtype": "BF16",
            "expert_execution_order": "ascending_expert_id",
        },
        "gate_up_bytes_per_expert": GATE_UP_BYTES,
        "down_bytes_per_expert": DOWN_BYTES,
        "bytes_per_expert": GATE_UP_BYTES + DOWN_BYTES,
        "logical_bytes_per_trace": logical_bytes,
        "shards": shards,
        "layers": layers,
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
    print(
        json.dumps(
            {
                "output": os.fspath(args.output),
                "layers": len(fixture["layers"]),
                "logical_bytes_per_trace": fixture["logical_bytes_per_trace"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
