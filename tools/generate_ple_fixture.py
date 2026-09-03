#!/usr/bin/env python3
"""Generate a sparse-row, two-step real layer-1 PLE fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, BinaryIO

import torch
import torch.nn.functional as F
import transformers

if __package__:
    from tools.generate_attention_residual_fixture import load_tensor, make_hyper_input
    from tools.generate_ngram_address_fixture import (
        checkpoint_revision,
        load_model_lock,
        reference_addresses,
        sha256_file,
    )
    from tools.generate_ngram_row_hash_fixture import read_exact_row, safetensor_payload_start
else:
    from generate_attention_residual_fixture import load_tensor, make_hyper_input  # type: ignore[no-redef]
    from generate_ngram_address_fixture import (  # type: ignore[no-redef]
        checkpoint_revision,
        load_model_lock,
        reference_addresses,
        sha256_file,
    )
    from generate_ngram_row_hash_fixture import (  # type: ignore[no-redef]
        read_exact_row,
        safetensor_payload_start,
    )


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_layer1_ple_cached_decode"
HIDDEN = 2560
HC_COUNT = 4
HC_HIDDEN = HIDDEN * HC_COUNT
HEADS = 16
HEAD_WIDTH = 160
CONTEXT = 2
CONV_STATE = 9
TOKENS = [42, 43]
INPUT_SPECS = [
    {"multiplier": 43, "add": 17, "modulus": 263, "center": 131, "divisor": 128, "sparse_stride": 1},
    {"multiplier": 61, "add": 29, "modulus": 277, "center": 138, "divisor": 128, "sparse_stride": 1},
]


def tensor_bytes(value: torch.Tensor) -> bytes:
    value = value.detach().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().tobytes()
    if value.dtype == torch.int64:
        return value.numpy().astype("<i8", copy=False).tobytes()
    raise ValueError(f"unsupported PLE capture dtype {value.dtype}")


def capture(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().contiguous()
    dtype = {torch.bfloat16: "BF16", torch.int64: "I64"}.get(value.dtype)
    if dtype is None:
        raise ValueError(f"unsupported PLE capture dtype {value.dtype}")
    return {
        "dtype": dtype,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(tensor_bytes(value)).hexdigest(),
    }


def grouped_rms(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    grouped = value.float().reshape(*value.shape[:-1], HC_COUNT, HIDDEN)
    normalized = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + 1.0e-6)
    return (normalized.flatten(-2) * (1.0 + weight.float())).to(value.dtype).contiguous()


def selected_embedding(
    checkpoint_dir: Path,
    address: dict[str, Any],
    rows: list[int],
    handles: dict[str, BinaryIO],
    payload_starts: dict[str, int],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    config = address["configuration"]
    parts = address["table_parts"]
    payload = bytearray()
    records = []
    for global_row in rows:
        part_index, local_row = divmod(global_row, config["rows_per_shard"])
        part = parts[part_index]
        shard = part["shard"]
        if shard not in handles:
            handles[shard] = (checkpoint_dir / shard).open("rb")
            payload_starts[shard] = safetensor_payload_start(handles[shard])
        row = read_exact_row(
            handles[shard],
            payload_starts[shard],
            part["data_offsets"][0],
            local_row,
            config["rows_per_shard"],
            HEAD_WIDTH * 2,
        )
        payload.extend(row)
        records.append(
            {
                "global_row": global_row,
                "part": part_index,
                "local_row": local_row,
                "tensor": part["tensor"],
                "shard": shard,
                "shard_bytes": part["shard_bytes"],
                "shard_sha256": part["shard_sha256"],
                "data_offsets": part["data_offsets"],
                "payload_sha256": hashlib.sha256(row).hexdigest(),
            }
        )
    values = torch.frombuffer(payload, dtype=torch.uint16).clone().view(torch.bfloat16)
    return values.reshape(1, 1, HIDDEN).contiguous(), records


def build_fixture(
    checkpoint_dir: Path,
    model_lock_path: Path,
    ngram_fixture_path: Path,
    ngram_row_fixture_path: Path,
    *,
    _return_outputs: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], list[torch.Tensor]]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    address = json.loads(ngram_fixture_path.read_text(encoding="utf-8"))
    row_authority = json.loads(ngram_row_fixture_path.read_text(encoding="utf-8"))
    if (
        revision != lock["revision"]
        or address.get("revision") != revision
        or address.get("semantic") != "qwen3_8_flash_next_ngram_addresses"
        or row_authority.get("revision") != revision
        or row_authority.get("address_fixture_sha256") != sha256_file(ngram_fixture_path)
    ):
        raise ValueError("PLE parent authority mismatch")
    ngram = address["configuration"]
    if (
        ngram["ngram_size"] != 3
        or ngram["heads_per_ngram"] != 8
        or ngram["ngram_heads"] != HEADS
        or ngram["head_width"] != HEAD_WIDTH
        or ngram["embedding_width"] != HIDDEN
        or ngram["eos_token_id"] != 248044
    ):
        raise ValueError("unsupported PLE n-gram configuration")

    config_path = checkpoint_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    if (
        raw_config["ple_layer_ids"] != [2]
        or raw_config["hidden_size"] != HIDDEN
        or raw_config["hc_count"] != HC_COUNT
        or raw_config["ple_embed_dim"] != HIDDEN
        or raw_config["ple_conv_kernel_size"] != 4
        or raw_config["ngram_size"] != 3
        or raw_config["rms_norm_eps"] != 1.0e-6
    ):
        raise ValueError("unsupported layer-1 PLE configuration")
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    prefix = "model.language_model.layers.1.ple"
    names = {
        "key_proj": (f"{prefix}.key_proj.weight", [HC_HIDDEN, HIDDEN]),
        "value_proj": (f"{prefix}.value_proj.weight", [HIDDEN, HIDDEN]),
        "norm_key": (f"{prefix}.norm_key.weight", [HC_HIDDEN]),
        "norm_query": (f"{prefix}.norm_query.weight", [HC_HIDDEN]),
        "norm_conv": (f"{prefix}.norm_conv.weight", [HC_HIDDEN]),
        "conv1d": (f"{prefix}.conv1d.weight", [HC_HIDDEN, 1, 4]),
    }
    tensors: dict[str, torch.Tensor] = {}
    tensor_records = {}
    for key, (name, shape) in names.items():
        value, record = load_tensor(checkpoint_dir, lock, weight_map, name, shape)
        tensors[key] = value
        tensor_records[key] = record

    multipliers = address["checkpoint_buffers"]["layer_multipliers"]["values"]
    sizes = address["checkpoint_buffers"]["ngram_heads_vocab_sizes"]["values"]
    offsets = address["checkpoint_buffers"]["ngram_heads_offsets"]["values"]
    previous_context = [ngram["eos_token_id"]] * CONTEXT
    conv_state = torch.zeros((1, HC_HIDDEN, CONV_STATE), dtype=torch.bfloat16)
    handles: dict[str, BinaryIO] = {}
    payload_starts: dict[str, int] = {}
    steps = []
    outputs = []
    try:
        for ordinal, (token, input_spec) in enumerate(zip(TOKENS, INPUT_SPECS, strict=True)):
            rows = reference_addresses(
                [token],
                previous_context,
                ngram["eos_token_id"],
                multipliers,
                sizes,
                offsets,
                ngram["heads_per_ngram"],
            )[0]
            embedding, row_records = selected_embedding(
                checkpoint_dir, address, rows, handles, payload_starts
            )
            hidden = make_hyper_input(input_spec)
            key_projection = F.linear(embedding, tensors["key_proj"]).contiguous()
            key_normed = grouped_rms(key_projection, tensors["norm_key"])
            value = F.linear(embedding, tensors["value_proj"]).contiguous()
            query_normed = grouped_rms(hidden, tensors["norm_query"])
            products = (
                key_normed.reshape(1, 1, HC_COUNT, HIDDEN)
                * query_normed.reshape(1, 1, HC_COUNT, HIDDEN)
            ).contiguous()
            gate = (products.sum(dim=-1, keepdim=True) / math.sqrt(HIDDEN)).contiguous()
            transformed_gate = (
                gate.abs().clamp_min(1.0e-6).sqrt() * gate.sign()
            ).contiguous()
            gate_sigmoid = torch.sigmoid(transformed_gate).contiguous()
            gated_value = (gate_sigmoid * value.unsqueeze(-2)).flatten(-2).contiguous()
            gated_value_normed = grouped_rms(gated_value, tensors["norm_conv"])

            current = gated_value_normed.transpose(1, 2).contiguous()
            if ordinal == 0:
                full_conv_state = F.pad(current, (CONV_STATE - 1, 0))
            else:
                full_conv_state = torch.cat([conv_state, current], dim=-1)
            conv_state = full_conv_state[..., -CONV_STATE:].contiguous()
            convolution_input = F.pad(full_conv_state, (CONV_STATE, 0))
            convolution_input = convolution_input[..., -(CONV_STATE + 1) :].contiguous()
            convolution = F.silu(
                F.conv1d(
                    convolution_input,
                    tensors["conv1d"],
                    groups=HC_HIDDEN,
                    dilation=raw_config["ngram_size"],
                )
            ).transpose(1, 2).contiguous()
            output = (gated_value + convolution).contiguous()
            outputs.append(output)

            previous_context = (previous_context + [token])[-CONTEXT:]
            context_state = torch.tensor([previous_context], dtype=torch.int64)
            captures = {
                "hidden_states": hidden,
                "embedding": embedding,
                "key_projection": key_projection,
                "key_normed": key_normed,
                "value": value,
                "query_normed": query_normed,
                "key_query_products": products,
                "gate": gate,
                "transformed_gate": transformed_gate,
                "gate_sigmoid": gate_sigmoid,
                "gated_value": gated_value,
                "gated_value_normed": gated_value_normed,
                "convolution_state": conv_state,
                "convolution": convolution,
                "output": output,
                "token_context_state": context_state,
            }
            steps.append(
                {
                    "ordinal": ordinal,
                    "mode": "initial_chunk" if ordinal == 0 else "cached_recurrent",
                    "token_id": token,
                    "previous_context": ([ngram["eos_token_id"]] * CONTEXT if ordinal == 0 else [ngram["eos_token_id"], TOKENS[0]]),
                    "input_spec": input_spec,
                    "rows": row_records,
                    "captures": {name: capture(value) for name, value in captures.items()},
                }
            )
    finally:
        for handle in handles.values():
            handle.close()

    fixture = {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "source_derived_huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextPLELayer.forward",
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
            "ngram_fixture_sha256": sha256_file(ngram_fixture_path),
            "ngram_row_fixture_sha256": sha256_file(ngram_row_fixture_path),
        },
        "configuration": {
            "layer": 1,
            "ple_layer_index": 0,
            "hidden_size": HIDDEN,
            "hc_count": HC_COUNT,
            "embedding_width": HIDDEN,
            "ngram_heads": HEADS,
            "head_width": HEAD_WIDTH,
            "context_length": CONTEXT,
            "conv_kernel_size": 4,
            "conv_dilation": 3,
            "conv_state_length": CONV_STATE,
            "boundary_dtype": "BF16",
            "token_state_dtype": "I64",
        },
        "case": {
            "name": "layer_1_two_token_ple",
            "tensors": tensor_records,
            "steps": steps,
        },
    }
    if _return_outputs:
        return fixture, outputs
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
    parser.add_argument("--ngram-fixture", required=True, type=Path)
    parser.add_argument("--ngram-row-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(
        args.checkpoint_dir,
        args.model_lock,
        args.ngram_fixture,
        args.ngram_row_fixture,
    )
    write_json(args.output, fixture)
    print(
        json.dumps(
            {
                "output": os.fspath(args.output),
                "tensors": len(fixture["case"]["tensors"]),
                "steps": len(fixture["case"]["steps"]),
                "rows": sum(len(step["rows"]) for step in fixture["case"]["steps"]),
                "captures_per_step": len(fixture["case"]["steps"][0]["captures"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
