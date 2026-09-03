#!/usr/bin/env python3
"""Generate checkpoint-backed Qwen3.8-Flash-Next n-gram address fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
import transformers
import transformers.conversion_mapping as conversion_mapping
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    _build_layer_multipliers,
    _find_nth_prime_after,
)


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_ngram_addresses"
PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding"
BUFFER_LOCATIONS = {
    "layer_multipliers": "model-00005-of-00131.safetensors",
    "ngram_heads_offsets": "model-00037-of-00131.safetensors",
    "ngram_heads_vocab_sizes": "model-00037-of-00131.safetensors",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_revision(checkpoint_dir: Path) -> str:
    trees = sorted((checkpoint_dir / ".cache" / "huggingface" / "trees").glob("*.json"))
    if len(trees) != 1 or not re.fullmatch(r"[0-9a-f]{40}", trees[0].stem):
        raise ValueError("checkpoint must contain exactly one 40-hex Hugging Face tree manifest")
    return trees[0].stem


def load_model_lock(model_lock_path: Path) -> dict[str, Any]:
    lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    if lock.get("model") != MODEL or not re.fullmatch(r"[0-9a-f]{40}", lock.get("revision", "")):
        raise ValueError("model lock does not identify the supported model and revision")
    return lock


def locked_file(lock: dict[str, Any], relative_path: str) -> dict[str, Any]:
    matches = [item for item in lock["files"] if item["path"] == relative_path]
    if len(matches) != 1:
        raise ValueError(f"model lock must contain exactly one entry for {relative_path}")
    return matches[0]


def checkpoint_buffer(checkpoint_dir: Path, shard: str, name: str) -> list[int]:
    with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as handle:
        value = handle.get_tensor(f"{PLE_PREFIX}.{name}")
    if value.dtype != torch.int64 or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int64 checkpoint tensor")
    return value.tolist()


def safetensor_descriptor(path: Path, tensor: str) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        if header_length <= 0 or header_length > 16 * 1024 * 1024:
            raise ValueError(f"unsupported safetensors header length in {path.name}")
        header = json.loads(handle.read(header_length))
    descriptor = header.get(tensor)
    if not isinstance(descriptor, dict):
        raise ValueError(f"{tensor} is absent from {path.name}")
    return descriptor


def shift_right_ignore_eos(token_ids: torch.Tensor, shift: int, eos_token_id: int) -> torch.Tensor:
    """Equation-level copy of the pinned Qwen4-Exp segmentation rule."""
    if shift == 0:
        return token_ids
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
    )
    segment_start = previous_eos + 1
    position_in_segment = positions.unsqueeze(0) - segment_start
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos_token_id))


def reference_addresses(
    input_ids: list[int],
    previous_context: list[int],
    eos_token_id: int,
    multipliers: list[int],
    head_vocab_sizes: list[int],
    head_offsets: list[int],
    heads_per_ngram: int,
) -> list[list[int]]:
    token_history = torch.tensor([previous_context + input_ids], dtype=torch.long)
    shifted = [shift_right_ignore_eos(token_history, shift, eos_token_id) for shift in range(3)]
    blocks = []
    for ngram in range(2, 4):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        sizes = torch.tensor(head_vocab_sizes[start:end], dtype=torch.long)
        offsets = torch.tensor(head_offsets[start:end], dtype=torch.long)
        blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes) + offsets)
    addresses = torch.cat(blocks, dim=-1)[:, -len(input_ids) :]
    return addresses[0].tolist()


def build_fixture(checkpoint_dir: Path, model_lock_path: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")

    config_path = checkpoint_dir / "config.json"
    config_hash = sha256_file(config_path)
    expected_config_hash = lock.get("local_small_file_sha256", {}).get("config.json")
    if config_hash != expected_config_hash:
        raise ValueError("checkpoint config hash does not match model lock")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    config = Qwen4ExpTextConfig(**raw_config)
    if (
        config.vocab_size != 248_320
        or config.ngram_size != 3
        or config.heads_per_ngram != 8
        or config.ngram_vocab_size_base != 20_000_000
        or config.make_ngram_vocab_size_divisible_by != 128
        or config.split_ngram_parts != 128
        or config.ple_layer_ids != [2]
        or config.ple_embed_dim != 2560
        or config.eos_token_id != 248_044
        or config.seed != 1234
    ):
        raise ValueError("unsupported Qwen n-gram configuration")

    ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram
    generated_multipliers = _build_layer_multipliers(
        config.vocab_size, config.ngram_size, 0, config.seed
    ).tolist()
    generated_sizes = [
        _find_nth_prime_after(config.ngram_vocab_size_base - 1, index + 1)
        for index in range(ngram_heads)
    ]
    generated_offsets = []
    running_offset = 0
    for size in generated_sizes:
        generated_offsets.append(running_offset)
        running_offset += size
    padded_rows = math.ceil(running_offset / config.make_ngram_vocab_size_divisible_by) * 128
    if padded_rows % config.split_ngram_parts:
        raise ValueError("padded n-gram table is not evenly divisible into checkpoint shards")
    rows_per_shard = padded_rows // config.split_ngram_parts

    index_path = checkpoint_dir / "model.safetensors.index.json"
    index_hash = sha256_file(index_path)
    expected_index_hash = lock.get("local_small_file_sha256", {}).get(index_path.name)
    if index_hash != expected_index_hash:
        raise ValueError("checkpoint tensor index hash does not match model lock")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    table_parts = []
    for part in range(config.split_ngram_parts):
        tensor = f"{PLE_PREFIX}.ngram_embedding.shard_{part}.weight"
        shard = weight_map.get(tensor)
        if not isinstance(shard, str):
            raise ValueError(f"tensor index is missing n-gram table part {part}")
        descriptor = safetensor_descriptor(checkpoint_dir / shard, tensor)
        if descriptor.get("dtype") != "BF16" or descriptor.get("shape") != [rows_per_shard, 160]:
            raise ValueError(f"n-gram table part {part} has an unsupported dtype or shape")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or offsets[1] - offsets[0] != rows_per_shard * 160 * 2
        ):
            raise ValueError(f"n-gram table part {part} has invalid data offsets")
        locked = locked_file(lock, shard)
        table_parts.append(
            {
                "part": part,
                "tensor": tensor,
                "shard": shard,
                "shard_bytes": locked["size"],
                "shard_sha256": locked["lfs_sha256"],
                "data_offsets": offsets,
            }
        )

    generated = {
        "layer_multipliers": generated_multipliers,
        "ngram_heads_offsets": generated_offsets,
        "ngram_heads_vocab_sizes": generated_sizes,
    }
    checkpoint_buffers = {}
    for name, shard in BUFFER_LOCATIONS.items():
        values = checkpoint_buffer(checkpoint_dir, shard, name)
        if values != generated[name]:
            raise ValueError(f"generated {name} does not match checkpoint payload")
        locked = locked_file(lock, shard)
        checkpoint_buffers[name] = {
            "tensor": f"{PLE_PREFIX}.{name}",
            "shard": shard,
            "shard_bytes": locked["size"],
            "shard_sha256": locked["lfs_sha256"],
            "values": values,
        }

    case_inputs = [
        ("initial_single", [42], [config.eos_token_id, config.eos_token_id]),
        ("ordinary_sequence", [1, 2, 3, 4], [config.eos_token_id, config.eos_token_id]),
        ("eos_segment_boundary", [17, config.eos_token_id, 23, 24], [9, 10]),
        ("incremental_context", [102, 103], [100, 101]),
        ("vocabulary_ceiling", [248_319, 248_318, 248_317], [248_316, 248_315]),
    ]
    cases = []
    for name, input_ids, previous_context in case_inputs:
        global_rows = reference_addresses(
            input_ids,
            previous_context,
            config.eos_token_id,
            generated_multipliers,
            generated_sizes,
            generated_offsets,
            config.heads_per_ngram,
        )
        cases.append(
            {
                "name": name,
                "input_ids": input_ids,
                "previous_context": previous_context,
                "global_rows": global_rows,
                "physical_rows": [
                    [
                        {"shard": row // rows_per_shard, "row": row % rows_per_shard}
                        for row in token_rows
                    ]
                    for token_rows in global_rows
                ],
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
            "config_sha256": config_hash,
            "conversion_mapping_sha256": sha256_file(Path(conversion_mapping.__file__)),
            "layout_source": "transformers.conversion_mapping.qwen4_exp_text.Concatenate(dim=0)",
            "model_lock_sha256": sha256_file(model_lock_path),
            "tensor_index_sha256": index_hash,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp",
        },
        "configuration": {
            "seed": config.seed,
            "eos_token_id": config.eos_token_id,
            "unigram_vocab_size": config.vocab_size,
            "ngram_size": config.ngram_size,
            "heads_per_ngram": config.heads_per_ngram,
            "ngram_heads": ngram_heads,
            "ngram_vocab_size_base": config.ngram_vocab_size_base,
            "embedding_width": config.ple_embed_dim,
            "head_width": config.ple_embed_dim // ngram_heads,
            "padded_rows": padded_rows,
            "split_parts": config.split_ngram_parts,
            "rows_per_shard": rows_per_shard,
            "useful_bf16_bytes_per_token": config.ple_embed_dim * 2,
        },
        "checkpoint_buffers": checkpoint_buffers,
        "table_parts": table_parts,
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
