#!/usr/bin/env python3
"""Generate real layer-3 full-attention fixtures including active QSA pruning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from transformers import DynamicCache
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    Qwen4ExpTextRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

if __package__:
    from tools.generate_attention_residual_fixture import load_tensor
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, sha256_file
else:
    from generate_attention_residual_fixture import load_tensor  # type: ignore[no-redef]
    from generate_ngram_address_fixture import (  # type: ignore[no-redef]
        checkpoint_revision,
        load_model_lock,
        sha256_file,
    )


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_layer3_full_attention_qsa"
LAYER = 3
HIDDEN = 2560
HEADS = 24
KV_HEADS = 2
HEAD_DIM = 256
INDEX_HEADS = 4
INDEX_KV_HEADS = 1
INDEX_DIM = 128
BUDGET = 2048
COMPRESS = 4
LONG_PAST = 2080

INPUT_SPECS = [
    {"multiplier": 47, "add": 19, "modulus": 269, "center": 134, "divisor": 128},
    {"multiplier": 71, "add": 31, "modulus": 281, "center": 140, "divisor": 128},
]
STATE_SPECS = {
    # The modulus must exceed the long fixture's flat row span sufficiently to
    # avoid repeating pooled blocks at QSA's top-k boundary.
    "indexer_keys": {
        "multiplier": 29,
        "add": 11,
        "modulus": 65521,
        "center": 32760,
        "divisor": 32768,
    },
    "key_states": {"multiplier": 37, "add": 23, "modulus": 263, "center": 131, "divisor": 256},
    "value_states": {"multiplier": 43, "add": 17, "modulus": 271, "center": 135, "divisor": 256},
}


def deterministic_bf16(shape: tuple[int, ...], spec: dict[str, int]) -> torch.Tensor:
    count = math.prod(shape)
    index = torch.arange(count, dtype=torch.int64)
    values = (index * spec["multiplier"] + spec["add"]) % spec["modulus"] - spec["center"]
    return values.float().div(spec["divisor"]).to(torch.bfloat16).reshape(shape).contiguous()


def tensor_bytes(value: torch.Tensor) -> bytes:
    value = value.detach().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().tobytes()
    if value.dtype == torch.float32:
        return value.numpy().astype("<f4", copy=False).tobytes()
    if value.dtype == torch.int64:
        return value.numpy().astype("<i8", copy=False).tobytes()
    if value.dtype == torch.int32:
        return value.numpy().astype("<i4", copy=False).tobytes()
    if value.dtype == torch.bool:
        return value.numpy().tobytes()
    raise ValueError(f"unsupported full-attention capture dtype {value.dtype}")


def capture(value: torch.Tensor) -> dict[str, Any]:
    dtype = {
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.bool: "BOOL",
    }.get(value.dtype)
    if dtype is None:
        raise ValueError(f"unsupported full-attention capture dtype {value.dtype}")
    return {
        "dtype": dtype,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(tensor_bytes(value)).hexdigest(),
    }


def prepare_cache(config: Qwen4ExpTextConfig, past_length: int) -> DynamicCache:
    cache = DynamicCache(config=config)
    if past_length:
        cache.update_indexer(
            deterministic_bf16((1, past_length, INDEX_DIM), STATE_SPECS["indexer_keys"]), LAYER
        )
        cache.update(
            deterministic_bf16((1, KV_HEADS, past_length, HEAD_DIM), STATE_SPECS["key_states"]),
            deterministic_bf16((1, KV_HEADS, past_length, HEAD_DIM), STATE_SPECS["value_states"]),
            LAYER,
        )
    return cache


def explicit_step(
    module: Qwen4ExpTextAttention,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    cache: DynamicCache,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    full_cos, full_sin = position_embeddings
    current_cos = full_cos[:, -1:, :]
    current_sin = full_sin[:, -1:, :]

    index_qk = module.indexer.index_qk_proj(hidden).contiguous()
    index_q, raw_current = torch.split(index_qk, [INDEX_HEADS * INDEX_DIM, INDEX_DIM], dim=-1)
    index_q = index_q.reshape(1, 1, INDEX_HEADS, INDEX_DIM).contiguous()
    raw_current = raw_current.reshape(1, 1, INDEX_DIM).contiguous()
    index_q_normed = module.indexer.q_layernorm(index_q).contiguous()
    index_q_rotated = apply_rotary_pos_emb(
        index_q_normed, cos=current_cos, sin=current_sin, unsqueeze_dim=2
    ).contiguous()
    raw_keys = cache.update_indexer(raw_current, LAYER).contiguous()

    visible = attention_mask == 0
    visible_indices = torch.nonzero(visible[0, 0, 0], as_tuple=False).flatten()
    complete_blocks = visible_indices.numel() // COMPRESS
    block_tokens = visible_indices[: complete_blocks * COMPRESS].view(complete_blocks, COMPRESS)
    if complete_blocks:
        key_groups = raw_keys[0].index_select(0, block_tokens.flatten()).view(
            complete_blocks, COMPRESS, INDEX_DIM
        )
        pooled_keys = key_groups.float().mean(dim=1).to(torch.bfloat16).contiguous()
        pooled_keys_normed = module.indexer.k_layernorm(pooled_keys).contiguous()
        group_starts = block_tokens[:, 0]
        block_keys_rotated = apply_rotary_pos_emb(
            pooled_keys_normed.unsqueeze(1),
            cos=full_cos[0].index_select(0, group_starts),
            sin=full_sin[0].index_select(0, group_starts),
        ).squeeze(1).contiguous()
        index_scores = torch.matmul(
            index_q_rotated[0, 0].float(), block_keys_rotated.float().transpose(-1, -2)
        ).transpose(-1, -2)
        index_scores = (torch.relu(index_scores).sum(-1) / math.sqrt(INDEX_DIM)).contiguous()
        selected_blocks = index_scores.topk(min(BUDGET // COMPRESS, complete_blocks), dim=0).indices
        selected_tokens = block_tokens.index_select(0, selected_blocks).flatten()
    else:
        pooled_keys = torch.empty((0, INDEX_DIM), dtype=torch.bfloat16)
        pooled_keys_normed = pooled_keys.clone()
        block_keys_rotated = pooled_keys.clone()
        index_scores = torch.empty((0,), dtype=torch.float32)
        selected_blocks = torch.empty((0,), dtype=torch.int64)
        selected_tokens = torch.empty((0,), dtype=torch.int64)
    tail = visible_indices[complete_blocks * COMPRESS :]
    selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int64)
    token_mask = torch.zeros(attention_mask.shape[-1], dtype=torch.bool)
    token_mask[selected_tokens] = True
    selected_mask = torch.where(
        token_mask.view(1, 1, 1, -1), attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min
    )
    excluded_blocks = torch.nonzero(
        ~torch.isin(torch.arange(complete_blocks), selected_blocks), as_tuple=False
    ).flatten()

    q_projection = module.q_proj(hidden).contiguous()
    query, gate = torch.chunk(q_projection.view(1, 1, -1, HEAD_DIM * 2), 2, dim=-1)
    gate = gate.reshape(1, 1, -1).contiguous()
    query_normed = module.q_norm(query.reshape(1, 1, HEADS, HEAD_DIM)).transpose(1, 2).contiguous()
    key_projection = module.k_proj(hidden).contiguous()
    key_normed = module.k_norm(key_projection.view(1, 1, KV_HEADS, HEAD_DIM)).transpose(1, 2).contiguous()
    value_projection = module.v_proj(hidden).view(1, 1, KV_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
    query_rotated, key_rotated = apply_rotary_pos_emb(
        query_normed, key_normed, current_cos, current_sin
    )
    query_rotated = query_rotated.contiguous()
    key_rotated = key_rotated.contiguous()
    key_cache, value_cache = cache.update(key_rotated, value_projection, LAYER)
    repeated_keys = repeat_kv(key_cache, HEADS // KV_HEADS).contiguous()
    repeated_values = repeat_kv(value_cache, HEADS // KV_HEADS).contiguous()
    attention_scores = torch.matmul(query_rotated, repeated_keys.transpose(2, 3)) / math.sqrt(HEAD_DIM)
    attention_scores = (attention_scores + attention_mask + selected_mask).contiguous()
    attention_probabilities = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(torch.bfloat16).contiguous()
    attention_value = torch.matmul(attention_probabilities, repeated_values)
    attention_value = attention_value.transpose(1, 2).reshape(1, 1, -1).contiguous()
    gate_sigmoid = torch.sigmoid(gate).contiguous()
    gated_value = (attention_value * gate_sigmoid).contiguous()
    output = module.o_proj(gated_value).contiguous()

    captures = {
        "hidden_states": hidden,
        "position_cos": full_cos,
        "position_sin": full_sin,
        "index_qk_projection": index_qk,
        "index_query_normed": index_q_normed,
        "index_query_rotated": index_q_rotated,
        "raw_indexer_cache": raw_keys,
        "pooled_indexer_keys": pooled_keys,
        "pooled_indexer_keys_normed": pooled_keys_normed,
        "block_indexer_keys_rotated": block_keys_rotated,
        "index_scores": index_scores,
        "selected_blocks": selected_blocks.to(torch.int64),
        "excluded_blocks": excluded_blocks.to(torch.int64),
        "selected_tokens": selected_tokens,
        "selected_token_mask": token_mask,
        "q_projection": q_projection,
        "query_normed": query_normed,
        "query_rotated": query_rotated,
        "key_projection": key_projection,
        "key_normed": key_normed,
        "key_rotated": key_rotated,
        "value_projection": value_projection,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "attention_scores": attention_scores,
        "attention_probabilities": attention_probabilities,
        "attention_value": attention_value,
        "gate": gate,
        "gate_sigmoid": gate_sigmoid,
        "gated_value": gated_value,
        "output": output,
    }
    return output, captures


def build_fixture(checkpoint_dir: Path, model_lock_path: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")
    config_path = checkpoint_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    required = (
        raw_config["hidden_size"] == HIDDEN
        and raw_config["num_attention_heads"] == HEADS
        and raw_config["num_key_value_heads"] == KV_HEADS
        and raw_config["head_dim"] == HEAD_DIM
        and raw_config["indexer_n_heads"] == INDEX_HEADS
        and raw_config["indexer_kv_heads"] == INDEX_KV_HEADS
        and raw_config["indexer_head_dim"] == INDEX_DIM
        and raw_config["indexer_budget"] == BUDGET
        and raw_config["indexer_compress_ratio"] == COMPRESS
        and raw_config["layer_types"][LAYER] == "full_attention"
        and raw_config["rope_parameters"]["partial_rotary_factor"] == 0.25
    )
    if not required:
        raise ValueError("unsupported layer-3 full-attention configuration")
    config = Qwen4ExpTextConfig(**raw_config)
    config._attn_implementation = "eager"
    module = Qwen4ExpTextAttention(config, LAYER).to(torch.bfloat16).eval()
    rotary = Qwen4ExpTextRotaryEmbedding(config).eval()

    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    shapes = {
        "q_proj.weight": [HEADS * HEAD_DIM * 2, HIDDEN],
        "k_proj.weight": [KV_HEADS * HEAD_DIM, HIDDEN],
        "v_proj.weight": [KV_HEADS * HEAD_DIM, HIDDEN],
        "o_proj.weight": [HIDDEN, HEADS * HEAD_DIM],
        "q_norm.weight": [HEAD_DIM],
        "k_norm.weight": [HEAD_DIM],
        "indexer.index_qk_proj.weight": [(INDEX_HEADS + INDEX_KV_HEADS) * INDEX_DIM, HIDDEN],
        "indexer.q_layernorm.weight": [INDEX_DIM],
        "indexer.k_layernorm.weight": [INDEX_DIM],
    }
    state = {}
    tensor_records = {}
    prefix = f"model.language_model.layers.{LAYER}.self_attn"
    for local_name, shape in shapes.items():
        value, record = load_tensor(checkpoint_dir, lock, weight_map, f"{prefix}.{local_name}", shape)
        state[local_name] = value
        tensor_records[local_name] = record
    module.load_state_dict(state, strict=True)

    cases = []
    for ordinal, (past_length, input_spec) in enumerate(zip((0, LONG_PAST), INPUT_SPECS, strict=True)):
        hidden = deterministic_bf16((1, 1, HIDDEN), input_spec)
        positions = torch.arange(past_length + 1, dtype=torch.int64).view(1, -1)
        position_embeddings = rotary(hidden, positions)
        attention_mask = torch.zeros((1, 1, 1, past_length + 1), dtype=torch.bfloat16)
        explicit_cache = prepare_cache(config, past_length)
        official_cache = prepare_cache(config, past_length)
        indexer_cache = prepare_cache(config, past_length)
        with torch.no_grad():
            output, captures = explicit_step(
                module, hidden, position_embeddings, attention_mask, explicit_cache
            )
            official_mask = module.indexer(
                hidden, position_embeddings, attention_mask, indexer_cache
            )
            official_output, _ = module(
                hidden, position_embeddings, attention_mask, past_key_values=official_cache
            )
        official_token_mask = official_mask == 0
        layer = official_cache.layers[LAYER]
        if not torch.equal(output, official_output):
            raise ValueError(f"explicit attention case {ordinal} disagrees with official output")
        if not torch.equal(captures["selected_token_mask"], official_token_mask[0, 0, 0]):
            raise ValueError(f"explicit attention case {ordinal} disagrees with official QSA mask")
        if not (
            torch.equal(captures["raw_indexer_cache"], layer.indexer_keys)
            and torch.equal(captures["key_cache"], layer.keys)
            and torch.equal(captures["value_cache"], layer.values)
        ):
            raise ValueError(f"explicit attention case {ordinal} disagrees with official cache")
        if past_length:
            scores = captures["index_scores"]
            boundary = scores.sort(descending=True).values
            if boundary[BUDGET // COMPRESS - 1] == boundary[BUDGET // COMPRESS]:
                raise ValueError(
                    "QSA top-k boundary is tied: "
                    f"value={boundary[BUDGET // COMPRESS].item()} "
                    f"unique={scores.unique().numel()} zeros={(scores == 0).sum().item()}"
                )
            expected_excluded = LONG_PAST // COMPRESS - BUDGET // COMPRESS
            if captures["excluded_blocks"].numel() != expected_excluded:
                raise ValueError(
                    f"long QSA case did not exclude exactly {expected_excluded} blocks"
                )
        cases.append(
            {
                "ordinal": ordinal,
                "mode": "initial" if not past_length else "active_qsa_pruning",
                "position": past_length,
                "past_length": past_length,
                "input_spec": input_spec,
                "state_specs": STATE_SPECS if past_length else {},
                "captures": {name: capture(value) for name, value in captures.items()},
            }
        )

    return {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "source_derived_and_official_huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": [
                "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextAttention.forward",
                "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextQSAIndexer.forward",
            ],
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
        },
        "configuration": {
            "layer": LAYER,
            "hidden_size": HIDDEN,
            "attention_heads": HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "rotary_dim": int(HEAD_DIM * raw_config["rope_parameters"]["partial_rotary_factor"]),
            "rope_theta": raw_config["rope_parameters"]["rope_theta"],
            "mrope_section": raw_config["rope_parameters"]["mrope_section"],
            "indexer_heads": INDEX_HEADS,
            "indexer_kv_heads": INDEX_KV_HEADS,
            "indexer_head_dim": INDEX_DIM,
            "indexer_budget": BUDGET,
            "indexer_compress_ratio": COMPRESS,
            "boundary_dtype": "BF16",
        },
        "tensors": tensor_records,
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
    print(json.dumps({"output": os.fspath(args.output), "tensors": len(fixture["tensors"]), "cases": len(fixture["cases"]), "captures_per_case": len(fixture["cases"][0]["captures"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
