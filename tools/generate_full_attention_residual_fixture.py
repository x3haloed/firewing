#!/usr/bin/env python3
"""Generate real layer-3 gated full-attention residual fixtures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import DynamicCache
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
)

if __package__:
    from tools.generate_attention_residual_fixture import (
        HYPER_LOCAL_TENSORS,
        HYPER_SHAPES,
        INPUT_SPECS,
        load_tensor,
        make_hyper_input,
    )
    from tools.generate_full_attention_fixture import (
        HEADS,
        HEAD_DIM,
        HIDDEN,
        INDEX_DIM,
        INDEX_HEADS,
        INDEX_KV_HEADS,
        KV_HEADS,
        LAYER,
        LONG_PAST,
        capture,
        explicit_step,
        prepare_cache,
    )
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, sha256_file
else:
    from generate_attention_residual_fixture import (  # type: ignore[no-redef]
        HYPER_LOCAL_TENSORS,
        HYPER_SHAPES,
        INPUT_SPECS,
        load_tensor,
        make_hyper_input,
    )
    from generate_full_attention_fixture import (  # type: ignore[no-redef]
        HEADS,
        HEAD_DIM,
        HIDDEN,
        INDEX_DIM,
        INDEX_HEADS,
        INDEX_KV_HEADS,
        KV_HEADS,
        LAYER,
        LONG_PAST,
        capture,
        explicit_step,
        prepare_cache,
    )
    from generate_ngram_address_fixture import (  # type: ignore[no-redef]
        checkpoint_revision,
        load_model_lock,
        sha256_file,
    )


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_layer3_full_attention_residual"
HC_COUNT = 4
HC_HIDDEN = HC_COUNT * HIDDEN


def build_fixture(
    checkpoint_dir: Path,
    model_lock_path: Path,
    full_attention_fixture_path: Path,
    *,
    _layer: int = LAYER,
    _hidden_overrides: list[torch.Tensor] | None = None,
    _past_lengths: tuple[int, int] = (0, LONG_PAST),
    _modes: tuple[str, str] = ("initial", "active_qsa_pruning"),
    _semantic: str = SEMANTIC,
    _reference_hashes: dict[str, str] | None = None,
    _require_committed_parent: bool = True,
    _sequential_cache: bool = False,
    _layer_prefix: str | None = None,
    _mtp_config: bool = False,
    _return_outputs: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], list[torch.Tensor]]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    parent = (
        json.loads(full_attention_fixture_path.read_text(encoding="utf-8"))
        if _require_committed_parent
        else None
    )
    if (
        revision != lock["revision"]
        or (_require_committed_parent and parent is not None and parent.get("revision") != revision)
        or (
            _require_committed_parent
            and parent is not None
            and parent.get("semantic") != "qwen3_8_flash_next_layer3_full_attention_qsa"
        )
        or (_require_committed_parent and _layer != LAYER)
        or (_hidden_overrides is not None and len(_hidden_overrides) != 2)
        or (_sequential_cache and _past_lengths != (0, 1))
        or (_mtp_config and _layer != 0)
    ):
        raise ValueError(f"layer-{_layer} attention-residual parent authority mismatch")
    config_path = checkpoint_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    effective_config = dict(raw_config)
    if _mtp_config:
        effective_config["num_hidden_layers"] = 1
        effective_config["layer_types"] = ["full_attention"]
        effective_config["full_attention_interval"] = 1
        effective_config["ple_layer_ids"] = []
    if (
        raw_config["hidden_size"] != HIDDEN
        or raw_config["hc_count"] != HC_COUNT
        or effective_config["layer_types"][_layer] != "full_attention"
        or _layer + 1 in effective_config["ple_layer_ids"]
    ):
        raise ValueError(f"unsupported layer-{_layer} attention-residual configuration")
    config = Qwen4ExpTextConfig(**effective_config)
    config._attn_implementation = "eager"
    hyper = Qwen4ExpTextGatedResidual(config).to(torch.bfloat16).eval()
    attention = Qwen4ExpTextAttention(config, _layer).to(torch.bfloat16).eval()
    rotary = Qwen4ExpTextRotaryEmbedding(config).eval()
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    tensors: dict[str, Any] = {}
    hyper_state = {}
    layer_prefix = _layer_prefix or f"model.language_model.layers.{_layer}"
    for key, local_name in HYPER_LOCAL_TENSORS.items():
        name = f"{layer_prefix}.attn_hyper_connection.{local_name}"
        value, record = load_tensor(checkpoint_dir, lock, weight_map, name, HYPER_SHAPES[key])
        hyper_state[local_name] = value
        tensors[f"attn_hyper_connection.{key}"] = record
    hyper.load_state_dict(hyper_state, strict=True)

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
    attention_state = {}
    prefix = f"{layer_prefix}.self_attn"
    for local_name, shape in shapes.items():
        value, record = load_tensor(
            checkpoint_dir, lock, weight_map, f"{prefix}.{local_name}", shape
        )
        attention_state[local_name] = value
        tensors[f"self_attn.{local_name}"] = record
    attention.load_state_dict(attention_state, strict=True)

    cases = []
    composed_outputs = []
    sequential_caches = (
        (DynamicCache(config=config), DynamicCache(config=config), DynamicCache(config=config))
        if _sequential_cache
        else None
    )
    for ordinal, (past_length, input_spec) in enumerate(zip(_past_lengths, INPUT_SPECS, strict=True)):
        hyper_input = (
            _hidden_overrides[ordinal]
            if _hidden_overrides is not None
            else make_hyper_input(input_spec)
        )
        positions = torch.arange(past_length + 1, dtype=torch.int64).view(1, -1)
        attention_mask = torch.zeros((1, 1, 1, past_length + 1), dtype=torch.bfloat16)
        if sequential_caches is None:
            explicit_cache = prepare_cache(config, past_length, _layer)
            official_cache = prepare_cache(config, past_length, _layer)
            indexer_cache = prepare_cache(config, past_length, _layer)
        else:
            explicit_cache, official_cache, indexer_cache = sequential_caches
        with torch.no_grad():
            mixed_input, preserved, injection_weights = hyper(hyper_input)
            position_embeddings = rotary(mixed_input, positions)
            attention_output, attention_captures = explicit_step(
                attention,
                mixed_input,
                position_embeddings,
                attention_mask,
                explicit_cache,
                _layer,
            )
            official_mask = attention.indexer(
                mixed_input, position_embeddings, attention_mask, indexer_cache
            )
            official_output, _ = attention(
                mixed_input,
                position_embeddings,
                attention_mask,
                past_key_values=official_cache,
            )
            injection_products = (
                attention_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
            ).contiguous()
            composed_output = (preserved + injection_products.flatten(-2)).contiguous()
        layer = official_cache.layers[_layer]
        if (
            not torch.equal(preserved, hyper_input)
            or not torch.equal(attention_output, official_output)
            or not torch.equal(attention_captures["selected_token_mask"], (official_mask == 0)[0, 0, 0])
            or not torch.equal(attention_captures["raw_indexer_cache"], layer.indexer_keys)
            or not torch.equal(attention_captures["key_cache"], layer.keys)
            or not torch.equal(attention_captures["value_cache"], layer.values)
        ):
            raise ValueError(f"layer-{_layer} attention-residual authority mismatch at case {ordinal}")
        captures = {
            "hyper_input": hyper_input,
            "mixed_input": mixed_input,
            "injection_weights": injection_weights,
            **{f"attention.{name}": value for name, value in attention_captures.items()},
            "injection_products": injection_products,
            "composed_output": composed_output,
        }
        cases.append(
            {
                "ordinal": ordinal,
                "mode": _modes[ordinal],
                "position": past_length,
                "past_length": past_length,
                "input_spec": input_spec,
                "captures": {name: capture(value) for name, value in captures.items()},
            }
        )
        composed_outputs.append(composed_output)

    fixture = {
        "schema_version": 1,
        "semantic": _semantic,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "source_derived_and_official_huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": [
                "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextGatedResidual.forward",
                "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextAttention.forward",
                "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward",
            ],
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
            **(
                _reference_hashes
                if _reference_hashes is not None
                else {"full_attention_fixture_sha256": sha256_file(full_attention_fixture_path)}
            ),
        },
        "configuration": {
            "layer": _layer,
            "layer_type": "full_attention",
            "hidden_size": HIDDEN,
            "hc_count": HC_COUNT,
            "boundary_dtype": "BF16",
            "active_qsa_past_length": _past_lengths[1],
        },
        "tensors": tensors,
        "cases": cases,
    }
    if _return_outputs:
        return fixture, composed_outputs
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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.checkpoint_dir, args.model_lock, args.full_attention_fixture)
    write_json(args.output, fixture)
    print(json.dumps({"output": os.fspath(args.output), "tensors": len(fixture["tensors"]), "cases": len(fixture["cases"]), "captures_per_case": len(fixture["cases"][0]["captures"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
