#!/usr/bin/env python3
"""Generate a token-derived two-step cached text-to-logits authority."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
import transformers
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

if __package__:
    from tools.generate_attention_residual_fixture import build_fixture as build_linear_attention
    from tools.generate_expert_fixture import capture_hash
    from tools.generate_full_attention_residual_fixture import build_fixture as build_full_attention
    from tools.generate_full_decoder_layer1_fixture import build_fixture as build_layer1
    from tools.generate_full_decoder_layer3_fixture import build_fixture as build_decoder, write_json
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, locked_file, sha256_file
    from tools.generate_text_output_fixture import MIXER_TENSORS, tensor_record
else:
    from generate_attention_residual_fixture import build_fixture as build_linear_attention  # type: ignore[no-redef]
    from generate_expert_fixture import capture_hash  # type: ignore[no-redef]
    from generate_full_attention_residual_fixture import build_fixture as build_full_attention  # type: ignore[no-redef]
    from generate_full_decoder_layer1_fixture import build_fixture as build_layer1  # type: ignore[no-redef]
    from generate_full_decoder_layer3_fixture import build_fixture as build_decoder, write_json  # type: ignore[no-redef]
    from generate_ngram_address_fixture import checkpoint_revision, load_model_lock, locked_file, sha256_file  # type: ignore[no-redef]
    from generate_text_output_fixture import MIXER_TENSORS, tensor_record  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_firewing_two_token_cached_text_logits"
TEXT = "Firewing"
TOKEN_IDS = [16_207, 22_856]
HIDDEN = 2560
HC_COUNT = 4
HC_HIDDEN = HIDDEN * HC_COUNT
HC_LOWRANK = 320
VOCAB = 248_320
ROOT = Path(__file__).resolve().parents[1]


def fixture_path(relative: str) -> Path:
    return ROOT / "fixtures" / relative


def embedding_roots(
    checkpoint_dir: Path,
    lock: dict[str, Any],
    weight_map: dict[str, str],
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    name = "model.language_model.embed_tokens.weight"
    shard = weight_map[name]
    with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as source:
        tensor = source.get_tensor(name)
        if tensor.dtype != torch.bfloat16 or list(tensor.shape) != [VOCAB, HIDDEN]:
            raise ValueError("unsupported token embedding tensor")
        rows = [tensor[token].clone().contiguous() for token in TOKEN_IDS]
    locked = locked_file(lock, shard)
    record = {
        "tensor": name,
        "shape": [VOCAB, HIDDEN],
        "shard": shard,
        "shard_bytes": locked["size"],
        "shard_sha256": locked["lfs_sha256"],
        "selected_rows": [
            {"token_id": token, "payload_sha256": capture_hash(row)}
            for token, row in zip(TOKEN_IDS, rows, strict=True)
        ],
    }
    roots = [row.repeat(HC_COUNT).reshape(1, 1, HC_HIDDEN).contiguous() for row in rows]
    return record, roots


def build_output(
    checkpoint_dir: Path,
    lock: dict[str, Any],
    raw_config: dict[str, Any],
    weight_map: dict[str, str],
    decoder_outputs: list[torch.Tensor],
) -> dict[str, Any]:
    prefix = "model.language_model.hyper_connection_mixer"
    module = Qwen4ExpTextGatedResidual(
        Qwen4ExpTextConfig(**raw_config), use_combine=False
    ).to(torch.bfloat16).eval()
    state: dict[str, torch.Tensor] = {}
    records: dict[str, Any] = {}
    for key, (local_name, shape) in MIXER_TENSORS.items():
        name = f"{prefix}.{local_name}"
        shard = weight_map[name]
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as source:
            value = source.get_tensor(name).contiguous()
        state[local_name] = value
        records[key] = tensor_record(lock, weight_map, name, shape, value)
    module.load_state_dict(state, strict=True)
    head_name = "lm_head.weight"
    head_shard = weight_map[head_name]
    with safe_open(checkpoint_dir / head_shard, framework="pt", device="cpu") as source:
        head = source.get_tensor(head_name).contiguous()
    records["lm_head"] = tensor_record(lock, weight_map, head_name, [VOCAB, HIDDEN], head)

    steps = []
    with torch.no_grad():
        for ordinal, hidden in enumerate(decoder_outputs):
            normalized = module.hc_norm(hidden).contiguous()
            down = module.input_mix_weight_down(normalized).contiguous()
            scaled = (down / HC_COUNT).contiguous()
            activated = torch.nn.functional.silu(scaled).contiguous()
            up = module.input_mix_weight_up(activated).contiguous()
            mix_weight = torch.sigmoid(up).contiguous()
            products = (
                mix_weight.unflatten(-1, (HC_COUNT, HIDDEN))
                * normalized.unflatten(-1, (HC_COUNT, HIDDEN))
            ).contiguous()
            mixed = products.mean(dim=-2).contiguous()
            if not torch.equal(module(hidden), mixed):
                raise ValueError("final mixer disagrees with official forward")
            logits = torch.nn.functional.linear(mixed, head).contiguous()
            flat = logits.reshape(-1)
            values, indices = torch.topk(flat, 20, sorted=True)
            cutoff = values[-1]
            above = torch.nonzero(flat > cutoff).reshape(-1)
            ties = torch.nonzero(flat == cutoff).reshape(-1)
            captures = {
                "decoder_output": hidden,
                "hyper_input_normed": normalized,
                "mix_down": down,
                "mix_down_scaled": scaled,
                "mix_down_silu": activated,
                "mix_up": up,
                "input_mix_weight": mix_weight,
                "mixed_products": products,
                "mixed_hidden": mixed,
                "logits": logits,
            }
            steps.append(
                {
                    "ordinal": ordinal,
                    "captures": {key: capture_hash(value) for key, value in captures.items()},
                    "top20_token_ids": indices.tolist(),
                    "top20_logit_bf16_u16": values.view(torch.uint16).tolist(),
                    "top20_cutoff_bf16_u16": cutoff.view(torch.uint16).item(),
                    "strictly_above_cutoff_token_ids": above.tolist(),
                    "cutoff_tie_token_ids": ties.tolist(),
                }
            )
            del logits
    del head
    gc.collect()
    return {"tensors": records, "steps": steps}


def build_fixture(
    checkpoint_dir: Path,
    model_lock_path: Path,
    *,
    _return_outputs: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], list[torch.Tensor]]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")
    config_path = checkpoint_dir / "config.json"
    index_path = checkpoint_dir / "model.safetensors.index.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    layer_types = raw_config["layer_types"]
    if (
        raw_config["hidden_size"] != HIDDEN
        or raw_config["hc_count"] != HC_COUNT
        or raw_config["vocab_size"] != VOCAB
        or raw_config["num_hidden_layers"] != 48
        or raw_config["ple_layer_ids"] != [2]
        or len(layer_types) != 48
        or any(kind != ("full_attention" if layer % 4 == 3 else "linear_attention") for layer, kind in enumerate(layer_types))
    ):
        raise ValueError("unsupported token text endpoint configuration")
    tokenizer = json.loads(fixture_path("tokenizer/qwen3_8_flash_next.json").read_text())
    raw_case = next(case for case in tokenizer["raw_cases"] if case["name"] == "ascii")
    if raw_case["text"] != TEXT or raw_case["token_ids"] != TOKEN_IDS:
        raise ValueError("tokenizer fixture no longer maps Firewing to the endpoint token IDs")
    embedding, current_outputs = embedding_roots(checkpoint_dir, lock, weight_map)
    embedding_root_hashes = [capture_hash(value) for value in current_outputs]

    base_reference = {
        "hidden_source": "token_embedding_repeated_across_four_streams",
        "tokenizer_fixture_sha256": sha256_file(fixture_path("tokenizer/qwen3_8_flash_next.json")),
    }
    attention_result = build_linear_attention(
        checkpoint_dir,
        model_lock_path,
        fixture_path("hyper_connection/qwen3_8_flash_next_layer0.json"),
        fixture_path("deltanet/qwen3_8_flash_next_layer0_decode.json"),
        _layer=0,
        _hidden_overrides=current_outputs,
        _semantic="qwen3_8_flash_next_token_layer0_attention",
        _reference_hashes=base_reference,
        _return_outputs=True,
    )
    if not isinstance(attention_result, tuple):
        raise AssertionError("token layer-0 attention outputs were not returned")
    attention, post_attention = attention_result
    decoder_result = build_decoder(
        checkpoint_dir,
        model_lock_path,
        fixture_path("attention_residual/qwen3_8_flash_next_layer0.json"),
        fixture_path("sparse_moe/qwen3_8_flash_next_layer0.json"),
        _parent_execution=(attention, post_attention),
        _parent_semantic="qwen3_8_flash_next_token_layer0_attention",
        _layer=0,
        _layer_type="linear_attention",
        _semantic="qwen3_8_flash_next_token_layer0_decoder",
        _reference_hashes=base_reference,
        _modes=("initial_chunk", "cached_recurrent"),
        _require_committed_parent=False,
        _return_outputs=True,
    )
    if not isinstance(decoder_result, tuple):
        raise AssertionError("token layer-0 decoder outputs were not returned")
    decoder, current_outputs = decoder_result
    layers = [{"layer": 0, "layer_type": "linear_attention", "attention": attention, "decoder": decoder}]

    layer1_result = build_layer1(
        checkpoint_dir,
        model_lock_path,
        fixture_path("ngram/qwen3_8_flash_next.json"),
        fixture_path("ngram/qwen3_8_flash_next_row_hashes.json"),
        fixture_path("ple/qwen3_8_flash_next_layer1_decode.json"),
        fixture_path("attention_residual/qwen3_8_flash_next_layer1_ple.json"),
        _hidden_overrides=current_outputs,
        _token_ids=TOKEN_IDS,
        _semantic="qwen3_8_flash_next_token_layer1_decoder",
        _reference_hashes=base_reference,
        _return_chain=True,
    )
    if not isinstance(layer1_result, tuple):
        raise AssertionError("token layer-1 outputs were not returned")
    decoder, current_outputs, attention, ple = layer1_result
    ple["semantic"] = "qwen3_8_flash_next_token_layer1_ple"
    attention["semantic"] = "qwen3_8_flash_next_token_layer1_attention"
    layers.append({"layer": 1, "layer_type": "linear_attention", "ple": ple, "attention": attention, "decoder": decoder})

    for layer in range(2, 48):
        kind = layer_types[layer]
        reference = {"hidden_source": f"token_endpoint.layer{layer - 1}_output"}
        if kind == "linear_attention":
            attention_result = build_linear_attention(
                checkpoint_dir,
                model_lock_path,
                fixture_path("hyper_connection/qwen3_8_flash_next_layer0.json"),
                fixture_path("deltanet/qwen3_8_flash_next_layer0_decode.json"),
                _layer=layer,
                _hidden_overrides=current_outputs,
                _semantic=f"qwen3_8_flash_next_token_layer{layer}_attention",
                _reference_hashes=reference,
                _return_outputs=True,
            )
            modes = ("initial_chunk", "cached_recurrent")
        else:
            attention_result = build_full_attention(
                checkpoint_dir,
                model_lock_path,
                fixture_path("full_attention/qwen3_8_flash_next_layer3.json"),
                _layer=layer,
                _hidden_overrides=current_outputs,
                _past_lengths=(0, 1),
                _modes=("initial", "cached_incremental"),
                _semantic=f"qwen3_8_flash_next_token_layer{layer}_attention",
                _reference_hashes=reference,
                _require_committed_parent=False,
                _sequential_cache=True,
                _return_outputs=True,
            )
            modes = ("initial", "cached_incremental")
        if not isinstance(attention_result, tuple):
            raise AssertionError(f"token layer-{layer} attention outputs were not returned")
        attention, post_attention = attention_result
        decoder_result = build_decoder(
            checkpoint_dir,
            model_lock_path,
            fixture_path("full_attention/qwen3_8_flash_next_layer3.json"),
            fixture_path("attention_residual/qwen3_8_flash_next_layer3.json"),
            _parent_execution=(attention, post_attention),
            _parent_semantic=f"qwen3_8_flash_next_token_layer{layer}_attention",
            _layer=layer,
            _layer_type=kind,
            _semantic=f"qwen3_8_flash_next_token_layer{layer}_decoder",
            _reference_hashes=reference,
            _modes=modes,
            _require_committed_parent=False,
            _return_outputs=True,
        )
        if not isinstance(decoder_result, tuple):
            raise AssertionError(f"token layer-{layer} decoder outputs were not returned")
        decoder, current_outputs = decoder_result
        layers.append({"layer": layer, "layer_type": kind, "attention": attention, "decoder": decoder})
        gc.collect()

    output = build_output(checkpoint_dir, lock, raw_config, weight_map, current_outputs)
    fixture = {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "source_derived_and_official_huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
            "tokenizer_fixture_sha256": sha256_file(fixture_path("tokenizer/qwen3_8_flash_next.json")),
        },
        "configuration": {
            "text": TEXT,
            "token_ids": TOKEN_IDS,
            "hidden_size": HIDDEN,
            "hc_count": HC_COUNT,
            "vocab_size": VOCAB,
            "layers": 48,
            "boundary_dtype": "BF16",
            "cache_mode": "sequential_incremental",
        },
        "embedding": embedding,
        "embedding_root_hashes": embedding_root_hashes,
        "layers": layers,
        "output": output,
    }
    if _return_outputs:
        return fixture, current_outputs
    return fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.checkpoint_dir, args.model_lock)
    write_json(args.output, fixture)
    print(json.dumps({"output": os.fspath(args.output), "layers": len(fixture["layers"]), "tokens": TOKEN_IDS}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
