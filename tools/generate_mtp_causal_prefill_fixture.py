#!/usr/bin/env python3
"""Generate the first target-derived Qwen4-Exp EAGLE prefill proposal authority."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

if __package__:
    from tools.generate_expert_fixture import capture_hash
    from tools.generate_full_attention_residual_fixture import build_fixture as build_attention
    from tools.generate_full_decoder_layer3_fixture import build_fixture as build_decoder
    from tools.generate_mtp_decoder_fixture import build_output_fixture, write_json
    from tools.generate_mtp_input_fusion_fixture import (
        HC_COUNT,
        HC_HIDDEN,
        HIDDEN,
        TENSORS,
        zero_centered_rms_norm,
    )
    from tools.generate_ngram_address_fixture import load_model_lock, locked_file, sha256_file
    from tools.generate_token_text_endpoint_fixture import build_fixture as build_endpoint
else:
    from generate_expert_fixture import capture_hash  # type: ignore[no-redef]
    from generate_full_attention_residual_fixture import build_fixture as build_attention  # type: ignore[no-redef]
    from generate_full_decoder_layer3_fixture import build_fixture as build_decoder  # type: ignore[no-redef]
    from generate_mtp_decoder_fixture import build_output_fixture, write_json  # type: ignore[no-redef]
    from generate_mtp_input_fusion_fixture import (  # type: ignore[no-redef]
        HC_COUNT,
        HC_HIDDEN,
        HIDDEN,
        TENSORS,
        zero_centered_rms_norm,
    )
    from generate_ngram_address_fixture import load_model_lock, locked_file, sha256_file  # type: ignore[no-redef]
    from generate_token_text_endpoint_fixture import build_fixture as build_endpoint  # type: ignore[no-redef]

MODEL = "Qwen/Qwen3.8-Flash-Next"
SGLANG_COMMIT = "78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385"
SEED_SEMANTIC = "qwen3_8_flash_next_target_derived_mtp_prefill_fusion"
ATTENTION_SEMANTIC = "qwen3_8_flash_next_target_derived_mtp_prefill_attention"
DECODER_SEMANTIC = "qwen3_8_flash_next_target_derived_mtp_prefill_decoder"
OUTPUT_SEMANTIC = "qwen3_8_flash_next_target_derived_mtp_prefill_logits"
LAYER_PREFIX = "mtp.layers.0"


def build_seed(
    checkpoint_dir: Path,
    model_lock_path: Path,
    mtp_source_lock_path: Path,
    scheduler_lock_path: Path,
    endpoint_fixture_path: Path,
    fusion_fixture_path: Path,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    endpoint_authority = json.loads(endpoint_fixture_path.read_text(encoding="utf-8"))
    endpoint_token_ids = endpoint_authority.get("configuration", {}).get("token_ids")
    endpoint_semantic = endpoint_authority.get("semantic")
    if (
        endpoint_authority.get("model") != MODEL
        or not isinstance(endpoint_token_ids, list)
        or len(endpoint_token_ids) < 2
        or not isinstance(endpoint_semantic, str)
    ):
        raise ValueError("unsupported target endpoint authority for MTP prefill")
    endpoint_result = build_endpoint(
        checkpoint_dir,
        model_lock_path,
        _return_outputs=True,
        _token_ids=endpoint_token_ids,
        _semantic=endpoint_semantic,
    )
    if not isinstance(endpoint_result, tuple):
        raise AssertionError("endpoint target hiddens were not returned")
    endpoint, target_hiddens = endpoint_result
    if endpoint != json.loads(endpoint_fixture_path.read_text(encoding="utf-8")):
        raise ValueError("regenerated target endpoint disagrees with committed authority")
    target_steps = endpoint["output"]["steps"]
    bonus_token = target_steps[-1]["top20_token_ids"][0]
    mtp_token_ids = endpoint["configuration"]["token_ids"][1:] + [bonus_token]

    scheduler_lock = json.loads(scheduler_lock_path.read_text(encoding="utf-8"))
    if scheduler_lock.get("commit") != SGLANG_COMMIT:
        raise ValueError("unsupported EAGLE scheduler source lock")
    source_files = {item["path"]: item for item in scheduler_lock.get("files", [])}
    if source_files.get("python/sglang/srt/speculative/eagle_worker_v2.py", {}).get("sha256") != "9a66d31868385646b9fb9f78053730f55d2e885e72382a8c8dc6db9f07709271":
        raise ValueError("unsupported EAGLE prefill implementation")

    lock = load_model_lock(model_lock_path)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    fusion_authority = json.loads(fusion_fixture_path.read_text(encoding="utf-8"))
    values: dict[str, torch.Tensor] = {}
    tensor_records: dict[str, Any] = {}
    for key, (name, shape) in TENSORS.items():
        authority = fusion_authority["case"]["tensors"][key]
        with safe_open(checkpoint_dir / weight_map[name], framework="pt", device="cpu") as source:
            value = source.get_tensor(name).contiguous()
        if list(value.shape) != shape or value.dtype != torch.bfloat16 or capture_hash(value) != authority["payload_sha256"]:
            raise ValueError(f"MTP fusion authority mismatch for {key}")
        values[key] = value
        locked = locked_file(lock, weight_map[name])
        tensor_records[key] = {
            "tensor": name,
            "shape": shape,
            "shard": weight_map[name],
            "shard_bytes": locked["size"],
            "shard_sha256": locked["lfs_sha256"],
            "payload_sha256": capture_hash(value),
        }

    embedding_name = "model.language_model.embed_tokens.weight"
    with safe_open(checkpoint_dir / weight_map[embedding_name], framework="pt", device="cpu") as source:
        embedding_table = source.get_tensor(embedding_name)
        embeddings = [embedding_table[token].clone().contiguous() for token in mtp_token_ids]
    embedding_locked = locked_file(lock, weight_map[embedding_name])

    fused_outputs: list[torch.Tensor] = []
    steps = []
    with torch.no_grad():
        for ordinal, (token_id, embedding, target_hidden) in enumerate(
            zip(mtp_token_ids, embeddings, target_hiddens, strict=True)
        ):
            embedding = embedding.reshape(HIDDEN).contiguous()
            target_hidden = target_hidden.reshape(HC_HIDDEN).contiguous()
            embedding_normed = zero_centered_rms_norm(
                embedding, values["pre_fc_norm_embedding"], 1e-6
            )
            hidden_normed = zero_centered_rms_norm(
                target_hidden, values["pre_fc_norm_hidden"], 1e-6
            )
            embedding_projected = torch.nn.functional.linear(
                embedding_normed, values["fc_embedding"]
            ).contiguous()
            hidden_projected = torch.nn.functional.linear(
                hidden_normed.view(HC_COUNT, HIDDEN), values["fc_hidden"]
            ).contiguous()
            fused = (embedding_projected.unsqueeze(0) + hidden_projected).contiguous().view(1, 1, HC_HIDDEN)
            captures = {
                "embedding": embedding,
                "target_hidden": target_hidden,
                "embedding_normed": embedding_normed,
                "target_hidden_normed": hidden_normed,
                "embedding_projected": embedding_projected,
                "target_hidden_projected": hidden_projected,
                "fused_hidden": fused,
            }
            fused_outputs.append(fused)
            steps.append(
                {
                    "ordinal": ordinal,
                    "mtp_input_token_id": token_id,
                    "target_hidden_endpoint_ordinal": ordinal,
                    "captures": {name: capture_hash(value) for name, value in captures.items()},
                }
            )

    seed = {
        "schema_version": 1,
        "semantic": SEED_SEMANTIC,
        "model": MODEL,
        "revision": lock["revision"],
        "reference": {
            "implementation": "sglang_eagle_prefill_rotation_and_qwen4_exp_mtp_fusion",
            "commit": SGLANG_COMMIT,
            "mtp_source_lock_sha256": sha256_file(mtp_source_lock_path),
            "scheduler_source_lock_sha256": sha256_file(scheduler_lock_path),
            "endpoint_fixture_sha256": sha256_file(endpoint_fixture_path),
            "fusion_fixture_sha256": sha256_file(fusion_fixture_path),
            "model_lock_sha256": sha256_file(model_lock_path),
            "tensor_index_sha256": sha256_file(index_path),
        },
        "configuration": {
            "target_input_token_ids": endpoint["configuration"]["token_ids"],
            "target_next_token_id": bonus_token,
            "mtp_prefill_token_ids": mtp_token_ids,
            "mtp_positions": list(range(len(mtp_token_ids))),
            "cache_mode": "sequential_mtp_prefill",
            "boundary_dtype": "BF16",
        },
        "embedding": {
            "tensor": embedding_name,
            "shape": [248320, HIDDEN],
            "shard": weight_map[embedding_name],
            "shard_bytes": embedding_locked["size"],
            "shard_sha256": embedding_locked["lfs_sha256"],
        },
        "tensors": tensor_records,
        "steps": steps,
    }
    return seed, fused_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--mtp-source-lock", required=True, type=Path)
    parser.add_argument("--scheduler-lock", required=True, type=Path)
    parser.add_argument("--endpoint-fixture", required=True, type=Path)
    parser.add_argument("--fusion-fixture", required=True, type=Path)
    parser.add_argument("--seed-output", required=True, type=Path)
    parser.add_argument("--attention-output", required=True, type=Path)
    parser.add_argument("--decoder-output", required=True, type=Path)
    parser.add_argument("--logits-output", required=True, type=Path)
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()

    seed, fused = build_seed(
        checkpoint_dir,
        args.model_lock,
        args.mtp_source_lock,
        args.scheduler_lock,
        args.endpoint_fixture,
        args.fusion_fixture,
    )
    write_json(args.seed_output, seed)
    shared_refs = {
        "mtp_source_lock_sha256": sha256_file(args.mtp_source_lock),
        "scheduler_source_lock_sha256": sha256_file(args.scheduler_lock),
        "endpoint_fixture_sha256": sha256_file(args.endpoint_fixture),
        "causal_seed_fixture_sha256": sha256_file(args.seed_output),
    }
    attention, post_attention = build_attention(
        checkpoint_dir,
        args.model_lock,
        args.seed_output,
        _layer=0,
        _hidden_overrides=fused,
        _past_lengths=(0, 1),
        _modes=("mtp_prefill_initial", "mtp_prefill_cached"),
        _semantic=ATTENTION_SEMANTIC,
        _reference_hashes=shared_refs,
        _require_committed_parent=False,
        _sequential_cache=True,
        _layer_prefix=LAYER_PREFIX,
        _mtp_config=True,
        _return_outputs=True,
    )
    write_json(args.attention_output, attention)
    decoder_refs = {**shared_refs, "attention_residual_fixture_sha256": sha256_file(args.attention_output)}
    decoder_result = build_decoder(
        checkpoint_dir,
        args.model_lock,
        args.attention_output,
        args.attention_output,
        _parent_execution=(attention, post_attention),
        _parent_semantic=ATTENTION_SEMANTIC,
        _layer=0,
        _layer_type="full_attention",
        _semantic=DECODER_SEMANTIC,
        _reference_hashes=decoder_refs,
        _modes=("mtp_prefill_initial", "mtp_prefill_cached"),
        _require_committed_parent=False,
        _layer_prefix=LAYER_PREFIX,
        _mtp_config=True,
        _return_outputs=True,
    )
    if not isinstance(decoder_result, tuple):
        raise AssertionError("causal MTP decoder outputs were not returned")
    decoder, decoder_outputs = decoder_result
    write_json(args.decoder_output, decoder)
    output = build_output_fixture(
        checkpoint_dir,
        args.model_lock,
        args.mtp_source_lock,
        args.seed_output,
        args.decoder_output,
        decoder_outputs,
        _semantic=OUTPUT_SEMANTIC,
        _reference_hashes=shared_refs,
    )
    write_json(args.logits_output, output)
    print(json.dumps({
        "mtp_prefill_token_ids": seed["configuration"]["mtp_prefill_token_ids"],
        "proposal_token_id": output["steps"][-1]["top20_token_ids"][0],
        "outputs": [os.fspath(args.seed_output), os.fspath(args.attention_output), os.fspath(args.decoder_output), os.fspath(args.logits_output)],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
