#!/usr/bin/env python3
"""Generate a source-faithful four-row Qwen4-Exp recursive MTP chain."""

from __future__ import annotations

import argparse
import copy
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
    from tools.generate_mtp_causal_prefill_fixture import build_seed
    from tools.generate_mtp_decoder_fixture import build_output_fixture, write_json
    from tools.generate_mtp_input_fusion_fixture import HC_COUNT, HC_HIDDEN, HIDDEN, TENSORS, zero_centered_rms_norm
    from tools.generate_ngram_address_fixture import load_model_lock, sha256_file
else:
    from generate_expert_fixture import capture_hash  # type: ignore[no-redef]
    from generate_full_attention_residual_fixture import build_fixture as build_attention  # type: ignore[no-redef]
    from generate_full_decoder_layer3_fixture import build_fixture as build_decoder  # type: ignore[no-redef]
    from generate_mtp_causal_prefill_fixture import build_seed  # type: ignore[no-redef]
    from generate_mtp_decoder_fixture import build_output_fixture, write_json  # type: ignore[no-redef]
    from generate_mtp_input_fusion_fixture import HC_COUNT, HC_HIDDEN, HIDDEN, TENSORS, zero_centered_rms_norm  # type: ignore[no-redef]
    from generate_ngram_address_fixture import load_model_lock, sha256_file  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SGLANG_COMMIT = "78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385"
SEED_SEMANTIC = "qwen3_8_flash_next_recursive_mtp_fusion"
ATTENTION_SEMANTIC = "qwen3_8_flash_next_recursive_mtp_attention"
DECODER_SEMANTIC = "qwen3_8_flash_next_recursive_mtp_decoder"
OUTPUT_SEMANTIC = "qwen3_8_flash_next_recursive_mtp_logits"
LAYER_PREFIX = "mtp.layers.0"


def fuse(
    token_id: int,
    source_hidden: torch.Tensor,
    embedding_table: torch.Tensor,
    values: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, str]]:
    embedding = embedding_table[token_id].clone().contiguous().reshape(HIDDEN)
    source_hidden = source_hidden.reshape(HC_HIDDEN).contiguous()
    embedding_normed = zero_centered_rms_norm(
        embedding, values["pre_fc_norm_embedding"], 1e-6
    )
    source_hidden_normed = zero_centered_rms_norm(
        source_hidden, values["pre_fc_norm_hidden"], 1e-6
    )
    embedding_projected = torch.nn.functional.linear(
        embedding_normed, values["fc_embedding"]
    ).contiguous()
    source_hidden_projected = torch.nn.functional.linear(
        source_hidden_normed.view(HC_COUNT, HIDDEN), values["fc_hidden"]
    ).contiguous()
    fused = (
        embedding_projected.unsqueeze(0) + source_hidden_projected
    ).contiguous().view(1, 1, HC_HIDDEN)
    captures = {
        "embedding": embedding,
        "source_hidden": source_hidden,
        "embedding_normed": embedding_normed,
        "source_hidden_normed": source_hidden_normed,
        "embedding_projected": embedding_projected,
        "source_hidden_projected": source_hidden_projected,
        "fused_hidden": fused,
    }
    return fused, {name: capture_hash(value) for name, value in captures.items()}


def build_components(
    checkpoint_dir: Path,
    model_lock_path: Path,
    parent_path: Path,
    fused: list[torch.Tensor],
    refs: dict[str, str],
    prefill_positions: int,
) -> tuple[dict[str, Any], dict[str, Any], list[torch.Tensor]]:
    count = len(fused)
    modes = tuple(
        "mtp_prefill_initial"
        if ordinal == 0
        else "mtp_prefill_cached"
        if ordinal < prefill_positions
        else "mtp_recursive_cached"
        for ordinal in range(count)
    )
    attention, post_attention = build_attention(
        checkpoint_dir,
        model_lock_path,
        parent_path,
        _layer=0,
        _hidden_overrides=fused,
        _past_lengths=tuple(range(count)),
        _modes=modes,
        _semantic=ATTENTION_SEMANTIC,
        _reference_hashes=refs,
        _require_committed_parent=False,
        _sequential_cache=True,
        _layer_prefix=LAYER_PREFIX,
        _mtp_config=True,
        _return_outputs=True,
    )
    decoder_result = build_decoder(
        checkpoint_dir,
        model_lock_path,
        parent_path,
        parent_path,
        _parent_execution=(attention, post_attention),
        _parent_semantic=ATTENTION_SEMANTIC,
        _layer=0,
        _layer_type="full_attention",
        _semantic=DECODER_SEMANTIC,
        _reference_hashes=refs,
        _modes=modes,
        _require_committed_parent=False,
        _layer_prefix=LAYER_PREFIX,
        _mtp_config=True,
        _return_outputs=True,
    )
    if not isinstance(decoder_result, tuple):
        raise AssertionError("recursive MTP decoder outputs were not returned")
    decoder, outputs = decoder_result
    return attention, decoder, outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--mtp-source-lock", required=True, type=Path)
    parser.add_argument("--scheduler-lock", required=True, type=Path)
    parser.add_argument("--recursive-lock", required=True, type=Path)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--endpoint-fixture", required=True, type=Path)
    parser.add_argument("--fusion-fixture", required=True, type=Path)
    parser.add_argument("--seed-output", required=True, type=Path)
    parser.add_argument("--attention-output", required=True, type=Path)
    parser.add_argument("--decoder-output", required=True, type=Path)
    parser.add_argument("--logits-output", required=True, type=Path)
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    if not 2 <= args.q <= 8:
        raise ValueError("recursive verification width must be between 2 and 8")

    recursive_lock = json.loads(args.recursive_lock.read_text(encoding="utf-8"))
    if recursive_lock.get("commit") != SGLANG_COMMIT:
        raise ValueError("unsupported recursive EAGLE source lock")
    files = {item["path"]: item for item in recursive_lock.get("files", [])}
    if (
        files.get("python/sglang/srt/speculative/eagle_worker_v2.py", {}).get("sha256")
        != "9a66d31868385646b9fb9f78053730f55d2e885e72382a8c8dc6db9f07709271"
        or files.get("python/sglang/srt/speculative/eagle_worker_common.py", {}).get("sha256")
        != "7d5bc17da41ad34230dfd76da34024496983eae5453f8b1c650a9f5f924e4934"
    ):
        raise ValueError("recursive EAGLE source identity mismatch")

    base_seed, fused = build_seed(
        checkpoint_dir,
        args.model_lock,
        args.mtp_source_lock,
        args.scheduler_lock,
        args.endpoint_fixture,
        args.fusion_fixture,
    )
    lock = load_model_lock(args.model_lock)
    weight_map = json.loads(
        (checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    values: dict[str, torch.Tensor] = {}
    for key, (name, _) in TENSORS.items():
        with safe_open(checkpoint_dir / weight_map[name], framework="pt", device="cpu") as source:
            values[key] = source.get_tensor(name).contiguous()
    embedding_name = "model.language_model.embed_tokens.weight"
    with safe_open(checkpoint_dir / weight_map[embedding_name], framework="pt", device="cpu") as source:
        embedding_table = source.get_tensor(embedding_name)

        recursive_steps = []
        for ordinal, step in enumerate(base_seed["steps"]):
            captures = dict(step["captures"])
            captures["source_hidden"] = captures.pop("target_hidden")
            captures["source_hidden_normed"] = captures.pop("target_hidden_normed")
            captures["source_hidden_projected"] = captures.pop("target_hidden_projected")
            recursive_steps.append(
                {
                    "ordinal": ordinal,
                    "mtp_input_token_id": step["mtp_input_token_id"],
                    "hidden_source_kind": "target_endpoint",
                    "hidden_source_ordinal": step["target_hidden_endpoint_ordinal"],
                    "captures": captures,
                }
            )

        token_ids = list(base_seed["configuration"]["mtp_prefill_token_ids"])
        prefill_positions = len(token_ids)
        total_positions = prefill_positions + args.q - 2
        provisional_refs = {"recursive_source_lock_sha256": sha256_file(args.recursive_lock)}
        for ordinal in range(prefill_positions, total_positions):
            _, provisional_decoder, decoder_outputs = build_components(
                checkpoint_dir,
                args.model_lock,
                args.endpoint_fixture,
                fused,
                provisional_refs,
                prefill_positions,
            )
            provisional_output = build_output_fixture(
                checkpoint_dir,
                args.model_lock,
                args.mtp_source_lock,
                args.endpoint_fixture,
                args.endpoint_fixture,
                decoder_outputs,
                _semantic=OUTPUT_SEMANTIC,
                _reference_hashes=provisional_refs,
            )
            next_input = provisional_output["steps"][-1]["top20_token_ids"][0]
            fused_hidden, captures = fuse(
                next_input, decoder_outputs[-1], embedding_table, values
            )
            token_ids.append(next_input)
            fused.append(fused_hidden)
            recursive_steps.append(
                {
                    "ordinal": ordinal,
                    "mtp_input_token_id": next_input,
                    "hidden_source_kind": "draft_decoder",
                    "hidden_source_ordinal": ordinal - 1,
                    "captures": captures,
                }
            )

    seed = copy.deepcopy(base_seed)
    seed["semantic"] = SEED_SEMANTIC
    seed["reference"]["implementation"] = "sglang_topk1_recursive_eagle_and_qwen4_exp_mtp_fusion"
    seed["reference"]["recursive_source_lock_sha256"] = sha256_file(args.recursive_lock)
    seed["configuration"] = {
        "target_input_token_ids": base_seed["configuration"]["target_input_token_ids"],
        "target_next_token_id": base_seed["configuration"]["target_next_token_id"],
        "mtp_input_token_ids": token_ids,
        "mtp_positions": list(range(total_positions)),
        "prefill_positions": prefill_positions,
        "recursive_positions": args.q - 2,
        "cache_mode": "sequential_mtp_prefill_then_recursive_decode",
        "boundary_dtype": "BF16",
    }
    seed["steps"] = recursive_steps
    write_json(args.seed_output, seed)

    shared_refs = {
        "mtp_source_lock_sha256": sha256_file(args.mtp_source_lock),
        "scheduler_source_lock_sha256": sha256_file(args.scheduler_lock),
        "recursive_source_lock_sha256": sha256_file(args.recursive_lock),
        "endpoint_fixture_sha256": sha256_file(args.endpoint_fixture),
        "recursive_seed_fixture_sha256": sha256_file(args.seed_output),
    }
    attention, decoder, decoder_outputs = build_components(
        checkpoint_dir,
        args.model_lock,
        args.seed_output,
        fused,
        shared_refs,
        prefill_positions,
    )
    write_json(args.attention_output, attention)
    decoder["reference"]["attention_residual_fixture_sha256"] = sha256_file(args.attention_output)
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
    proposals = [step["top20_token_ids"][0] for step in output["steps"]]
    print(
        json.dumps(
            {
                "mtp_input_token_ids": token_ids,
                "top1_token_ids": proposals,
                "proposal_vector": [base_seed["configuration"]["target_next_token_id"]]
                + proposals[prefill_positions - 1 :],
                "outputs": [
                    os.fspath(args.seed_output),
                    os.fspath(args.attention_output),
                    os.fspath(args.decoder_output),
                    os.fspath(args.logits_output),
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
