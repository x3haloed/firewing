#!/usr/bin/env python3
"""Generate source-pinned real-weight Qwen4-Exp MTP decoder fixtures."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

if __package__:
    from tools.generate_expert_fixture import capture_hash
    from tools.generate_full_attention_residual_fixture import build_fixture as build_attention
    from tools.generate_full_decoder_layer3_fixture import build_fixture as build_decoder
    from tools.generate_mtp_input_fusion_fixture import (
        HC_COUNT,
        HC_HIDDEN,
        HIDDEN,
        INPUT_SPECS,
        SGLANG_COMMIT,
        TENSORS,
        make_input,
        zero_centered_rms_norm,
    )
    from tools.generate_ngram_address_fixture import load_model_lock, sha256_file
    from tools.generate_text_output_fixture import HC_LOWRANK, MIXER_TENSORS, VOCAB, tensor_record
else:
    from generate_expert_fixture import capture_hash  # type: ignore[no-redef]
    from generate_full_attention_residual_fixture import build_fixture as build_attention  # type: ignore[no-redef]
    from generate_full_decoder_layer3_fixture import build_fixture as build_decoder  # type: ignore[no-redef]
    from generate_mtp_input_fusion_fixture import (  # type: ignore[no-redef]
        HC_COUNT,
        HC_HIDDEN,
        HIDDEN,
        INPUT_SPECS,
        SGLANG_COMMIT,
        TENSORS,
        make_input,
        zero_centered_rms_norm,
    )
    from generate_ngram_address_fixture import load_model_lock, sha256_file  # type: ignore[no-redef]
    from generate_text_output_fixture import HC_LOWRANK, MIXER_TENSORS, VOCAB, tensor_record  # type: ignore[no-redef]

ATTENTION_SEMANTIC = "qwen3_8_flash_next_mtp_full_attention_residual"
DECODER_SEMANTIC = "qwen3_8_flash_next_mtp_complete_decoder"
OUTPUT_SEMANTIC = "qwen3_8_flash_next_mtp_shared_head_logits"
LAYER_PREFIX = "mtp.layers.0"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reproduce_fused_hidden(checkpoint_dir: Path, fusion_fixture_path: Path) -> list[torch.Tensor]:
    fixture = json.loads(fusion_fixture_path.read_text(encoding="utf-8"))
    index = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    values: dict[str, torch.Tensor] = {}
    for key, (name, shape) in TENSORS.items():
        record = fixture["case"]["tensors"].get(key)
        if record is None or record["tensor"] != name or record["shape"] != shape:
            raise ValueError(f"MTP fusion fixture tensor mismatch for {key}")
        with safe_open(checkpoint_dir / index[name], framework="pt", device="cpu") as source:
            value = source.get_tensor(name).contiguous()
        if capture_hash(value) != record["payload_sha256"]:
            raise ValueError(f"MTP fusion tensor payload mismatch for {key}")
        values[key] = value

    def fuse(case: dict[str, object]) -> torch.Tensor:
        specs = case["input_specs"]
        assert isinstance(specs, dict)
        embedding = make_input(HIDDEN, specs["embedding"])
        target_hidden = make_input(HC_HIDDEN, specs["target_hidden"])
        embedding_normed = zero_centered_rms_norm(
            embedding, values["pre_fc_norm_embedding"], 1e-6
        )
        hidden_normed = zero_centered_rms_norm(
            target_hidden, values["pre_fc_norm_hidden"], 1e-6
        )
        embedding_projected = torch.nn.functional.linear(
            embedding_normed, values["fc_embedding"]
        )
        hidden_projected = torch.nn.functional.linear(
            hidden_normed.view(HC_COUNT, HIDDEN), values["fc_hidden"]
        )
        fused = (
            embedding_projected.unsqueeze(0) + hidden_projected
        ).contiguous().view(1, 1, HC_HIDDEN)
        captures = case["expected_bf16_sha256"]
        assert isinstance(captures, dict)
        if capture_hash(fused) != captures["fused_hidden"]:
            raise ValueError("reproduced MTP fused hidden does not match its committed authority")
        return fused

    sequence = fixture.get("sequence_case")
    if not isinstance(sequence, dict):
        raise ValueError("MTP fusion fixture lacks the second sequential case")
    return [fuse(fixture["case"]), fuse(sequence)]


def build_output_fixture(
    checkpoint_dir: Path,
    model_lock_path: Path,
    source_lock_path: Path,
    fusion_fixture_path: Path,
    decoder_fixture_path: Path,
    decoder_outputs: list[torch.Tensor],
) -> dict[str, object]:
    lock = load_model_lock(model_lock_path)
    config_path = checkpoint_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    module = Qwen4ExpTextGatedResidual(
        Qwen4ExpTextConfig(**raw_config), use_combine=False
    ).to(torch.bfloat16).eval()
    state: dict[str, torch.Tensor] = {}
    records: dict[str, object] = {}
    mixer_prefix = "mtp.hyper_connection_mixer"
    for key, (local_name, shape) in MIXER_TENSORS.items():
        name = f"{mixer_prefix}.{local_name}"
        with safe_open(checkpoint_dir / weight_map[name], framework="pt", device="cpu") as source:
            value = source.get_tensor(name).contiguous()
        state[local_name] = value
        records[key] = tensor_record(lock, weight_map, name, shape, value)
    module.load_state_dict(state, strict=True)

    head_name = "lm_head.weight"
    with safe_open(checkpoint_dir / weight_map[head_name], framework="pt", device="cpu") as source:
        head = source.get_tensor(head_name).contiguous()
    records["lm_head"] = tensor_record(lock, weight_map, head_name, [VOCAB, HIDDEN], head)

    steps = []
    with torch.no_grad():
        for ordinal, hidden in enumerate(decoder_outputs):
            hidden = hidden.reshape(1, 1, HC_HIDDEN).contiguous()
            normalized = module.hc_norm(hidden).contiguous()
            down = module.input_mix_weight_down(normalized).contiguous()
            scaled = (down / module.hc_count).contiguous()
            activated = torch.nn.functional.silu(scaled).contiguous()
            up = module.input_mix_weight_up(activated).contiguous()
            mix_weight = torch.sigmoid(up).contiguous()
            products = (
                mix_weight.unflatten(-1, (HC_COUNT, HIDDEN))
                * normalized.unflatten(-1, (HC_COUNT, HIDDEN))
            ).contiguous()
            mixed = products.mean(dim=-2).contiguous()
            if not torch.equal(module(hidden).contiguous(), mixed):
                raise ValueError("explicit MTP final mixer disagrees with source forward")
            logits = torch.nn.functional.linear(mixed, head).contiguous()
            values, indices = torch.topk(logits.reshape(-1), 20, sorted=True)
            cutoff = values[-1]
            strictly_above = torch.nonzero(logits.reshape(-1) > cutoff).reshape(-1)
            cutoff_ties = torch.nonzero(logits.reshape(-1) == cutoff).reshape(-1)
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
                    "captures": {name: capture_hash(value) for name, value in captures.items()},
                    "top20_token_ids": indices.tolist(),
                    "top20_logit_bf16_u16": values.view(torch.uint16).tolist(),
                    "top20_cutoff_bf16_u16": cutoff.view(torch.uint16).item(),
                    "strictly_above_cutoff_token_ids": strictly_above.tolist(),
                    "cutoff_tie_token_ids": cutoff_ties.tolist(),
                }
            )
            del logits
    del head
    gc.collect()
    return {
        "schema_version": 1,
        "semantic": OUTPUT_SEMANTIC,
        "model": "Qwen/Qwen3.8-Flash-Next",
        "revision": lock["revision"],
        "reference": {
            "implementation": "sglang_qwen4_exp_mtp_source_derived_and_huggingface_components",
            "commit": SGLANG_COMMIT,
            "source_lock_sha256": sha256_file(source_lock_path),
            "mtp_input_fusion_fixture_sha256": sha256_file(fusion_fixture_path),
            "decoder_fixture_sha256": sha256_file(decoder_fixture_path),
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
            "shared_head_tensor": head_name,
        },
        "configuration": {
            "hidden_size": HIDDEN,
            "hc_count": HC_COUNT,
            "hc_lowrank": HC_LOWRANK,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "boundary_dtype": "BF16",
        },
        "tensors": records,
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--fusion-fixture", required=True, type=Path)
    parser.add_argument("--attention-output", required=True, type=Path)
    parser.add_argument("--decoder-output", required=True, type=Path)
    parser.add_argument("--output-fixture", required=True, type=Path)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    source_lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    if source_lock.get("commit") != SGLANG_COMMIT:
        raise ValueError("unsupported MTP source lock")
    load_model_lock(args.model_lock)
    fused = reproduce_fused_hidden(checkpoint_dir, args.fusion_fixture)
    reference_hashes = {
        "source_lock_sha256": sha256_file(args.source_lock),
        "mtp_input_fusion_fixture_sha256": sha256_file(args.fusion_fixture),
    }
    attention, post_attention = build_attention(
        checkpoint_dir,
        args.model_lock,
        args.fusion_fixture,
        _layer=0,
        _hidden_overrides=fused,
        _past_lengths=(0, 1),
        _modes=("mtp_initial", "mtp_cached_decode"),
        _semantic=ATTENTION_SEMANTIC,
        _reference_hashes=reference_hashes,
        _require_committed_parent=False,
        _sequential_cache=True,
        _layer_prefix=LAYER_PREFIX,
        _mtp_config=True,
        _return_outputs=True,
    )
    write_json(args.attention_output, attention)
    decoder_reference = {
        **reference_hashes,
        "attention_residual_fixture_sha256": sha256_file(args.attention_output),
    }
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
        _reference_hashes=decoder_reference,
        _modes=("mtp_initial", "mtp_cached_decode"),
        _require_committed_parent=False,
        _layer_prefix=LAYER_PREFIX,
        _mtp_config=True,
        _return_outputs=True,
    )
    if not isinstance(decoder_result, tuple):
        raise AssertionError("MTP decoder outputs were not returned")
    decoder, decoder_outputs = decoder_result
    write_json(args.decoder_output, decoder)
    output = build_output_fixture(
        checkpoint_dir,
        args.model_lock,
        args.source_lock,
        args.fusion_fixture,
        args.decoder_output,
        decoder_outputs,
    )
    write_json(args.output_fixture, output)
    print(
        json.dumps(
            {
                "attention_output": os.fspath(args.attention_output),
                "decoder_output": os.fspath(args.decoder_output),
                "output_fixture": os.fspath(args.output_fixture),
                "attention_cases": len(attention["cases"]),
                "decoder_steps": len(decoder["steps"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
