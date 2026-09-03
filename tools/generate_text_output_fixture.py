#!/usr/bin/env python3
"""Generate the source-derived final mixer and LM-head authority for FW-0028."""

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
    from tools.generate_accumulated_layers4_47_fixture import build_fixture as build_decoder
    from tools.generate_expert_fixture import capture_hash
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, locked_file, sha256_file
else:
    from generate_accumulated_layers4_47_fixture import build_fixture as build_decoder  # type: ignore[no-redef]
    from generate_expert_fixture import capture_hash  # type: ignore[no-redef]
    from generate_ngram_address_fixture import checkpoint_revision, load_model_lock, locked_file, sha256_file  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_accumulated_decoder_final_mixer_logits"
HIDDEN = 2560
HC_COUNT = 4
HC_HIDDEN = HIDDEN * HC_COUNT
HC_LOWRANK = 320
VOCAB = 248_320
ROOT = Path(__file__).resolve().parents[1]

MIXER_TENSORS = {
    "hc_norm": ("hc_norm.weight", [HC_HIDDEN]),
    "input_mix_weight_down": ("input_mix_weight_down.weight", [HC_LOWRANK, HC_HIDDEN]),
    "input_mix_weight_up": ("input_mix_weight_up.weight", [HC_HIDDEN, HC_LOWRANK]),
}


def parent_paths() -> list[Path]:
    fixture = ROOT / "fixtures"
    return [
        fixture / "ngram/qwen3_8_flash_next.json",
        fixture / "ngram/qwen3_8_flash_next_row_hashes.json",
        fixture / "hyper_connection/qwen3_8_flash_next_layer0.json",
        fixture / "deltanet/qwen3_8_flash_next_layer0_decode.json",
        fixture / "attention_residual/qwen3_8_flash_next_layer0.json",
        fixture / "sparse_moe/qwen3_8_flash_next_layer0.json",
        fixture / "decoder_layer/qwen3_8_flash_next_layer0.json",
        fixture / "ple/qwen3_8_flash_next_layer1_decode.json",
        fixture / "attention_residual/qwen3_8_flash_next_layer1_ple.json",
        fixture / "decoder_layer/qwen3_8_flash_next_layer1_ple.json",
        fixture / "accumulated/qwen3_8_flash_next_layers0_1.json",
        fixture / "accumulated/qwen3_8_flash_next_layer2.json",
        fixture / "accumulated/qwen3_8_flash_next_layer3.json",
        fixture / "full_attention/qwen3_8_flash_next_layer3.json",
        fixture / "attention_residual/qwen3_8_flash_next_layer3.json",
    ]


def tensor_record(
    lock: dict[str, Any],
    weight_map: dict[str, str],
    name: str,
    shape: list[int],
    value: torch.Tensor,
) -> dict[str, Any]:
    shard = weight_map[name]
    locked = locked_file(lock, shard)
    if value.dtype != torch.bfloat16 or list(value.shape) != shape:
        raise ValueError(f"unsupported tensor {name}")
    return {
        "tensor": name,
        "shape": shape,
        "shard": shard,
        "shard_bytes": locked["size"],
        "shard_sha256": locked["lfs_sha256"],
        "payload_sha256": capture_hash(value),
    }


def build_fixture(checkpoint_dir: Path, model_lock_path: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")
    config_path = checkpoint_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    if (
        raw_config["hidden_size"] != HIDDEN
        or raw_config["hc_count"] != HC_COUNT
        or raw_config["hc_lowrank"] != HC_LOWRANK
        or raw_config["vocab_size"] != VOCAB
        or raw_config["tie_word_embeddings"] is not False
        or raw_config["rms_norm_eps"] != 1e-6
    ):
        raise ValueError("unsupported text output configuration")

    decoder_result = build_decoder(
        checkpoint_dir,
        model_lock_path,
        *parent_paths(),
        _return_outputs=True,
    )
    if not isinstance(decoder_result, tuple):
        raise AssertionError("decoder outputs were not returned")
    decoder_fixture, decoder_outputs = decoder_result
    committed_decoder = ROOT / "fixtures/accumulated/qwen3_8_flash_next_layers4_47.json"
    if decoder_fixture != json.loads(committed_decoder.read_text(encoding="utf-8")):
        raise ValueError("regenerated decoder stack disagrees with committed fixture")

    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
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
    records["lm_head"] = tensor_record(
        lock, weight_map, head_name, [VOCAB, HIDDEN], head
    )

    steps = []
    with torch.no_grad():
        for ordinal, hidden in enumerate(decoder_outputs):
            hidden = hidden.reshape(1, 1, HC_HIDDEN).contiguous()
            # capture_forward contains the exact source expression, but assumes the
            # injecting variant. The final mixer is the same prefix through products.
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
            official = module(hidden).contiguous()
            if not torch.equal(official, mixed):
                raise ValueError("explicit final-mixer captures disagree with source forward")
            logits = torch.nn.functional.linear(mixed, head).contiguous()
            if logits.dtype != torch.bfloat16 or list(logits.shape) != [1, 1, VOCAB]:
                raise ValueError("LM head did not preserve the expected BF16 boundary")
            flat_logits = logits.reshape(-1)
            values, indices = torch.topk(flat_logits, 20, sorted=True)
            cutoff = values[-1]
            strictly_above = torch.nonzero(flat_logits > cutoff).reshape(-1)
            cutoff_ties = torch.nonzero(flat_logits == cutoff).reshape(-1)
            if not (len(strictly_above) < 20 <= len(strictly_above) + len(cutoff_ties)):
                raise ValueError("invalid top-20 cutoff partition")
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
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "source_derived_and_official_huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "Qwen4ExpTextModel.forward; Qwen4ExpTextGatedResidual.forward; Qwen4ExpForCausalLM.forward",
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
            "decoder_fixture_sha256": sha256_file(committed_decoder),
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
    print(json.dumps({"output": os.fspath(args.output), "steps": len(fixture["steps"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
