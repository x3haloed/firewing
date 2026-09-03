#!/usr/bin/env python3
"""Generate a two-step layer-1 PLE plus attention residual fixture."""

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
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextGatedResidual,
)

if __package__:
    from tools.generate_attention_residual_fixture import load_tensor
    from tools.generate_deltanet_fixture import TENSOR_SHAPES as DELTANET_SHAPES, capture, explicit_step
    from tools.generate_hyper_connection_fixture import (
        EXPECTED_SHAPES as HYPER_SHAPES,
        LOCAL_TENSORS as HYPER_LOCAL_TENSORS,
    )
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, sha256_file
    from tools.generate_ple_fixture import build_fixture as build_ple_fixture
else:
    from generate_attention_residual_fixture import load_tensor  # type: ignore[no-redef]
    from generate_deltanet_fixture import TENSOR_SHAPES as DELTANET_SHAPES, capture, explicit_step  # type: ignore[no-redef]
    from generate_hyper_connection_fixture import (  # type: ignore[no-redef]
        EXPECTED_SHAPES as HYPER_SHAPES,
        LOCAL_TENSORS as HYPER_LOCAL_TENSORS,
    )
    from generate_ngram_address_fixture import checkpoint_revision, load_model_lock, sha256_file  # type: ignore[no-redef]
    from generate_ple_fixture import build_fixture as build_ple_fixture  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_layer1_ple_attention_residual_cached_decode"
HIDDEN = 2560
HC_COUNT = 4
HC_HIDDEN = HIDDEN * HC_COUNT


def build_fixture(
    checkpoint_dir: Path,
    model_lock_path: Path,
    ngram_fixture_path: Path,
    ngram_row_fixture_path: Path,
    ple_fixture_path: Path,
    *,
    _return_outputs: bool = False,
    _hidden_overrides: list[torch.Tensor] | None = None,
    _token_ids: list[int] | None = None,
    _return_chain: bool = False,
) -> (
    dict[str, Any]
    | tuple[dict[str, Any], list[torch.Tensor]]
    | tuple[dict[str, Any], list[torch.Tensor], dict[str, Any]]
):
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")

    ple_result = build_ple_fixture(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        _return_outputs=True,
        _hidden_overrides=_hidden_overrides,
        _token_ids=_token_ids,
    )
    if not isinstance(ple_result, tuple):
        raise AssertionError("PLE execution outputs were not returned")
    regenerated_ple, ple_outputs = ple_result
    committed_ple = json.loads(ple_fixture_path.read_text(encoding="utf-8"))
    if _hidden_overrides is None and regenerated_ple != committed_ple:
        raise ValueError("regenerated PLE parent disagrees with committed fixture")

    config_path = checkpoint_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    if (
        raw_config["hidden_size"] != HIDDEN
        or raw_config["hc_count"] != HC_COUNT
        or raw_config["layer_types"][1] != "linear_attention"
        or raw_config["ple_layer_ids"] != [2]
    ):
        raise ValueError("unsupported layer-1 PLE attention composition")
    config = Qwen4ExpTextConfig(**raw_config)
    hyper = Qwen4ExpTextGatedResidual(config).to(torch.bfloat16).eval()
    deltanet = Qwen4ExpTextGatedDeltaNet(config, 1).to(torch.bfloat16).eval()
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    tensors: dict[str, Any] = {}

    hyper_state: dict[str, torch.Tensor] = {}
    for key, local_name in HYPER_LOCAL_TENSORS.items():
        tensor_name = f"model.language_model.layers.1.attn_hyper_connection.{local_name}"
        value, record = load_tensor(checkpoint_dir, lock, weight_map, tensor_name, HYPER_SHAPES[key])
        hyper_state[local_name] = value
        tensors[f"attn_hyper_connection.{key}"] = record
    hyper.load_state_dict(hyper_state, strict=True)

    deltanet_state: dict[str, torch.Tensor] = {}
    for local_name, shape in DELTANET_SHAPES.items():
        tensor_name = f"model.language_model.layers.1.linear_attn.{local_name}"
        value, record = load_tensor(checkpoint_dir, lock, weight_map, tensor_name, shape)
        deltanet_state[local_name] = value
        tensors[f"linear_attn.{local_name}"] = record
    deltanet.load_state_dict(deltanet_state, strict=True)

    explicit_cache = DynamicCache(config=config)
    official_cache = DynamicCache(config=config)
    steps = []
    composed_outputs = []
    for ordinal, (ple_step, ple_output) in enumerate(
        zip(regenerated_ple["case"]["steps"], ple_outputs, strict=True)
    ):
        # The parent generator already froze and checked this deterministic input.
        spec = ple_step["input_spec"]
        if _hidden_overrides is None:
            from_input = torch.arange(HC_HIDDEN, dtype=torch.int64)
            hidden = (
                ((from_input * spec["multiplier"] + spec["add"]) % spec["modulus"] - spec["center"])
                .to(torch.float32)
                .div(spec["divisor"])
                .to(torch.bfloat16)
                .reshape(1, 1, HC_HIDDEN)
                .contiguous()
            )
        else:
            hidden = _hidden_overrides[ordinal]
        post_ple = (hidden + ple_output).contiguous()
        with torch.no_grad():
            mixed_input, official_hyper_input, injection_weights = hyper(post_ple)
            attention_output, delta_captures = explicit_step(
                deltanet, mixed_input, explicit_cache, cached=ordinal > 0
            )
            official_attention = deltanet(mixed_input, cache_params=official_cache)
        if not torch.equal(official_hyper_input, post_ple):
            raise ValueError("gated residual did not preserve post-PLE input")
        if (
            not torch.equal(attention_output, official_attention)
            or not torch.equal(delta_captures["convolution_state"], official_cache.layers[1].conv_states[0])
            or not torch.equal(delta_captures["recurrent_state"], official_cache.layers[1].recurrent_states[0])
        ):
            raise ValueError(f"explicit layer-1 DeltaNet step {ordinal} disagrees with official module")
        injection_products = (
            attention_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        ).contiguous()
        composed = (official_hyper_input + injection_products.flatten(-2)).contiguous()
        composed_outputs.append(composed)
        captures = {
            "hidden_states": hidden,
            "ple_output": ple_output,
            "post_ple": post_ple,
            "mixed_input": mixed_input,
            "injection_weights": injection_weights,
            "attention_output": attention_output,
            "convolution_state": delta_captures["convolution_state"],
            "recurrent_state": delta_captures["recurrent_state"],
            "injection_products": injection_products,
            "composed_output": composed,
        }
        steps.append(
            {
                "ordinal": ordinal,
                "mode": ple_step["mode"],
                "token_id": ple_step["token_id"],
                "input_spec": spec,
                "captures": {name: capture(value) for name, value in captures.items()},
            }
        )

    fixture = {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward",
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
            "ple_fixture_sha256": sha256_file(ple_fixture_path),
            "ngram_fixture_sha256": sha256_file(ngram_fixture_path),
            "ngram_row_fixture_sha256": sha256_file(ngram_row_fixture_path),
        },
        "configuration": {
            "layer": 1,
            "layer_type": "linear_attention",
            "ple_applied": True,
            "hidden_size": HIDDEN,
            "hc_count": HC_COUNT,
            "boundary_dtype": "BF16",
            "recurrent_state_dtype": "F32",
        },
        "case": {
            "name": f"layer_1_{'two' if len(steps) == 2 else 'three'}_token_ple_attention_residual",
            "tensors": tensors,
            "steps": steps,
        },
    }
    if _return_chain:
        return fixture, composed_outputs, regenerated_ple
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
    parser.add_argument("--ngram-fixture", required=True, type=Path)
    parser.add_argument("--ngram-row-fixture", required=True, type=Path)
    parser.add_argument("--ple-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(
        args.checkpoint_dir,
        args.model_lock,
        args.ngram_fixture,
        args.ngram_row_fixture,
        args.ple_fixture,
    )
    write_json(args.output, fixture)
    print(json.dumps({
        "output": os.fspath(args.output),
        "tensors": len(fixture["case"]["tensors"]),
        "steps": len(fixture["case"]["steps"]),
        "captures_per_step": len(fixture["case"]["steps"][0]["captures"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
