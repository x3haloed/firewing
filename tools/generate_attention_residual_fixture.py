#!/usr/bin/env python3
"""Generate a real two-step layer-0 attention residual fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import transformers
from safetensors import safe_open
from transformers import DynamicCache
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextGatedResidual,
)

if __package__:
    from tools.generate_deltanet_fixture import (
        TENSOR_SHAPES as DELTANET_SHAPES,
        capture,
        explicit_step,
    )
    from tools.generate_hyper_connection_fixture import (
        EXPECTED_SHAPES as HYPER_SHAPES,
        LOCAL_TENSORS as HYPER_LOCAL_TENSORS,
    )
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, locked_file, sha256_file
else:
    from generate_deltanet_fixture import TENSOR_SHAPES as DELTANET_SHAPES, capture, explicit_step  # type: ignore[no-redef]
    from generate_hyper_connection_fixture import (  # type: ignore[no-redef]
        EXPECTED_SHAPES as HYPER_SHAPES,
        LOCAL_TENSORS as HYPER_LOCAL_TENSORS,
    )
    from generate_ngram_address_fixture import checkpoint_revision, load_model_lock, locked_file, sha256_file  # type: ignore[no-redef]

MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_layer0_attention_residual_cached_decode"
HIDDEN = 2560
HC_COUNT = 4
HC_HIDDEN = HIDDEN * HC_COUNT
INPUT_SPECS = [
    {"multiplier": 43, "add": 17, "modulus": 263, "center": 131, "divisor": 128, "sparse_stride": 1},
    {"multiplier": 61, "add": 29, "modulus": 277, "center": 138, "divisor": 128, "sparse_stride": 1},
]


def tensor_hash(value: torch.Tensor) -> str:
    return capture(value)["sha256"]


def make_hyper_input(spec: dict[str, int]) -> torch.Tensor:
    index = torch.arange(HC_HIDDEN, dtype=torch.int64)
    value = (index * spec["multiplier"] + spec["add"]) % spec["modulus"] - spec["center"]
    if spec["sparse_stride"] > 1:
        value = torch.where(index % spec["sparse_stride"] == 0, value, 0)
    return value.to(torch.float32).div(spec["divisor"]).to(torch.bfloat16).reshape(1, 1, HC_HIDDEN).contiguous()


def load_tensor(
    checkpoint_dir: Path,
    lock: dict[str, Any],
    weight_map: dict[str, str],
    tensor_name: str,
    shape: list[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    shard = weight_map[tensor_name]
    with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as source:
        value = source.get_tensor(tensor_name).contiguous()
    if value.dtype != torch.bfloat16 or list(value.shape) != shape:
        raise ValueError(f"unsupported tensor {tensor_name}")
    locked = locked_file(lock, shard)
    return value, {
        "tensor": tensor_name,
        "dtype": "BF16",
        "shape": shape,
        "shard": shard,
        "shard_bytes": locked["size"],
        "shard_sha256": locked["lfs_sha256"],
        "payload_sha256": tensor_hash(value),
    }


def build_fixture(
    checkpoint_dir: Path,
    model_lock_path: Path,
    hyper_fixture_path: Path,
    deltanet_fixture_path: Path,
    *,
    _layer: int = 0,
    _hidden_overrides: list[torch.Tensor] | None = None,
    _input_specs: list[dict[str, int]] | None = None,
    _semantic: str = SEMANTIC,
    _reference_hashes: dict[str, str] | None = None,
    _return_outputs: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], list[torch.Tensor]]:
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
        or raw_config["layer_types"][_layer] != "linear_attention"
        or _layer + 1 in raw_config["ple_layer_ids"]
    ):
        raise ValueError(f"unsupported layer-{_layer} attention composition")
    input_specs = INPUT_SPECS if _input_specs is None else _input_specs
    if not input_specs or (_hidden_overrides is not None and (
        len(_hidden_overrides) != len(input_specs)
        or any(
            value.dtype != torch.bfloat16
            or list(value.shape) != [1, 1, HC_HIDDEN]
            or not value.is_contiguous()
            for value in _hidden_overrides
        )
    )):
        raise ValueError("unsupported attention hidden-state overrides")
    config = Qwen4ExpTextConfig(**raw_config)
    hyper = Qwen4ExpTextGatedResidual(config).to(torch.bfloat16).eval()
    deltanet = Qwen4ExpTextGatedDeltaNet(config, _layer).to(torch.bfloat16).eval()
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    tensors: dict[str, Any] = {}

    hyper_state: dict[str, torch.Tensor] = {}
    for key, local_name in HYPER_LOCAL_TENSORS.items():
        tensor_name = f"model.language_model.layers.{_layer}.attn_hyper_connection.{local_name}"
        value, record = load_tensor(checkpoint_dir, lock, weight_map, tensor_name, HYPER_SHAPES[key])
        hyper_state[local_name] = value
        tensors[f"attn_hyper_connection.{key}"] = record
    hyper.load_state_dict(hyper_state, strict=True)

    deltanet_state: dict[str, torch.Tensor] = {}
    for local_name, shape in DELTANET_SHAPES.items():
        tensor_name = f"model.language_model.layers.{_layer}.linear_attn.{local_name}"
        value, record = load_tensor(checkpoint_dir, lock, weight_map, tensor_name, shape)
        deltanet_state[local_name] = value
        tensors[f"linear_attn.{local_name}"] = record
    deltanet.load_state_dict(deltanet_state, strict=True)

    explicit_cache = DynamicCache(config=config)
    official_cache = DynamicCache(config=config)
    steps = []
    composed_outputs = []
    for ordinal, spec in enumerate(input_specs):
        hyper_input = (
            make_hyper_input(spec)
            if _hidden_overrides is None
            else _hidden_overrides[ordinal]
        )
        with torch.no_grad():
            mixed_input, official_hyper_input, injection_weights = hyper(hyper_input)
            attention_output, delta_captures = explicit_step(
                deltanet, mixed_input, explicit_cache, cached=ordinal > 0
            )
            official_attention = deltanet(mixed_input, cache_params=official_cache)
        if not torch.equal(official_hyper_input, hyper_input):
            raise ValueError("gated residual did not preserve hyper input")
        if (
            not torch.equal(attention_output, official_attention)
            or not torch.equal(delta_captures["convolution_state"], official_cache.layers[_layer].conv_states[0])
            or not torch.equal(delta_captures["recurrent_state"], official_cache.layers[_layer].recurrent_states[0])
        ):
            raise ValueError(f"explicit DeltaNet step {ordinal} disagrees with official module")
        injection_products = (
            attention_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        ).contiguous()
        composed = (hyper_input + injection_products.flatten(-2)).contiguous()
        composed_outputs.append(composed)
        captures = {
            "hyper_input": hyper_input,
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
                "mode": "initial_chunk" if ordinal == 0 else "cached_recurrent",
                "input_spec": spec,
                "captures": {name: capture(value) for name, value in captures.items()},
            }
        )

    token_count_word = {2: "two", 3: "three", 4: "four"}.get(len(steps))
    if token_count_word is None:
        raise ValueError("attention residual fixture supports only two, three, or four token steps")
    fixture = {
        "schema_version": 1,
        "semantic": _semantic,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward",
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
            **(
                _reference_hashes
                if _reference_hashes is not None
                else {
                    "hyper_fixture_sha256": sha256_file(hyper_fixture_path),
                    "deltanet_fixture_sha256": sha256_file(deltanet_fixture_path),
                }
            ),
        },
        "configuration": {
            "layer": _layer,
            "layer_type": "linear_attention",
            "ple_applied": False,
            "hidden_size": HIDDEN,
            "hc_count": HC_COUNT,
            "boundary_dtype": "BF16",
            "recurrent_state_dtype": "F32",
        },
        "case": {
            "name": f"layer_{_layer}_{token_count_word}_token_attention_residual",
            "tensors": tensors,
            "steps": steps,
        },
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
    parser.add_argument("--hyper-fixture", required=True, type=Path)
    parser.add_argument("--deltanet-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(
        args.checkpoint_dir,
        args.model_lock,
        args.hyper_fixture,
        args.deltanet_fixture,
    )
    write_json(args.output, fixture)
    print(
        json.dumps(
            {
                "output": os.fspath(args.output),
                "tensors": len(fixture["case"]["tensors"]),
                "steps": len(fixture["case"]["steps"]),
                "captures_per_step": len(fixture["case"]["steps"][0]["captures"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
