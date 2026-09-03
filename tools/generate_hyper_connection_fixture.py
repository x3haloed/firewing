#!/usr/bin/env python3
"""Generate a real Qwen gated hyper-connection fixture without payload bytes."""

from __future__ import annotations

import argparse
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
    from tools.generate_expert_fixture import capture_hash
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, locked_file, sha256_file
else:
    from generate_expert_fixture import capture_hash  # type: ignore[no-redef]
    from generate_ngram_address_fixture import checkpoint_revision, load_model_lock, locked_file, sha256_file  # type: ignore[no-redef]

MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_real_gated_hyper_connection"
HIDDEN = 2560
HC_COUNT = 4
HC_LOWRANK = 320
HC_HIDDEN = HIDDEN * HC_COUNT
INPUT_SPEC = {"multiplier": 43, "add": 17, "modulus": 263, "center": 131, "divisor": 128, "sparse_stride": 1}
LOCAL_TENSORS = {
    "hc_norm": "hc_norm.weight",
    "input_mix_weight_down": "input_mix_weight_down.weight",
    "input_mix_weight_up": "input_mix_weight_up.weight",
    "block_inject_weight": "block_inject_weight.weight",
}
EXPECTED_SHAPES = {
    "hc_norm": [HC_HIDDEN],
    "input_mix_weight_down": [HC_LOWRANK, HC_HIDDEN],
    "input_mix_weight_up": [HC_HIDDEN, HC_LOWRANK],
    "block_inject_weight": [HC_COUNT, HC_HIDDEN],
}


def make_hyper_input() -> torch.Tensor:
    index = torch.arange(HC_HIDDEN, dtype=torch.int64)
    value = ((index * INPUT_SPEC["multiplier"] + INPUT_SPEC["add"]) % INPUT_SPEC["modulus"] - INPUT_SPEC["center"])
    return value.to(torch.float32).div(INPUT_SPEC["divisor"]).to(torch.bfloat16).contiguous()


def capture_forward(module: Qwen4ExpTextGatedResidual, hyper_input: torch.Tensor) -> dict[str, torch.Tensor]:
    hyper_input_normed = module.hc_norm(hyper_input).contiguous()
    mix_down = module.input_mix_weight_down(hyper_input_normed).contiguous()
    mix_down_scaled = (mix_down / module.hc_count).contiguous()
    mix_down_silu = torch.nn.functional.silu(mix_down_scaled).contiguous()
    mix_up = module.input_mix_weight_up(mix_down_silu).contiguous()
    input_mix_weight = torch.sigmoid(mix_up).contiguous()
    mixed_products = (
        input_mix_weight.unflatten(-1, (module.hc_count, module.hidden_size))
        * hyper_input_normed.unflatten(-1, (module.hc_count, module.hidden_size))
    ).contiguous()
    mixed_input = mixed_products.mean(dim=-2).contiguous()
    inject_projection = module.block_inject_weight(hyper_input_normed).contiguous()
    inject_scaled = (inject_projection / module.hc_count).contiguous()
    inject_sigmoid = torch.sigmoid(inject_scaled).contiguous()
    injection_weights = (2 * inject_sigmoid).contiguous()
    outputs = {
        "hyper_input": hyper_input, "hyper_input_normed": hyper_input_normed,
        "mix_down": mix_down, "mix_down_scaled": mix_down_scaled,
        "mix_down_silu": mix_down_silu, "mix_up": mix_up,
        "input_mix_weight": input_mix_weight, "mixed_products": mixed_products,
        "mixed_input": mixed_input, "inject_projection": inject_projection,
        "inject_scaled": inject_scaled, "inject_sigmoid": inject_sigmoid,
        "injection_weights": injection_weights,
    }
    if any(value.dtype != torch.bfloat16 or not value.is_contiguous() for value in outputs.values()):
        raise ValueError("hyper-connection did not preserve BF16 capture boundaries")
    with torch.no_grad():
        official_mixed, official_hyper, official_injection = module(hyper_input)
    if not torch.equal(official_mixed, mixed_input) or not torch.equal(official_hyper, hyper_input) or not torch.equal(official_injection, injection_weights):
        raise ValueError("explicit captures disagree with official gated-residual forward")
    return outputs


def build_fixture(checkpoint_dir: Path, model_lock_path: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")
    config_path = checkpoint_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    if raw_config["hidden_size"] != HIDDEN or raw_config["hc_count"] != HC_COUNT or raw_config["hc_lowrank"] != HC_LOWRANK or raw_config["rms_norm_eps"] != 1e-6:
        raise ValueError("unsupported Qwen gated hyper-connection configuration")
    module = Qwen4ExpTextGatedResidual(Qwen4ExpTextConfig(**raw_config)).to(torch.bfloat16).eval()
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    prefix = "model.language_model.layers.0.attn_hyper_connection"
    state: dict[str, torch.Tensor] = {}
    records = {}
    for key, local_name in LOCAL_TENSORS.items():
        tensor_name = f"{prefix}.{local_name}"
        shard = weight_map[tensor_name]
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as source:
            value = source.get_tensor(tensor_name).contiguous()
        if value.dtype != torch.bfloat16 or list(value.shape) != EXPECTED_SHAPES[key]:
            raise ValueError(f"unsupported tensor {tensor_name}")
        state[local_name] = value
        locked = locked_file(lock, shard)
        records[key] = {"tensor": tensor_name, "shape": EXPECTED_SHAPES[key], "shard": shard, "shard_bytes": locked["size"], "shard_sha256": locked["lfs_sha256"], "payload_sha256": capture_hash(value)}
    module.load_state_dict(state, strict=True)
    captures = capture_forward(module, make_hyper_input())
    return {
        "schema_version": 1, "semantic": SEMANTIC, "model": MODEL, "revision": revision,
        "reference": {
            "implementation": "huggingface_transformers_qwen4_exp", "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextGatedResidual.forward",
            "rms_source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextRMSNorm.forward",
            "config_sha256": sha256_file(config_path), "tensor_index_sha256": sha256_file(index_path), "model_lock_sha256": sha256_file(model_lock_path),
        },
        "configuration": {"hidden_size": HIDDEN, "hc_count": HC_COUNT, "hc_lowrank": HC_LOWRANK, "rms_norm_eps": 1e-6, "boundary_dtype": "BF16", "use_combine": True},
        "case": {"name": "layer_0_attention_affine_mod_hyper_connection", "layer": 0, "kind": "attn_hyper_connection", "input_spec": INPUT_SPEC, "tensors": records, "expected_bf16_sha256": {name: capture_hash(value) for name, value in captures.items()}},
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
    print(json.dumps({"output": os.fspath(args.output), "captures": len(fixture["case"]["expected_bf16_sha256"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
