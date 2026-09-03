#!/usr/bin/env python3
"""Generate a real two-step Qwen Gated DeltaNet fixture without payload bytes."""

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
    causal_conv1d_fn,
    causal_conv1d_update,
    l2norm,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

if __package__:
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, locked_file, sha256_file
else:
    from generate_ngram_address_fixture import checkpoint_revision, load_model_lock, locked_file, sha256_file  # type: ignore[no-redef]

MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_real_gated_deltanet_cached_decode"
HIDDEN = 2560
K_HEADS = 16
V_HEADS = 48
HEAD_DIM = 128
KEY_DIM = K_HEADS * HEAD_DIM
VALUE_DIM = V_HEADS * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
CONV_KERNEL = 4
INPUT_SPECS = [
    {"multiplier": 47, "add": 23, "modulus": 269, "center": 134, "divisor": 128, "sparse_stride": 1},
    {"multiplier": 59, "add": 31, "modulus": 271, "center": 135, "divisor": 128, "sparse_stride": 1},
]
TENSOR_SHAPES = {
    "A_log": [V_HEADS],
    "conv1d.weight": [CONV_DIM, 1, CONV_KERNEL],
    "dt_bias": [V_HEADS],
    "in_proj_a.weight": [V_HEADS, HIDDEN],
    "in_proj_b.weight": [V_HEADS, HIDDEN],
    "in_proj_qkv.weight": [CONV_DIM, HIDDEN],
    "in_proj_z.weight": [VALUE_DIM, HIDDEN],
    "norm.weight": [HEAD_DIM],
    "out_proj.weight": [HIDDEN, VALUE_DIM],
}


def tensor_bytes(value: torch.Tensor) -> bytes:
    value = value.detach().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().tobytes()
    if value.dtype == torch.float32:
        return value.numpy().tobytes()
    raise ValueError(f"unsupported capture dtype {value.dtype}")


def capture(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().contiguous()
    dtype = {torch.bfloat16: "BF16", torch.float32: "F32"}.get(value.dtype)
    if dtype is None:
        raise ValueError(f"unsupported capture dtype {value.dtype}")
    return {"dtype": dtype, "shape": list(value.shape), "sha256": hashlib.sha256(tensor_bytes(value)).hexdigest()}


def make_input(spec: dict[str, int]) -> torch.Tensor:
    index = torch.arange(HIDDEN, dtype=torch.int64)
    value = ((index * spec["multiplier"] + spec["add"]) % spec["modulus"] - spec["center"])
    if spec["sparse_stride"] > 1:
        value = torch.where(index % spec["sparse_stride"] == 0, value, 0)
    return value.to(torch.float32).div(spec["divisor"]).to(torch.bfloat16).reshape(1, 1, HIDDEN).contiguous()


def explicit_step(
    module: Qwen4ExpTextGatedDeltaNet,
    hidden_states: torch.Tensor,
    cache: DynamicCache,
    cached: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mixed_qkv_projection = module.in_proj_qkv(hidden_states).contiguous()
    mixed_qkv_channels = mixed_qkv_projection.transpose(1, 2).contiguous()
    z = module.in_proj_z(hidden_states).reshape(1, 1, V_HEADS, HEAD_DIM).contiguous()
    b = module.in_proj_b(hidden_states).contiguous()
    a = module.in_proj_a(hidden_states).contiguous()
    if cached:
        conv_state = cache.layers[0].conv_states[0]
        convolved_channels = causal_conv1d_update(
            mixed_qkv_channels,
            conv_state,
            module.conv1d.weight.squeeze(1),
            module.conv1d.bias,
            module.activation,
        ).contiguous()
    else:
        full_conv = cache.update_conv_state(mixed_qkv_channels, 0, conv_kernel_size=CONV_KERNEL)
        convolved_channels = causal_conv1d_fn(
            full_conv,
            module.conv1d.weight.squeeze(1),
            module.conv1d.bias,
            activation=module.activation,
        )[:, :, -1:].contiguous()
    convolution_state = cache.layers[0].conv_states[0].clone().contiguous()
    convolved = convolved_channels.transpose(1, 2).contiguous()
    query, key, value = torch.split(convolved, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    query = query.reshape(1, 1, K_HEADS, HEAD_DIM).contiguous()
    key = key.reshape(1, 1, K_HEADS, HEAD_DIM).contiguous()
    value = value.reshape(1, 1, V_HEADS, HEAD_DIM).contiguous()
    beta = b.sigmoid().contiguous()
    decay = (-module.A_log.float().exp() * torch.nn.functional.softplus(a.float() + module.dt_bias)).contiguous()
    query_repeated = query.repeat_interleave(V_HEADS // K_HEADS, dim=2).contiguous()
    key_repeated = key.repeat_interleave(V_HEADS // K_HEADS, dim=2).contiguous()
    query_normalized = l2norm(query_repeated, dim=-1, eps=1e-6).contiguous()
    key_normalized = l2norm(key_repeated, dim=-1, eps=1e-6).contiguous()
    recurrent_before = cache.layers[0].recurrent_states[0] if cached else None
    if cached:
        core, recurrent_after = torch_recurrent_gated_delta_rule(
            query_repeated,
            key_repeated,
            value,
            g=decay,
            beta=beta,
            initial_state=recurrent_before,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        core, recurrent_after = torch_chunk_gated_delta_rule(
            query_repeated,
            key_repeated,
            value,
            g=decay,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    core = core.contiguous()
    recurrent_after = recurrent_after.contiguous()
    cache.update_recurrent_state(recurrent_after, 0)
    core_flat = core.reshape(-1, HEAD_DIM).contiguous()
    z_flat = z.reshape(-1, HEAD_DIM).contiguous()
    gated_norm = module.norm(core_flat, z_flat).contiguous()
    output = module.out_proj(gated_norm.reshape(1, 1, VALUE_DIM)).contiguous()
    values = {
        "hidden_states": hidden_states,
        "mixed_qkv_projection": mixed_qkv_projection,
        "z_projection": z,
        "b_projection": b,
        "a_projection": a,
        "convolution_state": convolution_state,
        "convolved_qkv": convolved,
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "decay": decay,
        "query_repeated": query_repeated,
        "key_repeated": key_repeated,
        "query_normalized": query_normalized,
        "key_normalized": key_normalized,
        "core_attention": core,
        "recurrent_state": cache.layers[0].recurrent_states[0].clone().contiguous(),
        "gated_norm": gated_norm,
        "output": output,
    }
    return output, values


def build_fixture(checkpoint_dir: Path, model_lock_path: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")
    config_path = checkpoint_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
    required = {
        "hidden_size": HIDDEN,
        "linear_num_key_heads": K_HEADS,
        "linear_num_value_heads": V_HEADS,
        "linear_key_head_dim": HEAD_DIM,
        "linear_value_head_dim": HEAD_DIM,
        "linear_conv_kernel_dim": CONV_KERNEL,
    }
    if any(raw_config[key] != value for key, value in required.items()) or raw_config["layer_types"][0] != "linear_attention" or raw_config["hidden_act"] != "silu" or raw_config["output_gate_type"] != "sigmoid":
        raise ValueError("unsupported Qwen Gated DeltaNet configuration")
    config = Qwen4ExpTextConfig(**raw_config)
    module = Qwen4ExpTextGatedDeltaNet(config, 0).to(torch.bfloat16).eval()
    index_path = checkpoint_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    prefix = "model.language_model.layers.0.linear_attn"
    state: dict[str, torch.Tensor] = {}
    tensors = {}
    for local_name, shape in TENSOR_SHAPES.items():
        tensor_name = f"{prefix}.{local_name}"
        shard = weight_map[tensor_name]
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as source:
            value = source.get_tensor(tensor_name).contiguous()
        if value.dtype != torch.bfloat16 or list(value.shape) != shape:
            raise ValueError(f"unsupported tensor {tensor_name}")
        state[local_name] = value
        locked = locked_file(lock, shard)
        tensors[local_name] = {
            "tensor": tensor_name,
            "dtype": "BF16",
            "shape": shape,
            "shard": shard,
            "shard_bytes": locked["size"],
            "shard_sha256": locked["lfs_sha256"],
            "payload_sha256": hashlib.sha256(tensor_bytes(value)).hexdigest(),
        }
    module.load_state_dict(state, strict=True)
    explicit_cache = DynamicCache(config=config)
    official_cache = DynamicCache(config=config)
    steps = []
    for ordinal, spec in enumerate(INPUT_SPECS):
        hidden = make_input(spec)
        with torch.no_grad():
            explicit_output, captures = explicit_step(module, hidden, explicit_cache, cached=ordinal > 0)
            official_output = module(hidden, cache_params=official_cache)
        official_conv = official_cache.layers[0].conv_states[0]
        official_recurrent = official_cache.layers[0].recurrent_states[0]
        if not torch.equal(explicit_output, official_output) or not torch.equal(captures["convolution_state"], official_conv) or not torch.equal(captures["recurrent_state"], official_recurrent):
            raise ValueError(f"explicit step {ordinal} disagrees with official Gated DeltaNet")
        steps.append({
            "ordinal": ordinal,
            "mode": "initial_chunk" if ordinal == 0 else "cached_recurrent",
            "input_spec": spec,
            "captures": {name: capture(value) for name, value in captures.items()},
        })
    return {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextGatedDeltaNet.forward",
            "config_sha256": sha256_file(config_path),
            "tensor_index_sha256": sha256_file(index_path),
            "model_lock_sha256": sha256_file(model_lock_path),
        },
        "configuration": {
            **required,
            "key_dim": KEY_DIM,
            "value_dim": VALUE_DIM,
            "conv_dim": CONV_DIM,
            "activation": "silu",
            "output_gate": "sigmoid",
            "weight_dtype": "BF16",
            "recurrent_state_dtype": "F32",
        },
        "case": {
            "name": "layer_0_two_token_cached_decode",
            "layer": 0,
            "tensors": tensors,
            "steps": steps,
        },
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
    print(json.dumps({"output": os.fspath(args.output), "steps": len(fixture["case"]["steps"]), "captures_per_step": len(fixture["case"]["steps"][0]["captures"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
