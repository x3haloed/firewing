#!/usr/bin/env python3
"""Screen modified block-scaled weights on Firewing's real layer-0 mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

if __package__:
    from tools import analyze_q2_lossless_experts as common
    from tools.generate_expert_fixture import capture_hash, expert_forward
    from tools.generate_mixture_fixture import accumulate_bf16_in_expert_order
    from tools.generate_ngram_address_fixture import load_model_lock, locked_file
    from tools.generate_router_fixture import make_hidden, tensor_bytes
else:
    import analyze_q2_lossless_experts as common
    from generate_expert_fixture import capture_hash, expert_forward
    from generate_mixture_fixture import accumulate_bf16_in_expert_order
    from generate_ngram_address_fixture import load_model_lock, locked_file
    from generate_router_fixture import make_hidden, tensor_bytes


MODEL_LOCK_SHA256 = "f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444"
MIXTURE_SHA256 = "975a9982919297d37dd077f774693c782295cba496542c6adf278182e27b4d89"
DEFAULT_BLOCK = 128
FP8_MAX = 448.0
INT8_MAX = 127.0


def block_fp8_weight(
    weight: torch.Tensor, block: int = DEFAULT_BLOCK
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if (
        weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or block <= 0
        or weight.shape[0] % block
        or weight.shape[1] % block
        or not torch.isfinite(weight.float()).all()
    ):
        raise common.AnalysisError("block-FP8 weight must be finite aligned BF16 matrix")
    rows, columns = weight.shape
    blocks = (
        weight.float()
        .reshape(rows // block, block, columns // block, block)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    maximums = blocks.abs().amax(dim=(2, 3))
    scales = torch.clamp(maximums, min=1.0e-10) / FP8_MAX
    normalized = torch.clamp(blocks / scales[:, :, None, None], -FP8_MAX, FP8_MAX)
    codes = normalized.to(torch.float8_e4m3fn)
    decoded = (codes.float() * scales[:, :, None, None]).to(torch.bfloat16)
    decoded = decoded.permute(0, 2, 1, 3).reshape(rows, columns).contiguous()
    artifact_bytes = weight.numel() + scales.numel() * 4
    return decoded, scales.contiguous(), artifact_bytes


def block_int8_weight(
    weight: torch.Tensor, block: int = DEFAULT_BLOCK
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if (
        weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or block <= 0
        or weight.shape[0] % block
        or weight.shape[1] % block
        or not torch.isfinite(weight.float()).all()
    ):
        raise common.AnalysisError("block-INT8 weight must be finite aligned BF16 matrix")
    rows, columns = weight.shape
    blocks = (
        weight.float()
        .reshape(rows // block, block, columns // block, block)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    maximums = blocks.abs().amax(dim=(2, 3))
    scales = torch.clamp(maximums, min=1.0e-10) / INT8_MAX
    codes = torch.clamp(torch.round(blocks / scales[:, :, None, None]), -127, 127).to(
        torch.int8
    )
    decoded = (codes.float() * scales[:, :, None, None]).to(torch.bfloat16)
    decoded = decoded.permute(0, 2, 1, 3).reshape(rows, columns).contiguous()
    artifact_bytes = weight.numel() + scales.numel() * 4
    return decoded, scales.contiguous(), artifact_bytes


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    if actual.shape != reference.shape:
        raise common.AnalysisError("block-weight metric shape mismatch")
    difference = actual.float().double() - reference.float().double()
    denominator = torch.linalg.vector_norm(reference.float().double()).item()
    if denominator == 0:
        raise common.AnalysisError("block-weight metric reference norm is zero")
    return {
        "relative_l2": torch.linalg.vector_norm(difference).item() / denominator,
        "maximum_absolute_error": difference.abs().max().item(),
        "bf16_equality_fraction": torch.eq(actual, reference).double().mean().item(),
    }


def analyze(
    checkpoint_dir: Path,
    model_lock_path: Path,
    mixture_path: Path,
    implementation_commit: str,
    weight_format: str = "block_fp8",
    block_size: int = DEFAULT_BLOCK,
) -> dict[str, Any]:
    common.require_clean_commit(implementation_commit)
    common.require_hash(model_lock_path, MODEL_LOCK_SHA256)
    common.require_hash(mixture_path, MIXTURE_SHA256)
    lock = load_model_lock(model_lock_path)
    fixture = common.read_json(mixture_path)
    case = fixture.get("case", {})
    if (
        fixture.get("model") != common.MODEL
        or fixture.get("revision") != common.REVISION
        or fixture.get("semantic") != "qwen3_8_flash_next_real_top10_expert_mixture"
        or case.get("layer") != 0
        or len(case.get("experts", [])) != 10
    ):
        raise common.AnalysisError("block-FP8 mixture authority mismatch")
    hidden = make_hidden(2560, case["input_spec"])
    if capture_hash(hidden) != case.get("input_bf16_sha256"):
        raise common.AnalysisError("block-FP8 input authority mismatch")
    gate = case["gate_up"]
    down = case["down"]
    for bank in (gate, down):
        locked = locked_file(lock, bank["shard"])
        path = checkpoint_dir / bank["shard"]
        if locked["size"] != bank["shard_bytes"] or locked["lfs_sha256"] != bank["shard_sha256"]:
            raise common.AnalysisError("block-FP8 locked shard identity mismatch")
        if path.stat().st_size != locked["size"]:
            raise common.AnalysisError("block-FP8 live shard size mismatch")

    score_by_expert = {
        row["expert"]: torch.tensor(row["route_weight_bf16"], dtype=torch.bfloat16)
        for row in case["experts"]
    }
    reference_contributions = []
    candidate_contributions = []
    expert_rows = []
    source_bytes = 0
    artifact_bytes = 0
    exact_baseline_hashes = 0
    if weight_format == "block_fp8":
        quantize = lambda weight: block_fp8_weight(weight, block_size)
        mode = (
            "modified_block_fp8_weight_only"
            if block_size == DEFAULT_BLOCK
            else f"modified_block{block_size}_fp8_weight_only"
        )
        format_description = (
            f"e4m3fn_per_{block_size}x{block_size}_absmax_f32_scale"
        )
    elif weight_format == "block_int8":
        quantize = lambda weight: block_int8_weight(weight, block_size)
        mode = (
            "modified_block_int8_weight_only"
            if block_size == DEFAULT_BLOCK
            else f"modified_block{block_size}_int8_weight_only"
        )
        format_description = (
            f"symmetric_int8_per_{block_size}x{block_size}_absmax_f32_scale"
        )
    else:
        raise common.AnalysisError("unknown modified block weight format")
    with safe_open(checkpoint_dir / gate["shard"], framework="pt", device="cpu") as gate_file:
        with safe_open(checkpoint_dir / down["shard"], framework="pt", device="cpu") as down_file:
            for expected in case["experts"]:
                expert = expected["expert"]
                gate_up = gate_file.get_slice(gate["tensor"])[expert].contiguous()
                down_weight = down_file.get_slice(down["tensor"])[expert].contiguous()
                if (
                    hashlib.sha256(tensor_bytes(gate_up)).hexdigest()
                    != expected["gate_up_payload_sha256"]
                    or hashlib.sha256(tensor_bytes(down_weight)).hexdigest()
                    != expected["down_payload_sha256"]
                ):
                    raise common.AnalysisError("block-FP8 source payload mismatch")
                reference = expert_forward(
                    hidden, gate_up, down_weight, score_by_expert[expert]
                )
                if capture_hash(reference["weighted_down"]) != expected["weighted_down_bf16_sha256"]:
                    raise common.AnalysisError("block-FP8 exact baseline mismatch")
                exact_baseline_hashes += 1
                candidate_gate_up, gate_scales, gate_bytes = quantize(gate_up)
                candidate_down, down_scales, down_bytes = quantize(down_weight)
                candidate = expert_forward(
                    hidden, candidate_gate_up, candidate_down, score_by_expert[expert]
                )
                reference_contributions.append(reference["weighted_down"])
                candidate_contributions.append(candidate["weighted_down"])
                source_bytes += gate_up.numel() * 2 + down_weight.numel() * 2
                artifact_bytes += gate_bytes + down_bytes
                expert_rows.append(
                    {
                        "expert": expert,
                        "gate_up_scale_blocks": gate_scales.numel(),
                        "down_scale_blocks": down_scales.numel(),
                        "weighted_down": error_metrics(
                            candidate["weighted_down"], reference["weighted_down"]
                        ),
                    }
                )
    reference_mixture = accumulate_bf16_in_expert_order(reference_contributions)
    if capture_hash(reference_mixture) != case.get("mixture_bf16_sha256"):
        raise common.AnalysisError("block-FP8 mixture baseline mismatch")
    candidate_mixture = accumulate_bf16_in_expert_order(candidate_contributions)
    mixture_metrics = error_metrics(candidate_mixture, reference_mixture)
    maximum_expert_relative_l2 = max(
        row["weighted_down"]["relative_l2"] for row in expert_rows
    )
    passes = mixture_metrics["relative_l2"] <= 0.01 and maximum_expert_relative_l2 <= 0.02
    return {
        "schema_version": 1,
        "semantic": f"qwen3_8_flash_next_{mode}_layer0_top10_fidelity_screen",
        "mode": mode,
        "implementation_commit": implementation_commit,
        "model": common.MODEL,
        "revision": common.REVISION,
        "model_lock_sha256": MODEL_LOCK_SHA256,
        "mixture_fixture_sha256": MIXTURE_SHA256,
        "layer": 0,
        "block_size": block_size,
        "experts": expert_rows,
        "exact_baseline_hashes": exact_baseline_hashes + 1,
        "weight_format": format_description,
        "activation_format": "bf16_unmodified_favorable_grant",
        "boundary_format": "bf16",
        "source_weight_bytes": source_bytes,
        "artifact_weight_and_scale_bytes": artifact_bytes,
        "artifact_to_source_ratio": artifact_bytes / source_bytes,
        "mixture": mixture_metrics,
        "maximum_expert_weighted_down_relative_l2": maximum_expert_relative_l2,
        "continuation_gates": {
            "mixture_relative_l2_maximum": 0.01,
            "each_expert_weighted_down_relative_l2_maximum": 0.02,
        },
        "passes_continuation_gates": passes,
        "decision": f"{'continue' if passes else 'reject'}_{mode}",
        "limitations": [
            "one layer and one real routed input only",
            "weight-only quantization grants exact BF16 activations",
            "no accumulated logits distributional hosted or endpoint fidelity",
            "no runtime artifact kernel or performance measurement",
        ],
        "performance_claim": None,
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
    parser.add_argument("model_lock", type=Path)
    parser.add_argument("mixture_fixture", type=Path)
    parser.add_argument("implementation_commit")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--weight-format",
        choices=("block_fp8", "block_int8"),
        default="block_fp8",
    )
    parser.add_argument(
        "--block-size", type=int, choices=(4, 8, 16, 32, 128), default=DEFAULT_BLOCK
    )
    args = parser.parse_args()
    report = analyze(
        args.checkpoint_dir,
        args.model_lock,
        args.mixture_fixture,
        args.implementation_commit,
        args.weight_format,
        args.block_size,
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
