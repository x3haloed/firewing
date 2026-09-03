#!/usr/bin/env python3
"""Screen compact INT8 topology on authenticated early, middle, and late layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

if __package__:
    from tools import analyze_q2_lossless_experts as common
    from tools.analyze_block_fp8_weight_fidelity import (
        block_int8_weight,
        error_metrics,
        write_json,
    )
    from tools.generate_accumulated_layers4_47_fixture import build_fixture
    from tools.generate_expert_fixture import expert_forward
    from tools.generate_mixture_fixture import accumulate_bf16_in_expert_order
    from tools.generate_ngram_address_fixture import sha256_file
else:
    import analyze_q2_lossless_experts as common
    from analyze_block_fp8_weight_fidelity import block_int8_weight, error_metrics, write_json
    from generate_accumulated_layers4_47_fixture import build_fixture
    from generate_expert_fixture import expert_forward
    from generate_mixture_fixture import accumulate_bf16_in_expert_order
    from generate_ngram_address_fixture import sha256_file


ROOT = Path(__file__).parents[1]
MODEL_LOCK_SHA256 = "f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444"
ACCUMULATED_FIXTURE_SHA256 = "6ed2e16da1e64fb8001c26608d7972f4910190f74768055b5778dc7891ebf525"
SELECTED_LAYERS = (4, 24, 46)
DEFAULT_BLOCK_SHAPE = (4, 4)


def quantized_mixture(
    hidden: torch.Tensor,
    selection: list[int],
    scores: list[float],
    gate_up_file: Any,
    down_file: Any,
    gate_up_name: str,
    down_name: str,
    reference_mixture: torch.Tensor,
    block_shape: tuple[int, int] = DEFAULT_BLOCK_SHAPE,
) -> dict[str, Any]:
    if (
        hidden.dtype != torch.bfloat16
        or hidden.ndim != 1
        or len(selection) != len(scores)
        or len(selection) != len(set(selection))
        or reference_mixture.dtype != torch.bfloat16
    ):
        raise common.AnalysisError("INT8 real-layer mixture authority mismatch")
    if len(block_shape) != 2 or block_shape[0] * block_shape[1] != 16:
        raise common.AnalysisError("INT8 real-layer scale topology must cover 16 weights")
    score_by_expert = dict(zip(selection, scores, strict=True))
    contributions = []
    expert_rows = []
    source_bytes = 0
    artifact_bytes = 0
    for expert in sorted(selection):
        gate_up = gate_up_file.get_slice(gate_up_name)[expert].contiguous()
        down = down_file.get_slice(down_name)[expert].contiguous()
        route_weight = torch.tensor(score_by_expert[expert], dtype=torch.bfloat16)
        reference = expert_forward(hidden, gate_up, down, route_weight)["weighted_down"]
        quantized_gate_up, gate_scales, gate_bytes = block_int8_weight(
            gate_up, block_shape
        )
        quantized_down, down_scales, down_bytes = block_int8_weight(
            down, block_shape
        )
        candidate = expert_forward(
            hidden, quantized_gate_up, quantized_down, route_weight
        )["weighted_down"]
        contributions.append(candidate)
        expert_rows.append(
            {
                "expert": expert,
                "gate_up_scale_blocks": gate_scales.numel(),
                "down_scale_blocks": down_scales.numel(),
                "weighted_down": error_metrics(candidate, reference),
            }
        )
        source_bytes += gate_up.numel() * 2 + down.numel() * 2
        artifact_bytes += gate_bytes + down_bytes
    candidate_mixture = accumulate_bf16_in_expert_order(contributions)
    return {
        "experts": expert_rows,
        "source_weight_bytes": source_bytes,
        "artifact_weight_and_scale_bytes": artifact_bytes,
        "artifact_to_source_ratio": artifact_bytes / source_bytes,
        "mixture": error_metrics(candidate_mixture, reference_mixture),
        "maximum_expert_weighted_down_relative_l2": max(
            row["weighted_down"]["relative_l2"] for row in expert_rows
        ),
    }


def analyze(
    checkpoint_dir: Path,
    model_lock_path: Path,
    implementation_commit: str,
    block_rows: int = DEFAULT_BLOCK_SHAPE[0],
    block_columns: int = DEFAULT_BLOCK_SHAPE[1],
) -> dict[str, Any]:
    common.require_clean_commit(implementation_commit)
    common.require_hash(model_lock_path, MODEL_LOCK_SHA256)
    accumulated_path = ROOT / "fixtures/accumulated/qwen3_8_flash_next_layers4_47.json"
    common.require_hash(accumulated_path, ACCUMULATED_FIXTURE_SHA256)
    observations: list[dict[str, Any]] = []
    block_shape = (block_rows, block_columns)
    if block_rows * block_columns != 16:
        raise common.AnalysisError("INT8 real-layer scale topology must cover 16 weights")

    def observe(**values: Any) -> None:
        layer = values["layer"]
        if layer not in SELECTED_LAYERS:
            return
        result = quantized_mixture(
            values["hidden"],
            values["selection"],
            values["scores"],
            values["gate_up_file"],
            values["down_file"],
            values["gate_up_name"],
            values["down_name"],
            values["reference_mixture"],
            block_shape,
        )
        observations.append(
            {"layer": layer, "ordinal": values["ordinal"], **result}
        )

    fixture = ROOT / "fixtures"
    generated = build_fixture(
        checkpoint_dir,
        model_lock_path,
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
        _mixture_observer=observe,
    )
    committed = json.loads(accumulated_path.read_text(encoding="utf-8"))
    if generated != committed:
        raise common.AnalysisError("regenerated accumulated authority disagrees")
    expected_pairs = [(layer, ordinal) for layer in SELECTED_LAYERS for ordinal in range(2)]
    if [(row["layer"], row["ordinal"]) for row in observations] != expected_pairs:
        raise common.AnalysisError("INT8 real-layer observation coverage mismatch")
    maximum_mixture = max(row["mixture"]["relative_l2"] for row in observations)
    maximum_expert = max(
        row["maximum_expert_weighted_down_relative_l2"] for row in observations
    )
    passes = maximum_mixture <= 0.01 and maximum_expert <= 0.02
    topology_tag = (
        "block4"
        if block_shape == DEFAULT_BLOCK_SHAPE
        else f"block{block_rows}x{block_columns}"
    )
    return {
        "schema_version": 1,
        "semantic": (
            f"qwen3_8_flash_next_modified_{topology_tag}_int8_"
            "source_accumulated_real_layer_screen"
        ),
        "mode": f"modified_{topology_tag}_int8_weight_only",
        "implementation_commit": implementation_commit,
        "model": common.MODEL,
        "revision": common.REVISION,
        "model_lock_sha256": MODEL_LOCK_SHA256,
        "accumulated_fixture_sha256": ACCUMULATED_FIXTURE_SHA256,
        "selected_layers": list(SELECTED_LAYERS),
        "block_shape": [block_rows, block_columns],
        "observations": observations,
        "maximum_mixture_relative_l2": maximum_mixture,
        "maximum_expert_weighted_down_relative_l2": maximum_expert,
        "continuation_gates": {
            "each_mixture_relative_l2_maximum": 0.01,
            "each_expert_weighted_down_relative_l2_maximum": 0.02,
        },
        "passes_continuation_gates": passes,
        "decision": f"{'continue' if passes else 'reject'}_modified_{topology_tag}_int8_weight_only",
        "limitations": [
            "six source-accumulated real layer-local inputs only",
            "source routes are held fixed",
            "weight-only quantization grants exact BF16 activations",
            "no candidate-accumulated logits hosted or endpoint fidelity",
            "no runtime artifact kernel or performance measurement",
        ],
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("model_lock", type=Path)
    parser.add_argument("implementation_commit")
    parser.add_argument("output", type=Path)
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_SHAPE[0])
    parser.add_argument("--block-columns", type=int, default=DEFAULT_BLOCK_SHAPE[1])
    args = parser.parse_args()
    report = analyze(
        args.checkpoint_dir,
        args.model_lock,
        args.implementation_commit,
        args.block_rows,
        args.block_columns,
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
