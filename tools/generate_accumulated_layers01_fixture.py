#!/usr/bin/env python3
"""Generate a two-token accumulated layer-0 through layer-1 fixture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import transformers

if __package__:
    from tools.generate_decoder_layer_fixture import build_fixture as build_layer0
    from tools.generate_full_decoder_layer1_fixture import build_fixture as build_layer1
    from tools.generate_full_decoder_layer3_fixture import write_json
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, sha256_file
else:
    from generate_decoder_layer_fixture import build_fixture as build_layer0  # type: ignore[no-redef]
    from generate_full_decoder_layer1_fixture import build_fixture as build_layer1  # type: ignore[no-redef]
    from generate_full_decoder_layer3_fixture import write_json  # type: ignore[no-redef]
    from generate_ngram_address_fixture import checkpoint_revision, load_model_lock, sha256_file  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_accumulated_layers0_1_cached_decode"
LAYER1_DECODER_SEMANTIC = "qwen3_8_flash_next_layer1_complete_accumulated_from_layer0"


def build_fixture(
    checkpoint_dir: Path,
    model_lock_path: Path,
    ngram_fixture_path: Path,
    ngram_row_fixture_path: Path,
    layer0_attention_fixture_path: Path,
    layer0_sparse_moe_fixture_path: Path,
    layer0_fixture_path: Path,
    ple_fixture_path: Path,
    layer1_attention_fixture_path: Path,
    layer1_fixture_path: Path,
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")

    layer0_result = build_layer0(
        checkpoint_dir,
        model_lock_path,
        layer0_attention_fixture_path,
        layer0_sparse_moe_fixture_path,
        _return_outputs=True,
    )
    if not isinstance(layer0_result, tuple):
        raise AssertionError("layer-0 execution outputs were not returned")
    generated_layer0, layer0_outputs = layer0_result
    committed_layer0 = json.loads(layer0_fixture_path.read_text(encoding="utf-8"))
    if generated_layer0 != committed_layer0:
        raise ValueError("regenerated layer-0 parent disagrees with committed fixture")

    reference_hashes = {
        "ngram_fixture_sha256": sha256_file(ngram_fixture_path),
        "ngram_row_fixture_sha256": sha256_file(ngram_row_fixture_path),
        "ple_fixture_sha256": sha256_file(ple_fixture_path),
        "attention_residual_fixture_sha256": sha256_file(layer1_attention_fixture_path),
        "source_layer1_fixture_sha256": sha256_file(layer1_fixture_path),
        "hidden_source": "accumulated.layer0_output",
    }
    layer1_result = build_layer1(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        ple_fixture_path,
        layer1_attention_fixture_path,
        _hidden_overrides=layer0_outputs,
        _semantic=LAYER1_DECODER_SEMANTIC,
        _reference_hashes=reference_hashes,
        _return_chain=True,
    )
    if not isinstance(layer1_result, tuple):
        raise AssertionError("accumulated layer-1 chain was not returned")
    layer1_decoder, layer1_outputs, layer1_attention, layer1_ple = layer1_result
    layer1_ple["semantic"] = "qwen3_8_flash_next_layer1_ple_accumulated_from_layer0"
    layer1_ple["reference"]["hidden_source"] = "accumulated.layer0_output"
    layer1_attention["semantic"] = (
        "qwen3_8_flash_next_layer1_ple_attention_residual_accumulated_from_layer0"
    )
    layer1_attention["reference"]["hidden_source"] = "accumulated.layer0_output"

    steps = []
    for ordinal in range(2):
        layer0_step = generated_layer0["case"]["steps"][ordinal]
        layer1_step = layer1_decoder["steps"][ordinal]
        steps.append(
            {
                "ordinal": ordinal,
                "mode": "initial_chunk" if ordinal == 0 else "cached_recurrent",
                "token_id": [42, 43][ordinal],
                "input_spec": generated_layer0["case"]["steps"][ordinal].get(
                    "input_spec",
                    [
                        {"multiplier": 43, "add": 17, "modulus": 263, "center": 131, "divisor": 128, "sparse_stride": 1},
                        {"multiplier": 61, "add": 29, "modulus": 277, "center": 138, "divisor": 128, "sparse_stride": 1},
                    ][ordinal],
                ),
                "layer0_selected_experts": layer0_step["selected_experts"],
                "layer1_selected_experts": layer1_step["selected_experts"],
                "captures": {
                    "layer0_output": layer0_step["captures"]["layer_output"],
                    "layer1_ple_output": layer1_ple["case"]["steps"][ordinal]["captures"]["output"],
                    "layer1_post_attention": layer1_attention["case"]["steps"][ordinal]["captures"]["composed_output"],
                    "layer1_output": layer1_step["captures"]["layer_output"],
                },
            }
        )

    return {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "source_derived_huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward",
            "model_lock_sha256": sha256_file(model_lock_path),
            "ngram_fixture_sha256": sha256_file(ngram_fixture_path),
            "ngram_row_fixture_sha256": sha256_file(ngram_row_fixture_path),
            "layer0_attention_fixture_sha256": sha256_file(layer0_attention_fixture_path),
            "layer0_sparse_moe_fixture_sha256": sha256_file(layer0_sparse_moe_fixture_path),
            "layer0_fixture_sha256": sha256_file(layer0_fixture_path),
            "source_ple_fixture_sha256": sha256_file(ple_fixture_path),
            "source_layer1_attention_fixture_sha256": sha256_file(layer1_attention_fixture_path),
            "source_layer1_fixture_sha256": sha256_file(layer1_fixture_path),
        },
        "configuration": {
            "first_layer": 0,
            "last_layer": 1,
            "tokens": [42, 43],
            "hidden_size": 2560,
            "hc_count": 4,
            "boundary_dtype": "BF16",
            "layer_types": ["linear_attention", "linear_attention"],
            "ple_layers": [1],
        },
        "layer1_ple": layer1_ple,
        "layer1_attention": layer1_attention,
        "layer1_decoder": layer1_decoder,
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--ngram-fixture", required=True, type=Path)
    parser.add_argument("--ngram-row-fixture", required=True, type=Path)
    parser.add_argument("--layer0-attention-fixture", required=True, type=Path)
    parser.add_argument("--layer0-sparse-moe-fixture", required=True, type=Path)
    parser.add_argument("--layer0-fixture", required=True, type=Path)
    parser.add_argument("--ple-fixture", required=True, type=Path)
    parser.add_argument("--layer1-attention-fixture", required=True, type=Path)
    parser.add_argument("--layer1-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(
        args.checkpoint_dir,
        args.model_lock,
        args.ngram_fixture,
        args.ngram_row_fixture,
        args.layer0_attention_fixture,
        args.layer0_sparse_moe_fixture,
        args.layer0_fixture,
        args.ple_fixture,
        args.layer1_attention_fixture,
        args.layer1_fixture,
    )
    write_json(args.output, fixture)
    print(json.dumps({
        "output": os.fspath(args.output),
        "steps": len(fixture["steps"]),
        "layer0_routes": [step["layer0_selected_experts"] for step in fixture["steps"]],
        "layer1_routes": [step["layer1_selected_experts"] for step in fixture["steps"]],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
