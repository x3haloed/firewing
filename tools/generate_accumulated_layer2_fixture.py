#!/usr/bin/env python3
"""Generate accumulated layer-2 linear-decoder evidence from FW-0024 outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import transformers

if __package__:
    from tools.generate_accumulated_layers01_fixture import build_fixture as build_layers01
    from tools.generate_attention_residual_fixture import build_fixture as build_attention
    from tools.generate_full_decoder_layer3_fixture import (
        build_fixture as build_decoder,
        write_json,
    )
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, sha256_file
else:
    from generate_accumulated_layers01_fixture import build_fixture as build_layers01  # type: ignore[no-redef]
    from generate_attention_residual_fixture import build_fixture as build_attention  # type: ignore[no-redef]
    from generate_full_decoder_layer3_fixture import (  # type: ignore[no-redef]
        build_fixture as build_decoder,
        write_json,
    )
    from generate_ngram_address_fixture import checkpoint_revision, load_model_lock, sha256_file  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_accumulated_layer2_cached_decode"
ATTENTION_SEMANTIC = "qwen3_8_flash_next_layer2_attention_accumulated_from_layer1"
DECODER_SEMANTIC = "qwen3_8_flash_next_layer2_complete_accumulated_from_layer1"


def build_fixture(
    checkpoint_dir: Path,
    model_lock_path: Path,
    ngram_fixture_path: Path,
    ngram_row_fixture_path: Path,
    layer0_hyper_fixture_path: Path,
    layer0_deltanet_fixture_path: Path,
    layer0_attention_fixture_path: Path,
    layer0_sparse_moe_fixture_path: Path,
    layer0_fixture_path: Path,
    ple_fixture_path: Path,
    layer1_attention_fixture_path: Path,
    layer1_fixture_path: Path,
    layers01_fixture_path: Path,
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")
    layers01_result = build_layers01(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        layer0_attention_fixture_path,
        layer0_sparse_moe_fixture_path,
        layer0_fixture_path,
        ple_fixture_path,
        layer1_attention_fixture_path,
        layer1_fixture_path,
        _return_outputs=True,
    )
    if not isinstance(layers01_result, tuple):
        raise AssertionError("FW-0024 outputs were not returned")
    generated_layers01, layer1_outputs = layers01_result
    if generated_layers01 != json.loads(layers01_fixture_path.read_text(encoding="utf-8")):
        raise ValueError("regenerated FW-0024 parent disagrees with committed fixture")

    attention_reference = {
        "layers01_fixture_sha256": sha256_file(layers01_fixture_path),
        "hidden_source": "accumulated.layer1_output",
    }
    attention_result = build_attention(
        checkpoint_dir,
        model_lock_path,
        layer0_hyper_fixture_path,
        layer0_deltanet_fixture_path,
        _layer=2,
        _hidden_overrides=layer1_outputs,
        _semantic=ATTENTION_SEMANTIC,
        _reference_hashes=attention_reference,
        _return_outputs=True,
    )
    if not isinstance(attention_result, tuple):
        raise AssertionError("layer-2 attention outputs were not returned")
    layer2_attention, post_attention = attention_result
    decoder_result = build_decoder(
        checkpoint_dir,
        model_lock_path,
        layer0_attention_fixture_path,
        layers01_fixture_path,
        _parent_execution=(layer2_attention, post_attention),
        _parent_semantic=ATTENTION_SEMANTIC,
        _layer=2,
        _layer_type="linear_attention",
        _semantic=DECODER_SEMANTIC,
        _reference_hashes={
            "layers01_fixture_sha256": sha256_file(layers01_fixture_path),
            "hidden_source": "accumulated.layer1_output",
        },
        _modes=("initial_chunk", "cached_recurrent"),
        _require_committed_parent=False,
        _return_outputs=True,
    )
    if not isinstance(decoder_result, tuple):
        raise AssertionError("layer-2 decoder outputs were not returned")
    layer2_decoder, layer2_outputs = decoder_result

    steps = []
    for ordinal in range(2):
        steps.append(
            {
                "ordinal": ordinal,
                "mode": "initial_chunk" if ordinal == 0 else "cached_recurrent",
                "input_spec": [
                    {"multiplier": 43, "add": 17, "modulus": 263, "center": 131, "divisor": 128, "sparse_stride": 1},
                    {"multiplier": 61, "add": 29, "modulus": 277, "center": 138, "divisor": 128, "sparse_stride": 1},
                ][ordinal],
                "selected_experts": layer2_decoder["steps"][ordinal]["selected_experts"],
                "captures": {
                    "layer1_output": generated_layers01["steps"][ordinal]["captures"]["layer1_output"],
                    "post_attention": layer2_attention["case"]["steps"][ordinal]["captures"]["composed_output"],
                    "layer2_output": layer2_decoder["steps"][ordinal]["captures"]["layer_output"],
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
            "layers01_fixture_sha256": sha256_file(layers01_fixture_path),
        },
        "configuration": {
            "layer": 2,
            "layer_type": "linear_attention",
            "ple_applied": False,
            "hidden_size": 2560,
            "hc_count": 4,
            "boundary_dtype": "BF16",
            "recurrent_state_dtype": "F32",
        },
        "attention": layer2_attention,
        "decoder": layer2_decoder,
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--ngram-fixture", required=True, type=Path)
    parser.add_argument("--ngram-row-fixture", required=True, type=Path)
    parser.add_argument("--layer0-hyper-fixture", required=True, type=Path)
    parser.add_argument("--layer0-deltanet-fixture", required=True, type=Path)
    parser.add_argument("--layer0-attention-fixture", required=True, type=Path)
    parser.add_argument("--layer0-sparse-moe-fixture", required=True, type=Path)
    parser.add_argument("--layer0-fixture", required=True, type=Path)
    parser.add_argument("--ple-fixture", required=True, type=Path)
    parser.add_argument("--layer1-attention-fixture", required=True, type=Path)
    parser.add_argument("--layer1-fixture", required=True, type=Path)
    parser.add_argument("--layers01-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(
        args.checkpoint_dir,
        args.model_lock,
        args.ngram_fixture,
        args.ngram_row_fixture,
        args.layer0_hyper_fixture,
        args.layer0_deltanet_fixture,
        args.layer0_attention_fixture,
        args.layer0_sparse_moe_fixture,
        args.layer0_fixture,
        args.ple_fixture,
        args.layer1_attention_fixture,
        args.layer1_fixture,
        args.layers01_fixture,
    )
    write_json(args.output, fixture)
    print(json.dumps({
        "output": os.fspath(args.output),
        "steps": len(fixture["steps"]),
        "selected_experts": [step["selected_experts"] for step in fixture["steps"]],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
