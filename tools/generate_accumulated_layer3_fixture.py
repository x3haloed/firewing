#!/usr/bin/env python3
"""Generate accumulated layer-3 full-attention evidence from FW-0025 outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import transformers

if __package__:
    from tools.generate_accumulated_layer2_fixture import build_fixture as build_layer2
    from tools.generate_full_attention_residual_fixture import build_fixture as build_attention
    from tools.generate_full_decoder_layer3_fixture import build_fixture as build_decoder, write_json
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, sha256_file
else:
    from generate_accumulated_layer2_fixture import build_fixture as build_layer2  # type: ignore[no-redef]
    from generate_full_attention_residual_fixture import build_fixture as build_attention  # type: ignore[no-redef]
    from generate_full_decoder_layer3_fixture import (  # type: ignore[no-redef]
        build_fixture as build_decoder,
        write_json,
    )
    from generate_ngram_address_fixture import checkpoint_revision, load_model_lock, sha256_file  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_accumulated_layer3_cached_decode"
ATTENTION_SEMANTIC = "qwen3_8_flash_next_layer3_attention_accumulated_from_layer2"
DECODER_SEMANTIC = "qwen3_8_flash_next_layer3_complete_accumulated_from_layer2"


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
    layer2_fixture_path: Path,
    full_attention_fixture_path: Path,
    attention_residual_fixture_path: Path,
    *,
    _return_outputs: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], list[Any]]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")

    layer2_result = build_layer2(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        layer0_hyper_fixture_path,
        layer0_deltanet_fixture_path,
        layer0_attention_fixture_path,
        layer0_sparse_moe_fixture_path,
        layer0_fixture_path,
        ple_fixture_path,
        layer1_attention_fixture_path,
        layer1_fixture_path,
        layers01_fixture_path,
        _return_outputs=True,
    )
    if not isinstance(layer2_result, tuple):
        raise AssertionError("FW-0025 outputs were not returned")
    generated_layer2, layer2_outputs = layer2_result
    if generated_layer2 != json.loads(layer2_fixture_path.read_text(encoding="utf-8")):
        raise ValueError("regenerated FW-0025 parent disagrees with committed fixture")

    parent_reference = {
        "layer2_fixture_sha256": sha256_file(layer2_fixture_path),
        "hidden_source": "accumulated.layer2_output",
        "cache_source": "sequential_layer3_decode",
    }
    attention_result = build_attention(
        checkpoint_dir,
        model_lock_path,
        full_attention_fixture_path,
        _hidden_overrides=layer2_outputs,
        _past_lengths=(0, 1),
        _modes=("initial", "cached_incremental"),
        _semantic=ATTENTION_SEMANTIC,
        _reference_hashes=parent_reference,
        _require_committed_parent=False,
        _sequential_cache=True,
        _return_outputs=True,
    )
    if not isinstance(attention_result, tuple):
        raise AssertionError("layer-3 attention outputs were not returned")
    layer3_attention, post_attention = attention_result
    decoder_result = build_decoder(
        checkpoint_dir,
        model_lock_path,
        full_attention_fixture_path,
        attention_residual_fixture_path,
        _parent_execution=(layer3_attention, post_attention),
        _parent_semantic=ATTENTION_SEMANTIC,
        _layer=3,
        _layer_type="full_attention",
        _semantic=DECODER_SEMANTIC,
        _reference_hashes=parent_reference,
        _modes=("initial", "cached_incremental"),
        _require_committed_parent=False,
        _return_outputs=True,
    )
    if not isinstance(decoder_result, tuple):
        raise AssertionError("layer-3 decoder outputs were not returned")
    layer3_decoder, layer3_outputs = decoder_result

    steps = []
    for ordinal in range(2):
        steps.append(
            {
                "ordinal": ordinal,
                "mode": "initial" if ordinal == 0 else "cached_incremental",
                "position": ordinal,
                "past_length": ordinal,
                "selected_experts": layer3_decoder["steps"][ordinal]["selected_experts"],
                "captures": {
                    "layer2_output": generated_layer2["steps"][ordinal]["captures"]["layer2_output"],
                    "post_attention": layer3_attention["cases"][ordinal]["captures"]["composed_output"],
                    "layer3_output": layer3_decoder["steps"][ordinal]["captures"]["layer_output"],
                },
            }
        )
    fixture = {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "source_derived_and_official_huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "source": "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward",
            "model_lock_sha256": sha256_file(model_lock_path),
            "layer2_fixture_sha256": sha256_file(layer2_fixture_path),
        },
        "configuration": {
            "layer": 3,
            "layer_type": "full_attention",
            "ple_applied": False,
            "hidden_size": 2560,
            "hc_count": 4,
            "boundary_dtype": "BF16",
            "cache_lengths": [0, 1],
        },
        "attention": layer3_attention,
        "decoder": layer3_decoder,
        "steps": steps,
    }
    if _return_outputs:
        return fixture, layer3_outputs
    return fixture


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
    parser.add_argument("--layer2-fixture", required=True, type=Path)
    parser.add_argument("--full-attention-fixture", required=True, type=Path)
    parser.add_argument("--attention-residual-fixture", required=True, type=Path)
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
        args.layer2_fixture,
        args.full_attention_fixture,
        args.attention_residual_fixture,
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
