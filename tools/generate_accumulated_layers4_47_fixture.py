#!/usr/bin/env python3
"""Generate hash-only accumulated decoder evidence for layers 4 through 47."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import transformers

if __package__:
    from tools.generate_accumulated_layer3_fixture import build_fixture as build_layer3
    from tools.generate_attention_residual_fixture import build_fixture as build_linear_attention
    from tools.generate_full_attention_residual_fixture import build_fixture as build_full_attention
    from tools.generate_full_decoder_layer3_fixture import build_fixture as build_decoder, write_json
    from tools.generate_ngram_address_fixture import checkpoint_revision, load_model_lock, sha256_file
else:
    from generate_accumulated_layer3_fixture import build_fixture as build_layer3  # type: ignore[no-redef]
    from generate_attention_residual_fixture import build_fixture as build_linear_attention  # type: ignore[no-redef]
    from generate_full_attention_residual_fixture import build_fixture as build_full_attention  # type: ignore[no-redef]
    from generate_full_decoder_layer3_fixture import (  # type: ignore[no-redef]
        build_fixture as build_decoder,
        write_json,
    )
    from generate_ngram_address_fixture import checkpoint_revision, load_model_lock, sha256_file  # type: ignore[no-redef]


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_accumulated_layers4_47_cached_decode"
FIRST_LAYER = 4
LAST_LAYER = 47


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
    layer3_fixture_path: Path,
    full_attention_fixture_path: Path,
    attention_residual_fixture_path: Path,
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock = load_model_lock(model_lock_path)
    revision = checkpoint_revision(checkpoint_dir)
    if revision != lock["revision"]:
        raise ValueError("checkpoint revision does not match model lock")
    raw_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))[
        "text_config"
    ]
    layer_types = raw_config["layer_types"]
    if (
        raw_config["num_hidden_layers"] != 48
        or len(layer_types) != 48
        or raw_config["ple_layer_ids"] != [2]
        or any(
            layer_type != ("full_attention" if layer % 4 == 3 else "linear_attention")
            for layer, layer_type in enumerate(layer_types)
        )
    ):
        raise ValueError("unsupported decoder layer schedule")

    parent_result = build_layer3(
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
        layer2_fixture_path,
        full_attention_fixture_path,
        attention_residual_fixture_path,
        _return_outputs=True,
    )
    if not isinstance(parent_result, tuple):
        raise AssertionError("FW-0026 outputs were not returned")
    generated_parent, current_outputs = parent_result
    if generated_parent != json.loads(layer3_fixture_path.read_text(encoding="utf-8")):
        raise ValueError("regenerated FW-0026 parent disagrees with committed fixture")

    layers = []
    for layer in range(FIRST_LAYER, LAST_LAYER + 1):
        layer_type = layer_types[layer]
        attention_semantic = f"qwen3_8_flash_next_layer{layer}_attention_accumulated"
        decoder_semantic = f"qwen3_8_flash_next_layer{layer}_complete_accumulated"
        reference = {
            "layer3_fixture_sha256": sha256_file(layer3_fixture_path),
            "parent_layer": str(layer - 1),
            "hidden_source": f"accumulated.layer{layer - 1}_output",
            "cache_source": f"sequential_layer{layer}_decode",
        }
        if layer_type == "linear_attention":
            attention_result = build_linear_attention(
                checkpoint_dir,
                model_lock_path,
                layer0_hyper_fixture_path,
                layer0_deltanet_fixture_path,
                _layer=layer,
                _hidden_overrides=current_outputs,
                _semantic=attention_semantic,
                _reference_hashes=reference,
                _return_outputs=True,
            )
            modes = ("initial_chunk", "cached_recurrent")
        else:
            attention_result = build_full_attention(
                checkpoint_dir,
                model_lock_path,
                full_attention_fixture_path,
                _layer=layer,
                _hidden_overrides=current_outputs,
                _past_lengths=(0, 1),
                _modes=("initial", "cached_incremental"),
                _semantic=attention_semantic,
                _reference_hashes=reference,
                _require_committed_parent=False,
                _sequential_cache=True,
                _return_outputs=True,
            )
            modes = ("initial", "cached_incremental")
        if not isinstance(attention_result, tuple):
            raise AssertionError(f"layer-{layer} attention outputs were not returned")
        attention, post_attention = attention_result
        decoder_result = build_decoder(
            checkpoint_dir,
            model_lock_path,
            full_attention_fixture_path,
            attention_residual_fixture_path,
            _parent_execution=(attention, post_attention),
            _parent_semantic=attention_semantic,
            _layer=layer,
            _layer_type=layer_type,
            _semantic=decoder_semantic,
            _reference_hashes=reference,
            _modes=modes,
            _require_committed_parent=False,
            _return_outputs=True,
        )
        if not isinstance(decoder_result, tuple):
            raise AssertionError(f"layer-{layer} decoder outputs were not returned")
        decoder, next_outputs = decoder_result
        steps = []
        for ordinal in range(2):
            attention_step = (
                attention["case"]["steps"][ordinal]
                if layer_type == "linear_attention"
                else attention["cases"][ordinal]
            )
            steps.append(
                {
                    "ordinal": ordinal,
                    "mode": modes[ordinal],
                    "selected_experts": decoder["steps"][ordinal]["selected_experts"],
                    "captures": {
                        "layer_input": attention_step["captures"]["hyper_input"],
                        "post_attention": attention_step["captures"]["composed_output"],
                        "layer_output": decoder["steps"][ordinal]["captures"]["layer_output"],
                    },
                }
            )
        layers.append(
            {
                "layer": layer,
                "layer_type": layer_type,
                "attention": attention,
                "decoder": decoder,
                "steps": steps,
            }
        )
        current_outputs = next_outputs
        del attention_result, post_attention, decoder_result, next_outputs
        gc.collect()

    return {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": revision,
        "reference": {
            "implementation": "source_derived_and_official_huggingface_transformers_qwen4_exp",
            "transformers_version": transformers.__version__,
            "model_lock_sha256": sha256_file(model_lock_path),
            "layer3_fixture_sha256": sha256_file(layer3_fixture_path),
        },
        "configuration": {
            "first_layer": FIRST_LAYER,
            "last_layer": LAST_LAYER,
            "layer_types": layer_types,
            "ple_layer_ids": raw_config["ple_layer_ids"],
            "hidden_size": 2560,
            "hc_count": 4,
            "boundary_dtype": "BF16",
        },
        "layers": layers,
        "final_outputs": [
            layers[-1]["steps"][ordinal]["captures"]["layer_output"] for ordinal in range(2)
        ],
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
    parser.add_argument("--layer2-fixture", required=True, type=Path)
    parser.add_argument("--layer3-fixture", required=True, type=Path)
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
        args.layer3_fixture,
        args.full_attention_fixture,
        args.attention_residual_fixture,
    )
    write_json(args.output, fixture)
    print(
        json.dumps(
            {
                "output": os.fspath(args.output),
                "layers": len(fixture["layers"]),
                "routes": len(fixture["layers"]) * 2,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
