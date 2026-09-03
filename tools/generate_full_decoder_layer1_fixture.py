#!/usr/bin/env python3
"""Generate a complete real layer-1 PLE-bearing decoder fixture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

if __package__:
    from tools.generate_full_decoder_layer3_fixture import (
        build_fixture as build_decoder_fixture,
        write_json,
    )
    from tools.generate_ngram_address_fixture import sha256_file
    from tools.generate_ple_attention_residual_fixture import build_fixture as build_attention_fixture
else:
    from generate_full_decoder_layer3_fixture import (  # type: ignore[no-redef]
        build_fixture as build_decoder_fixture,
        write_json,
    )
    from generate_ngram_address_fixture import sha256_file  # type: ignore[no-redef]
    from generate_ple_attention_residual_fixture import build_fixture as build_attention_fixture  # type: ignore[no-redef]


SEMANTIC = "qwen3_8_flash_next_layer1_ple_complete_decoder"
PARENT_SEMANTIC = "qwen3_8_flash_next_layer1_ple_attention_residual_cached_decode"


def build_fixture(
    checkpoint_dir: Path,
    model_lock_path: Path,
    ngram_fixture_path: Path,
    ngram_row_fixture_path: Path,
    ple_fixture_path: Path,
    attention_residual_fixture_path: Path,
    *,
    _hidden_overrides: list[torch.Tensor] | None = None,
    _token_ids: list[int] | None = None,
    _semantic: str = SEMANTIC,
    _reference_hashes: dict[str, str] | None = None,
    _return_chain: bool = False,
) -> dict | tuple[dict, list[torch.Tensor], dict, dict]:
    parent_result = build_attention_fixture(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        ple_fixture_path,
        _hidden_overrides=_hidden_overrides,
        _token_ids=_token_ids,
        _return_chain=True,
    )
    if not isinstance(parent_result, tuple) or len(parent_result) != 3:
        raise AssertionError("layer-1 attention outputs were not returned")
    generated, post_attention, generated_ple = parent_result
    if (
        _hidden_overrides is None
        and generated != json.loads(attention_residual_fixture_path.read_text(encoding="utf-8"))
    ):
        raise ValueError("layer-1 attention parent fixture mismatch")
    reference_hashes = _reference_hashes or {
        "ngram_fixture_sha256": sha256_file(ngram_fixture_path),
        "ngram_row_fixture_sha256": sha256_file(ngram_row_fixture_path),
        "ple_fixture_sha256": sha256_file(ple_fixture_path),
        "attention_residual_fixture_sha256": sha256_file(attention_residual_fixture_path),
    }
    decoder_result = build_decoder_fixture(
        checkpoint_dir,
        model_lock_path,
        ple_fixture_path,
        attention_residual_fixture_path,
        _parent_execution=(generated, post_attention),
        _parent_semantic=PARENT_SEMANTIC,
        _layer=1,
        _layer_type="linear_attention",
        _semantic=_semantic,
        _reference_hashes=reference_hashes,
        _modes=tuple(
            "initial_chunk" if ordinal == 0 else "cached_recurrent"
            for ordinal in range(len(post_attention))
        ),
        _require_committed_parent=_hidden_overrides is None,
        _return_outputs=True,
    )
    if not isinstance(decoder_result, tuple):
        raise AssertionError("layer-1 decoder outputs were not returned")
    fixture, outputs = decoder_result
    if _return_chain:
        return fixture, outputs, generated, generated_ple
    return fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--ngram-fixture", required=True, type=Path)
    parser.add_argument("--ngram-row-fixture", required=True, type=Path)
    parser.add_argument("--ple-fixture", required=True, type=Path)
    parser.add_argument("--attention-residual-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(
        args.checkpoint_dir,
        args.model_lock,
        args.ngram_fixture,
        args.ngram_row_fixture,
        args.ple_fixture,
        args.attention_residual_fixture,
    )
    write_json(args.output, fixture)
    print(json.dumps({"output": os.fspath(args.output), "tensors": len(fixture["tensors"]), "steps": len(fixture["steps"]), "selected_experts": [step["selected_experts"] for step in fixture["steps"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
