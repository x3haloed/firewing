#!/usr/bin/env python3
"""Generate deterministic Qwen3.8-Flash-Next tokenizer/template fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import transformers
from transformers import AutoTokenizer


MODEL = "Qwen/Qwen3.8-Flash-Next"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_revision(checkpoint_dir: Path) -> str:
    trees = sorted((checkpoint_dir / ".cache" / "huggingface" / "trees").glob("*.json"))
    if len(trees) != 1 or not re.fullmatch(r"[0-9a-f]{40}", trees[0].stem):
        raise ValueError("checkpoint must contain exactly one 40-hex Hugging Face tree manifest")
    return trees[0].stem


def normalize_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if not isinstance(encoded, list) or not all(isinstance(value, int) for value in encoded):
        raise TypeError(f"unexpected tokenizer output: {type(encoded).__name__}")
    return encoded


def build_fixture(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    raw_texts = [
        ("ascii", "Firewing"),
        ("whitespace", " leading\nline\tend "),
        ("multilingual", "Hello, 世界 — مرحبا — Привет 👋"),
        ("special-looking-text", "literal <|im_start|> marker"),
    ]
    raw_cases = [
        {
            "name": name,
            "text": value,
            "add_special_tokens": False,
            "token_ids": tokenizer.encode(value, add_special_tokens=False),
        }
        for name, value in raw_texts
    ]

    chats = [
        {
            "name": "simple_no_thinking",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Say hello to Firewing."},
            ],
            "options": {"add_generation_prompt": True, "enable_thinking": False},
        },
        {
            "name": "simple_thinking",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "options": {"add_generation_prompt": True, "enable_thinking": True},
        },
        {
            "name": "completed_turn",
            "messages": [
                {"role": "user", "content": "Return one word."},
                {"role": "assistant", "content": "Done"},
            ],
            "options": {"add_generation_prompt": False, "enable_thinking": False},
        },
    ]
    chat_cases = []
    for case in chats:
        rendered = tokenizer.apply_chat_template(
            case["messages"], tokenize=False, **case["options"]
        )
        tokenized = tokenizer.apply_chat_template(
            case["messages"], tokenize=True, **case["options"]
        )
        chat_cases.append(
            {
                **case,
                "rendered": rendered,
                "rendered_utf8_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "token_ids": normalize_ids(tokenized),
            }
        )

    return {
        "schema_version": 1,
        "semantic": "qwen3_8_flash_next_tokenizer_and_chat_template",
        "model": MODEL,
        "revision": checkpoint_revision(checkpoint_dir),
        "reference": {
            "implementation": "huggingface_transformers",
            "transformers_version": transformers.__version__,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_json_sha256": sha256_file(checkpoint_dir / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(checkpoint_dir / "tokenizer_config.json"),
            "chat_template_sha256": sha256_file(checkpoint_dir / "chat_template.jinja"),
        },
        "special_tokens": tokenizer.special_tokens_map,
        "raw_cases": raw_cases,
        "chat_cases": chat_cases,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.checkpoint_dir)
    write_json(args.output, fixture)
    print(json.dumps({"output": os.fspath(args.output), "raw_cases": len(fixture["raw_cases"]), "chat_cases": len(fixture["chat_cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
