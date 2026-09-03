#!/usr/bin/env python3
"""Hash sparse Qwen n-gram rows without loading or committing model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO


MODEL = "Qwen/Qwen3.8-Flash-Next"
SEMANTIC = "qwen3_8_flash_next_ngram_row_hashes"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safetensor_payload_start(handle: BinaryIO) -> int:
    handle.seek(0)
    raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError("truncated safetensors header length")
    header_length = int.from_bytes(raw, "little")
    if header_length <= 0 or header_length > 16 * 1024 * 1024:
        raise ValueError("unsupported safetensors header length")
    return 8 + header_length


def read_exact_row(
    handle: BinaryIO,
    payload_start: int,
    tensor_start: int,
    local_row: int,
    rows_per_part: int,
    row_bytes: int,
) -> bytes:
    if not 0 <= local_row < rows_per_part:
        raise ValueError("n-gram local row is out of bounds")
    handle.seek(payload_start + tensor_start + local_row * row_bytes)
    value = handle.read(row_bytes)
    if len(value) != row_bytes:
        raise ValueError("truncated n-gram row payload")
    return value


def build_fixture(checkpoint_dir: Path, address_fixture_path: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    address = json.loads(address_fixture_path.read_text(encoding="utf-8"))
    if (
        address.get("schema_version") != 1
        or address.get("semantic") != "qwen3_8_flash_next_ngram_addresses"
        or address.get("model") != MODEL
    ):
        raise ValueError("unsupported n-gram address fixture")
    config = address["configuration"]
    row_bytes = config["head_width"] * 2
    if row_bytes != 320 or config["rows_per_shard"] != 2_500_012:
        raise ValueError("unsupported n-gram physical row layout")
    parts = address["table_parts"]
    if len(parts) != config["split_parts"] or any(part["part"] != index for index, part in enumerate(parts)):
        raise ValueError("n-gram table parts are not in numeric order")

    handles: dict[str, BinaryIO] = {}
    payload_starts: dict[str, int] = {}
    try:
        cases = []
        for case in address["cases"]:
            row_hashes = []
            for token_rows, physical_rows in zip(
                case["global_rows"], case["physical_rows"], strict=True
            ):
                if len(token_rows) != config["ngram_heads"]:
                    raise ValueError("address fixture has an unexpected head count")
                token_hashes = []
                for global_row, physical in zip(token_rows, physical_rows, strict=True):
                    expected_part = global_row // config["rows_per_shard"]
                    expected_row = global_row % config["rows_per_shard"]
                    if physical != {"shard": expected_part, "row": expected_row}:
                        raise ValueError("address fixture physical mapping is inconsistent")
                    part = parts[expected_part]
                    shard = part["shard"]
                    if shard not in handles:
                        handles[shard] = (checkpoint_dir / shard).open("rb")
                        payload_starts[shard] = safetensor_payload_start(handles[shard])
                    value = read_exact_row(
                        handles[shard],
                        payload_starts[shard],
                        part["data_offsets"][0],
                        expected_row,
                        config["rows_per_shard"],
                        row_bytes,
                    )
                    token_hashes.append(hashlib.sha256(value).hexdigest())
                row_hashes.append(token_hashes)
            cases.append({"name": case["name"], "row_sha256": row_hashes})
    finally:
        for handle in handles.values():
            handle.close()

    return {
        "schema_version": 1,
        "semantic": SEMANTIC,
        "model": MODEL,
        "revision": address["revision"],
        "address_fixture_sha256": sha256_file(address_fixture_path),
        "row_bytes": row_bytes,
        "cases": cases,
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
    parser.add_argument("address_fixture", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture = build_fixture(args.checkpoint_dir, args.address_fixture)
    write_json(args.output, fixture)
    print(json.dumps({"output": os.fspath(args.output), "cases": len(fixture["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
