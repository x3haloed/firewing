#!/usr/bin/env python3
"""Build FW-0049's exact page-aligned sequential BF16-shuffle container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import zstandard

if __package__:
    from tools import analyze_q2_lossless_experts as lossless
    from tools import analyze_sequential_q2_zstd_oracle as sequential
    from tools.build_q2_zstd_container import PAGE_BYTES, align_up
else:
    import analyze_q2_lossless_experts as lossless
    import analyze_sequential_q2_zstd_oracle as sequential
    from build_q2_zstd_container import PAGE_BYTES, align_up

FW0048_RECEIPT_SHA256 = (
    "592d5b4e4c45f3733977a9a068c660dd23f90c877f5df1e0afa960841f1f1e89"
)
COMPRESSED_BYTES = 7_381_296_763
FIRST_SEMANTIC = "qwen3_8_flash_next_firewing_four_token_cached_text_logits"
SECOND_SEMANTIC = "qwen3_8_flash_next_firewing_six_token_cached_text_logits"


def target_rows(
    endpoint: dict[str, Any], semantic: str, ordinals: tuple[int, int]
) -> list[list[list[str]]]:
    """Return ordered layer events while reusing the strict route validator."""
    expected = 687 if ordinals == (2, 3) else 731
    sequential.route_records(endpoint, semantic, ordinals, expected)
    rows: list[list[list[str]]] = []
    for ordinal in ordinals:
        row = []
        for layer_index, layer in enumerate(endpoint["layers"]):
            selected = layer["decoder"]["steps"][ordinal]["selected_experts"]
            row.append([f"{layer_index}:{expert}" for expert in selected])
        rows.append(row)
    return rows


def build(
    checkpoint: Path,
    model_lock_path: Path,
    first_endpoint_path: Path,
    second_endpoint_path: Path,
    first_transaction_path: Path,
    second_transaction_path: Path,
    fw0048_receipt_path: Path,
    implementation_commit: str,
    container_path: Path,
) -> dict[str, Any]:
    lossless.require_clean_commit(implementation_commit)
    for path, expected in (
        (model_lock_path, sequential.MODEL_LOCK_SHA256),
        (first_endpoint_path, sequential.FIRST_ENDPOINT_SHA256),
        (second_endpoint_path, sequential.SECOND_ENDPOINT_SHA256),
        (first_transaction_path, sequential.FIRST_TRANSACTION_SHA256),
        (second_transaction_path, sequential.SECOND_TRANSACTION_SHA256),
        (fw0048_receipt_path, FW0048_RECEIPT_SHA256),
    ):
        lossless.require_hash(path, expected)
    lock = lossless.read_json(model_lock_path)
    first_endpoint = lossless.read_json(first_endpoint_path)
    second_endpoint = lossless.read_json(second_endpoint_path)
    first_transaction = lossless.read_json(first_transaction_path)
    second_transaction = lossless.read_json(second_transaction_path)
    fw0048 = lossless.read_json(fw0048_receipt_path)
    if (
        lock.get("model") != sequential.MODEL
        or lock.get("revision") != sequential.REVISION
        or first_transaction.get("decision", {}).get("accepted_tokens") != 2
        or second_transaction.get("decision", {}).get("accepted_tokens") != 2
        or first_transaction.get("configuration", {}).get("q") != 2
        or second_transaction.get("configuration", {}).get("q") != 2
        or fw0048.get("compressed_bytes") != COMPRESSED_BYTES
        or fw0048.get("exact_transform") != "bf16_even_bytes_then_odd_bytes"
        or fw0048.get("sequential_union_experts") != 1097
    ):
        raise lossless.AnalysisError("sequential container authority mismatch")

    first = sequential.route_records(first_endpoint, FIRST_SEMANTIC, (2, 3), 687)
    second = sequential.route_records(second_endpoint, SECOND_SEMANTIC, (4, 5), 731)
    for identity in first.keys() & second.keys():
        if first[identity] != second[identity]:
            raise lossless.AnalysisError("sequential container repeated hash mismatch")
    union = first | second
    if len(union) != 1097 or len(first.keys() & second.keys()) != 321:
        raise lossless.AnalysisError("sequential container union mismatch")
    transactions = [
        {
            "ordinal": 0,
            "accepted_tokens": 2,
            "target_rows": target_rows(first_endpoint, FIRST_SEMANTIC, (2, 3)),
        },
        {
            "ordinal": 1,
            "accepted_tokens": 2,
            "target_rows": target_rows(second_endpoint, SECOND_SEMANTIC, (4, 5)),
        },
    ]

    locked = {entry["path"]: entry for entry in lock["files"]}
    layouts: dict[int, tuple[Path, int, Path, int]] = {}
    for layer_index, layer in enumerate(second_endpoint["layers"]):
        banks = layer["decoder"]["expert_banks"]
        gate = banks["gate_up"]
        down = banks["down"]
        for bank in (gate, down):
            entry = locked.get(bank["shard"])
            path = checkpoint / bank["shard"]
            if (
                entry is None
                or entry.get("size") != bank.get("shard_bytes")
                or entry.get("lfs_sha256") != bank.get("shard_sha256")
                or path.stat().st_size != bank.get("shard_bytes")
            ):
                raise lossless.AnalysisError("sequential container expert bank mismatch")
        gate_offset, gate_bytes = lossless.tensor_layout(
            checkpoint / gate["shard"], gate["tensor"]
        )
        down_offset, down_bytes = lossless.tensor_layout(
            checkpoint / down["shard"], down["tensor"]
        )
        if (
            gate_bytes != sequential.EXPERTS * 6_553_600
            or down_bytes != sequential.EXPERTS * 3_276_800
        ):
            raise lossless.AnalysisError("sequential container expert layout mismatch")
        layouts[layer_index] = (
            checkpoint / gate["shard"],
            gate_offset,
            checkpoint / down["shard"],
            down_offset,
        )

    container_path.parent.mkdir(parents=True, exist_ok=True)
    partial = container_path.with_name(container_path.name + ".partial")
    if partial.exists():
        partial.unlink()
    compressor = zstandard.ZstdCompressor(level=1)
    decompressor = zstandard.ZstdDecompressor()
    records = []
    container_digest = hashlib.sha256()
    compressed_total = 0
    physical_total = 0
    zeroes = bytes(PAGE_BYTES)
    with partial.open("xb") as output:
        for (layer, expert), hashes in sorted(union.items()):
            gate_path, gate_base, down_path, down_base = layouts[layer]
            gate = lossless.read_exact(gate_path, gate_base + expert * 6_553_600, 6_553_600)
            down = lossless.read_exact(down_path, down_base + expert * 3_276_800, 3_276_800)
            if (
                lossless.sha256_bytes(gate) != hashes["gate_up"]
                or lossless.sha256_bytes(down) != hashes["down"]
            ):
                raise lossless.AnalysisError(
                    f"sequential container payload mismatch: {layer}:{expert}"
                )
            source = gate + down
            encoded = compressor.compress(sequential.shuffle_bf16(source))
            decoded = decompressor.decompress(encoded, max_output_size=sequential.EXPERT_BYTES)
            if sequential.unshuffle_bf16(decoded) != source:
                raise lossless.AnalysisError(
                    f"sequential container round trip mismatch: {layer}:{expert}"
                )
            offset = physical_total
            physical_bytes = align_up(len(encoded))
            padding = physical_bytes - len(encoded)
            output.write(encoded)
            container_digest.update(encoded)
            while padding:
                chunk = min(padding, len(zeroes))
                output.write(zeroes[:chunk])
                container_digest.update(zeroes[:chunk])
                padding -= chunk
            records.append(
                {
                    "identity": f"{layer}:{expert}",
                    "layer": layer,
                    "expert": expert,
                    "offset": offset,
                    "compressed_bytes": len(encoded),
                    "physical_bytes": physical_bytes,
                    "frame_sha256": lossless.sha256_bytes(encoded),
                    "source_sha256": lossless.sha256_bytes(source),
                }
            )
            compressed_total += len(encoded)
            physical_total += physical_bytes
        output.flush()
        os.fsync(output.fileno())
    if compressed_total != COMPRESSED_BYTES:
        partial.unlink()
        raise lossless.AnalysisError("sequential container compressed ledger mismatch")
    os.replace(partial, container_path)
    return {
        "schema_version": 1,
        "semantic": "qwen3_8_flash_next_two_q2_exact_bf16_shuffle_zstd1_page_aligned_expert_container",
        "implementation_commit": implementation_commit,
        "model": sequential.MODEL,
        "revision": sequential.REVISION,
        "authorities": {
            "model_lock_sha256": sequential.MODEL_LOCK_SHA256,
            "first_endpoint_sha256": sequential.FIRST_ENDPOINT_SHA256,
            "second_endpoint_sha256": sequential.SECOND_ENDPOINT_SHA256,
            "first_transaction_sha256": sequential.FIRST_TRANSACTION_SHA256,
            "second_transaction_sha256": sequential.SECOND_TRANSACTION_SHA256,
            "fw_0048_receipt_sha256": FW0048_RECEIPT_SHA256,
        },
        "codec": {
            "name": "zstandard",
            "python_package_version": zstandard.__version__,
            "level": 1,
            "frame_content_size": True,
            "independent_frames": True,
        },
        "exact_transform": "bf16_even_bytes_then_odd_bytes_per_expert",
        "page_bytes": PAGE_BYTES,
        "source_bytes_per_expert": sequential.EXPERT_BYTES,
        "fixed_resident_bytes": lossless.FIXED_BYTES,
        "resident_limit_bytes": lossless.RESIDENT_LIMIT_BYTES,
        "compressed_cache_bytes": sequential.CACHE_BYTES,
        "records": records,
        "transactions": transactions,
        "source_bytes": len(records) * sequential.EXPERT_BYTES,
        "compressed_bytes": compressed_total,
        "physical_bytes": physical_total,
        "container_file": container_path.name,
        "container_sha256": container_digest.hexdigest(),
        "exact_round_trips": len(records),
        "batch_size": 1,
        "concurrency": 1,
        "sampling": "greedy",
        "q": 2,
        "A": 4,
        "sum_equivalent_U": (697 + 741) / 480,
        "performance_claim": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model_lock", type=Path)
    parser.add_argument("first_endpoint", type=Path)
    parser.add_argument("second_endpoint", type=Path)
    parser.add_argument("first_transaction", type=Path)
    parser.add_argument("second_transaction", type=Path)
    parser.add_argument("fw0048_receipt", type=Path)
    parser.add_argument("implementation_commit")
    parser.add_argument("container", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = build(
        args.checkpoint,
        args.model_lock,
        args.first_endpoint,
        args.second_endpoint,
        args.first_transaction,
        args.second_transaction,
        args.fw0048_receipt,
        args.implementation_commit,
        args.container,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
