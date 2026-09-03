#!/usr/bin/env python3
"""Compute an exact two-transaction zstd-1 fractional-cache storage oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import zstandard

if __package__:
    from tools import analyze_q2_lossless_experts as lossless
else:
    import analyze_q2_lossless_experts as lossless

FIRST_ENDPOINT_SHA256 = lossless.ENDPOINT_SHA256
FIRST_TRANSACTION_SHA256 = lossless.TRANSACTION_SHA256
MODEL_LOCK_SHA256 = lossless.MODEL_LOCK_SHA256
MODEL = lossless.MODEL
REVISION = lossless.REVISION
EXPERTS = lossless.EXPERTS
EXPERT_BYTES = lossless.EXPERT_BYTES
CACHE_BYTES = lossless.RESIDENT_LIMIT_BYTES - lossless.FIXED_BYTES
SECOND_ENDPOINT_SHA256 = "114dae658be2edf772d3f8b3e4ef7c9ac669387d42c35a26cf739323733ee130"
SECOND_TRANSACTION_SHA256 = "897cf8ad8278847a41e645cc97db79acc5fe1c7b19e0fa34975bbb896b78d573"
FIRST_MANIFEST_SHA256 = "893fa5739e4d4e22f23f5306d2e32ef33bb17af54a7e631fdf5b1286e63cc863"
FW0046_RECEIPT_SHA256 = "fa27310db856c9a2ef2cde1ce2f1a66e0be29f7db1a0b0dfe8357f83216c2c51"
RAW_PHYSICAL_BYTES_PER_SECOND = 3_501_482_752.6893535


def route_records(
    endpoint: dict[str, Any], semantic: str, steps: tuple[int, int], expected: int
) -> dict[tuple[int, int], dict[str, str]]:
    if (
        endpoint.get("schema_version") != 1
        or endpoint.get("semantic") != semantic
        or endpoint.get("model") != MODEL
        or endpoint.get("revision") != REVISION
        or len(endpoint.get("layers", [])) != 48
    ):
        raise lossless.AnalysisError("sequential endpoint identity mismatch")
    records: dict[tuple[int, int], dict[str, str]] = {}
    for layer_index, layer in enumerate(endpoint["layers"]):
        decoder = layer.get("decoder", {})
        if layer.get("layer") != layer_index or len(decoder.get("steps", [])) != max(steps) + 1:
            raise lossless.AnalysisError("sequential endpoint schedule mismatch")
        for ordinal in steps:
            step = decoder["steps"][ordinal]
            selected = step.get("selected_experts")
            entries = step.get("experts")
            if (
                step.get("ordinal") != ordinal
                or not isinstance(selected, list)
                or len(selected) != 10
                or len(set(selected)) != 10
                or sorted(selected) != step.get("expert_execution_order")
                or [entry.get("expert") for entry in entries] != sorted(selected)
            ):
                raise lossless.AnalysisError("sequential endpoint route mismatch")
            for entry in entries:
                identity = (layer_index, entry["expert"])
                hashes = {
                    "gate_up": entry["gate_up_payload_sha256"],
                    "down": entry["down_payload_sha256"],
                }
                if identity in records and records[identity] != hashes:
                    raise lossless.AnalysisError("sequential repeated expert hash mismatch")
                records[identity] = hashes
    if len(records) != expected:
        raise lossless.AnalysisError("sequential target union mismatch")
    return records


def analyze(
    checkpoint: Path,
    model_lock_path: Path,
    first_endpoint_path: Path,
    second_endpoint_path: Path,
    first_transaction_path: Path,
    second_transaction_path: Path,
    first_manifest_path: Path,
    fw0046_receipt_path: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    lossless.require_clean_commit(implementation_commit)
    for path, expected in (
        (model_lock_path, MODEL_LOCK_SHA256),
        (first_endpoint_path, FIRST_ENDPOINT_SHA256),
        (second_endpoint_path, SECOND_ENDPOINT_SHA256),
        (first_transaction_path, FIRST_TRANSACTION_SHA256),
        (second_transaction_path, SECOND_TRANSACTION_SHA256),
        (first_manifest_path, FIRST_MANIFEST_SHA256),
        (fw0046_receipt_path, FW0046_RECEIPT_SHA256),
    ):
        lossless.require_hash(path, expected)
    lock = lossless.read_json(model_lock_path)
    first_endpoint = lossless.read_json(first_endpoint_path)
    second_endpoint = lossless.read_json(second_endpoint_path)
    first_transaction = lossless.read_json(first_transaction_path)
    second_transaction = lossless.read_json(second_transaction_path)
    first_manifest = lossless.read_json(first_manifest_path)
    fw0046 = lossless.read_json(fw0046_receipt_path)
    if (
        lock.get("model") != MODEL
        or lock.get("revision") != REVISION
        or first_transaction.get("decision", {}).get("accepted_tokens") != 2
        or second_transaction.get("decision", {}).get("accepted_tokens") != 2
        or first_transaction.get("configuration", {}).get("q") != 2
        or second_transaction.get("configuration", {}).get("q") != 2
        or first_manifest.get("compressed_bytes") != 5_251_840_172
        or fw0046.get("miss_frames") != 130
    ):
        raise lossless.AnalysisError("sequential supporting authority mismatch")
    first = route_records(
        first_endpoint,
        "qwen3_8_flash_next_firewing_four_token_cached_text_logits",
        (2, 3),
        687,
    )
    second = route_records(
        second_endpoint,
        "qwen3_8_flash_next_firewing_six_token_cached_text_logits",
        (4, 5),
        731,
    )
    for identity in first.keys() & second.keys():
        if first[identity] != second[identity]:
            raise lossless.AnalysisError("sequential cross-transaction expert hash mismatch")
    union = first | second
    if len(union) != 1097 or len(first.keys() & second.keys()) != 321:
        raise lossless.AnalysisError("sequential union/intersection mismatch")

    first_sizes = {
        (record["layer"], record["expert"]): record["compressed_bytes"]
        for record in first_manifest["records"]
    }
    if set(first_sizes) != set(first) or sum(first_sizes.values()) != 5_251_840_172:
        raise lossless.AnalysisError("first container record ledger mismatch")
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
                raise lossless.AnalysisError("sequential expert bank mismatch")
        gate_offset, gate_bytes = lossless.tensor_layout(
            checkpoint / gate["shard"], gate["tensor"]
        )
        down_offset, down_bytes = lossless.tensor_layout(
            checkpoint / down["shard"], down["tensor"]
        )
        if gate_bytes != EXPERTS * 6_553_600 or down_bytes != EXPERTS * 3_276_800:
            raise lossless.AnalysisError("sequential expert layout mismatch")
        layouts[layer_index] = (
            checkpoint / gate["shard"],
            gate_offset,
            checkpoint / down["shard"],
            down_offset,
        )

    compressor = zstandard.ZstdCompressor(level=1)
    decompressor = zstandard.ZstdDecompressor()
    sizes = dict(first_sizes)
    new_source_bytes = 0
    for (layer, expert), hashes in sorted(second.items()):
        if (layer, expert) in sizes:
            continue
        gate_path, gate_base, down_path, down_base = layouts[layer]
        gate = lossless.read_exact(gate_path, gate_base + expert * 6_553_600, 6_553_600)
        down = lossless.read_exact(down_path, down_base + expert * 3_276_800, 3_276_800)
        if (
            lossless.sha256_bytes(gate) != hashes["gate_up"]
            or lossless.sha256_bytes(down) != hashes["down"]
        ):
            raise lossless.AnalysisError("sequential new expert payload mismatch")
        source = gate + down
        encoded = compressor.compress(source)
        if decompressor.decompress(encoded, max_output_size=EXPERT_BYTES) != source:
            raise lossless.AnalysisError("sequential new expert round trip mismatch")
        sizes[(layer, expert)] = len(encoded)
        new_source_bytes += len(source)
    if len(sizes) != 1097 or new_source_bytes != 410 * EXPERT_BYTES:
        raise lossless.AnalysisError("sequential compressed union ledger mismatch")

    union_compressed_bytes = sum(sizes.values())
    minimum_miss_bytes = max(0, union_compressed_bytes - CACHE_BYTES)
    storage_seconds = minimum_miss_bytes / RAW_PHYSICAL_BYTES_PER_SECOND
    accepted = 4
    return {
        "schema_version": 1,
        "semantic": "qwen3_8_flash_next_two_q2_transactions_zstd1_fractional_cache_storage_oracle",
        "implementation_commit": implementation_commit,
        "model": MODEL,
        "revision": REVISION,
        "authorities": {
            "model_lock_sha256": MODEL_LOCK_SHA256,
            "first_endpoint_sha256": FIRST_ENDPOINT_SHA256,
            "second_endpoint_sha256": SECOND_ENDPOINT_SHA256,
            "first_transaction_sha256": FIRST_TRANSACTION_SHA256,
            "second_transaction_sha256": SECOND_TRANSACTION_SHA256,
            "first_manifest_sha256": FIRST_MANIFEST_SHA256,
            "fw_0046_receipt_sha256": FW0046_RECEIPT_SHA256,
        },
        "codec": "zstandard_0.25.0_level_1_independent_expert_frames",
        "first_transaction_experts": len(first),
        "second_transaction_experts": len(second),
        "cross_transaction_experts": len(first.keys() & second.keys()),
        "sequential_union_experts": len(union),
        "new_experts_compressed_and_round_tripped": 410,
        "source_bytes": len(union) * EXPERT_BYTES,
        "compressed_bytes": union_compressed_bytes,
        "compressed_ratio": union_compressed_bytes / (len(union) * EXPERT_BYTES),
        "compressed_cache_bytes": CACHE_BYTES,
        "optimistic_fractional_miss_bytes": minimum_miss_bytes,
        "favorable_physical_bytes_per_second": RAW_PHYSICAL_BYTES_PER_SECOND,
        "optimistic_storage_seconds": storage_seconds,
        "optimistic_storage_only_accepted_tps": accepted / storage_seconds,
        "transaction_count": 2,
        "batch_size": 1,
        "concurrency": 1,
        "q": 2,
        "A": accepted,
        "sum_equivalent_U": (697 + 741) / 480,
        "favorable_grants": [
            "the complete two-transaction future is known",
            "the initial cache is free and may contain fractional expert frames",
            "every distinct expert outside the initial cache is loaded at most once",
            "compressed bytes have no page frame metadata or request amplification",
            "FW-0044 raw physical bandwidth scales perfectly to compressed bytes",
            "decompression Metal MTP fixed endpoint work cache management and synchronization are free",
            "storage overlaps perfectly across both transactions",
        ],
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
    parser.add_argument("first_manifest", type=Path)
    parser.add_argument("fw0046_receipt", type=Path)
    parser.add_argument("implementation_commit")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.checkpoint,
        args.model_lock,
        args.first_endpoint,
        args.second_endpoint,
        args.first_transaction,
        args.second_transaction,
        args.first_manifest,
        args.fw0046_receipt,
        args.implementation_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
