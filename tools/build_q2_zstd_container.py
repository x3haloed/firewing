#!/usr/bin/env python3
"""Build the external page-aligned exact q2 zstd expert container for FW-0046."""

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
else:
    import analyze_q2_lossless_experts as lossless

ENDPOINT_SHA256 = lossless.ENDPOINT_SHA256
EXPERT_BYTES = lossless.EXPERT_BYTES
EXPERTS = lossless.EXPERTS
FIXED_BYTES = lossless.FIXED_BYTES
FW0044_SHA256 = lossless.FW0044_SHA256
LAYERS = lossless.LAYERS
MODEL = lossless.MODEL
MODEL_LOCK_SHA256 = lossless.MODEL_LOCK_SHA256
RESIDENT_LIMIT_BYTES = lossless.RESIDENT_LIMIT_BYTES
REVISION = lossless.REVISION
TRANSACTION_SHA256 = lossless.TRANSACTION_SHA256
AnalysisError = lossless.AnalysisError
read_exact = lossless.read_exact
read_json = lossless.read_json
require_clean_commit = lossless.require_clean_commit
require_hash = lossless.require_hash
route_authority = lossless.route_authority
sha256_bytes = lossless.sha256_bytes
tensor_layout = lossless.tensor_layout

PAGE_BYTES = 16 * 1024
FW0045_COMPRESSED_BYTES = 5_251_840_172


def align_up(value: int, alignment: int = PAGE_BYTES) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise AnalysisError("invalid alignment request")
    return (value + alignment - 1) & -alignment


def q2_events(endpoint: dict[str, Any]) -> list[list[list[str]]]:
    route_authority(endpoint)
    rows: list[list[list[str]]] = [[], []]
    for row_index, ordinal in enumerate((2, 3)):
        for layer_index, layer in enumerate(endpoint["layers"]):
            selected = layer["decoder"]["steps"][ordinal]["selected_experts"]
            rows[row_index].append([f"{layer_index}:{expert}" for expert in selected])
    if len(rows[0]) != LAYERS or len(rows[1]) != LAYERS:
        raise AnalysisError("q2 event count mismatch")
    return rows


def build(
    checkpoint: Path,
    model_lock_path: Path,
    endpoint_path: Path,
    transaction_path: Path,
    fw0044_path: Path,
    implementation_commit: str,
    container_path: Path,
) -> dict[str, Any]:
    require_clean_commit(implementation_commit)
    require_hash(model_lock_path, MODEL_LOCK_SHA256)
    require_hash(endpoint_path, ENDPOINT_SHA256)
    require_hash(transaction_path, TRANSACTION_SHA256)
    require_hash(fw0044_path, FW0044_SHA256)
    endpoint = read_json(endpoint_path)
    routes = route_authority(endpoint)
    events = q2_events(endpoint)
    lock = read_json(model_lock_path)
    transaction = read_json(transaction_path)
    if (
        lock.get("model") != MODEL
        or lock.get("revision") != REVISION
        or transaction.get("decision", {}).get("accepted_tokens") != 2
        or transaction.get("configuration", {}).get("q") != 2
    ):
        raise AnalysisError("container authority mismatch")
    locked = {entry["path"]: entry for entry in lock["files"]}
    layouts: dict[int, tuple[Path, int, Path, int]] = {}
    for layer_index, layer in enumerate(endpoint["layers"]):
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
                raise AnalysisError("expert bank identity mismatch")
        gate_offset, gate_bytes = tensor_layout(checkpoint / gate["shard"], gate["tensor"])
        down_offset, down_bytes = tensor_layout(checkpoint / down["shard"], down["tensor"])
        if gate_bytes != EXPERTS * 6_553_600 or down_bytes != EXPERTS * 3_276_800:
            raise AnalysisError("expert bank layout mismatch")
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
    records = []
    container_digest = hashlib.sha256()
    compressed_total = 0
    physical_total = 0
    zeroes = bytes(PAGE_BYTES)
    with partial.open("xb") as output:
        for (layer, expert), hashes in sorted(routes.items()):
            gate_path, gate_base, down_path, down_base = layouts[layer]
            gate = read_exact(gate_path, gate_base + expert * 6_553_600, 6_553_600)
            down = read_exact(down_path, down_base + expert * 3_276_800, 3_276_800)
            if sha256_bytes(gate) != hashes["gate_up"] or sha256_bytes(down) != hashes["down"]:
                raise AnalysisError(f"expert payload mismatch: layer {layer} expert {expert}")
            source = gate + down
            encoded = compressor.compress(source)
            if zstandard.ZstdDecompressor().decompress(
                encoded, max_output_size=EXPERT_BYTES
            ) != source:
                raise AnalysisError(f"container round trip mismatch: layer {layer} expert {expert}")
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
                    "frame_sha256": sha256_bytes(encoded),
                    "source_sha256": sha256_bytes(source),
                }
            )
            compressed_total += len(encoded)
            physical_total += physical_bytes
        output.flush()
        os.fsync(output.fileno())
    if compressed_total != FW0045_COMPRESSED_BYTES:
        partial.unlink()
        raise AnalysisError("container compressed-byte ledger differs from FW-0045")
    os.replace(partial, container_path)
    return {
        "schema_version": 1,
        "semantic": "qwen3_8_flash_next_q2_exact_zstd1_page_aligned_expert_container",
        "implementation_commit": implementation_commit,
        "model": MODEL,
        "revision": REVISION,
        "authorities": {
            "model_lock_sha256": MODEL_LOCK_SHA256,
            "endpoint_fixture_sha256": ENDPOINT_SHA256,
            "transaction_fixture_sha256": TRANSACTION_SHA256,
            "fw_0044_receipt_sha256": FW0044_SHA256,
        },
        "codec": {
            "name": "zstandard",
            "python_package_version": zstandard.__version__,
            "level": 1,
            "frame_content_size": True,
            "independent_frames": True,
        },
        "page_bytes": PAGE_BYTES,
        "source_bytes_per_expert": EXPERT_BYTES,
        "fixed_resident_bytes": FIXED_BYTES,
        "resident_limit_bytes": RESIDENT_LIMIT_BYTES,
        "compressed_cache_bytes": RESIDENT_LIMIT_BYTES - FIXED_BYTES,
        "records": records,
        "target_rows": events,
        "source_bytes": len(records) * EXPERT_BYTES,
        "compressed_bytes": compressed_total,
        "physical_bytes": physical_total,
        "container_file": container_path.name,
        "container_sha256": container_digest.hexdigest(),
        "exact_round_trips": len(records),
        "batch_size": 1,
        "concurrency": 1,
        "sampling": "greedy",
        "q": 2,
        "A": 2,
        "U": 697 / 480,
        "performance_claim": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model_lock", type=Path)
    parser.add_argument("endpoint", type=Path)
    parser.add_argument("transaction", type=Path)
    parser.add_argument("fw0044_receipt", type=Path)
    parser.add_argument("implementation_commit")
    parser.add_argument("container", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = build(
        args.checkpoint,
        args.model_lock,
        args.endpoint,
        args.transaction,
        args.fw0044_receipt,
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
