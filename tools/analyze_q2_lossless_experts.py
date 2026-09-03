#!/usr/bin/env python3
"""Measure exact per-expert zstd coding and an impossible-favorable q2 bound."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import zstandard

MODEL = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
ENDPOINT_SHA256 = "e2ccf01a37cc5cb2cf44a30185850b8910b06233bc32d7ddaaeb537204daa899"
TRANSACTION_SHA256 = "9954668a28b64944c0830760a799383082e834be22106ec1613df12d748b9757"
MODEL_LOCK_SHA256 = "f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444"
FW0044_SHA256 = "6aa3f7cc04d35a8686ceb6c3c0b55f22b548129b67db242975b6693c79d5d6f9"
LAYERS = 48
EXPERTS = 512
EXPERT_BYTES = 9_830_400
FIXED_BYTES = 8_623_999_000
RESIDENT_LIMIT_BYTES = 12 * 1024**3
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class AnalysisError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc


def require_hash(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise AnalysisError(f"authority hash mismatch: {path}")


def require_clean_commit(commit: str) -> None:
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not GIT_COMMIT_RE.fullmatch(commit) or commit != head or dirty:
        raise AnalysisError("implementation commit must be exact clean Git HEAD")


def tensor_layout(path: Path, tensor: str) -> tuple[int, int]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise AnalysisError(f"truncated safetensors prefix: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 0 < header_bytes <= 64 * 1024 * 1024:
            raise AnalysisError(f"invalid safetensors header length: {path}")
        header = json.loads(handle.read(header_bytes))
    entry = header.get(tensor)
    if not isinstance(entry, dict) or entry.get("dtype") != "BF16":
        raise AnalysisError(f"missing BF16 tensor {tensor}")
    offsets = entry.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise AnalysisError(f"invalid offsets for {tensor}")
    return 8 + header_bytes + offsets[0], offsets[1] - offsets[0]


def route_authority(endpoint: dict[str, Any]) -> dict[tuple[int, int], dict[str, str]]:
    if (
        endpoint.get("schema_version") != 1
        or endpoint.get("semantic")
        != "qwen3_8_flash_next_firewing_four_token_cached_text_logits"
        or endpoint.get("model") != MODEL
        or endpoint.get("revision") != REVISION
        or len(endpoint.get("layers", [])) != LAYERS
    ):
        raise AnalysisError("endpoint identity mismatch")
    records: dict[tuple[int, int], dict[str, str]] = {}
    for layer_index, layer in enumerate(endpoint["layers"]):
        decoder = layer.get("decoder", {})
        if layer.get("layer") != layer_index or len(decoder.get("steps", [])) != 4:
            raise AnalysisError("endpoint schedule mismatch")
        for ordinal in (2, 3):
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
                raise AnalysisError("endpoint route mismatch")
            for entry in entries:
                identity = (layer_index, entry["expert"])
                hashes = {
                    "gate_up": entry["gate_up_payload_sha256"],
                    "down": entry["down_payload_sha256"],
                }
                if identity in records and records[identity] != hashes:
                    raise AnalysisError("repeated expert hash mismatch")
                records[identity] = hashes
    if len(records) != 687:
        raise AnalysisError("q2 target union differs from FW-0040")
    return records


def read_exact(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb", buffering=0) as handle:
        handle.seek(offset)
        data = handle.read(size)
    if len(data) != size:
        raise AnalysisError(f"short tensor read: {path}")
    return data


def fw0044_constants(prior: dict[str, Any]) -> tuple[list[int], list[float], int]:
    physical: list[int] = []
    storage_ms: list[float] = []
    target_compute_ns = 0
    trials = prior.get("trials", [])
    for token in (0, 1):
        controls = [
            row
            for row in trials
            if row.get("mode") == "storage_only_control" and row.get("token_ordinal") == token
        ]
        candidates = [
            row
            for row in trials
            if row.get("mode") == "storage_compute_overlap" and row.get("token_ordinal") == token
        ]
        if len(controls) != 3 or len(candidates) != 3:
            raise AnalysisError("FW-0044 trial count mismatch")
        physical_values = {row.get("process_disk_bytes_read") for row in controls + candidates}
        if len(physical_values) != 1 or None in physical_values:
            raise AnalysisError("FW-0044 physical-byte ledger mismatch")
        physical.append(physical_values.pop())
        control_walls = sorted(row["complete_wall_time_ns"] for row in controls)
        compute_walls = sorted(row["compute_wall_time_ns"] for row in candidates)
        storage_ms.append(control_walls[1] / 1_000_000)
        target_compute_ns += compute_walls[1]
    return physical, storage_ms, target_compute_ns


def analyze(
    checkpoint: Path,
    model_lock_path: Path,
    endpoint_path: Path,
    transaction_path: Path,
    fw0044_path: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    require_clean_commit(implementation_commit)
    require_hash(model_lock_path, MODEL_LOCK_SHA256)
    require_hash(endpoint_path, ENDPOINT_SHA256)
    require_hash(transaction_path, TRANSACTION_SHA256)
    require_hash(fw0044_path, FW0044_SHA256)
    lock = read_json(model_lock_path)
    transaction = read_json(transaction_path)
    prior = read_json(fw0044_path)
    if (
        lock.get("model") != MODEL
        or lock.get("revision") != REVISION
        or transaction.get("decision", {}).get("accepted_tokens") != 2
        or transaction.get("configuration", {}).get("q") != 2
        or prior.get("A") != 2
        or prior.get("misses_by_target_row") != [47, 207]
    ):
        raise AnalysisError("supporting authority mismatch")
    locked = {entry["path"]: entry for entry in lock["files"]}
    endpoint = read_json(endpoint_path)
    routes = route_authority(endpoint)
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

    compressor = zstandard.ZstdCompressor(level=1)
    decompressor = zstandard.ZstdDecompressor()
    compressed_sizes: list[int] = []
    decompression_ns: list[int] = []
    for (layer, expert), hashes in sorted(routes.items()):
        gate_path, gate_base, down_path, down_base = layouts[layer]
        gate = read_exact(gate_path, gate_base + expert * 6_553_600, 6_553_600)
        down = read_exact(down_path, down_base + expert * 3_276_800, 3_276_800)
        if sha256_bytes(gate) != hashes["gate_up"] or sha256_bytes(down) != hashes["down"]:
            raise AnalysisError(f"expert payload mismatch: layer {layer} expert {expert}")
        source = gate + down
        encoded = compressor.compress(source)
        started = time.perf_counter_ns()
        decoded = decompressor.decompress(encoded, max_output_size=EXPERT_BYTES)
        elapsed = time.perf_counter_ns() - started
        if decoded != source:
            raise AnalysisError(f"lossless round trip mismatch: layer {layer} expert {expert}")
        compressed_sizes.append(len(encoded))
        decompression_ns.append(elapsed)

    source_bytes = len(routes) * EXPERT_BYTES
    compressed_bytes = sum(compressed_sizes)
    cache_bytes = RESIDENT_LIMIT_BYTES - FIXED_BYTES
    optimistic_miss_bytes = max(0, compressed_bytes - cache_bytes)
    physical_by_row, storage_ms_by_row, target_compute_ns = fw0044_constants(prior)
    raw_physical_bytes = sum(physical_by_row)
    raw_storage_ns = sum(storage_ms_by_row) * 1_000_000
    physical_bytes_per_second = raw_physical_bytes / (raw_storage_ns / 1e9)
    optimistic_storage_ns = optimistic_miss_bytes / physical_bytes_per_second * 1e9
    measured_decode_ns = sum(decompression_ns)
    optimistic_decode_ns = measured_decode_ns * optimistic_miss_bytes / compressed_bytes
    optimistic_wall_ns = max(optimistic_storage_ns, optimistic_decode_ns, target_compute_ns)
    return {
        "schema_version": 1,
        "semantic": "qwen3_8_flash_next_q2_exact_per_expert_zstd1_favorable_bound",
        "implementation_commit": implementation_commit,
        "model": MODEL,
        "revision": REVISION,
        "authorities": {
            "model_lock_sha256": MODEL_LOCK_SHA256,
            "endpoint_fixture_sha256": ENDPOINT_SHA256,
            "transaction_fixture_sha256": TRANSACTION_SHA256,
            "fw_0044_receipt_sha256": FW0044_SHA256,
        },
        "codec": {"name": "zstandard", "python_package_version": zstandard.__version__, "level": 1},
        "independent_expert_frames": len(routes),
        "source_bytes": source_bytes,
        "compressed_bytes": compressed_bytes,
        "compressed_ratio": compressed_bytes / source_bytes,
        "minimum_frame_bytes": min(compressed_sizes),
        "median_frame_bytes": sorted(compressed_sizes)[len(compressed_sizes) // 2],
        "maximum_frame_bytes": max(compressed_sizes),
        "exact_round_trips": len(routes),
        "resident_limit_bytes": RESIDENT_LIMIT_BYTES,
        "fixed_resident_bytes": FIXED_BYTES,
        "compressed_cache_bytes": cache_bytes,
        "optimistic_fractional_miss_bytes": optimistic_miss_bytes,
        "measured_full_union_decompression_ns": measured_decode_ns,
        "optimistic_fractional_miss_decompression_ns": optimistic_decode_ns,
        "fw_0044_physical_bytes_per_second": physical_bytes_per_second,
        "optimistic_compressed_storage_ns": optimistic_storage_ns,
        "exact_target_metal_ns": target_compute_ns,
        "optimistic_perfect_three_way_overlap_wall_ns": optimistic_wall_ns,
        "optimistic_accepted_bound_tps": 2e9 / optimistic_wall_ns,
        "batch_size": 1,
        "concurrency": 1,
        "sampling": "greedy",
        "q": 2,
        "A": 2,
        "U": 697 / 480,
        "favorable_grants": [
            "fractional compressed cache allocation with free future-known initial contents",
            "compressed miss bytes have no page, frame, metadata, or request amplification",
            "FW-0044 raw physical bandwidth scales perfectly to compressed bytes",
            "storage, decompression, and exact target Metal compute overlap perfectly",
            "only the fractional missing share of measured union decompression is charged",
            "MTP, fixed endpoint work, cache management, synchronization, and sampling are free",
        ],
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
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.checkpoint,
        args.model_lock,
        args.endpoint,
        args.transaction,
        args.fw0044_receipt,
        args.implementation_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
