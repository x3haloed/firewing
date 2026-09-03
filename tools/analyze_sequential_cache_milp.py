#!/usr/bin/env python3
"""Optimize an offline capacity-respecting cache for FW-0050."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix
import scipy

if __package__:
    from tools import analyze_q2_lossless_experts as lossless
    from tools.build_sequential_shuffle_container import (
        COMPRESSED_BYTES,
        FW0048_RECEIPT_SHA256,
    )
else:
    import analyze_q2_lossless_experts as lossless
    from build_sequential_shuffle_container import COMPRESSED_BYTES, FW0048_RECEIPT_SHA256

MANIFEST_SHA256 = "6759e772d2c9a4560ab39ae80a3b4f4e1a24552adafbf30a396e84166b9c71ca"
CONTAINER_SHA256 = "b14d0f9827a001b97495b97f11d111495f94e8c7392e0ec7d9e7f39095a372bb"
BUILDER_COMMIT = "6271f3dd845f00d183d5b76053718859be0f14bd"
PHYSICAL_BYTES = 7_388_381_184
PAGE_BYTES = 16 * 1024
FAVORABLE_PHYSICAL_BYTES_PER_SECOND = 3_501_482_752.6893535
NODE_LIMIT = 10_000


@dataclass(frozen=True)
class Record:
    identity: str
    compressed_bytes: int
    physical_bytes: int


@dataclass(frozen=True)
class RetentionInterval:
    identity: str
    after_event: int
    hit_event: int


def build_intervals(events: list[list[str]]) -> list[RetentionInterval]:
    occurrences: dict[str, list[int]] = defaultdict(list)
    for event_index, event in enumerate(events):
        if len(event) != len(set(event)):
            raise lossless.AnalysisError("cache event contains a repeated identity")
        for identity in event:
            occurrences[identity].append(event_index)
    intervals = []
    for identity, positions in sorted(occurrences.items()):
        previous = -1
        for position in positions:
            intervals.append(RetentionInterval(identity, previous, position))
            previous = position
    return intervals


def solve_retention(
    records: dict[str, Record],
    events: list[list[str]],
    capacity_bytes: int,
    node_limit: int = NODE_LIMIT,
) -> tuple[list[RetentionInterval], Any]:
    if capacity_bytes <= 0 or node_limit <= 0:
        raise lossless.AnalysisError("cache solver bounds must be positive")
    intervals = build_intervals(events)
    if any(interval.identity not in records for interval in intervals):
        raise lossless.AnalysisError("cache interval references an unknown record")
    rows: list[int] = []
    columns: list[int] = []
    coefficients: list[float] = []
    for column, interval in enumerate(intervals):
        coefficient = records[interval.identity].compressed_bytes / capacity_bytes
        for boundary in range(interval.after_event + 1, interval.hit_event + 1):
            rows.append(boundary)
            columns.append(column)
            coefficients.append(coefficient)
    matrix = csc_matrix(
        (coefficients, (rows, columns)),
        shape=(len(events), len(intervals)),
        dtype=float,
    )
    objective_scale = min(record.physical_bytes for record in records.values())
    objective = -numpy.asarray(
        [records[interval.identity].physical_bytes / objective_scale for interval in intervals],
        dtype=float,
    )
    result = milp(
        objective,
        integrality=numpy.ones(len(intervals), dtype=numpy.uint8),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, 0, 1),
        options={
            "mip_rel_gap": 0.0,
            "node_limit": node_limit,
            "presolve": True,
        },
    )
    if result.x is None or result.fun is None or result.mip_dual_bound is None:
        raise lossless.AnalysisError(f"cache solver produced no certificate: {result.message}")
    return intervals, result


def validate_manifest(manifest: dict[str, Any]) -> tuple[dict[str, Record], list[list[str]], int]:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("semantic")
        != "qwen3_8_flash_next_two_q2_exact_bf16_shuffle_zstd1_page_aligned_expert_container"
        or manifest.get("implementation_commit") != BUILDER_COMMIT
        or manifest.get("model") != lossless.MODEL
        or manifest.get("revision") != lossless.REVISION
        or manifest.get("authorities", {}).get("model_lock_sha256")
        != lossless.MODEL_LOCK_SHA256
        or manifest.get("authorities", {}).get("fw_0048_receipt_sha256")
        != FW0048_RECEIPT_SHA256
        or manifest.get("exact_transform") != "bf16_even_bytes_then_odd_bytes_per_expert"
        or manifest.get("page_bytes") != PAGE_BYTES
        or manifest.get("compressed_bytes") != COMPRESSED_BYTES
        or manifest.get("physical_bytes") != PHYSICAL_BYTES
        or manifest.get("container_sha256") != CONTAINER_SHA256
        or manifest.get("compressed_cache_bytes") != lossless.RESIDENT_LIMIT_BYTES - lossless.FIXED_BYTES
        or manifest.get("performance_claim") is not None
    ):
        raise lossless.AnalysisError("cache manifest authority mismatch")
    raw_records = manifest.get("records")
    transactions = manifest.get("transactions")
    if not isinstance(raw_records, list) or len(raw_records) != 1097:
        raise lossless.AnalysisError("cache manifest record count mismatch")
    records = {
        row["identity"]: Record(
            row["identity"], row["compressed_bytes"], row["physical_bytes"]
        )
        for row in raw_records
    }
    if len(records) != len(raw_records):
        raise lossless.AnalysisError("cache manifest identity collision")
    if any(
        record.compressed_bytes <= 0
        or record.physical_bytes < record.compressed_bytes
        or record.physical_bytes % PAGE_BYTES
        for record in records.values()
    ):
        raise lossless.AnalysisError("cache manifest record ledger mismatch")
    if not isinstance(transactions, list) or len(transactions) != 2:
        raise lossless.AnalysisError("cache manifest transaction count mismatch")
    events = []
    for ordinal, transaction in enumerate(transactions):
        rows = transaction.get("target_rows")
        if (
            transaction.get("ordinal") != ordinal
            or transaction.get("accepted_tokens") != 2
            or not isinstance(rows, list)
            or len(rows) != 2
        ):
            raise lossless.AnalysisError("cache transaction authority mismatch")
        for row in rows:
            if len(row) != 48 or any(len(event) != 10 for event in row):
                raise lossless.AnalysisError("cache event shape mismatch")
            events.extend(row)
    if (
        len(events) != 192
        or sum(map(len, events)) != 1920
        or {identity for event in events for identity in event} != set(records)
    ):
        raise lossless.AnalysisError("cache access union mismatch")
    return records, events, manifest["compressed_cache_bytes"]


def analyze(manifest_path: Path, implementation_commit: str) -> dict[str, Any]:
    lossless.require_clean_commit(implementation_commit)
    lossless.require_hash(manifest_path, MANIFEST_SHA256)
    manifest = lossless.read_json(manifest_path)
    records, events, capacity_bytes = validate_manifest(manifest)
    intervals, result = solve_retention(records, events, capacity_bytes)
    selected = [value >= 0.5 for value in result.x]
    if any(abs(value - round(value)) > 1e-7 for value in result.x):
        raise lossless.AnalysisError("cache solver incumbent is not integral")

    boundary_bytes = [0] * len(events)
    selected_rows = []
    misses_by_event: list[list[str]] = [[] for _ in events]
    for interval, retained in zip(intervals, selected, strict=True):
        if retained:
            record = records[interval.identity]
            for boundary in range(interval.after_event + 1, interval.hit_event + 1):
                boundary_bytes[boundary] += record.compressed_bytes
            selected_rows.append(
                {
                    "identity": interval.identity,
                    "after_event": interval.after_event,
                    "hit_event": interval.hit_event,
                }
            )
        else:
            misses_by_event[interval.hit_event].append(interval.identity)
    if max(boundary_bytes) > capacity_bytes:
        raise lossless.AnalysisError("cache solver certificate exceeds capacity")
    miss_identities = [identity for event in misses_by_event for identity in event]
    hit_count = len(intervals) - len(miss_identities)
    miss_compressed_bytes = sum(records[identity].compressed_bytes for identity in miss_identities)
    miss_physical_bytes = sum(records[identity].physical_bytes for identity in miss_identities)
    total_physical_bytes = sum(
        records[identity].physical_bytes for event in events for identity in event
    )
    objective_scale = min(record.physical_bytes for record in records.values())
    incumbent_from_solver = total_physical_bytes + result.fun * objective_scale
    if not math.isclose(incumbent_from_solver, miss_physical_bytes, abs_tol=0.5):
        raise lossless.AnalysisError("cache solver objective disagrees with replay")
    optimistic_lower_bound = total_physical_bytes + result.mip_dual_bound * objective_scale
    storage_seconds = miss_physical_bytes / FAVORABLE_PHYSICAL_BYTES_PER_SECOND
    return {
        "schema_version": 1,
        "semantic": "qwen3_8_flash_next_two_q2_bf16_shuffle_capacity_respecting_offline_cache_milp",
        "implementation_commit": implementation_commit,
        "model": lossless.MODEL,
        "revision": lossless.REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "container_sha256": CONTAINER_SHA256,
        "solver": {
            "name": "scipy.optimize.milp_highs",
            "scipy_version": scipy.__version__,
            "numpy_version": numpy.__version__,
            "status": int(result.status),
            "message": result.message,
            "node_limit": NODE_LIMIT,
            "nodes_processed": int(result.mip_node_count),
            "relative_gap": float(result.mip_gap),
            "incumbent_objective": float(result.fun),
            "dual_bound": float(result.mip_dual_bound),
        },
        "cache_semantic": "whole_compressed_expert_frames_retained_between_ordered_layer_events",
        "initial_cache_semantic": "free_offline_future_known_capacity_respecting",
        "event_order": "transaction_then_target_row_then_layer_with_simultaneous_top10_event",
        "events": len(events),
        "accesses": len(intervals),
        "unique_experts": len(records),
        "retention_intervals_selected": hit_count,
        "misses": len(miss_identities),
        "compressed_cache_bytes": capacity_bytes,
        "maximum_boundary_resident_compressed_bytes": max(boundary_bytes),
        "miss_compressed_bytes": miss_compressed_bytes,
        "miss_physical_bytes": miss_physical_bytes,
        "total_uncached_physical_bytes": total_physical_bytes,
        "certified_optimistic_minimum_miss_physical_bytes": optimistic_lower_bound,
        "favorable_physical_bytes_per_second": FAVORABLE_PHYSICAL_BYTES_PER_SECOND,
        "incumbent_storage_seconds": storage_seconds,
        "incumbent_storage_only_accepted_tps": 4 / storage_seconds,
        "selected_retention_intervals": selected_rows,
        "misses_by_event": misses_by_event,
        "batch_size": 1,
        "concurrency": 1,
        "q": 2,
        "A": 4,
        "sum_equivalent_U": (697 + 741) / 480,
        "favorable_grants": [
            "the complete two-transaction route future is known",
            "selected initial compressed frames are installed for free",
            "the solver optimizes physical bytes avoided rather than a causal policy",
            "decompression inverse shuffle Metal fixed endpoint work and synchronization are free",
        ],
        "performance_claim": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("implementation_commit")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.manifest, args.implementation_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
