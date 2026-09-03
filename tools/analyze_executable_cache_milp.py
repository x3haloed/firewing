#!/usr/bin/env python3
"""Optimize an offline mixed compressed/decoded cache for FW-0053."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

if __package__:
    from tools import analyze_q2_lossless_experts as lossless
    from tools.analyze_sequential_cache_milp import (
        CONTAINER_SHA256,
        FAVORABLE_PHYSICAL_BYTES_PER_SECOND,
        MANIFEST_SHA256,
        Record,
        RetentionInterval,
        build_intervals,
        validate_manifest,
    )
else:
    import analyze_q2_lossless_experts as lossless
    from analyze_sequential_cache_milp import (
        CONTAINER_SHA256,
        FAVORABLE_PHYSICAL_BYTES_PER_SECOND,
        MANIFEST_SHA256,
        Record,
        RetentionInterval,
        build_intervals,
        validate_manifest,
    )

FW0051_SHA256 = "59dcef0b2c78da0dbb7521ce0c824632b86d894bc9db8b6140a0ef24294d0644"
FW0052_SHA256 = "d8561eae477282e59cf1ed32828f993ef0f99bafd6457f053e53f6df3221100b"
SOURCE_BYTES_PER_EXPERT = 9_830_400
EVENTS = 192
ACCESSES = 1_920
ACCEPTED = 4
NODE_LIMIT = 10_000


def solve_retention(
    records: dict[str, Record],
    events: list[list[str]],
    capacity_bytes: int,
    source_bytes_per_expert: int,
    physical_bytes_per_second: float,
    decoded_source_bytes_per_second: float,
    node_limit: int = NODE_LIMIT,
) -> tuple[list[RetentionInterval], Any]:
    if (
        capacity_bytes <= 0
        or source_bytes_per_expert <= 0
        or physical_bytes_per_second <= 0
        or decoded_source_bytes_per_second <= 0
        or node_limit <= 0
    ):
        raise lossless.AnalysisError("executable-cache solver bounds must be positive")
    intervals = build_intervals(events)
    count = len(intervals)
    if any(interval.identity not in records for interval in intervals):
        raise lossless.AnalysisError("executable-cache interval references an unknown record")

    rows: list[int] = []
    columns: list[int] = []
    coefficients: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    # Variables [0,count) retain compressed bytes; [count,2*count) retain
    # decoded executable BF16; the final continuous variable is wall seconds.
    for boundary in range(len(events)):
        for column, interval in enumerate(intervals):
            if interval.after_event < boundary <= interval.hit_event:
                rows.extend((row, row))
                columns.extend((column, count + column))
                coefficients.extend(
                    (
                        records[interval.identity].compressed_bytes / capacity_bytes,
                        source_bytes_per_expert / capacity_bytes,
                    )
                )
        lower.append(-numpy.inf)
        upper.append(1.0)
        row += 1

    for column in range(count):
        rows.extend((row, row))
        columns.extend((column, count + column))
        coefficients.extend((1.0, 1.0))
        lower.append(-numpy.inf)
        upper.append(1.0)
        row += 1

    uncached_physical_seconds = sum(
        records[interval.identity].physical_bytes / physical_bytes_per_second
        for interval in intervals
    )
    for column, interval in enumerate(intervals):
        avoided = records[interval.identity].physical_bytes / physical_bytes_per_second
        rows.extend((row, row))
        columns.extend((column, count + column))
        coefficients.extend((-avoided, -avoided))
    rows.append(row)
    columns.append(2 * count)
    coefficients.append(-1.0)
    lower.append(-numpy.inf)
    upper.append(-uncached_physical_seconds)
    row += 1

    all_decode_seconds = (
        count * source_bytes_per_expert / decoded_source_bytes_per_second
    )
    for column in range(count):
        rows.append(row)
        columns.append(count + column)
        coefficients.append(-source_bytes_per_expert / decoded_source_bytes_per_second)
    rows.append(row)
    columns.append(2 * count)
    coefficients.append(-1.0)
    lower.append(-numpy.inf)
    upper.append(-all_decode_seconds)
    row += 1

    matrix = csc_matrix(
        (coefficients, (rows, columns)),
        shape=(row, 2 * count + 1),
        dtype=float,
    )
    objective = numpy.zeros(2 * count + 1)
    objective[-1] = 1.0
    integrality = numpy.zeros(2 * count + 1, dtype=numpy.uint8)
    integrality[: 2 * count] = 1
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(
            numpy.zeros(2 * count + 1),
            numpy.r_[numpy.ones(2 * count), numpy.inf],
        ),
        constraints=LinearConstraint(
            matrix,
            numpy.asarray(lower),
            numpy.asarray(upper),
        ),
        options={
            "mip_rel_gap": 0.0,
            "node_limit": node_limit,
            "presolve": True,
        },
    )
    if result.x is None or result.fun is None or result.mip_dual_bound is None:
        raise lossless.AnalysisError(
            f"executable-cache solver produced no certificate: {result.message}"
        )
    if (
        not math.isfinite(result.mip_gap)
        or result.mip_gap < 0
        or result.mip_dual_bound > result.fun + 1e-7
    ):
        raise lossless.AnalysisError("executable-cache solver bound is invalid")
    return intervals, result


def decode_rate_from_receipt(receipt: dict[str, Any]) -> tuple[float, list[float]]:
    if (
        receipt.get("schema_version") != 1
        or receipt.get("semantic")
        != "qwen3_8_flash_next_two_q2_exact_bf16_shuffle_zstd1_capacity_cache_parallel_physical_metal_overlap_favorable_bound"
        or receipt.get("miss_frames") != 464
        or receipt.get("miss_source_bytes") != 4_561_305_600
        or receipt.get("A") != ACCEPTED
        or receipt.get("performance_claim") is not None
    ):
        raise lossless.AnalysisError("FW-0051 receipt authority mismatch")
    controls = [
        trial
        for trial in receipt.get("interleaved_trials", [])
        if trial.get("mode") == "parallel_storage_decode_inverse_shuffle_control"
    ]
    if len(controls) != 3 or any(trial.get("workers") != 8 for trial in controls):
        raise lossless.AnalysisError("FW-0051 control trial ledger mismatch")
    ideal_seconds = [
        (
            trial["summed_decompression_time_ns"]
            + trial["summed_inverse_shuffle_time_ns"]
        )
        / trial["workers"]
        / 1_000_000_000
        for trial in controls
    ]
    if any(seconds <= 0 for seconds in ideal_seconds):
        raise lossless.AnalysisError("FW-0051 decode timing is not positive")
    return receipt["miss_source_bytes"] / min(ideal_seconds), ideal_seconds


def validate_compute_receipt(receipt: dict[str, Any]) -> int:
    if (
        receipt.get("schema_version") != 2
        or receipt.get("semantic")
        != "qwen3_8_flash_next_real_top10_moe_exact_resident_metal_one_transaction_lut_swiglu"
        or receipt.get("candidate_median_wall_time_ns") != 3_164_709
        or receipt.get("command_buffers_per_candidate") != 1
        or receipt.get("exact_candidate_measurements") != 33
        or receipt.get("performance_claim") is not None
    ):
        raise lossless.AnalysisError("FW-0052 receipt authority mismatch")
    return receipt["candidate_median_wall_time_ns"]


def analyze(
    manifest_path: Path,
    fw0051_path: Path,
    fw0052_path: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    lossless.require_clean_commit(implementation_commit)
    lossless.require_hash(manifest_path, MANIFEST_SHA256)
    lossless.require_hash(fw0051_path, FW0051_SHA256)
    lossless.require_hash(fw0052_path, FW0052_SHA256)
    manifest = lossless.read_json(manifest_path)
    records, events, capacity_bytes = validate_manifest(manifest)
    if (
        manifest.get("source_bytes_per_expert") != SOURCE_BYTES_PER_EXPERT
        or len(events) != EVENTS
        or sum(map(len, events)) != ACCESSES
    ):
        raise lossless.AnalysisError("executable-cache manifest shape mismatch")
    fw0051 = lossless.read_json(fw0051_path)
    fw0052 = lossless.read_json(fw0052_path)
    decode_bytes_per_second, ideal_decode_control_seconds = decode_rate_from_receipt(fw0051)
    metal_median_ns = validate_compute_receipt(fw0052)

    intervals, result = solve_retention(
        records,
        events,
        capacity_bytes,
        SOURCE_BYTES_PER_EXPERT,
        FAVORABLE_PHYSICAL_BYTES_PER_SECOND,
        decode_bytes_per_second,
    )
    count = len(intervals)
    compressed = [value >= 0.5 for value in result.x[:count]]
    decoded = [value >= 0.5 for value in result.x[count : 2 * count]]
    if any(abs(value - round(value)) > 1e-7 for value in result.x[: 2 * count]):
        raise lossless.AnalysisError("executable-cache incumbent is not integral")
    if any(left and right for left, right in zip(compressed, decoded, strict=True)):
        raise lossless.AnalysisError("executable-cache interval has two representations")

    boundary_bytes = [0] * len(events)
    selections: list[dict[str, Any]] = []
    misses_by_event: list[list[str]] = [[] for _ in events]
    compressed_hits_by_event: list[list[str]] = [[] for _ in events]
    decoded_hits_by_event: list[list[str]] = [[] for _ in events]
    for interval, keep_compressed, keep_decoded in zip(
        intervals, compressed, decoded, strict=True
    ):
        representation = None
        residency_bytes = 0
        if keep_compressed:
            representation = "compressed"
            residency_bytes = records[interval.identity].compressed_bytes
            compressed_hits_by_event[interval.hit_event].append(interval.identity)
        elif keep_decoded:
            representation = "decoded_bf16"
            residency_bytes = SOURCE_BYTES_PER_EXPERT
            decoded_hits_by_event[interval.hit_event].append(interval.identity)
        else:
            misses_by_event[interval.hit_event].append(interval.identity)
        if representation is not None:
            for boundary in range(interval.after_event + 1, interval.hit_event + 1):
                boundary_bytes[boundary] += residency_bytes
            selections.append(
                {
                    "identity": interval.identity,
                    "after_event": interval.after_event,
                    "hit_event": interval.hit_event,
                    "representation": representation,
                    "residency_bytes": residency_bytes,
                }
            )
    if max(boundary_bytes) > capacity_bytes:
        raise lossless.AnalysisError("executable-cache certificate exceeds capacity")

    misses = [identity for event in misses_by_event for identity in event]
    compressed_hits = [identity for event in compressed_hits_by_event for identity in event]
    decoded_hits = [identity for event in decoded_hits_by_event for identity in event]
    if len(misses) + len(compressed_hits) + len(decoded_hits) != ACCESSES:
        raise lossless.AnalysisError("executable-cache access partition mismatch")
    physical_bytes = sum(records[identity].physical_bytes for identity in misses)
    decode_accesses = len(misses) + len(compressed_hits)
    decoded_source_bytes = decode_accesses * SOURCE_BYTES_PER_EXPERT
    physical_seconds = physical_bytes / FAVORABLE_PHYSICAL_BYTES_PER_SECOND
    decode_seconds = decoded_source_bytes / decode_bytes_per_second
    acquisition_seconds = max(physical_seconds, decode_seconds)
    if not math.isclose(result.fun, acquisition_seconds, abs_tol=1e-7):
        raise lossless.AnalysisError("executable-cache solver objective disagrees with replay")
    metal_seconds = metal_median_ns * EVENTS / 1_000_000_000
    optimistic_wall_seconds = max(acquisition_seconds, metal_seconds)
    initial_compressed = sum(
        interval.after_event == -1 and keep
        for interval, keep in zip(intervals, compressed, strict=True)
    )
    initial_decoded = sum(
        interval.after_event == -1 and keep
        for interval, keep in zip(intervals, decoded, strict=True)
    )

    return {
        "schema_version": 1,
        "semantic": "qwen3_8_flash_next_two_q2_mixed_compressed_decoded_executable_cache_offline_milp",
        "implementation_commit": implementation_commit,
        "model": lossless.MODEL,
        "revision": lossless.REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "container_sha256": CONTAINER_SHA256,
        "fw_0051_receipt_sha256": FW0051_SHA256,
        "fw_0052_receipt_sha256": FW0052_SHA256,
        "solver": {
            "name": "scipy.optimize.milp_highs",
            "scipy_version": scipy.__version__,
            "numpy_version": numpy.__version__,
            "status": int(result.status),
            "message": result.message,
            "node_limit": NODE_LIMIT,
            "nodes_processed": int(result.mip_node_count),
            "relative_gap": float(result.mip_gap),
            "incumbent_objective_seconds": float(result.fun),
            "dual_bound_seconds": float(result.mip_dual_bound),
            "dual_bound_scope": "solver diagnostic only; the integer incumbent is independently replayed",
        },
        "incumbent_certificate": {
            "integral_within_absolute_tolerance": 1e-7,
            "replayed_capacity_boundaries": len(boundary_bytes),
            "maximum_capacity_violation_bytes": 0,
            "objective_matches_replay_within_seconds": 1e-7,
        },
        "cache_semantic": "each_retention_interval_selects_compressed_decoded_bf16_or_absent",
        "initial_cache_semantic": "free_offline_future_known_representation_and_contents",
        "event_order": "transaction_then_target_row_then_layer_with_simultaneous_top10_event",
        "events": len(events),
        "accesses": len(intervals),
        "unique_experts": len(records),
        "compressed_retention_intervals": len(compressed_hits),
        "decoded_retention_intervals": len(decoded_hits),
        "misses": len(misses),
        "compressed_cache_bytes": capacity_bytes,
        "free_initial_compressed_frames": initial_compressed,
        "free_initial_decoded_frames": initial_decoded,
        "maximum_boundary_resident_bytes": max(boundary_bytes),
        "physical_miss_bytes": physical_bytes,
        "decode_accesses": decode_accesses,
        "decoded_source_bytes": decoded_source_bytes,
        "favorable_physical_bytes_per_second": FAVORABLE_PHYSICAL_BYTES_PER_SECOND,
        "fw_0051_ideal_eight_worker_decode_control_seconds": ideal_decode_control_seconds,
        "favorable_decoded_source_bytes_per_second": decode_bytes_per_second,
        "physical_seconds": physical_seconds,
        "ideal_decode_seconds": decode_seconds,
        "optimistic_acquisition_seconds": acquisition_seconds,
        "fw_0052_median_metal_execution_ns": metal_median_ns,
        "metal_executions": EVENTS,
        "metal_seconds": metal_seconds,
        "optimistic_perfect_overlap_wall_seconds": optimistic_wall_seconds,
        "optimistic_accepted_tps": ACCEPTED / optimistic_wall_seconds,
        "four_tps_headroom_seconds": 1.0 - optimistic_wall_seconds,
        "selected_retention_intervals": selections,
        "misses_by_event": misses_by_event,
        "compressed_hits_by_event": compressed_hits_by_event,
        "decoded_hits_by_event": decoded_hits_by_event,
        "batch_size": 1,
        "concurrency": 1,
        "q": 2,
        "A": ACCEPTED,
        "sum_equivalent_U": (697 + 741) / 480,
        "favorable_grants": [
            "the complete two-transaction route future and initial cache representation are known",
            "selected initial compressed or decoded frames are installed for free",
            "eight decode workers have ideal load balance at the fastest measured aggregate CPU-work rate",
            "physical reads decode inverse shuffle and exact routed Metal overlap perfectly",
            "cache metadata movement decoded-buffer installation and every fixed endpoint operation are free",
        ],
        "performance_claim": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("fw0051_receipt", type=Path)
    parser.add_argument("fw0052_receipt", type=Path)
    parser.add_argument("implementation_commit")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.manifest,
        args.fw0051_receipt,
        args.fw0052_receipt,
        args.implementation_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
