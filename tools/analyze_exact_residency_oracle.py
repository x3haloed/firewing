#!/usr/bin/env python3
"""Evaluate an impossible-favorable exact BF16 residency bound for FW-0034."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
ENDPOINT_SHA256 = "2eb3712c7959837a24fe6db5b5d7b1f87c9926b4dd80eae524590e43287331ca"
CENSUS_SHA256 = "043b5b45edd1f4aeb628a66b00fec60c035204c30a48d34d0a95f3e10d0bd937"
ACQUISITION_SHA256 = "b8e5a175c0402bced494ebb1cc4a61f903f2ff8a1a094fa8a17d043311f942b5"
EXPERT_BYTES = 9_830_400
FIXED_CATEGORIES = (
    "gated_deltanet",
    "gated_residual",
    "lm_head",
    "qwen_sparse_attention",
    "routers_and_expert_gates",
    "shared_experts",
    "ngram_projection",
)
CAPACITIES = (10 * 1024**3, 12 * 1024**3)


class AnalysisError(ValueError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise AnalysisError(f"authority hash mismatch: {path}")


def route_events(endpoint: dict[str, Any]) -> list[set[tuple[int, int]]]:
    if (
        endpoint.get("schema_version") != 1
        or endpoint.get("model") != MODEL
        or endpoint.get("revision") != REVISION
        or endpoint.get("semantic")
        != "qwen3_8_flash_next_firewing_two_token_cached_text_logits"
        or len(endpoint.get("layers", [])) != 48
    ):
        raise AnalysisError("endpoint authority identity mismatch")
    by_token: list[list[set[tuple[int, int]]]] = [[], []]
    for expected_layer, layer in enumerate(endpoint["layers"]):
        decoder = layer.get("decoder", {})
        if layer.get("layer") != expected_layer or len(decoder.get("steps", [])) != 2:
            raise AnalysisError("endpoint layer schedule mismatch")
        for token, step in enumerate(decoder["steps"]):
            selected = step.get("selected_experts")
            execution = step.get("expert_execution_order")
            records = step.get("experts")
            if (
                step.get("ordinal") != token
                or not isinstance(selected, list)
                or len(selected) != 10
                or len(set(selected)) != 10
                or any(not isinstance(value, int) or not 0 <= value < 512 for value in selected)
                or sorted(selected) != execution
                or [record.get("expert") for record in records] != execution
            ):
                raise AnalysisError("endpoint route authority mismatch")
            by_token[token].append({(expected_layer, expert) for expert in selected})
    return by_token[0] + by_token[1]


def belady(events: list[set[tuple[int, int]]], capacity: int) -> dict[str, Any]:
    if not events or capacity < max(map(len, events)):
        raise AnalysisError("expert capacity cannot serve one layer")
    future: dict[tuple[int, int], deque[int]] = defaultdict(deque)
    for position, demand in enumerate(events):
        if not demand:
            raise AnalysisError("route event is empty")
        for identity in demand:
            future[identity].append(position)
    identities = sorted(future)
    initial = sorted(identities, key=lambda item: (future[item][0], item))[:capacity]
    resident = set(initial)
    misses_by_event: list[int] = []
    for position, demand in enumerate(events):
        misses = demand - resident
        misses_by_event.append(len(misses))
        resident.update(misses)
        for identity in demand:
            if not future[identity] or future[identity][0] != position:
                raise AnalysisError("future-use replay mismatch")
            future[identity].popleft()
        while len(resident) > capacity:
            candidates = resident - demand
            if not candidates:
                raise AnalysisError("capacity cannot retain current demand")
            victim = max(
                candidates,
                key=lambda item: (
                    future[item][0] if future[item] else math.inf,
                    item,
                ),
            )
            resident.remove(victim)
    return {
        "capacity_experts": capacity,
        "distinct_layer_experts": len(identities),
        "free_initial_experts": len(initial),
        "misses": sum(misses_by_event),
        "misses_by_event": misses_by_event,
    }


def analyze(endpoint_path: Path, census_path: Path, acquisition_path: Path) -> dict[str, Any]:
    require_hash(endpoint_path, ENDPOINT_SHA256)
    require_hash(census_path, CENSUS_SHA256)
    require_hash(acquisition_path, ACQUISITION_SHA256)
    endpoint = read_json(endpoint_path)
    census = read_json(census_path)
    acquisition = read_json(acquisition_path)
    if (
        census.get("model") != MODEL
        or census.get("revision") != REVISION
        or census.get("status") != "headers_complete_payload_hashes_pending"
        or census.get("tree_manifest", {}).get("sha256")
        != "6042846bc80da9b7946c9b5814d791e899ac162c8cf4ae5a35985dcbee180542"
        or census.get("observed", {}).get("parsed_tensor_count") != 1658
        or acquisition.get("model") != MODEL
        or acquisition.get("revision") != REVISION
        or acquisition.get("logical_bytes_per_trace") != 4_718_592_000
    ):
        raise AnalysisError("census or acquisition authority mismatch")
    categories = census["observed"]["category_bytes"]
    fixed_bytes = sum(categories[name] for name in FIXED_CATEGORIES)
    if fixed_bytes != 8_623_999_000:
        raise AnalysisError("ordinary fixed-byte ledger mismatch")
    cold_eight = [
        row
        for row in acquisition.get("summaries", [])
        if row.get("workers") == 8
        and row.get("cache_state") == "range_invalidated_page_aligned_f_nocache_f_rdahead_zero"
    ]
    if len(cold_eight) != 1 or cold_eight[0].get("samples") != 3:
        raise AnalysisError("eight-worker transport authority missing")
    transport_ns = cold_eight[0]["maximum_worker_pread_ms_median"] * 1_000_000
    optimistic_bytes_per_second = acquisition["logical_bytes_per_trace"] / (transport_ns / 1e9)
    events = route_events(endpoint)
    scenarios = []
    for resident_limit in CAPACITIES:
        available = resident_limit - fixed_bytes
        result = belady(events, available // EXPERT_BYTES)
        misses_by_token = [
            sum(result["misses_by_event"][:48]),
            sum(result["misses_by_event"][48:]),
        ]
        del result["misses_by_event"]
        result.update(
            {
                "resident_limit_bytes": resident_limit,
                "fixed_resident_bytes": fixed_bytes,
                "expert_resident_bytes": result["capacity_experts"] * EXPERT_BYTES,
                "unallocated_bytes": available % EXPERT_BYTES,
                "miss_bytes": result["misses"] * EXPERT_BYTES,
                "misses_by_token": misses_by_token,
                "miss_bytes_by_token": [value * EXPERT_BYTES for value in misses_by_token],
            }
        )
        storage_seconds = result["miss_bytes"] / optimistic_bytes_per_second
        token_tps = [
            "infinite" if value == 0 else optimistic_bytes_per_second / value
            for value in result["miss_bytes_by_token"]
        ]
        result["optimistic_storage_only_aggregate_tps"] = 2 / storage_seconds
        result["optimistic_storage_only_tps_by_token"] = token_tps
        result["passes_four_tps_aggregate"] = result["optimistic_storage_only_aggregate_tps"] >= 4
        result["passes_three_tps_min_token"] = all(
            value == "infinite" or value >= 3 for value in token_tps
        )
        scenarios.append(result)
    return {
        "schema_version": 1,
        "semantic": "firewing_exact_bf16_future_aware_residency_oracle",
        "model": MODEL,
        "revision": REVISION,
        "authorities": {
            "endpoint_fixture_sha256": ENDPOINT_SHA256,
            "checkpoint_census_sha256": CENSUS_SHA256,
            "expert_acquisition_sha256": ACQUISITION_SHA256,
        },
        "schedule": "two_exact_positions_token_major_48_layer_ten_expert_sets",
        "expert_bytes": EXPERT_BYTES,
        "fixed_categories": list(FIXED_CATEGORIES),
        "fixed_bytes": fixed_bytes,
        "route_events": len(events),
        "optimistic_parallel_transport_bytes_per_second": optimistic_bytes_per_second,
        "favorable_grants": [
            "all fixed matrices remain resident",
            "expert cache initial contents are free and future-aware",
            "Belady evictions have perfect future knowledge and zero cost",
            "resident allocation granularity and runtime buffers are free",
            "all compute synchronization ngram traffic and prefetch cost are zero",
        ],
        "scenarios": scenarios,
        "scope": "two-position development trace only; not a route-distribution or endpoint result",
        "batch_size": 1,
        "concurrency": 1,
        "accepted_tokens": 0,
        "A": 0,
        "U": 0,
        "performance_claim": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, type=Path)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--acquisition", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = analyze(args.endpoint, args.census, args.acquisition)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except AnalysisError as exc:
        print(f"exact-residency-oracle: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps({"output": os.fspath(args.output), "scenarios": report["scenarios"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
