#!/usr/bin/env python3
"""Bound mixed-cache schedules by mandatory unified-memory traffic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from tools import analyze_q2_lossless_experts as common
else:
    import analyze_q2_lossless_experts as common


MANIFEST_SHA256 = "6759e772d2c9a4560ab39ae80a3b4f4e1a24552adafbf30a396e84166b9c71ca"
FW0034_SHA256 = "19bd38ecc103a80fafc0085063123b86ddaa2aa5365c2fbdf147dae73c6168da"
FW0053_SHA256 = "a61d498af5512c3fcbdc3447d8217383ddacf39361cdea553a41a79d9a10cb3f"
FW0055_4GB_SHA256 = "7c9136e878cefa5c1689285b167b25f3ffd340a808ea3282f0caae84946c100f"
EXPERT_BYTES = 9_830_400
TARGET_ROWS = 4
ACCESSES = 1_920
ACCEPTED = 4
GRANTED_FABRIC_BYTES_PER_SECOND = 68_250_000_000.0
GRANTED_ON_CHIP_CACHE_BYTES = 1_000_000_000
ROW_TRANSITIONS = TARGET_ROWS - 1


def traffic_floor(
    fixed_bytes: int,
    compressed_input_bytes: int,
    decoded_write_bytes: int,
    on_chip_cache_bytes: int = GRANTED_ON_CHIP_CACHE_BYTES,
    fabric_bytes_per_second: float = GRANTED_FABRIC_BYTES_PER_SECOND,
) -> dict[str, Any]:
    values = (
        fixed_bytes,
        compressed_input_bytes,
        decoded_write_bytes,
        on_chip_cache_bytes,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise common.AnalysisError("memory-floor byte counts must be nonnegative integers")
    if fabric_bytes_per_second <= 0:
        raise common.AnalysisError("memory-floor fabric rate must be positive")
    fixed_matrix_reads = fixed_bytes * TARGET_ROWS
    routed_expert_reads = ACCESSES * EXPERT_BYTES
    raw_bytes = (
        fixed_matrix_reads
        + routed_expert_reads
        + compressed_input_bytes
        + decoded_write_bytes
    )
    cache_reuse_grant = on_chip_cache_bytes * ROW_TRANSITIONS
    adjusted_bytes = max(0, raw_bytes - cache_reuse_grant)
    seconds = adjusted_bytes / fabric_bytes_per_second
    return {
        "target_fixed_matrix_reads_bytes": fixed_matrix_reads,
        "routed_expert_weight_reads_bytes": routed_expert_reads,
        "compressed_decoder_input_reads_bytes": compressed_input_bytes,
        "materialized_decoded_weight_writes_bytes": decoded_write_bytes,
        "raw_mandatory_fabric_bytes": raw_bytes,
        "granted_on_chip_cache_bytes_per_row_transition": on_chip_cache_bytes,
        "row_transitions": ROW_TRANSITIONS,
        "granted_cross_row_cache_reuse_bytes": cache_reuse_grant,
        "adjusted_mandatory_fabric_bytes": adjusted_bytes,
        "granted_fabric_bytes_per_second": fabric_bytes_per_second,
        "minimum_wall_seconds": seconds,
        "maximum_accepted_tps": ACCEPTED / seconds if seconds else None,
        "passes_four_tps": seconds <= 1.0,
    }


def validate_schedule(
    receipt: dict[str, Any], records: dict[str, dict[str, Any]], capacity: int
) -> tuple[int, int]:
    if (
        receipt.get("schema_version") != 1
        or receipt.get("semantic")
        != "qwen3_8_flash_next_two_q2_mixed_compressed_decoded_executable_cache_offline_milp"
        or receipt.get("model") != common.MODEL
        or receipt.get("revision") != common.REVISION
        or receipt.get("manifest_sha256") != MANIFEST_SHA256
        or receipt.get("mixed_representation_capacity_bytes") != capacity
        or receipt.get("events") != 192
        or receipt.get("accesses") != ACCESSES
        or receipt.get("A") != ACCEPTED
        or receipt.get("performance_claim") is not None
    ):
        raise common.AnalysisError("mixed-cache schedule authority mismatch")
    misses = receipt.get("misses_by_event")
    compressed = receipt.get("compressed_hits_by_event")
    decoded = receipt.get("decoded_hits_by_event")
    if not all(
        isinstance(rows, list) and len(rows) == 192
        for rows in (misses, compressed, decoded)
    ):
        raise common.AnalysisError("mixed-cache event ledger mismatch")
    compressed_identities = [identity for rows in misses + compressed for identity in rows]
    decoded_identities = [identity for rows in decoded for identity in rows]
    if len(compressed_identities) + len(decoded_identities) != ACCESSES:
        raise common.AnalysisError("mixed-cache access count mismatch")
    try:
        compressed_bytes = sum(
            records[identity]["compressed_bytes"] for identity in compressed_identities
        )
    except (KeyError, TypeError) as exc:
        raise common.AnalysisError("mixed-cache receipt references unknown frame") from exc
    decoded_write_bytes = len(compressed_identities) * EXPERT_BYTES
    if (
        len(compressed_identities) != receipt.get("decode_accesses")
        or decoded_write_bytes != receipt.get("decoded_source_bytes")
    ):
        raise common.AnalysisError("mixed-cache decode ledger mismatch")
    return compressed_bytes, decoded_write_bytes


def analyze(
    manifest_path: Path,
    fw0034_path: Path,
    fw0053_path: Path,
    fw0055_4gb_path: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    common.require_clean_commit(implementation_commit)
    for path, digest in (
        (manifest_path, MANIFEST_SHA256),
        (fw0034_path, FW0034_SHA256),
        (fw0053_path, FW0053_SHA256),
        (fw0055_4gb_path, FW0055_4GB_SHA256),
    ):
        common.require_hash(path, digest)
    manifest = common.read_json(manifest_path)
    prior = common.read_json(fw0034_path)
    if (
        manifest.get("model") != common.MODEL
        or manifest.get("revision") != common.REVISION
        or manifest.get("source_bytes_per_expert") != EXPERT_BYTES
        or len(manifest.get("records", [])) != 1_097
        or prior.get("semantic") != "firewing_exact_bf16_future_aware_residency_oracle"
        or prior.get("fixed_bytes") != 8_623_999_000
        or prior.get("expert_bytes") != EXPERT_BYTES
        or prior.get("fixed_categories")
        != [
            "gated_deltanet",
            "gated_residual",
            "lm_head",
            "qwen_sparse_attention",
            "routers_and_expert_gates",
            "shared_experts",
            "ngram_projection",
        ]
        or prior.get("performance_claim") is not None
    ):
        raise common.AnalysisError("memory-floor source authority mismatch")
    records = {row["identity"]: row for row in manifest["records"]}
    if len(records) != len(manifest["records"]):
        raise common.AnalysisError("memory-floor manifest identity mismatch")
    schedules = []
    for name, path, capacity in (
        ("fw_0053_full_budget", fw0053_path, 4_260_902_888),
        ("fw_0055_4gb", fw0055_4gb_path, 4_000_000_000),
    ):
        receipt = common.read_json(path)
        compressed_bytes, decoded_write_bytes = validate_schedule(receipt, records, capacity)
        schedules.append(
            {
                "name": name,
                "capacity_bytes": capacity,
                "receipt_sha256": common.sha256_file(path),
                **traffic_floor(prior["fixed_bytes"], compressed_bytes, decoded_write_bytes),
            }
        )
    return {
        "schema_version": 1,
        "semantic": "qwen3_8_flash_next_q2_materialized_mixed_cache_unified_memory_favorable_floor",
        "implementation_commit": implementation_commit,
        "model": common.MODEL,
        "revision": common.REVISION,
        "authorities": {
            "manifest_sha256": MANIFEST_SHA256,
            "fw_0034_receipt_sha256": FW0034_SHA256,
            "fw_0053_receipt_sha256": FW0053_SHA256,
            "fw_0055_4gb_receipt_sha256": FW0055_4GB_SHA256,
            "apple_m2_bandwidth_source": "https://developer.apple.com/videos/play/wwdc2022/101/?time=977",
            "apple_m1_transistor_source": "https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/",
        },
        "bandwidth_derivation": {
            "apple_m2_decimal_gb_per_second": 100.0,
            "apple_stated_m2_increase_over_m1": 0.50,
            "implied_m1_decimal_gb_per_second": 100.0 / 1.5,
            "granted_m1_decimal_gb_per_second": GRANTED_FABRIC_BYTES_PER_SECOND / 1e9,
        },
        "on_chip_cache_grant": {
            "bytes": GRANTED_ON_CHIP_CACHE_BYTES,
            "bits": GRANTED_ON_CHIP_CACHE_BYTES * 8,
            "six_transistor_sram_transistors": GRANTED_ON_CHIP_CACHE_BYTES * 8 * 6,
            "apple_published_total_m1_transistors": 16_000_000_000,
            "interpretation": "the cache-reuse grant alone would require three times the entire published M1 transistor count as ideal 6T SRAM",
        },
        "schedules": schedules,
        "decision": "reject_materialized_mixed_cache_representation_for_firewing4"
        if all(not row["passes_four_tps"] for row in schedules)
        else "materialized_mixed_cache_frontier_open",
        "favorable_grants": [
            "68.25 decimal GB/s exceeds the M1 bandwidth implied by Apple's 100 GB/s M2 and 50-percent uplift statement",
            "one impossible-favorable decimal GB of weights remains on chip across each target-row transition",
            "all mandatory fabric transfers overlap perfectly and sustain the granted peak continuously",
            "SSD DMA writes activation traffic metadata synchronization drafter work and every non-weight output are free",
        ],
        "scope": "materialized BF16 mixed-cache representation on the frozen two-transaction q2 trace; not all exact representations or a production route distribution",
        "batch_size": 1,
        "concurrency": 1,
        "q": 2,
        "A": ACCEPTED,
        "sum_equivalent_U": (697 + 741) / 480,
        "performance_claim": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("fw0034_receipt", type=Path)
    parser.add_argument("fw0053_receipt", type=Path)
    parser.add_argument("fw0055_4gb_receipt", type=Path)
    parser.add_argument("implementation_commit")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.manifest,
        args.fw0034_receipt,
        args.fw0053_receipt,
        args.fw0055_4gb_receipt,
        args.implementation_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
