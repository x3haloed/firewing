#!/usr/bin/env python3
"""Build a metadata-only, fail-closed census of a Hugging Face local-dir download.

The tool reads the local Hugging Face tree manifest and Safetensors headers. It
never reads tensor payload bytes and never performs network access. Payload
hashes from the tree manifest are recorded as expected identities, not as local
verification results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MAX_HEADER_BYTES = 64 * 1024 * 1024
SHARD_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
DTYPE_BYTES = {
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2FNUZ": 1,
    "I8": 1,
    "U8": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


class CensusError(RuntimeError):
    """Raised when an input violates the checkpoint census contract."""


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_id_file(path: Path, size: int, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CensusError(f"cannot read JSON {path}: {exc}") from exc


def find_tree_manifest(checkpoint_dir: Path) -> tuple[str, Path, dict[str, Any]]:
    tree_dir = checkpoint_dir / ".cache" / "huggingface" / "trees"
    candidates = sorted(tree_dir.glob("*.json"))
    if len(candidates) != 1:
        raise CensusError(
            f"expected exactly one Hugging Face tree manifest in {tree_dir}; "
            f"found {len(candidates)}"
        )
    path = candidates[0]
    revision = path.stem
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise CensusError(f"tree manifest filename is not a 40-hex revision: {path.name}")
    tree = read_json(path)
    if tree.get("format_version") != 1 or not isinstance(tree.get("files"), dict):
        raise CensusError(f"unsupported Hugging Face tree manifest schema: {path}")
    return revision, path, tree


def tensor_category(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision"
    if ".ple.ple_embedding.ngram_embedding." in name:
        return "ngram_embedding"
    if ".mtp" in name or name.startswith("mtp"):
        return "mtp"
    if ".mlp.experts." in name:
        return "routed_experts"
    if ".mlp.shared_expert." in name:
        return "shared_experts"
    if ".mlp.gate." in name or ".mlp.shared_expert_gate." in name:
        return "routers_and_expert_gates"
    if ".linear_attn." in name:
        return "gated_deltanet"
    if ".self_attn." in name or ".qsa." in name:
        return "qwen_sparse_attention"
    if "hyper_connection" in name:
        return "gated_residual"
    if "embed_tokens" in name:
        return "token_embeddings"
    if "lm_head" in name:
        return "lm_head"
    if ".ple." in name:
        return "ngram_projection"
    if name.startswith("model.language_model."):
        return "language_other"
    return "other"


def tensor_nbytes(entry: dict[str, Any], name: str) -> int:
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if dtype not in DTYPE_BYTES:
        raise CensusError(f"unknown dtype {dtype!r} for tensor {name}")
    if not isinstance(shape, list) or not all(isinstance(v, int) and v >= 0 for v in shape):
        raise CensusError(f"invalid shape for tensor {name}: {shape!r}")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(v, int) and v >= 0 for v in offsets)
        or offsets[1] < offsets[0]
    ):
        raise CensusError(f"invalid data offsets for tensor {name}: {offsets!r}")
    elements = 1
    for dimension in shape:
        elements *= dimension
    expected = elements * DTYPE_BYTES[dtype]
    actual = offsets[1] - offsets[0]
    if expected != actual:
        raise CensusError(
            f"tensor {name} byte mismatch: shape/dtype imply {expected}, offsets contain {actual}"
        )
    return actual


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise CensusError(f"truncated Safetensors prefix: {path}")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if header_bytes == 0 or header_bytes > MAX_HEADER_BYTES:
                raise CensusError(f"invalid Safetensors header size {header_bytes}: {path}")
            raw_header = handle.read(header_bytes)
            if len(raw_header) != header_bytes:
                raise CensusError(f"truncated Safetensors header: {path}")
        header = json.loads(raw_header)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CensusError(f"cannot read Safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise CensusError(f"Safetensors header is not an object: {path}")
    return header_bytes, header


def inspect_shard(path: Path, expected_size: int) -> dict[str, Any]:
    header_bytes, header = read_safetensors_header(path)
    payload_bytes = expected_size - 8 - header_bytes
    if payload_bytes < 0:
        raise CensusError(f"header exceeds expected shard size: {path}")

    tensors: list[dict[str, Any]] = []
    intervals: list[tuple[int, int, str]] = []
    category_bytes: Counter[str] = Counter()
    category_parameters: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    total_parameters = 0

    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise CensusError(f"tensor entry is not an object for {name} in {path}")
        nbytes = tensor_nbytes(entry, name)
        start, end = entry["data_offsets"]
        if end > payload_bytes:
            raise CensusError(f"tensor {name} extends beyond payload in {path}")
        parameters = nbytes // DTYPE_BYTES[entry["dtype"]]
        category = tensor_category(name)
        intervals.append((start, end, name))
        category_bytes[category] += nbytes
        category_parameters[category] += parameters
        dtype_counts[entry["dtype"]] += 1
        total_parameters += parameters
        tensors.append(
            {
                "name": name,
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "data_offsets": entry["data_offsets"],
                "bytes": nbytes,
                "parameters": parameters,
                "category": category,
            }
        )

    intervals.sort()
    cursor = 0
    for start, end, name in intervals:
        if start != cursor:
            relation = "overlap" if start < cursor else "gap"
            raise CensusError(f"tensor payload {relation} before {name} in {path}: {cursor} -> {start}")
        cursor = end
    if cursor != payload_bytes:
        raise CensusError(
            f"tensor payload does not cover shard {path}: tensors end at {cursor}, payload is {payload_bytes}"
        )

    return {
        "header_bytes": header_bytes,
        "payload_bytes": payload_bytes,
        "tensor_count": len(tensors),
        "parameter_count": total_parameters,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "category_bytes": dict(sorted(category_bytes.items())),
        "category_parameters": dict(sorted(category_parameters.items())),
        "tensors": tensors,
    }


def expected_inventory(tree: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    files: list[dict[str, Any]] = []
    total_bytes = 0
    shard_count = 0
    shard_denominators: set[int] = set()
    shard_indices: set[int] = set()
    for relative_path, metadata in sorted(tree["files"].items()):
        if not isinstance(metadata, dict) or not isinstance(metadata.get("size"), int):
            raise CensusError(f"invalid tree entry for {relative_path}")
        size = metadata["size"]
        if size < 0:
            raise CensusError(f"negative size for {relative_path}")
        match = SHARD_RE.fullmatch(relative_path)
        is_shard = match is not None
        is_lfs = metadata.get("lfs_sha256") is not None or metadata.get("lfs_size") is not None
        if not re.fullmatch(r"[0-9a-f]{40}", str(metadata.get("blob_id", ""))):
            raise CensusError(f"missing or invalid Git blob identity for {relative_path}")
        if is_lfs:
            if metadata.get("lfs_size") != size or not re.fullmatch(
                r"[0-9a-f]{64}", str(metadata.get("lfs_sha256", ""))
            ):
                raise CensusError(f"missing or invalid LFS identity for {relative_path}")
        if match:
            shard_count += 1
            shard_indices.add(int(match.group(1)))
            shard_denominators.add(int(match.group(2)))
        files.append(
            {
                "path": relative_path,
                "size": size,
                "git_blob": metadata.get("blob_id"),
                "lfs_sha256": metadata.get("lfs_sha256"),
                "xet_hash": metadata.get("xet_hash"),
                "kind": (
                    "weight_shard" if is_shard else "lfs_artifact" if is_lfs else "metadata"
                ),
            }
        )
        total_bytes += size
    if shard_count:
        if shard_denominators != {shard_count} or shard_indices != set(range(1, shard_count + 1)):
            raise CensusError(
                f"weight shard sequence is not closed: count={shard_count}, "
                f"denominators={sorted(shard_denominators)}"
            )
    return files, total_bytes, shard_count


def build_census(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_dir = checkpoint_dir.resolve()
    revision, tree_path, tree = find_tree_manifest(checkpoint_dir)
    files, expected_total_bytes, expected_shards = expected_inventory(tree)

    local_complete = 0
    local_complete_bytes = 0
    missing: list[str] = []
    size_mismatches: list[dict[str, Any]] = []
    content_integrity_mismatches: list[dict[str, Any]] = []
    completed_shards: list[dict[str, Any]] = []
    aggregate_category_bytes: Counter[str] = Counter()
    aggregate_category_parameters: Counter[str] = Counter()
    aggregate_dtype_counts: Counter[str] = Counter()
    observed_tensor_names: set[str] = set()
    small_file_sha256: dict[str, str] = {}

    for entry in files:
        path = checkpoint_dir / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
            continue
        actual_size = path.stat().st_size
        if actual_size != entry["size"]:
            size_mismatches.append(
                {"path": entry["path"], "expected_size": entry["size"], "actual_size": actual_size}
            )
            continue
        if entry["kind"] == "metadata":
            actual_blob = git_blob_id_file(path, actual_size)
            if actual_blob != entry["git_blob"]:
                content_integrity_mismatches.append(
                    {
                        "path": entry["path"],
                        "expected_git_blob": entry["git_blob"],
                        "actual_git_blob": actual_blob,
                    }
                )
                continue
            small_file_sha256[entry["path"]] = sha256_file(path)
            local_complete += 1
            local_complete_bytes += actual_size
            continue
        if entry["kind"] == "lfs_artifact":
            actual_sha256 = sha256_file(path)
            if actual_sha256 != entry["lfs_sha256"]:
                content_integrity_mismatches.append(
                    {
                        "path": entry["path"],
                        "expected_sha256": entry["lfs_sha256"],
                        "actual_sha256": actual_sha256,
                    }
                )
                continue
            local_complete += 1
            local_complete_bytes += actual_size
            continue
        shard = inspect_shard(path, entry["size"])
        for tensor in shard["tensors"]:
            if tensor["name"] in observed_tensor_names:
                raise CensusError(f"duplicate tensor name across completed shards: {tensor['name']}")
            observed_tensor_names.add(tensor["name"])
        aggregate_category_bytes.update(shard["category_bytes"])
        aggregate_category_parameters.update(shard["category_parameters"])
        aggregate_dtype_counts.update(shard["dtype_tensor_counts"])
        completed_shards.append(
            {
                "path": entry["path"],
                "expected_sha256": entry["lfs_sha256"],
                "payload_sha256_verified": False,
                **shard,
            }
        )
        local_complete += 1
        local_complete_bytes += actual_size

    expected_shard_names = {entry["path"] for entry in files if entry["kind"] == "weight_shard"}
    completed_shard_names = {entry["path"] for entry in completed_shards}
    completed_weight_bytes = sum(
        entry["size"] for entry in files if entry["path"] in completed_shard_names
    )
    expected_weight_bytes = sum(entry["size"] for entry in files if entry["kind"] == "weight_shard")
    download_complete = not missing and not size_mismatches and not content_integrity_mismatches
    status = "headers_complete_payload_hashes_pending" if download_complete else "incomplete_download"

    config_path = checkpoint_dir / "config.json"
    config = read_json(config_path) if config_path.is_file() else None
    architecture = None
    if isinstance(config, dict):
        architecture = {
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "language_model_only": config.get("language_model_only"),
            "text_config": config.get("text_config"),
            "vision_config": config.get("vision_config"),
        }

    census = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen3.8-Flash-Next",
        "revision": revision,
        "checkpoint_dir_observed": os.fspath(checkpoint_dir),
        "tree_manifest": {
            "path": os.fspath(tree_path),
            "sha256": sha256_file(tree_path),
        },
        "status": status,
        "network_access_performed": False,
        "tensor_payload_bytes_read": 0,
        "payload_sha256_verification_complete": False,
        "expected": {
            "file_count": len(files),
            "weight_shard_count": expected_shards,
            "total_bytes": expected_total_bytes,
            "weight_shard_bytes": expected_weight_bytes,
        },
        "observed": {
            "complete_file_count": local_complete,
            "complete_file_bytes": local_complete_bytes,
            "complete_weight_shard_count": len(completed_shards),
            "complete_weight_shard_bytes": completed_weight_bytes,
            "missing_file_count": len(missing),
            "size_mismatch_count": len(size_mismatches),
            "content_integrity_mismatch_count": len(content_integrity_mismatches),
            "parsed_tensor_count": len(observed_tensor_names),
            "parsed_parameter_count": sum(aggregate_category_parameters.values()),
            "parsed_tensor_bytes": sum(aggregate_category_bytes.values()),
            "category_bytes": dict(sorted(aggregate_category_bytes.items())),
            "category_parameters": dict(sorted(aggregate_category_parameters.items())),
            "dtype_tensor_counts": dict(sorted(aggregate_dtype_counts.items())),
        },
        "missing_files": missing,
        "size_mismatches": size_mismatches,
        "content_integrity_mismatches": content_integrity_mismatches,
        "architecture_from_local_config": architecture,
        "small_file_sha256": small_file_sha256,
        "completed_shards": completed_shards,
        "limitations": [
            "Only complete local shards were inspected.",
            "Safetensors payload bytes were not read or hashed.",
            "Expected LFS SHA-256 values are remote identities, not local verification results.",
            "A complete tensor census requires all 131 shard headers.",
        ],
        "performance_claim": None,
        "accepted_tokens": 0,
    }

    model_lock = {
        "schema_version": SCHEMA_VERSION,
        "model": "Qwen/Qwen3.8-Flash-Next",
        "revision": revision,
        "source_tree_manifest_sha256": census["tree_manifest"]["sha256"],
        "expected_file_count": len(files),
        "expected_total_bytes": expected_total_bytes,
        "expected_weight_shard_count": expected_shards,
        "expected_weight_shard_bytes": expected_weight_bytes,
        "files": files,
        "local_small_file_sha256": small_file_sha256,
        "payload_verification_status": "pending_full_download_and_sha256",
        "lock_status": "source_revision_and_expected_inventory_pinned",
    }
    return census, model_lock


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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="census report JSON")
    parser.add_argument("--model-lock", type=Path, help="expected-inventory model lock JSON")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="exit nonzero unless every expected file is present at its expected size",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        census, model_lock = build_census(args.checkpoint_dir)
        write_json(args.output, census)
        if args.model_lock:
            write_json(args.model_lock, model_lock)
    except CensusError as exc:
        print(f"checkpoint-census: {exc}", file=sys.stderr)
        return 2
    if args.require_complete and census["status"] == "incomplete_download":
        print(
            "checkpoint-census: checkpoint download is incomplete; report preserved",
            file=sys.stderr,
        )
        return 3
    print(
        json.dumps(
            {
                "status": census["status"],
                "revision": census["revision"],
                "complete_weight_shards": census["observed"]["complete_weight_shard_count"],
                "expected_weight_shards": census["expected"]["weight_shard_count"],
                "parsed_tensors": census["observed"]["parsed_tensor_count"],
                "output": os.fspath(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
