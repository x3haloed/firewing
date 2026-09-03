#!/usr/bin/env python3
"""Verify a completed checkpoint copy against a Firewing model lock.

Unlike checkpoint_census.py, this tool intentionally reads every expected file
byte. Run it only after the download or copy is complete. It performs no
network access and writes a machine-readable report even when verification
fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")


class VerificationError(RuntimeError):
    """Raised when a model lock is malformed or cannot be used safely."""


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_lock(lock: Any) -> list[dict[str, Any]]:
    if not isinstance(lock, dict) or lock.get("schema_version") != SCHEMA_VERSION:
        raise VerificationError("unsupported model lock schema")
    if lock.get("model") != "Qwen/Qwen3.8-Flash-Next":
        raise VerificationError(f"unexpected model identity: {lock.get('model')!r}")
    if not REVISION_RE.fullmatch(str(lock.get("revision", ""))):
        raise VerificationError("model lock revision is not a 40-hex commit")
    files = lock.get("files")
    if not isinstance(files, list) or len(files) != lock.get("expected_file_count"):
        raise VerificationError("model lock file inventory is missing or incomplete")

    seen: set[str] = set()
    expected_total = 0
    weight_count = 0
    weight_bytes = 0
    small_hashes = lock.get("local_small_file_sha256")
    if not isinstance(small_hashes, dict):
        raise VerificationError("model lock local_small_file_sha256 is missing")
    for entry in files:
        if not isinstance(entry, dict):
            raise VerificationError("model lock contains a non-object file entry")
        relative = entry.get("path")
        size = entry.get("size")
        kind = entry.get("kind")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen
        ):
            raise VerificationError(f"unsafe or duplicate path in model lock: {relative!r}")
        if not isinstance(size, int) or size < 0:
            raise VerificationError(f"invalid size for {relative}")
        if kind == "weight_shard":
            if not SHA256_RE.fullmatch(str(entry.get("lfs_sha256", ""))):
                raise VerificationError(f"weight shard lacks an expected SHA-256: {relative}")
            weight_count += 1
            weight_bytes += size
        elif kind == "lfs_artifact":
            if not SHA256_RE.fullmatch(str(entry.get("lfs_sha256", ""))):
                raise VerificationError(f"LFS artifact lacks an expected SHA-256: {relative}")
        elif kind == "metadata":
            if not SHA256_RE.fullmatch(str(small_hashes.get(relative, ""))):
                raise VerificationError(
                    f"metadata hash is absent for {relative}; regenerate the lock from a complete download"
                )
        else:
            raise VerificationError(f"unknown file kind for {relative}: {kind!r}")
        seen.add(relative)
        expected_total += size

    declared = (
        ("expected_total_bytes", expected_total),
        ("expected_weight_shard_count", weight_count),
        ("expected_weight_shard_bytes", weight_bytes),
    )
    for field, calculated in declared:
        if lock.get(field) != calculated:
            raise VerificationError(
                f"model lock {field} mismatch: declared {lock.get(field)!r}, calculated {calculated}"
            )
    return files


def verify_checkpoint(checkpoint_dir: Path, lock_path: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock_path = lock_path.resolve()
    lock = read_json(lock_path)
    files = validate_lock(lock)
    small_hashes = lock["local_small_file_sha256"]

    results: list[dict[str, Any]] = []
    bytes_hashed = 0
    missing_files = 0
    size_mismatches = 0
    hash_mismatches = 0
    verified_files = 0

    for entry in files:
        relative = entry["path"]
        path = checkpoint_dir / relative
        result: dict[str, Any] = {
            "path": relative,
            "kind": entry["kind"],
            "expected_size": entry["size"],
        }
        if not path.is_file():
            result["status"] = "missing"
            missing_files += 1
            results.append(result)
            continue
        actual_size = path.stat().st_size
        result["actual_size"] = actual_size
        if actual_size != entry["size"]:
            result["status"] = "size_mismatch"
            size_mismatches += 1
            results.append(result)
            continue

        expected_hash = (
            small_hashes[relative]
            if entry["kind"] == "metadata"
            else entry["lfs_sha256"]
        )
        actual_hash = sha256_file(path)
        bytes_hashed += actual_size
        result["expected_sha256"] = expected_hash
        result["actual_sha256"] = actual_hash
        if actual_hash != expected_hash:
            result["status"] = "sha256_mismatch"
            hash_mismatches += 1
        else:
            result["status"] = "verified"
            verified_files += 1
        results.append(result)

    passed = not (missing_files or size_mismatches or hash_mismatches)
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified" if passed else "failed",
        "model": lock["model"],
        "revision": lock["revision"],
        "checkpoint_dir_observed": os.fspath(checkpoint_dir),
        "model_lock": {
            "path": os.fspath(lock_path),
            "sha256": sha256_file(lock_path),
        },
        "network_access_performed": False,
        "expected_file_count": len(files),
        "verified_file_count": verified_files,
        "missing_file_count": missing_files,
        "size_mismatch_count": size_mismatches,
        "sha256_mismatch_count": hash_mismatches,
        "bytes_hashed": bytes_hashed,
        "files": results,
        "performance_claim": None,
        "accepted_tokens": 0,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = verify_checkpoint(args.checkpoint_dir, args.model_lock)
        write_json(args.output, report)
    except VerificationError as exc:
        print(f"checkpoint-verify: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "verified_files": report["verified_file_count"],
                "expected_files": report["expected_file_count"],
                "bytes_hashed": report["bytes_hashed"],
                "output": os.fspath(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "verified" else 3


if __name__ == "__main__":
    raise SystemExit(main())
