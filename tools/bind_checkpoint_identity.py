#!/usr/bin/env python3
"""Bind a completed SHA-256 checkpoint receipt to current filesystem identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from tools.checkpoint_verify import VerificationError, validate_lock, write_json
else:
    from checkpoint_verify import VerificationError, validate_lock, write_json


SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp_ns(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise VerificationError("verification captured_at_utc is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise VerificationError("verification captured_at_utc lacks a timezone")
    return int(parsed.timestamp() * 1_000_000_000)


def expected_sha256(lock: dict[str, Any], entry: dict[str, Any]) -> str:
    if entry["kind"] == "metadata":
        return lock["local_small_file_sha256"][entry["path"]]
    return entry["lfs_sha256"]


def bind_checkpoint_identity(
    checkpoint_dir: Path, lock_path: Path, verification_path: Path
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    lock_path = lock_path.resolve()
    verification_path = verification_path.resolve()
    lock = read_json(lock_path)
    lock_files = validate_lock(lock)
    verification = read_json(verification_path)
    verification_hash = sha256_file(verification_path)
    lock_hash = sha256_file(lock_path)
    if (
        not isinstance(verification, dict)
        or verification.get("schema_version") != 1
        or verification.get("status") != "verified"
        or verification.get("model") != lock["model"]
        or verification.get("revision") != lock["revision"]
        or verification.get("checkpoint_dir_observed") != os.fspath(checkpoint_dir)
        or verification.get("model_lock", {}).get("sha256") != lock_hash
        or verification.get("expected_file_count") != len(lock_files)
        or verification.get("verified_file_count") != len(lock_files)
        or verification.get("bytes_hashed") != lock["expected_total_bytes"]
    ):
        raise VerificationError("checkpoint verification receipt does not bind this lock and tree")
    captured_ns = timestamp_ns(str(verification.get("captured_at_utc", "")))
    records = verification.get("files")
    if not isinstance(records, list) or len(records) != len(lock_files):
        raise VerificationError("checkpoint verification file inventory is incomplete")
    by_path = {record.get("path"): record for record in records if isinstance(record, dict)}
    if len(by_path) != len(lock_files):
        raise VerificationError("checkpoint verification paths are duplicate or malformed")

    identities = []
    for entry in lock_files:
        relative = entry["path"]
        record = by_path.get(relative)
        expected_hash = expected_sha256(lock, entry)
        if (
            not isinstance(record, dict)
            or record.get("status") != "verified"
            or record.get("expected_size") != entry["size"]
            or record.get("actual_size") != entry["size"]
            or record.get("expected_sha256") != expected_hash
            or record.get("actual_sha256") != expected_hash
            or not SHA256_RE.fullmatch(expected_hash)
        ):
            raise VerificationError(f"verification record mismatch for {relative}")
        path = checkpoint_dir / relative
        try:
            info = path.lstat()
        except OSError as exc:
            raise VerificationError(f"cannot stat {path}: {exc}") from exc
        if (
            not stat.S_ISREG(info.st_mode)
            or path.is_symlink()
            or info.st_size != entry["size"]
            or info.st_mtime_ns > captured_ns
        ):
            raise VerificationError(
                f"{relative} is not the unchanged regular file covered by verification"
            )
        identities.append(
            {
                "path": relative,
                "bytes": info.st_size,
                "sha256": expected_hash,
                "device": info.st_dev,
                "inode": info.st_ino,
                "modified_ns": info.st_mtime_ns,
                "changed_ns": info.st_ctime_ns,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "semantic": "firewing_verified_checkpoint_live_identity_binding",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": lock["model"],
        "revision": lock["revision"],
        "checkpoint_dir": os.fspath(checkpoint_dir),
        "model_lock_sha256": lock_hash,
        "verification_receipt": {
            "path": os.fspath(verification_path),
            "sha256": verification_hash,
            "captured_at_utc": verification["captured_at_utc"],
            "bytes_hashed": verification["bytes_hashed"],
        },
        "files": identities,
        "file_count": len(identities),
        "total_bytes": sum(item["bytes"] for item in identities),
        "network_access_performed": False,
        "accepted_tokens": 0,
        "performance_claim": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = bind_checkpoint_identity(
            args.checkpoint_dir, args.model_lock, args.verification
        )
        write_json(args.output, report)
    except VerificationError as exc:
        write_json(
            args.output,
            {"schema_version": SCHEMA_VERSION, "status": "failed", "error": str(exc)},
        )
        print(f"checkpoint-identity: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": os.fspath(args.output),
                "files": report["file_count"],
                "bytes": report["total_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
