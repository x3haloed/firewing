#!/usr/bin/env python3
"""Fail-closed capacity check for placing a locked checkpoint on a filesystem."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CapacityError(RuntimeError):
    """Raised when capacity inputs cannot be trusted."""


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapacityError(f"cannot read JSON {path}: {exc}") from exc


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


def inspect_capacity(destination: Path, lock_path: Path, reserve_bytes: int) -> dict[str, Any]:
    destination = destination.resolve()
    lock_path = lock_path.resolve()
    if reserve_bytes < 0:
        raise CapacityError("reserve bytes cannot be negative")
    if not destination.is_dir():
        raise CapacityError(f"destination directory does not exist: {destination}")
    lock = read_json(lock_path)
    required_bytes = lock.get("expected_total_bytes") if isinstance(lock, dict) else None
    revision = lock.get("revision") if isinstance(lock, dict) else None
    if not isinstance(required_bytes, int) or required_bytes <= 0:
        raise CapacityError("model lock has no positive expected_total_bytes")
    if not re.fullmatch(r"[0-9a-f]{40}", str(revision or "")):
        raise CapacityError("model lock revision is not a 40-hex commit")

    stat = os.statvfs(destination)
    filesystem_available_bytes = stat.f_bavail * stat.f_frsize
    filesystem_total_bytes = stat.f_blocks * stat.f_frsize
    required_with_reserve = required_bytes + reserve_bytes
    margin_bytes = filesystem_available_bytes - required_with_reserve
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if margin_bytes >= 0 else "insufficient_capacity",
        "model": lock.get("model"),
        "revision": revision,
        "destination": os.fspath(destination),
        "destination_device": os.stat(destination).st_dev,
        "filesystem_total_bytes": filesystem_total_bytes,
        "filesystem_available_bytes": filesystem_available_bytes,
        "checkpoint_required_bytes": required_bytes,
        "declared_reserve_bytes": reserve_bytes,
        "required_with_reserve_bytes": required_with_reserve,
        "margin_bytes": margin_bytes,
        "network_access_performed": False,
        "performance_claim": None,
        "accepted_tokens": 0,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--model-lock", required=True, type=Path)
    parser.add_argument("--reserve-bytes", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = inspect_capacity(args.destination, args.model_lock, args.reserve_bytes)
        write_json(args.output, report)
    except CapacityError as exc:
        print(f"checkpoint-capacity: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
