import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.bind_checkpoint_identity import bind_checkpoint_identity, main
from tools.checkpoint_verify import VerificationError, verify_checkpoint


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class CheckpointIdentityTests(unittest.TestCase):
    def make_case(self) -> tuple[Path, Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        shard = b"verified-shard"
        (root / "model.safetensors").write_bytes(shard)
        lock = {
            "schema_version": 1,
            "model": "Qwen/Qwen3.8-Flash-Next",
            "revision": "d" * 40,
            "expected_file_count": 1,
            "expected_total_bytes": len(shard),
            "expected_weight_shard_count": 1,
            "expected_weight_shard_bytes": len(shard),
            "files": [{"path": "model.safetensors", "size": len(shard), "kind": "weight_shard", "lfs_sha256": sha256(shard)}],
            "local_small_file_sha256": {},
        }
        lock_path = root / "lock.json"
        write_json(lock_path, lock)
        verification = verify_checkpoint(root, lock_path)
        verification_path = root / "verification.json"
        write_json(verification_path, verification)
        return root, lock_path, verification_path

    def test_binds_verified_content_to_live_identity(self) -> None:
        root, lock_path, verification_path = self.make_case()
        report = bind_checkpoint_identity(root, lock_path, verification_path)
        self.assertEqual(report["file_count"], 1)
        self.assertEqual(report["files"][0]["inode"], (root / "model.safetensors").stat().st_ino)
        self.assertEqual(report["verification_receipt"]["bytes_hashed"], 14)

    def test_post_verification_mutation_fails_closed(self) -> None:
        root, lock_path, verification_path = self.make_case()
        shard = root / "model.safetensors"
        shard.write_bytes(b"changed-content")
        future = datetime.now(timezone.utc) + timedelta(seconds=2)
        os.utime(shard, ns=(int(future.timestamp() * 1e9),) * 2)
        with self.assertRaisesRegex(VerificationError, "unchanged regular file"):
            bind_checkpoint_identity(root, lock_path, verification_path)

    def test_receipt_hash_mismatch_fails_closed(self) -> None:
        root, lock_path, verification_path = self.make_case()
        verification = json.loads(verification_path.read_text())
        verification["model_lock"]["sha256"] = "0" * 64
        write_json(verification_path, verification)
        with self.assertRaisesRegex(VerificationError, "does not bind"):
            bind_checkpoint_identity(root, lock_path, verification_path)

    def test_failure_is_preserved(self) -> None:
        root, lock_path, verification_path = self.make_case()
        output = root / "failure.json"
        result = main([str(root), "--model-lock", str(lock_path), "--verification", str(root / "missing.json"), "--output", str(output)])
        self.assertEqual(result, 2)
        self.assertEqual(json.loads(output.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
