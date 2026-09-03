import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.checkpoint_verify import VerificationError, main, verify_checkpoint


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class CheckpointVerifyTests(unittest.TestCase):
    def make_checkpoint(self) -> tuple[Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        metadata = b'{"model_type":"qwen4_exp"}\n'
        weights = b"fake-safetensors-shard"
        (root / "config.json").write_bytes(metadata)
        (root / "model-00001-of-00001.safetensors").write_bytes(weights)
        lock = {
            "schema_version": 1,
            "model": "Qwen/Qwen3.8-Flash-Next",
            "revision": "d" * 40,
            "expected_file_count": 2,
            "expected_total_bytes": len(metadata) + len(weights),
            "expected_weight_shard_count": 1,
            "expected_weight_shard_bytes": len(weights),
            "files": [
                {
                    "path": "config.json",
                    "size": len(metadata),
                    "kind": "metadata",
                    "lfs_sha256": None,
                },
                {
                    "path": "model-00001-of-00001.safetensors",
                    "size": len(weights),
                    "kind": "weight_shard",
                    "lfs_sha256": sha256(weights),
                },
            ],
            "local_small_file_sha256": {"config.json": sha256(metadata)},
        }
        lock_path = root / "model-lock.json"
        write_json(lock_path, lock)
        return root, lock_path

    def test_verifies_complete_checkpoint(self) -> None:
        root, lock_path = self.make_checkpoint()
        report = verify_checkpoint(root, lock_path)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["verified_file_count"], 2)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(report["bytes_hashed"], lock["expected_total_bytes"])

    def test_hash_mismatch_fails_and_preserves_report(self) -> None:
        root, lock_path = self.make_checkpoint()
        shard = root / "model-00001-of-00001.safetensors"
        shard.write_bytes(b"x" * shard.stat().st_size)
        output = root / "verification.json"
        result = main([str(root), "--model-lock", str(lock_path), "--output", str(output)])
        self.assertEqual(result, 3)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["sha256_mismatch_count"], 1)

    def test_preliminary_lock_fails_closed(self) -> None:
        root, lock_path = self.make_checkpoint()
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["local_small_file_sha256"] = {}
        write_json(lock_path, lock)
        with self.assertRaisesRegex(VerificationError, "regenerate the lock"):
            verify_checkpoint(root, lock_path)


if __name__ == "__main__":
    unittest.main()
