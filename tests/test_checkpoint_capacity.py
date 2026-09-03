import json
import tempfile
import unittest
from pathlib import Path

from tools.checkpoint_capacity import CapacityError, inspect_capacity, main


class CheckpointCapacityTests(unittest.TestCase):
    def make_lock(self, root: Path, required_bytes: int) -> Path:
        lock = root / "model-lock.json"
        lock.write_text(
            json.dumps(
                {
                    "model": "Qwen/Qwen3.8-Flash-Next",
                    "revision": "d" * 40,
                    "expected_total_bytes": required_bytes,
                }
            ),
            encoding="utf-8",
        )
        return lock

    def test_small_checkpoint_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = inspect_capacity(root, self.make_lock(root, 1), 0)
            self.assertEqual(report["status"], "pass")
            self.assertGreaterEqual(report["margin_bytes"], 0)

    def test_impossible_checkpoint_fails_and_preserves_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = self.make_lock(root, 10**18)
            output = root / "capacity.json"
            result = main([str(root), "--model-lock", str(lock), "--output", str(output)])
            self.assertEqual(result, 3)
            self.assertEqual(json.loads(output.read_text())["status"], "insufficient_capacity")

    def test_negative_reserve_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(CapacityError, "negative"):
                inspect_capacity(root, self.make_lock(root, 1), -1)


if __name__ == "__main__":
    unittest.main()
