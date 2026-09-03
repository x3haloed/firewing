import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.checkpoint_census import CensusError, build_census, main


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_safetensors(path: Path, tensors: dict[str, dict[str, object]], payload: bytes) -> None:
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    padding = (-len(header)) % 8
    header += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def git_blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


class CheckpointCensusTests(unittest.TestCase):
    def make_checkpoint(self, *, include_second_shard: bool = False) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        first = root / "model-00001-of-00002.safetensors"
        second = root / "model-00002-of-00002.safetensors"
        write_safetensors(
            first,
            {
                "model.language_model.layers.0.mlp.experts.gate_up_proj": {
                    "dtype": "BF16",
                    "shape": [2, 2],
                    "data_offsets": [0, 8],
                }
            },
            b"\0" * 8,
        )
        if include_second_shard:
            write_safetensors(
                second,
                {
                    "model.visual.patch_embed.proj.weight": {
                        "dtype": "F32",
                        "shape": [1],
                        "data_offsets": [0, 4],
                    }
                },
                b"\0" * 4,
            )
        write_json(root / "config.json", {"model_type": "qwen4_exp"})
        files = {
            "config.json": {
                "size": (root / "config.json").stat().st_size,
                "blob_id": git_blob_id(root / "config.json"),
            },
            first.name: {
                "size": first.stat().st_size,
                "blob_id": "b" * 40,
                "lfs_size": first.stat().st_size,
                "lfs_sha256": "1" * 64,
                "xet_hash": "2" * 64,
            },
            second.name: {
                "size": second.stat().st_size if second.exists() else 100,
                "blob_id": "c" * 40,
                "lfs_size": second.stat().st_size if second.exists() else 100,
                "lfs_sha256": "3" * 64,
                "xet_hash": "4" * 64,
            },
        }
        write_json(
            root / ".cache" / "huggingface" / "trees" / ("d" * 40 + ".json"),
            {"format_version": 1, "files": files},
        )
        return root

    def test_partial_download_parses_only_complete_shards(self) -> None:
        root = self.make_checkpoint()
        census, model_lock = build_census(root)
        self.assertEqual(census["status"], "incomplete_download")
        self.assertEqual(census["expected"]["weight_shard_count"], 2)
        self.assertEqual(census["observed"]["complete_weight_shard_count"], 1)
        self.assertEqual(census["observed"]["parsed_tensor_count"], 1)
        self.assertEqual(census["observed"]["category_bytes"]["routed_experts"], 8)
        self.assertFalse(census["payload_sha256_verification_complete"])
        self.assertEqual(model_lock["payload_verification_status"], "pending_full_download_and_sha256")

    def test_complete_download_closes_header_census(self) -> None:
        root = self.make_checkpoint(include_second_shard=True)
        census, _ = build_census(root)
        self.assertEqual(census["status"], "headers_complete_payload_hashes_pending")
        self.assertEqual(census["observed"]["category_bytes"]["vision"], 4)
        self.assertEqual(census["observed"]["parsed_parameter_count"], 5)

    def test_require_complete_preserves_report_and_fails(self) -> None:
        root = self.make_checkpoint()
        output = root / "report.json"
        result = main([str(root), "--output", str(output), "--require-complete"])
        self.assertEqual(result, 3)
        self.assertTrue(output.exists())

    def test_tensor_shape_offset_disagreement_fails_closed(self) -> None:
        root = self.make_checkpoint()
        shard = root / "model-00001-of-00002.safetensors"
        write_safetensors(
            shard,
            {
                "bad": {
                    "dtype": "BF16",
                    "shape": [2, 2],
                    "data_offsets": [0, 6],
                }
            },
            b"\0" * 6,
        )
        tree_path = next((root / ".cache" / "huggingface" / "trees").glob("*.json"))
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        tree["files"][shard.name]["size"] = shard.stat().st_size
        tree["files"][shard.name]["lfs_size"] = shard.stat().st_size
        write_json(tree_path, tree)
        with self.assertRaisesRegex(CensusError, "byte mismatch"):
            build_census(root)

    def test_metadata_git_blob_disagreement_fails_closed(self) -> None:
        root = self.make_checkpoint()
        config = root / "config.json"
        config.write_bytes(config.read_bytes().replace(b"qwen4_exp", b"qwen4_bad"))
        census, model_lock = build_census(root)
        self.assertEqual(census["observed"]["content_integrity_mismatch_count"], 1)
        self.assertEqual(census["content_integrity_mismatches"][0]["path"], "config.json")
        self.assertNotIn("config.json", model_lock["local_small_file_sha256"])

    def test_non_shard_lfs_content_uses_lfs_sha256(self) -> None:
        root = self.make_checkpoint()
        tokenizer = root / "tokenizer.json"
        tokenizer.write_bytes(b"tokenizer-lfs-content")
        tree_path = next((root / ".cache" / "huggingface" / "trees").glob("*.json"))
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        tree["files"][tokenizer.name] = {
            "size": tokenizer.stat().st_size,
            "blob_id": "a" * 40,
            "lfs_size": tokenizer.stat().st_size,
            "lfs_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            "xet_hash": "b" * 64,
        }
        write_json(tree_path, tree)

        census, model_lock = build_census(root)
        entry = next(item for item in model_lock["files"] if item["path"] == tokenizer.name)
        self.assertEqual(entry["kind"], "lfs_artifact")
        self.assertEqual(census["observed"]["content_integrity_mismatch_count"], 0)

        tokenizer.write_bytes(b"x" * tokenizer.stat().st_size)
        census, _ = build_census(root)
        self.assertEqual(census["observed"]["content_integrity_mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()
