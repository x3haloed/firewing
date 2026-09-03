from __future__ import annotations

import unittest

import torch

from tools.analyze_block_fp8_weight_fidelity import (
    block_fp8_weight,
    block_int8_weight,
    error_metrics,
)
from tools.analyze_q2_lossless_experts import AnalysisError


class BlockFp8WeightFidelityTests(unittest.TestCase):
    def test_constant_aligned_block_rounds_exactly_and_accounts_scale(self) -> None:
        weight = torch.full((128, 128), 2.0, dtype=torch.bfloat16)
        decoded, scales, artifact_bytes = block_fp8_weight(weight)
        self.assertTrue(torch.equal(decoded, weight))
        self.assertEqual(tuple(scales.shape), (1, 1))
        self.assertEqual(artifact_bytes, 128 * 128 + 4)

    def test_unknown_shape_and_zero_reference_fail_closed(self) -> None:
        with self.assertRaises(AnalysisError):
            block_fp8_weight(torch.zeros((127, 128), dtype=torch.bfloat16))
        with self.assertRaises(AnalysisError):
            error_metrics(torch.zeros(1), torch.zeros(1))

    def test_symmetric_int8_block_preserves_scale_grid_closely(self) -> None:
        scale = 2.0 / 127.0
        values = torch.arange(-64, 64, dtype=torch.float32) * scale
        weight = values.repeat(128, 1).to(torch.bfloat16)
        decoded, scales, artifact_bytes = block_int8_weight(weight)
        self.assertEqual(tuple(scales.shape), (1, 1))
        self.assertEqual(artifact_bytes, 128 * 128 + 4)
        self.assertLess(error_metrics(decoded, weight)["relative_l2"], 0.01)

    def test_symmetric_int8_block32_has_exact_scale_ledger(self) -> None:
        weight = torch.full((128, 128), 2.0, dtype=torch.bfloat16)
        decoded, scales, artifact_bytes = block_int8_weight(weight, 32)
        self.assertTrue(torch.equal(decoded, weight))
        self.assertEqual(tuple(scales.shape), (4, 4))
        self.assertEqual(artifact_bytes, 128 * 128 + 16 * 4)

    def test_invalid_block_size_fails_closed(self) -> None:
        with self.assertRaises(AnalysisError):
            block_int8_weight(torch.zeros((128, 128), dtype=torch.bfloat16), 0)

    def test_finer_int8_scale_ledgers_are_monotonic(self) -> None:
        weight = torch.full((128, 128), 2.0, dtype=torch.bfloat16)
        _, scales16, bytes16 = block_int8_weight(weight, 16)
        _, scales8, bytes8 = block_int8_weight(weight, 8)
        self.assertEqual(tuple(scales16.shape), (8, 8))
        self.assertEqual(tuple(scales8.shape), (16, 16))
        self.assertEqual(bytes16, 128 * 128 + 64 * 4)
        self.assertEqual(bytes8, 128 * 128 + 256 * 4)
        self.assertLess(bytes16, bytes8)

    def test_block4_is_last_square_grid_smaller_than_bf16(self) -> None:
        weight = torch.full((128, 128), 2.0, dtype=torch.bfloat16)
        decoded4, scales4, bytes4 = block_int8_weight(weight, 4)
        _, scales2, bytes2 = block_int8_weight(weight, 2)
        self.assertTrue(torch.equal(decoded4, weight))
        self.assertEqual(tuple(scales4.shape), (32, 32))
        self.assertEqual(tuple(scales2.shape), (64, 64))
        self.assertEqual(bytes4, weight.numel() * 5 // 4)
        self.assertEqual(bytes2, weight.numel() * 2)


if __name__ == "__main__":
    unittest.main()
