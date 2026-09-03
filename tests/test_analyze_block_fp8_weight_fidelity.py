from __future__ import annotations

import unittest

import torch

from tools.analyze_block_fp8_weight_fidelity import (
    block_affine_uint8_weight,
    block_affine_uint8_with_exact_groups,
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

    def test_rectangular_int8_grid_preserves_scale_ledger(self) -> None:
        weight = torch.full((128, 128), 2.0, dtype=torch.bfloat16)
        decoded, scales, artifact_bytes = block_int8_weight(weight, (1, 16))
        self.assertTrue(torch.equal(decoded, weight))
        self.assertEqual(tuple(scales.shape), (128, 8))
        self.assertEqual(artifact_bytes, weight.numel() * 5 // 4)

    def test_clipped_int8_grid_changes_range_without_changing_bytes(self) -> None:
        weight = torch.ones((4, 4), dtype=torch.bfloat16)
        weight[0, 0] = 4.0
        decoded, scales, artifact_bytes = block_int8_weight(weight, (4, 4), 0.5)
        self.assertEqual(tuple(scales.shape), (1, 1))
        self.assertEqual(artifact_bytes, weight.numel() + 4)
        self.assertLess(decoded[0, 0].item(), weight[0, 0].item())
        with self.assertRaises(AnalysisError):
            block_int8_weight(weight, (4, 4), 1.01)

    def test_affine_uint8_grid_accounts_zero_points_and_constants(self) -> None:
        weight = torch.full((2, 16), 2.0, dtype=torch.bfloat16)
        weight[1] = torch.linspace(-1.0, 3.0, 16).to(torch.bfloat16)
        decoded, scales, artifact_bytes = block_affine_uint8_weight(
            weight, (1, 16)
        )
        self.assertTrue(torch.equal(decoded[0], weight[0]))
        self.assertEqual(tuple(scales.shape), (2, 1))
        self.assertEqual(artifact_bytes, weight.numel() + scales.numel() * 5)
        self.assertLess(error_metrics(decoded, weight)["relative_l2"], 0.01)

    def test_affine_exact_group_restores_largest_error_and_accounts_ordinal(self) -> None:
        weight = torch.linspace(-2.0, 3.0, 32).reshape(2, 16).to(torch.bfloat16)
        affine, scales, core_bytes = block_affine_uint8_weight(weight, (1, 16))
        restored, restored_scales, artifact_bytes = (
            block_affine_uint8_with_exact_groups(weight, (1, 16), 5000)
        )
        self.assertTrue(torch.equal(scales, restored_scales))
        self.assertLess(
            error_metrics(restored, weight)["relative_l2"],
            error_metrics(affine, weight)["relative_l2"],
        )
        self.assertEqual(artifact_bytes, core_bytes + 16 * 2 + 4)


if __name__ == "__main__":
    unittest.main()
