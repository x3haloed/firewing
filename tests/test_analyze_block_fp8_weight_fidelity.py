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


if __name__ == "__main__":
    unittest.main()
