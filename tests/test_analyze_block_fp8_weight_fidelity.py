from __future__ import annotations

import unittest

import torch

from tools.analyze_block_fp8_weight_fidelity import block_fp8_weight, error_metrics
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


if __name__ == "__main__":
    unittest.main()
