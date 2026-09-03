from __future__ import annotations

import unittest

from tools.analyze_materialized_memory_floor import traffic_floor
from tools.analyze_q2_lossless_experts import AnalysisError


class MaterializedMemoryFloorTests(unittest.TestCase):
    def test_floor_charges_four_fixed_walks_and_three_cache_grants(self) -> None:
        report = traffic_floor(
            fixed_bytes=10,
            compressed_input_bytes=20,
            decoded_write_bytes=30,
            on_chip_cache_bytes=5,
            fabric_bytes_per_second=100.0,
        )
        self.assertEqual(report["target_fixed_matrix_reads_bytes"], 40)
        self.assertEqual(report["routed_expert_weight_reads_bytes"], 18_874_368_000)
        self.assertEqual(report["granted_cross_row_cache_reuse_bytes"], 15)
        self.assertEqual(report["adjusted_mandatory_fabric_bytes"], 18_874_368_075)

    def test_floor_fails_closed_on_invalid_bounds(self) -> None:
        with self.assertRaises(AnalysisError):
            traffic_floor(-1, 0, 0)
        with self.assertRaises(AnalysisError):
            traffic_floor(0, 0, 0, fabric_bytes_per_second=0)


if __name__ == "__main__":
    unittest.main()
