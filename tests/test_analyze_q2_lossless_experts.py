import json
import unittest
from pathlib import Path

from tools.analyze_q2_lossless_experts import AnalysisError, fw0044_constants, route_authority


class Q2LosslessExpertTests(unittest.TestCase):
    def test_route_authority_has_exact_q2_union(self) -> None:
        endpoint = json.loads(
            Path("fixtures/endpoint/qwen3_8_flash_next_firewing_four_token.json").read_text()
        )
        routes = route_authority(endpoint)
        self.assertEqual(len(routes), 687)
        self.assertTrue(all(0 <= layer < 48 and 0 <= expert < 512 for layer, expert in routes))

    def test_unknown_semantic_fails_closed(self) -> None:
        endpoint = json.loads(
            Path("fixtures/endpoint/qwen3_8_flash_next_firewing_four_token.json").read_text()
        )
        endpoint["semantic"] = "modified"
        with self.assertRaises(AnalysisError):
            route_authority(endpoint)

    def test_raw_fw0044_trial_ledger_yields_frozen_constants(self) -> None:
        trials = []
        for token, physical in enumerate((100, 200)):
            for mode, walls, compute in (
                ("storage_only_control", (30, 10, 20), (None, None, None)),
                ("storage_compute_overlap", (40, 50, 60), (7, 9, 8)),
            ):
                trials.extend(
                    {
                        "token_ordinal": token,
                        "mode": mode,
                        "process_disk_bytes_read": physical,
                        "complete_wall_time_ns": wall,
                        "compute_wall_time_ns": compute_ns,
                    }
                    for wall, compute_ns in zip(walls, compute, strict=True)
                )
        prior = {"trials": trials}
        physical, storage_ms, compute_ns = fw0044_constants(prior)
        self.assertEqual(physical, [100, 200])
        self.assertEqual(storage_ms, [0.00002, 0.00002])
        self.assertEqual(compute_ns, 16)


if __name__ == "__main__":
    unittest.main()
