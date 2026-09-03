import json
import unittest
from pathlib import Path

from tools.analyze_q2_lossless_experts import AnalysisError, route_authority


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


if __name__ == "__main__":
    unittest.main()
