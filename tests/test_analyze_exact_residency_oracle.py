import json
import unittest
from pathlib import Path

from tools.analyze_exact_residency_oracle import AnalysisError, belady, route_events


class ExactResidencyOracleTests(unittest.TestCase):
    def test_future_aware_cache_reuses_and_evicts_exactly(self) -> None:
        events = [
            {(0, 0), (0, 1)},
            {(1, 0), (1, 1)},
            {(0, 0), (1, 0)},
        ]
        result = belady(events, 2)
        self.assertEqual(result["distinct_layer_experts"], 4)
        self.assertEqual(result["free_initial_experts"], 2)
        self.assertEqual(result["misses"], 3)
        self.assertEqual(result["misses_by_event"], [0, 2, 1])

    def test_capacity_smaller_than_route_fails_closed(self) -> None:
        with self.assertRaises(AnalysisError):
            belady([{(0, value) for value in range(10)}], 9)

    def test_committed_endpoint_route_authority_is_complete(self) -> None:
        endpoint = json.loads(
            Path("fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json").read_text()
        )
        events = route_events(endpoint)
        self.assertEqual(len(events), 96)
        self.assertTrue(all(len(event) == 10 for event in events))
        self.assertEqual(len(set().union(*events)), 859)


if __name__ == "__main__":
    unittest.main()
