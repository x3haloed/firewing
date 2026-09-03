from __future__ import annotations

import json
import unittest
from pathlib import Path


class FullDecoderLayer1FixtureTests(unittest.TestCase):
    def test_committed_fixture_uses_dynamic_layer1_routes(self) -> None:
        fixture = json.loads(
            Path("fixtures/decoder_layer/qwen3_8_flash_next_layer1_ple.json").read_text()
        )
        self.assertEqual(fixture["configuration"]["layer"], 1)
        self.assertEqual(fixture["configuration"]["layer_type"], "linear_attention")
        self.assertTrue(
            all(
                record["tensor"].startswith("model.language_model.layers.1.")
                for record in fixture["tensors"].values()
            )
        )
        self.assertEqual([step["mode"] for step in fixture["steps"]], ["initial_chunk", "cached_recurrent"])
        routes = [step["selected_experts"] for step in fixture["steps"]]
        self.assertEqual(len(routes[0]), 10)
        self.assertEqual(len(routes[1]), 10)
        self.assertNotEqual(routes[0], routes[1])
        for step in fixture["steps"]:
            self.assertEqual(step["expert_execution_order"], sorted(step["selected_experts"]))
            self.assertEqual(len(step["captures"]), 16)


if __name__ == "__main__":
    unittest.main()
