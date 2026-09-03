from __future__ import annotations

import json
import unittest
from pathlib import Path


class AccumulatedLayers01FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(
            Path("fixtures/accumulated/qwen3_8_flash_next_layers0_1.json").read_text()
        )

    def test_layer1_consumes_exact_layer0_output_hash(self) -> None:
        for ordinal, step in enumerate(self.fixture["steps"]):
            ple_step = self.fixture["layer1_ple"]["case"]["steps"][ordinal]
            self.assertEqual(
                step["captures"]["layer0_output"],
                ple_step["captures"]["hidden_states"],
            )
            self.assertEqual(
                step["captures"]["layer1_output"],
                self.fixture["layer1_decoder"]["steps"][ordinal]["captures"]["layer_output"],
            )

    def test_accumulation_changes_both_layer1_routes(self) -> None:
        standalone = json.loads(
            Path("fixtures/decoder_layer/qwen3_8_flash_next_layer1_ple.json").read_text()
        )
        accumulated_routes = [step["layer1_selected_experts"] for step in self.fixture["steps"]]
        standalone_routes = [step["selected_experts"] for step in standalone["steps"]]
        self.assertNotEqual(accumulated_routes[0], standalone_routes[0])
        self.assertNotEqual(accumulated_routes[1], standalone_routes[1])

    def test_every_accumulated_layer1_dense_tensor_has_layer1_identity(self) -> None:
        tensors = self.fixture["layer1_decoder"]["tensors"].values()
        self.assertTrue(
            all(record["tensor"].startswith("model.language_model.layers.1.") for record in tensors)
        )


if __name__ == "__main__":
    unittest.main()
