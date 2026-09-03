from __future__ import annotations

import json
import unittest
from pathlib import Path


class AccumulatedLayer2FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(
            Path("fixtures/accumulated/qwen3_8_flash_next_layer2.json").read_text()
        )

    def test_layer2_consumes_exact_accumulated_layer1_output(self) -> None:
        for ordinal, step in enumerate(self.fixture["steps"]):
            attention_step = self.fixture["attention"]["case"]["steps"][ordinal]
            self.assertEqual(
                step["captures"]["layer1_output"],
                attention_step["captures"]["hyper_input"],
            )

    def test_every_tensor_has_layer2_identity(self) -> None:
        attention_tensors = self.fixture["attention"]["case"]["tensors"].values()
        decoder_tensors = self.fixture["decoder"]["tensors"].values()
        self.assertTrue(
            all(
                record["tensor"].startswith("model.language_model.layers.2.")
                for record in [*attention_tensors, *decoder_tensors]
            )
        )

    def test_two_real_routes_are_disjoint(self) -> None:
        routes = [set(step["selected_experts"]) for step in self.fixture["steps"]]
        self.assertTrue(routes[0].isdisjoint(routes[1]))


if __name__ == "__main__":
    unittest.main()
