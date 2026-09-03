from __future__ import annotations

import json
import unittest
from pathlib import Path


class AccumulatedLayer3FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(
            Path("fixtures/accumulated/qwen3_8_flash_next_layer3.json").read_text()
        )

    def test_layer3_consumes_exact_accumulated_layer2_output(self) -> None:
        for ordinal, step in enumerate(self.fixture["steps"]):
            attention_case = self.fixture["attention"]["cases"][ordinal]
            self.assertEqual(
                step["captures"]["layer2_output"],
                attention_case["captures"]["hyper_input"],
            )

    def test_every_tensor_has_layer3_identity(self) -> None:
        attention_tensors = self.fixture["attention"]["tensors"].values()
        decoder_tensors = self.fixture["decoder"]["tensors"].values()
        self.assertTrue(
            all(
                record["tensor"].startswith("model.language_model.layers.3.")
                for record in [*attention_tensors, *decoder_tensors]
            )
        )

    def test_second_step_retains_first_step_full_attention_cache(self) -> None:
        captures = self.fixture["attention"]["cases"][1]["captures"]
        self.assertEqual(captures["attention.raw_indexer_cache"]["shape"], [1, 2, 128])
        self.assertEqual(captures["attention.key_cache"]["shape"], [1, 2, 2, 256])
        self.assertEqual(captures["attention.value_cache"]["shape"], [1, 2, 2, 256])
        self.assertEqual(captures["attention.selected_tokens"]["shape"], [2])
        self.assertEqual(captures["attention.excluded_blocks"]["shape"], [0])

    def test_two_real_routes_are_disjoint(self) -> None:
        routes = [set(step["selected_experts"]) for step in self.fixture["steps"]]
        self.assertTrue(routes[0].isdisjoint(routes[1]))


if __name__ == "__main__":
    unittest.main()
