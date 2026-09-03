from __future__ import annotations

import json
import unittest
from pathlib import Path


class AccumulatedLayers4Through47FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(
            Path("fixtures/accumulated/qwen3_8_flash_next_layers4_47.json").read_text()
        )
        cls.parent = json.loads(
            Path("fixtures/accumulated/qwen3_8_flash_next_layer3.json").read_text()
        )

    def test_exact_remaining_layer_schedule(self) -> None:
        layers = self.fixture["layers"]
        self.assertEqual([entry["layer"] for entry in layers], list(range(4, 48)))
        self.assertEqual(
            [entry["layer"] for entry in layers if entry["layer_type"] == "full_attention"],
            list(range(7, 48, 4)),
        )

    def test_every_layer_consumes_the_exact_previous_output(self) -> None:
        previous_steps = self.parent["steps"]
        for layer in self.fixture["layers"]:
            for ordinal in range(2):
                self.assertEqual(
                    layer["steps"][ordinal]["captures"]["layer_input"],
                    previous_steps[ordinal]["captures"]["layer3_output" if layer["layer"] == 4 else "layer_output"],
                )
            previous_steps = layer["steps"]

    def test_every_tensor_has_its_layer_identity(self) -> None:
        for layer in self.fixture["layers"]:
            number = layer["layer"]
            if layer["layer_type"] == "linear_attention":
                attention_tensors = layer["attention"]["case"]["tensors"].values()
            else:
                attention_tensors = layer["attention"]["tensors"].values()
            decoder_tensors = layer["decoder"]["tensors"].values()
            self.assertTrue(
                all(
                    record["tensor"].startswith(f"model.language_model.layers.{number}.")
                    for record in [*attention_tensors, *decoder_tensors]
                )
            )

    def test_all_routes_and_final_outputs_are_frozen(self) -> None:
        selections = sum(
            len(step["selected_experts"])
            for layer in self.fixture["layers"]
            for step in layer["steps"]
        )
        self.assertEqual(selections, 880)
        self.assertEqual(
            self.fixture["final_outputs"],
            [step["captures"]["layer_output"] for step in self.fixture["layers"][-1]["steps"]],
        )


if __name__ == "__main__":
    unittest.main()
