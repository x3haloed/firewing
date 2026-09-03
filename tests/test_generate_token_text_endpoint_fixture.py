from __future__ import annotations

import json
import unittest
from pathlib import Path


class TokenTextEndpointFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.endpoint = json.loads(
            Path("fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json").read_text()
        )
        cls.tokenizer = json.loads(
            Path("fixtures/tokenizer/qwen3_8_flash_next.json").read_text()
        )

    def test_text_and_tokens_are_bound_to_tokenizer_fixture(self) -> None:
        case = next(case for case in self.tokenizer["raw_cases"] if case["name"] == "ascii")
        self.assertEqual(case["text"], self.endpoint["configuration"]["text"])
        self.assertEqual(case["token_ids"], self.endpoint["configuration"]["token_ids"])

    def test_every_attention_input_links_to_embedding_or_previous_layer(self) -> None:
        previous = self.endpoint["embedding_root_hashes"]
        for layer in self.endpoint["layers"]:
            if layer["layer"] == 1:
                actual = [
                    step["captures"]["hidden_states"]["sha256"]
                    for step in layer["attention"]["case"]["steps"]
                ]
            elif layer["layer_type"] == "linear_attention":
                actual = [
                    step["captures"]["hyper_input"]["sha256"]
                    for step in layer["attention"]["case"]["steps"]
                ]
            else:
                actual = [
                    case["captures"]["hyper_input"]["sha256"]
                    for case in layer["attention"]["cases"]
                ]
            self.assertEqual(actual, previous, f"layer {layer['layer']} input")
            previous = [step["captures"]["layer_output"]["sha256"] for step in layer["decoder"]["steps"]]

    def test_complete_logits_link_to_layer47(self) -> None:
        final = [
            step["captures"]["layer_output"]["sha256"]
            for step in self.endpoint["layers"][-1]["decoder"]["steps"]
        ]
        self.assertEqual(
            [step["captures"]["decoder_output"] for step in self.endpoint["output"]["steps"]],
            final,
        )
        self.assertTrue(all(len(step["captures"]["logits"]) == 64 for step in self.endpoint["output"]["steps"]))


if __name__ == "__main__":
    unittest.main()
