from __future__ import annotations

import json
import unittest
from pathlib import Path


class TextOutputFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(
            Path(
                "fixtures/accumulated/qwen3_8_flash_next_final_mixer_logits.json"
            ).read_text()
        )
        cls.decoder = json.loads(
            Path("fixtures/accumulated/qwen3_8_flash_next_layers4_47.json").read_text()
        )

    def test_output_starts_at_exact_decoder_boundaries(self) -> None:
        for step, decoder in zip(self.fixture["steps"], self.decoder["final_outputs"]):
            self.assertEqual(step["captures"]["decoder_output"], decoder["sha256"])

    def test_fixture_freezes_full_vocab_hashes_without_logits_payload(self) -> None:
        self.assertEqual(self.fixture["configuration"]["vocab_size"], 248_320)
        for step in self.fixture["steps"]:
            self.assertEqual(len(step["captures"]["logits"]), 64)
            self.assertNotIn("full_logits", step)

    def test_ranked_diagnostic_preserves_cutoff_ties(self) -> None:
        for step in self.fixture["steps"]:
            above = step["strictly_above_cutoff_token_ids"]
            ties = step["cutoff_tie_token_ids"]
            self.assertLess(len(above), 20)
            self.assertGreaterEqual(len(above) + len(ties), 20)
            self.assertTrue(set(step["top20_token_ids"]).issubset(set(above) | set(ties)))
        self.assertGreater(len(self.fixture["steps"][0]["cutoff_tie_token_ids"]), 1)


if __name__ == "__main__":
    unittest.main()
