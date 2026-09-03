import json
import unittest
from pathlib import Path


class FullAttentionResidualFixtureTests(unittest.TestCase):
    def test_committed_fixture_composes_active_qsa_attention(self) -> None:
        fixture = json.loads(
            Path("fixtures/attention_residual/qwen3_8_flash_next_layer3.json").read_text()
        )
        self.assertEqual(fixture["configuration"]["layer"], 3)
        self.assertEqual(fixture["configuration"]["layer_type"], "full_attention")
        self.assertEqual(len(fixture["tensors"]), 13)
        self.assertEqual(len(fixture["cases"]), 2)
        initial, pruning = fixture["cases"]
        self.assertEqual(initial["captures"]["composed_output"]["shape"], [1, 1, 10240])
        self.assertEqual(pruning["past_length"], 2080)
        self.assertEqual(pruning["captures"]["attention.selected_blocks"]["shape"], [512])
        self.assertEqual(pruning["captures"]["attention.excluded_blocks"]["shape"], [8])
        for case in fixture["cases"]:
            self.assertEqual(len(case["captures"]), 36)
            self.assertEqual(case["captures"]["mixed_input"]["shape"], [1, 1, 2560])
            self.assertEqual(case["captures"]["injection_products"]["shape"], [1, 1, 4, 2560])


if __name__ == "__main__":
    unittest.main()
