import json
import unittest
from pathlib import Path

import torch

from tools.generate_full_decoder_layer3_fixture import apply_mixture_transformer


class FullDecoderLayer3FixtureTests(unittest.TestCase):
    def test_mixture_transformer_has_strict_bf16_boundary(self) -> None:
        reference = torch.arange(8, dtype=torch.bfloat16)
        self.assertIs(apply_mixture_transformer(reference, None), reference)

        transformed = apply_mixture_transformer(
            reference,
            lambda **values: values["reference_mixture"] + 1,
            layer=4,
        )
        self.assertTrue(torch.equal(transformed, reference + 1))
        self.assertTrue(transformed.is_contiguous())

        for invalid in (
            lambda **_: torch.zeros(7, dtype=torch.bfloat16),
            lambda **_: torch.zeros(8, dtype=torch.float32),
            lambda **_: "not a tensor",
        ):
            with self.assertRaisesRegex(ValueError, "BF16 routed-mixture boundary"):
                apply_mixture_transformer(reference, invalid)

    def test_committed_fixture_uses_distinct_routes_after_full_attention(self) -> None:
        fixture = json.loads(
            Path("fixtures/decoder_layer/qwen3_8_flash_next_layer3.json").read_text()
        )
        self.assertEqual(fixture["configuration"]["layer"], 3)
        self.assertEqual(fixture["configuration"]["layer_type"], "full_attention")
        self.assertEqual(len(fixture["tensors"]), 9)
        self.assertEqual(len(fixture["expert_banks"]), 2)
        self.assertEqual(len(fixture["steps"]), 2)
        first, second = fixture["steps"]
        self.assertEqual(first["mode"], "initial")
        self.assertEqual(second["mode"], "active_qsa_pruning")
        self.assertEqual(len(first["selected_experts"]), 10)
        self.assertEqual(len(second["selected_experts"]), 10)
        self.assertTrue(set(first["selected_experts"]).isdisjoint(second["selected_experts"]))
        for step in fixture["steps"]:
            self.assertEqual(len(step["experts"]), 10)
            self.assertEqual(len(step["captures"]), 16)
            self.assertEqual(step["captures"]["layer_output"]["shape"], [1, 1, 10240])


if __name__ == "__main__":
    unittest.main()
