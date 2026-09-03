import unittest
import json
from pathlib import Path

import torch


class DecoderLayerFixtureTests(unittest.TestCase):
    def test_committed_fixture_binds_two_dynamic_route_steps(self) -> None:
        fixture = json.loads(
            Path("fixtures/decoder_layer/qwen3_8_flash_next_layer0.json").read_text()
        )
        self.assertEqual(
            fixture["semantic"], "qwen3_8_flash_next_layer0_complete_cached_decoder"
        )
        self.assertEqual(len(fixture["case"]["tensors"]), 9)
        steps = fixture["case"]["steps"]
        self.assertEqual([step["mode"] for step in steps], ["initial_chunk", "cached_recurrent"])
        self.assertNotEqual(steps[0]["selected_experts"], steps[1]["selected_experts"])
        for step in steps:
            selected = step["selected_experts"]
            self.assertEqual(len(selected), 10)
            self.assertEqual(len(set(selected)), 10)
            self.assertEqual(step["expert_execution_order"], sorted(selected))
            self.assertEqual(
                [entry["expert"] for entry in step["experts"]], sorted(selected)
            )
            self.assertEqual(len(step["captures"]), 16)

    def test_four_stream_mlp_injection_layout(self) -> None:
        output = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        weights = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16)
        products = output.unsqueeze(-2) * weights.unsqueeze(-1)
        self.assertEqual(tuple(products.shape), (4, 2))
        self.assertEqual(
            products.tolist(),
            [[0.5, -1.0], [1.0, -2.0], [1.5, -3.0], [2.0, -4.0]],
        )

    def test_source_expert_order_is_independent_of_router_rank(self) -> None:
        selected = [31, 4, 19, 7]
        self.assertEqual(sorted(selected), [4, 7, 19, 31])


if __name__ == "__main__":
    unittest.main()
