import unittest

import torch

from tools.generate_sparse_moe_block_fixture import shared_expert_forward


class SparseMoeBlockFixtureTests(unittest.TestCase):
    def test_tiny_shared_expert_and_gate(self) -> None:
        hidden = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        gate = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
        up = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.bfloat16)
        down = torch.eye(2, dtype=torch.bfloat16)
        shared_gate = torch.tensor([[0.0, 0.0]], dtype=torch.bfloat16)
        routed = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
        outputs = shared_expert_forward(hidden, gate, up, down, shared_gate, routed)
        self.assertEqual(outputs["shared_gate_sigmoid"].item(), 0.5)
        self.assertTrue(
            torch.equal(outputs["gated_shared"], outputs["shared_down"] * 0.5)
        )
        self.assertTrue(
            torch.equal(outputs["combined"], routed + outputs["gated_shared"])
        )


if __name__ == "__main__":
    unittest.main()
