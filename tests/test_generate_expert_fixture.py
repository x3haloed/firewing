import unittest

import torch

from tools.generate_expert_fixture import capture_hash, expert_forward


class ExpertFixtureTests(unittest.TestCase):
    def test_tiny_expert_matches_auditable_equation(self) -> None:
        hidden = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        gate_up = torch.tensor(
            [[1.0, 2.0], [0.5, -0.5], [2.0, 0.0], [-1.0, 1.0]],
            dtype=torch.bfloat16,
        )
        down = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16
        )
        weight = torch.tensor(0.5, dtype=torch.bfloat16)
        outputs = expert_forward(hidden, gate_up, down, weight)
        expected_gate_up = torch.tensor([-3.0, 1.5, 2.0, -3.0], dtype=torch.bfloat16)
        expected_swiglu = (
            torch.nn.functional.silu(expected_gate_up[:2]) * expected_gate_up[2:]
        )
        self.assertTrue(torch.equal(outputs["gate_up"], expected_gate_up))
        self.assertTrue(torch.equal(outputs["swiglu"], expected_swiglu))
        self.assertTrue(torch.equal(outputs["down"], expected_swiglu))
        self.assertTrue(torch.equal(outputs["weighted_down"], expected_swiglu * weight))

    def test_capture_hash_is_over_bf16_payload(self) -> None:
        value = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        self.assertEqual(
            capture_hash(value),
            "7b429b1e3fd37fd03505ae4982471ea2c830392213b48a4e69976b5ebebce8e4",
        )


if __name__ == "__main__":
    unittest.main()
