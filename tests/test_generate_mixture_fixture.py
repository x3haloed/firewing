import unittest

import torch

from tools.generate_mixture_fixture import accumulate_bf16_in_expert_order


class MixtureFixtureTests(unittest.TestCase):
    def test_mixture_rounds_after_each_expert(self) -> None:
        contributions = [
            torch.tensor([256.0], dtype=torch.bfloat16),
            torch.tensor([1.0], dtype=torch.bfloat16),
            torch.tensor([-256.0], dtype=torch.bfloat16),
        ]
        # 256 + 1 rounds back to 256 in BF16, so the final value is zero.
        self.assertEqual(accumulate_bf16_in_expert_order(contributions).item(), 0.0)
        self.assertEqual(sum(value.float() for value in contributions).item(), 1.0)

    def test_mixture_order_is_observable(self) -> None:
        first = [
            torch.tensor([256.0], dtype=torch.bfloat16),
            torch.tensor([-256.0], dtype=torch.bfloat16),
            torch.tensor([1.0], dtype=torch.bfloat16),
        ]
        second = [first[0], first[2], first[1]]
        self.assertEqual(accumulate_bf16_in_expert_order(first).item(), 1.0)
        self.assertEqual(accumulate_bf16_in_expert_order(second).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
