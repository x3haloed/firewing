from __future__ import annotations

import unittest

import torch

from tools.generate_attention_residual_fixture import make_hyper_input
from tools.generate_ple_fixture import INPUT_SPECS


class PleAttentionResidualFixtureTests(unittest.TestCase):
    def test_bf16_ple_addition_is_staged_before_attention(self) -> None:
        hidden = make_hyper_input(INPUT_SPECS[0])
        ple = torch.full_like(hidden, 0.00390625)
        staged = (hidden + ple).contiguous()
        self.assertEqual(staged.dtype, torch.bfloat16)
        self.assertEqual(tuple(staged.shape), (1, 1, 10240))
        self.assertTrue(staged.is_contiguous())
        self.assertFalse(torch.equal(hidden, staged))

    def test_parent_inputs_match_attention_input_formula(self) -> None:
        first = make_hyper_input(INPUT_SPECS[0])
        second = make_hyper_input(INPUT_SPECS[1])
        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
