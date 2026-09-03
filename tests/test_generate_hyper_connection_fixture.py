from __future__ import annotations

import unittest

import torch

from tools.generate_hyper_connection_fixture import HC_COUNT, HC_HIDDEN, HIDDEN, INPUT_SPEC, make_hyper_input
from tools.generate_router_fixture import tensor_bytes


class HyperConnectionFixtureTests(unittest.TestCase):
    def test_input_is_deterministic_bf16(self) -> None:
        first = make_hyper_input()
        second = make_hyper_input()
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertTrue(first.is_contiguous())
        self.assertEqual(list(first.shape), [HC_HIDDEN])
        self.assertEqual(tensor_bytes(first), tensor_bytes(second))

    def test_input_affine_formula_and_group_shape(self) -> None:
        values = make_hyper_input().float()
        for index in (0, 1, 127, 2559, 2560, HC_HIDDEN - 1):
            expected = ((index * INPUT_SPEC["multiplier"] + INPUT_SPEC["add"]) % INPUT_SPEC["modulus"] - INPUT_SPEC["center"]) / INPUT_SPEC["divisor"]
            self.assertEqual(values[index].item(), torch.tensor(expected).to(torch.bfloat16).item())
        self.assertEqual(HC_HIDDEN, HC_COUNT * HIDDEN)


if __name__ == "__main__":
    unittest.main()
