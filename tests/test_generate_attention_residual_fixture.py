from __future__ import annotations

import unittest

import torch

from tools.generate_attention_residual_fixture import (
    HC_HIDDEN,
    INPUT_SPECS,
    make_hyper_input,
)
from tools.generate_deltanet_fixture import tensor_bytes


class AttentionResidualFixtureTests(unittest.TestCase):
    def test_inputs_are_distinct_deterministic_bf16_hyper_states(self) -> None:
        inputs = [make_hyper_input(spec) for spec in INPUT_SPECS]
        self.assertTrue(all(value.dtype == torch.bfloat16 for value in inputs))
        self.assertTrue(all(list(value.shape) == [1, 1, HC_HIDDEN] for value in inputs))
        self.assertTrue(all(value.is_contiguous() for value in inputs))
        self.assertNotEqual(tensor_bytes(inputs[0]), tensor_bytes(inputs[1]))
        self.assertEqual(tensor_bytes(inputs[0]), tensor_bytes(make_hyper_input(INPUT_SPECS[0])))

    def test_input_formula_covers_all_four_streams(self) -> None:
        spec = INPUT_SPECS[1]
        values = make_hyper_input(spec).flatten().float()
        for index in (0, 1, 2559, 2560, 7679, HC_HIDDEN - 1):
            expected = (
                (index * spec["multiplier"] + spec["add"]) % spec["modulus"]
                - spec["center"]
            ) / spec["divisor"]
            self.assertEqual(values[index].item(), torch.tensor(expected).to(torch.bfloat16).item())


if __name__ == "__main__":
    unittest.main()
