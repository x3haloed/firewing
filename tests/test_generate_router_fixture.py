import unittest

import torch

from tools.generate_router_fixture import make_hidden


class RouterFixtureTests(unittest.TestCase):
    def test_affine_mod_input_is_deterministic_bf16(self) -> None:
        spec = {"multiplier": 3, "add": 1, "modulus": 7, "center": 3, "divisor": 2, "sparse_stride": 1}
        self.assertEqual(make_hidden(5, spec).tolist(), [-1.0, 0.5, -1.5, 0.0, 1.5])

    def test_sparse_stride_zeros_other_positions(self) -> None:
        spec = {"multiplier": 3, "add": 1, "modulus": 7, "center": 3, "divisor": 2, "sparse_stride": 2}
        value = make_hidden(5, spec)
        self.assertEqual(value.dtype, torch.bfloat16)
        self.assertEqual(value.tolist(), [-1.0, 0.0, -1.5, 0.0, 1.5])


if __name__ == "__main__":
    unittest.main()
