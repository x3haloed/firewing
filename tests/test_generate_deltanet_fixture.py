from __future__ import annotations

import hashlib
import unittest

import torch

from tools.generate_deltanet_fixture import HIDDEN, INPUT_SPECS, capture, make_input, tensor_bytes


class DeltaNetFixtureTests(unittest.TestCase):
    def test_inputs_are_distinct_deterministic_bf16(self) -> None:
        values = [make_input(spec) for spec in INPUT_SPECS]
        self.assertTrue(all(value.dtype == torch.bfloat16 for value in values))
        self.assertTrue(all(list(value.shape) == [1, 1, HIDDEN] for value in values))
        self.assertNotEqual(tensor_bytes(values[0]), tensor_bytes(values[1]))
        self.assertEqual(tensor_bytes(values[0]), tensor_bytes(make_input(INPUT_SPECS[0])))

    def test_capture_preserves_dtype_shape_and_raw_hash(self) -> None:
        bf16 = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        f32 = torch.tensor([0.25, -0.5], dtype=torch.float32)
        self.assertEqual(capture(bf16), {"dtype": "BF16", "shape": [2], "sha256": hashlib.sha256(tensor_bytes(bf16)).hexdigest()})
        self.assertEqual(capture(f32), {"dtype": "F32", "shape": [2], "sha256": hashlib.sha256(tensor_bytes(f32)).hexdigest()})


if __name__ == "__main__":
    unittest.main()
