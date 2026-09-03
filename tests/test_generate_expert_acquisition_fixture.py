import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.generate_expert_acquisition_fixture import input_spec, sha256_range, tensor_layout


class ExpertAcquisitionFixtureTests(unittest.TestCase):
    def test_layer_input_specs_are_deterministic_and_distinct(self) -> None:
        self.assertEqual(input_spec(0)["multiplier"], 37)
        self.assertEqual(input_spec(47)["multiplier"], 131)
        self.assertNotEqual(input_spec(0), input_spec(47))
        with self.assertRaises(ValueError):
            input_spec(48)

    def test_tensor_layout_and_bounded_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.safetensors"
            value = torch.arange(12, dtype=torch.float32).to(torch.bfloat16).reshape(3, 4)
            save_file({"weight": value}, path)
            layout = tensor_layout(path, "weight")
            self.assertEqual(layout["shape"], [3, 4])
            self.assertEqual(layout["payload_bytes"], 24)
            with path.open("rb") as handle:
                digest = sha256_range(handle, layout["absolute_offset"], 24)
            expected = __import__("hashlib").sha256(value.view(torch.uint16).numpy().tobytes()).hexdigest()
            self.assertEqual(digest, expected)


if __name__ == "__main__":
    unittest.main()
