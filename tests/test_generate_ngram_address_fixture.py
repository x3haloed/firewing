import unittest

import torch

from tools.generate_ngram_address_fixture import reference_addresses, shift_right_ignore_eos


class NGramAddressFixtureTests(unittest.TestCase):
    def test_shift_does_not_cross_eos(self) -> None:
        tokens = torch.tensor([[9, 10, 99, 20, 21]], dtype=torch.long)
        self.assertEqual(
            shift_right_ignore_eos(tokens, 2, 99).tolist(),
            [[99, 99, 9, 99, 99]],
        )

    def test_reference_shape_is_tokens_by_heads(self) -> None:
        values = reference_addresses(
            [1, 2],
            [99, 99],
            99,
            [3, 5, 7],
            [101] * 16,
            [index * 101 for index in range(16)],
            8,
        )
        self.assertEqual(len(values), 2)
        self.assertTrue(all(len(row) == 16 for row in values))
        self.assertTrue(all(index * 101 <= values[0][index] < (index + 1) * 101 for index in range(16)))


if __name__ == "__main__":
    unittest.main()
