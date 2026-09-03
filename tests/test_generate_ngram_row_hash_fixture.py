import io
import unittest

from tools.generate_ngram_row_hash_fixture import read_exact_row, safetensor_payload_start


class NGramRowHashFixtureTests(unittest.TestCase):
    def test_reads_only_selected_row(self) -> None:
        header = b'{"tensor":{"dtype":"U8","shape":[3,4],"data_offsets":[0,12]}} '
        file_bytes = len(header).to_bytes(8, "little") + header + bytes(range(12))
        handle = io.BytesIO(file_bytes)
        self.assertEqual(safetensor_payload_start(handle), 8 + len(header))
        self.assertEqual(read_exact_row(handle, 8 + len(header), 0, 1, 3, 4), bytes([4, 5, 6, 7]))

    def test_rejects_out_of_bounds_row(self) -> None:
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            read_exact_row(io.BytesIO(), 0, 0, 3, 3, 4)


if __name__ == "__main__":
    unittest.main()
