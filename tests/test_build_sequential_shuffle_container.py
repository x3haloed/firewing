import json
import unittest
from pathlib import Path

from tools.build_sequential_shuffle_container import (
    FIRST_SEMANTIC,
    SECOND_SEMANTIC,
    target_rows,
)


class SequentialShuffleContainerTests(unittest.TestCase):
    def test_four_target_rows_bind_the_frozen_sequential_union(self) -> None:
        first = json.loads(
            Path("fixtures/endpoint/qwen3_8_flash_next_firewing_four_token.json").read_text()
        )
        second = json.loads(
            Path(
                "fixtures/endpoint/qwen3_8_flash_next_firewing_second_q2_six_token.json"
            ).read_text()
        )
        rows = target_rows(first, FIRST_SEMANTIC, (2, 3)) + target_rows(
            second, SECOND_SEMANTIC, (4, 5)
        )
        self.assertEqual([len(row) for row in rows], [48, 48, 48, 48])
        self.assertTrue(all(len(event) == 10 for row in rows for event in row))
        self.assertEqual(
            len({identity for row in rows for event in row for identity in event}), 1097
        )


if __name__ == "__main__":
    unittest.main()
