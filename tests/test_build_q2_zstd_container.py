import json
import unittest
from pathlib import Path

from tools.analyze_q2_lossless_experts import AnalysisError
from tools.build_q2_zstd_container import PAGE_BYTES, align_up, q2_events


class Q2ZstdContainerTests(unittest.TestCase):
    def test_alignment_is_exact_and_fails_closed(self) -> None:
        self.assertEqual(align_up(0), 0)
        self.assertEqual(align_up(1), PAGE_BYTES)
        self.assertEqual(align_up(PAGE_BYTES), PAGE_BYTES)
        with self.assertRaises(AnalysisError):
            align_up(1, 3)

    def test_q2_events_bind_all_layer_demands(self) -> None:
        endpoint = json.loads(
            Path("fixtures/endpoint/qwen3_8_flash_next_firewing_four_token.json").read_text()
        )
        rows = q2_events(endpoint)
        self.assertEqual([len(row) for row in rows], [48, 48])
        self.assertTrue(all(len(event) == 10 for row in rows for event in row))
        self.assertEqual(len({identity for row in rows for event in row for identity in event}), 687)


if __name__ == "__main__":
    unittest.main()
