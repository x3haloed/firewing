import json
import unittest
from pathlib import Path

import torch

from tools.generate_full_attention_fixture import (
    BUDGET,
    COMPRESS,
    INDEX_DIM,
    LONG_PAST,
    STATE_SPECS,
    deterministic_bf16,
)


class FullAttentionFixtureTests(unittest.TestCase):
    def test_committed_fixture_crosses_qsa_pruning_boundary(self) -> None:
        fixture = json.loads(
            Path("fixtures/full_attention/qwen3_8_flash_next_layer3.json").read_text()
        )
        self.assertEqual(fixture["configuration"]["layer"], 3)
        self.assertEqual(len(fixture["tensors"]), 9)
        initial, pruning = fixture["cases"]
        self.assertEqual(initial["mode"], "initial")
        self.assertEqual(initial["captures"]["selected_tokens"]["shape"], [1])
        self.assertEqual(pruning["mode"], "active_qsa_pruning")
        self.assertEqual(pruning["past_length"], LONG_PAST)
        self.assertEqual(pruning["captures"]["selected_blocks"]["shape"], [BUDGET // COMPRESS])
        self.assertEqual(
            pruning["captures"]["excluded_blocks"]["shape"],
            [LONG_PAST // COMPRESS - BUDGET // COMPRESS],
        )
        self.assertEqual(pruning["captures"]["selected_tokens"]["shape"], [BUDGET + 1])
        self.assertEqual(pruning["captures"]["raw_indexer_cache"]["shape"], [1, LONG_PAST + 1, INDEX_DIM])

    def test_synthetic_state_is_deterministic_and_row_distinct(self) -> None:
        spec = STATE_SPECS["indexer_keys"]
        first = deterministic_bf16((1, LONG_PAST, INDEX_DIM), spec)
        second = deterministic_bf16((1, LONG_PAST, INDEX_DIM), spec)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first[:, 0], first[:, -1]))


if __name__ == "__main__":
    unittest.main()
