import unittest
import json
from pathlib import Path

import torch

from tools.generate_ple_fixture import CONV_STATE, HC_HIDDEN, grouped_rms


class PleFixtureTests(unittest.TestCase):
    def test_committed_fixture_has_independent_two_step_state(self) -> None:
        fixture = json.loads(
            Path("fixtures/ple/qwen3_8_flash_next_layer1_decode.json").read_text()
        )
        steps = fixture["case"]["steps"]
        self.assertEqual([step["token_id"] for step in steps], [42, 43])
        self.assertEqual(
            [step["previous_context"] for step in steps],
            [[248044, 248044], [248044, 42]],
        )
        for step in steps:
            self.assertEqual(len(step["rows"]), 16)
            self.assertEqual(len(step["captures"]), 16)
            self.assertEqual(
                step["captures"]["convolution_state"]["shape"], [1, 10240, 9]
            )

    def test_grouped_rms_keeps_streams_independent(self) -> None:
        values = torch.cat(
            [torch.full((2560,), float(index + 1), dtype=torch.bfloat16) for index in range(4)]
        ).reshape(1, 1, -1)
        result = grouped_rms(values, torch.zeros(10240, dtype=torch.bfloat16))
        self.assertTrue(torch.equal(result, torch.ones_like(result)))

    def test_cached_dilated_convolution_state_has_nine_positions(self) -> None:
        state = torch.zeros((1, 4, CONV_STATE), dtype=torch.bfloat16)
        first = torch.ones((1, 4, 1), dtype=torch.bfloat16)
        state = torch.nn.functional.pad(first, (CONV_STATE - 1, 0))
        self.assertEqual(tuple(state.shape), (1, 4, 9))
        second = torch.full((1, 4, 1), 2.0, dtype=torch.bfloat16)
        state = torch.cat([state, second], dim=-1)[..., -CONV_STATE:]
        self.assertEqual(state[0, 0, -2:].tolist(), [1.0, 2.0])

    def test_accumulated_hidden_override_contract_is_full_four_stream_bf16(self) -> None:
        value = torch.zeros((1, 1, HC_HIDDEN), dtype=torch.bfloat16).contiguous()
        self.assertEqual(list(value.shape), [1, 1, 10240])
        self.assertTrue(value.is_contiguous())


if __name__ == "__main__":
    unittest.main()
