import json
import unittest
from pathlib import Path

from tools.analyze_q2_lossless_experts import AnalysisError
from tools.analyze_sequential_q2_zstd_oracle import route_records


class SequentialQ2ZstdOracleTests(unittest.TestCase):
    def test_second_transaction_route_union_is_frozen(self) -> None:
        endpoint = json.loads(
            Path(
                "fixtures/endpoint/qwen3_8_flash_next_firewing_second_q2_six_token.json"
            ).read_text()
        )
        records = route_records(
            endpoint,
            "qwen3_8_flash_next_firewing_six_token_cached_text_logits",
            (4, 5),
            731,
        )
        self.assertEqual(len(records), 731)

    def test_wrong_step_count_fails_closed(self) -> None:
        endpoint = json.loads(
            Path(
                "fixtures/endpoint/qwen3_8_flash_next_firewing_second_q2_six_token.json"
            ).read_text()
        )
        endpoint["layers"][0]["decoder"]["steps"].pop()
        with self.assertRaises(AnalysisError):
            route_records(
                endpoint,
                "qwen3_8_flash_next_firewing_six_token_cached_text_logits",
                (4, 5),
                731,
            )


if __name__ == "__main__":
    unittest.main()
