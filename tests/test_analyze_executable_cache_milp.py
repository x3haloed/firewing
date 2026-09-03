from __future__ import annotations

import itertools
import unittest

from tools.analyze_executable_cache_milp import solve_retention
from tools.analyze_sequential_cache_milp import Record, build_intervals


class ExecutableCacheMilpTests(unittest.TestCase):
    def test_mixed_cache_solver_matches_exhaustive_tiny_oracle(self) -> None:
        records = {
            "a": Record("a", compressed_bytes=3, physical_bytes=4),
            "b": Record("b", compressed_bytes=2, physical_bytes=3),
            "c": Record("c", compressed_bytes=4, physical_bytes=5),
        }
        events = [["a", "b"], ["c"], ["a"], ["b", "c"]]
        capacity = 6
        source_bytes = 5
        physical_rate = 10.0
        decode_rate = 12.0
        intervals = build_intervals(events)

        best = float("inf")
        for choices in itertools.product(range(3), repeat=len(intervals)):
            # 0 absent, 1 compressed, 2 decoded.
            if any(
                sum(
                    0
                    if choices[index] == 0
                    else records[interval.identity].compressed_bytes
                    if choices[index] == 1
                    else source_bytes
                    for index, interval in enumerate(intervals)
                    if interval.after_event < boundary <= interval.hit_event
                )
                > capacity
                for boundary in range(len(events))
            ):
                continue
            physical = sum(
                records[interval.identity].physical_bytes
                for interval, choice in zip(intervals, choices, strict=True)
                if choice == 0
            )
            decodes = sum(choice != 2 for choice in choices)
            best = min(
                best,
                max(
                    physical / physical_rate,
                    decodes * source_bytes / decode_rate,
                ),
            )

        actual_intervals, result = solve_retention(
            records,
            events,
            capacity,
            source_bytes,
            physical_rate,
            decode_rate,
            node_limit=10_000,
        )
        self.assertEqual(actual_intervals, intervals)
        self.assertLess(abs(result.fun - best), 1e-8)


if __name__ == "__main__":
    unittest.main()
