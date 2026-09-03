import itertools
import unittest

from tools.analyze_sequential_cache_milp import Record, build_intervals, solve_retention


class SequentialCacheMilpTests(unittest.TestCase):
    def test_milp_matches_exhaustive_tiny_interval_cache(self) -> None:
        records = {
            "a": Record("a", 3, 4),
            "b": Record("b", 2, 3),
            "c": Record("c", 2, 2),
        }
        events = [["a"], ["b"], ["a", "c"], ["b"]]
        intervals = build_intervals(events)
        best_saved = -1
        for selected in itertools.product((False, True), repeat=len(intervals)):
            resident = [0] * len(events)
            saved = 0
            for interval, retained in zip(intervals, selected, strict=True):
                if not retained:
                    continue
                record = records[interval.identity]
                saved += record.physical_bytes
                for boundary in range(interval.after_event + 1, interval.hit_event + 1):
                    resident[boundary] += record.compressed_bytes
            if max(resident) <= 4:
                best_saved = max(best_saved, saved)
        solved_intervals, result = solve_retention(records, events, 4, node_limit=100)
        selected_saved = sum(
            records[interval.identity].physical_bytes
            for interval, value in zip(solved_intervals, result.x, strict=True)
            if value >= 0.5
        )
        self.assertEqual(selected_saved, best_saved)
        self.assertEqual(result.status, 0)

    def test_intervals_include_free_initial_retention(self) -> None:
        intervals = build_intervals([["a"], ["b"], ["a"]])
        self.assertEqual(
            [(row.identity, row.after_event, row.hit_event) for row in intervals],
            [("a", -1, 0), ("a", 0, 2), ("b", -1, 1)],
        )


if __name__ == "__main__":
    unittest.main()
