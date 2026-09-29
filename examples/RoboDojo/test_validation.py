"""Validation must cover tasks and remain fixed across checkpoints."""
from collections import Counter
import unittest

from examples.RoboDojo.validation import balanced_validation_subset


class FakeActions:
    episodes = [{"task_index": 0, "length": 100}, {"task_index": 1, "length": 10},
                {"task_index": 2, "length": 10}]

    def __len__(self):
        return sum(row["length"] for row in self.episodes)


class FakeVQA:
    rows = [{"task_index": task} for task in range(35) for _ in range(10)]

    def __len__(self):
        return len(self.rows)


class ValidationSamplingTest(unittest.TestCase):
    def test_action_frame_selection_balances_tasks(self):
        first, report = balanced_validation_subset(FakeActions(), 9, 42)
        second, _ = balanced_validation_subset(FakeActions(), 9, 42)
        self.assertEqual(first.indices, second.indices)
        self.assertEqual(len(set(first.indices)), 9)
        counts = Counter(0 if i < 100 else 1 if i < 110 else 2 for i in first.indices)
        self.assertEqual(counts, {0: 3, 1: 3, 2: 3})
        self.assertEqual(report["samples_by_task"], dict(counts))

    def test_formal_vqa_validation_covers_all_35_tasks(self):
        selected, report = balanced_validation_subset(FakeVQA(), 64, 42)
        self.assertEqual(len(selected), 64)
        self.assertEqual(len(set(selected.indices)), 64)
        self.assertEqual(len(report["samples_by_task"]), 35)
        self.assertEqual(set(report["samples_by_task"].values()), {1, 2})

    def test_small_validation_is_not_duplicated_by_sampler(self):
        data = FakeActions()
        selected, report = balanced_validation_subset(data, 200, 42)
        self.assertEqual(len(selected), len(data))
        self.assertEqual(len(set(selected.indices)), len(data))
        self.assertTrue(report["complete_validation_set"])


if __name__ == "__main__":
    unittest.main()
