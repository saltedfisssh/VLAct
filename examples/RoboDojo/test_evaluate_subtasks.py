"""Offline metric masking, weighting and split isolation."""
import unittest
from types import SimpleNamespace

import numpy as np

from examples.RoboDojo.evaluate_subtasks import ActionErrors, check_split, selected_indices


class OfflineEvaluationTest(unittest.TestCase):
    def test_valid_steps_weighted_not_chunk_means(self):
        errors = ActionErrors()
        errors.add(np.ones((3, 14)), np.zeros((3, 14)), [1, 0, 0])
        errors.add(np.full((3, 14), 3.), np.zeros((3, 14)), [1, 1, 1])
        report = errors.report()
        self.assertEqual(report["normalized_l1"], 2.5)
        self.assertEqual(report["valid_action_steps"], 4)
        self.assertEqual(report["frames"], 2)
        bad_target = np.zeros((3, 14))
        bad_target[1:] = np.nan
        fresh = ActionErrors()
        fresh.add(np.ones((3, 14)), bad_target, [1, 0, 0])
        self.assertEqual(fresh.report()["normalized_l1"], 1.)
        with self.assertRaises(ValueError):
            fresh.add(np.full((3, 14), np.nan), np.zeros((3, 14)), [1, 1, 1])

    def test_selection_keeps_episode_offsets(self):
        data = SimpleNamespace(episodes=[{"length": 4}, {"length": 3}, {"length": 2}])
        self.assertEqual(selected_indices(data, 2, 0, 0), [0, 2, 4, 6, 7])
        self.assertEqual(selected_indices(data, 2, 2, 1), [0, 4])

    def test_original_episode_lineage_not_reindexed_episode(self):
        spec = {"source_sha256": "source-digest", "language_mode": "subtask"}
        trained = {"spec": spec, "episodes": [
            {"split": "train", "source_episode_index": 0, "original_episode_index": 100}]}
        validation = {"spec": spec, "episodes": [
            {"split": "val", "source_episode_index": 0, "original_episode_index": 101}]}
        check_split(validation, trained)
        validation["episodes"][0]["original_episode_index"] = 100
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_split(validation, trained)


if __name__ == "__main__":
    unittest.main()
