"""Contract tests: action ordering, train-only stats, and temporal label boundaries."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.RoboDojo.data import RoboDojoDataset, current_subtask, normalize, select_episodes, write_json


class DataContractTest(unittest.TestCase):
    def test_q99_matches_native_clipping_and_constant_dimensions(self):
        stats = {"q01": [0, .5, -2], "q99": [2, .5, 2]}
        actual = normalize(np.array([[-1, .5, 1], [4, .5, -4]]), stats)
        np.testing.assert_allclose(actual, [[-1, .5, .5], [1, .5, -1]])

    def test_only_current_subtask_used(self):
        messages = [{"style": "plan", "content": "future answer", "timestamp": 0},
                    {"style": "subtask", "content": "pick", "timestamp": 0},
                    {"style": "subtask", "content": "place", "timestamp": 3}]
        self.assertEqual(current_subtask(messages, 2), "pick")
        self.assertEqual(current_subtask(messages, 3), "place")
        with self.assertRaises(ValueError):
            current_subtask(messages, -1)

    def test_conflicting_simultaneous_subtasks_rejected(self):
        messages = [{"style": "subtask", "content": text, "timestamp": 0} for text in ["pick", "place"]]
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            current_subtask(messages, 0)

    def test_one_per_task_holdout_is_disjoint(self):
        tasks = {i: [{"episode_index": i}] for i in range(35)}
        selected = select_episodes(tasks, seed=42, episodes_per_task=1, max_tasks=0,
                                   validation_fraction=.1, split_unit="task")
        train = {task for _, task, split in selected if split == "train"}
        val = {task for _, task, split in selected if split == "val"}
        self.assertEqual((len(train), len(val)), (31, 4))
        self.assertFalse(train & val)
        with self.assertRaisesRegex(ValueError, "two episodes"):
            select_episodes(tasks, seed=42, episodes_per_task=1, max_tasks=0,
                            validation_fraction=.1, split_unit="episode")

    def test_episode_and_subtask_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "manifest.json", {"spec": {"source": directory}, "episodes": [
                {"offset": 0, "length": 3, "split": "train", "task": "goal"},
                {"offset": 3, "length": 2, "split": "train", "task": "other"}]})
            action = np.arange(5 * 14, dtype=np.float32).reshape(5, 14) / 100
            np.save(root / "action.npy", action)
            np.save(root / "timestamps.npy", np.arange(5) / 25)
            write_json(root / "dataset_statistics.json", {"arx_x5": {"action": {"q01": [0] * 14, "q99": [1] * 14}}})
            write_json(root / "subtasks.json", ["pick", "pick", "place", "left", "left"])
            data = RoboDojoDataset(root, horizon=4)
            _, _, language, target, mask = data.target(1)
            self.assertEqual(language, "goal\nCurrent subtask: pick")
            self.assertEqual(mask.tolist(), [True, False, False, False])
            np.testing.assert_allclose(target[0], 2 * action[1] - 1)
            np.testing.assert_array_equal(target[0], target[-1])
            _, _, language, _, mask = data.target(2)
            self.assertEqual(language, "goal\nCurrent subtask: place")
            self.assertEqual(mask.tolist(), [True, False, False, False])
            _, _, language, _, _ = data.target(3)
            self.assertEqual(language, "other\nCurrent subtask: left")
            # Repeated identical text still denotes a new atomic segment when its start changes.
            write_json(root / "subtasks.json", ["pick", "pick", "pick", "left", "left"])
            np.save(root / "subtask_starts.npy", np.array([0., 0., .08, 0., 0.]))
            data = RoboDojoDataset(root, horizon=4)
            self.assertEqual(data.target(1)[-1].tolist(), [True, False, False, False])

    def test_real_oft_loss_ignores_actions_after_boundary(self):
        from types import SimpleNamespace
        import torch
        from starVLA.model.framework.QwenOFT import Qwenvl_OFT
        owner = SimpleNamespace(_infer_shortest_angular_joint_mask=lambda *args: None)
        predicted = torch.ones(1, 4, 14, requires_grad=True)
        target = torch.zeros_like(predicted)
        target[:, 2:] = 1000000
        loss = Qwenvl_OFT._compute_action_loss(owner, predicted, target,
                                              action_valid_mask=torch.tensor([[1, 1, 0, 0]]))
        self.assertEqual(loss.item(), 1.)
        loss.backward()
        self.assertTrue(torch.all(predicted.grad[:, 2:] == 0))
        self.assertTrue(torch.all(predicted.grad[:, :2] > 0))

    def test_real_smoke_statistics_exclude_validation(self):
        root = Path('results/RoboDojo/data_smoke')
        if not root.exists():
            self.skipTest('Prepare the smoke cache first')
        manifest = json.loads((root / 'manifest.json').read_text())
        train = [r for r in manifest['episodes'] if r['split'] == 'train']
        val = [r for r in manifest['episodes'] if r['split'] == 'val']
        self.assertFalse({r['source_episode_index'] for r in train} & {r['source_episode_index'] for r in val})
        actions = np.load(root / 'action.npy')
        selected = np.concatenate([actions[r['offset']:r['offset'] + r['length']] for r in train])
        stats = json.loads((root / 'dataset_statistics.json').read_text())['arx_x5']['action']
        np.testing.assert_allclose(stats['q01'], np.quantile(selected, .01, axis=0))
        np.testing.assert_allclose(stats['q99'], np.quantile(selected, .99, axis=0))


if __name__ == '__main__':
    unittest.main()
