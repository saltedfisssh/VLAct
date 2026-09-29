"""Pseudo-label authorization and deferred-training gates, without GPUs."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from examples.RoboDojo.data import prepare
from examples.RoboDojo.after_annotation_train import process_start, ready_report, verify_cache


class FullTrainingGatesTest(unittest.TestCase):
    def make_source(self, root):
        source = root / "source"
        (source / "meta/episodes").mkdir(parents=True)
        (source / "data").mkdir()
        info = {
            "codebase_version": "v3.0", "fps": 25,
            "features": {"action": {"shape": [14]}, "observation.state": {"shape": [14]}},
            "data_path": "data/{file_index}.parquet", "video_path": "{video_key}/{file_index}.mp4",
        }
        (source / "meta/info.json").write_text(json.dumps(info))
        tasks = pd.DataFrame({"task_index": [0]}, index=["Put the object in the bowl."])
        pq.write_table(pa.Table.from_pandas(tasks), source / "meta/tasks.parquet")
        rows, metadata = [], []
        for episode in range(2):
            meta = {"episode_index": episode, "tasks": tasks.index.tolist(), "length": 2,
                    "data/chunk_index": 0, "data/file_index": 0}
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                for key, value in (("chunk_index", 0), ("file_index", 0), ("from_timestamp", episode)):
                    meta[f"videos/observation.images.{camera}/{key}"] = value
            metadata.append(meta)
            for frame in range(2):
                rows.append({"episode_index": episode, "task_index": 0, "frame_index": frame,
                             "timestamp": frame / 25, "action": [float(episode + frame)] * 14,
                             "observation.state": [float(episode + frame)] * 14,
                             "language_persistent": [{"style": "subtask", "timestamp": 0.,
                                                       "content": "Pick up the object."}]})
        pq.write_table(pa.Table.from_pylist(rows), source / "data/0.parquet")
        pq.write_table(pa.Table.from_pylist(metadata), source / "meta/episodes/0.parquet")
        return source

    def test_explicit_full_opt_in_preserves_unreviewed_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            common = dict(source=source, language_mode="subtask", include_vqa=False,
                          annotations_reviewed=False, seed=42, episodes_per_task=0, max_tasks=0,
                          validation_fraction=.5, split_unit="episode")
            with self.assertRaisesRegex(ValueError, "explicitly authorize"):
                prepare(argparse.Namespace(**common, output=root / "forbidden",
                                           allow_unreviewed_smoke=False, allow_unreviewed_training=False))
            for full in (False, True):
                output = root / ("full" if full else "smoke")
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare(argparse.Namespace(**common, output=output,
                                               allow_unreviewed_smoke=not full, allow_unreviewed_training=full))
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertFalse(manifest["spec"]["annotations_reviewed"])
                self.assertEqual(manifest["spec"]["smoke_only"], not full)
                self.assertEqual(manifest["spec"].get("allow_unreviewed_training", False), full)
                self.assertEqual({x["split"] for x in manifest["episodes"]}, {"train", "val"})
            self.assertNotIn("language_events", pq.read_schema(source / "data/0.parquet").names)

    def test_pid_identity_and_annotation_failure_gate(self):
        self.assertTrue(process_start(os.getpid()).isdigit())
        self.assertIsNone(process_start(2147483647))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(json.dumps({"total_episodes": 3500, "total_frames": 10}))
            report = {"status": "passed", "errors": [], "episodes": 3500, "frames": 10,
                      "per_episode": {str(i): {} for i in range(3500)}}
            (root / "annotation_report.json").write_text(json.dumps(report))
            self.assertEqual(ready_report(root)["episodes"], 3500)
            report["status"] = "failed"
            (root / "annotation_report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                ready_report(root)

    def test_full_split_gate_rejects_repeated_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = [{"task_index": task, "original_episode_index": task * 100 + episode,
                         "split": "train" if episode < 90 else "val"}
                        for task in range(35) for episode in range(100)]
            manifest = {"episodes": episodes, "frames": 100, "spec": {
                "annotations_reviewed": False, "allow_unreviewed_training": True,
                "language_mode": "subtask", "include_vqa": True},
                "vqa": {"train_pairs": 10, "val_pairs": 10}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_cache(root)["train_episodes"], 3150)
            manifest["episodes"][90]["original_episode_index"] = 0
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "leakage"):
                verify_cache(root)


if __name__ == "__main__":
    unittest.main()
