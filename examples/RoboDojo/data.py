"""RoboDojo v3 cache: original 14D order, continuous grippers, train-only q01/q99."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

from examples.RoboDojo.video import read_rgb_frame

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def normalize(values, stats):
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    result = np.array(values, dtype=np.float32, copy=True)
    variable = low != high
    result[..., variable] = 2 * (values[..., variable] - low[variable]) / (high[variable] - low[variable]) - 1
    return np.clip(result, -1, 1).astype(np.float32)


def resolve_subtask(messages, timestamp):
    candidates = [m for m in (messages or []) if m.get("style") == "subtask"
                  and m.get("content") and m.get("timestamp") is not None
                  and float(m["timestamp"]) <= timestamp + 1e-5]
    if not candidates:
        raise ValueError(f"Missing current subtask at t={timestamp}; review annotations first")
    latest = max(float(m["timestamp"]) for m in candidates)
    current = [m for m in candidates if float(m["timestamp"]) == latest]
    if len({m["content"] for m in current}) != 1:
        raise ValueError(f"Conflicting subtask labels at t={latest}")
    return current[0]


def current_subtask(messages, timestamp):
    return resolve_subtask(messages, timestamp)["content"]


def select_episodes(by_task, *, seed, episodes_per_task, max_tasks, validation_fraction, split_unit):
    """Keep entire episodes together; task holdout supports one annotated episode per task."""
    selected, rng = [], np.random.default_rng(seed)
    tasks = sorted(by_task)[:max_tasks or None]
    held_out = set()
    if split_unit == "task" and validation_fraction:
        n_val = max(1, round(len(tasks) * validation_fraction))
        if n_val >= len(tasks):
            raise ValueError("Task holdout needs at least two tasks")
        held_out = set(rng.permutation(tasks)[:n_val].tolist())
    for task in tasks:
        rows = sorted(by_task[task], key=lambda r: r["episode_index"])
        rows = [rows[i] for i in rng.permutation(len(rows))]
        if episodes_per_task:
            rows = rows[:episodes_per_task]
        n_val = 0
        if split_unit == "episode" and validation_fraction:
            n_val = max(1, round(len(rows) * validation_fraction))
            if n_val >= len(rows):
                raise ValueError("Episode holdout needs two episodes per task; use --split-unit task for one-per-task annotations")
        for i, row in enumerate(rows):
            is_val = task in held_out if split_unit == "task" else i < n_val
            selected.append((row, task, "val" if is_val else "train"))
    if not selected:
        raise ValueError("No episodes selected")
    return selected


def prepare(args):
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and cache must be separate, non-nested directories")
    pseudo_labels = (args.language_mode == "subtask" or args.include_vqa) and not args.annotations_reviewed
    allow_unreviewed_training = getattr(args, "allow_unreviewed_training", False)
    smoke_only = pseudo_labels and not allow_unreviewed_training
    if smoke_only and not args.allow_unreviewed_smoke:
        raise ValueError("Use --annotations-reviewed after review, --allow-unreviewed-smoke for diagnostics, "
                         "or explicitly authorize pseudo-label training with --allow-unreviewed-training")
    info = json.loads((source / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["features"]["action"]["shape"] == [14]
    assert info["features"]["observation.state"]["shape"] == [14]
    assert 0 <= args.validation_fraction < 1
    fingerprint_files = [source / "meta/info.json", source / "meta/tasks.parquet",
                         *sorted((source / "meta/episodes").rglob("*.parquet")),
                         *sorted((source / "data").rglob("*.parquet"))]
    fingerprint = hashlib.sha256()
    for path in fingerprint_files:
        fingerprint.update(str(path.relative_to(source)).encode())
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                fingerprint.update(block)
    spec = {"format_version": 1, "source": str(source), "source_sha256": fingerprint.hexdigest(),
            "seed": args.seed, "episodes_per_task": args.episodes_per_task,
            "max_tasks": args.max_tasks, "validation_fraction": args.validation_fraction,
            "language_mode": args.language_mode, "annotations_reviewed": args.annotations_reviewed}
    if allow_unreviewed_training:
        spec.update(allow_unreviewed_training=True, annotation_quality="unreviewed_pseudo_labels")
    if args.language_mode == "subtask" or args.split_unit != "episode":
        spec.update(format_version=2, split_unit=args.split_unit, smoke_only=smoke_only)
    if args.include_vqa:
        spec.update(format_version=3, include_vqa=True, vqa_types=["count", "attribute", "spatial"], smoke_only=smoke_only)
    if output.exists():
        marker = output / "manifest.json"
        if marker.exists() and json.loads(marker.read_text())["spec"] == spec:
            print(f"Reusing cache: {output}", flush=True)
            return
        raise FileExistsError(f"Refusing to overwrite incomplete/different cache: {output}")
    tasks_df = pq.read_table(source / "meta/tasks.parquet").to_pandas()
    tasks = {int(row.task_index): str(text) for text, row in tasks_df.iterrows()}
    metadata = [r for p in sorted((source / "meta/episodes").rglob("*.parquet"))
                for r in pq.read_table(p).to_pylist()]
    by_task = {}
    for row in metadata:
        if len(row["tasks"]) != 1:
            raise ValueError("Expected exactly one task per episode")
        task = next(k for k, v in tasks.items() if v == row["tasks"][0])
        by_task.setdefault(task, []).append(row)
    selected = select_episodes(by_task, seed=args.seed, episodes_per_task=args.episodes_per_task,
                               max_tasks=args.max_tasks, validation_fraction=args.validation_fraction,
                               split_unit=args.split_unit)
    lineage_file = source / "sample_selection.json"
    lineage = ({r["episode_index"]: r["source_episode_index"]
                for r in json.loads(lineage_file.read_text())["samples"]} if lineage_file.exists() else {})
    output.mkdir(parents=True)
    records, vectors, timestamps, languages = [], {"action": [], "state": []}, [], []
    subtask_starts = []
    vqa_records, vqa_excluded = [], {}
    offset = 0
    # Read each source shard once, never write to the source dataset.
    shards = {}
    for item in selected:
        row = item[0]
        shards.setdefault((row["data/chunk_index"], row["data/file_index"]), []).append(item)
    for (chunk, file), items in sorted(shards.items()):
        path = source / info["data_path"].format(chunk_index=chunk, file_index=file)
        columns = ["episode_index", "frame_index", "task_index", "timestamp", "action", "observation.state"]
        if args.language_mode == "subtask":
            columns.append("language_persistent")
        if args.include_vqa:
            columns.append("language_events")
        table = pq.read_table(path, columns=columns).to_pandas()
        for row, task, split in sorted(items, key=lambda item: item[0]["episode_index"]):
            ep = table[table.episode_index == row["episode_index"]].sort_values("frame_index")
            n = len(ep)
            assert n == row["length"] and np.array_equal(ep.frame_index.to_numpy(), np.arange(n))
            assert set(ep.task_index) == {task}
            for key, col in (("action", "action"), ("state", "observation.state")):
                values = np.stack(ep[col]).astype(np.float32)
                assert values.shape == (n, 14) and np.isfinite(values).all()
                vectors[key].append(values)
            timestamps.append(ep.timestamp.to_numpy(dtype=np.float64))
            if args.language_mode == "subtask":
                current = [resolve_subtask(list(msgs), float(t)) for msgs, t in
                           zip(ep.language_persistent, ep.timestamp)]
                labels = [m["content"] for m in current]
                subtask_starts.extend(float(m["timestamp"]) for m in current)
            else:
                labels = [""] * n
            languages.extend(labels)
            if args.include_vqa:
                from examples.RoboDojo.vqa import extract_vqa_pairs
                for frame, events in enumerate(ep.language_events):
                    pairs, excluded = extract_vqa_pairs(list(events) if events is not None else [])
                    for kind, count in excluded.items():
                        vqa_excluded[kind] = vqa_excluded.get(kind, 0) + count
                    for pair in pairs:
                        vqa_records.append({**pair, "source_episode_index": int(row["episode_index"]),
                                            "original_episode_index": lineage.get(row["episode_index"], row["episode_index"]),
                                            "task_index": task, "frame_index": frame, "absolute_index": offset + frame,
                                            "timestamp": float(ep.timestamp.iloc[frame]), "split": split})
            video = {}
            for camera in CAMERAS:
                key = f"observation.images.{camera}"
                video[camera] = {"path": info["video_path"].format(video_key=key,
                    chunk_index=row[f"videos/{key}/chunk_index"], file_index=row[f"videos/{key}/file_index"]),
                    "from_timestamp": row[f"videos/{key}/from_timestamp"]}
            records.append({"source_episode_index": row["episode_index"], "task_index": task,
                            "original_episode_index": lineage.get(row["episode_index"], row["episode_index"]),
                            "task": tasks[task], "offset": offset, "length": n, "split": split, "videos": video})
            offset += n
        print(f"Prepared shard {chunk}/{file}: {offset} frames", flush=True)
    arrays = {key: np.concatenate(value) for key, value in vectors.items()}
    train_mask = np.zeros(offset, dtype=bool)
    for record in records:
        if record["split"] == "train":
            train_mask[record["offset"]:record["offset"] + record["length"]] = True
    statistics = {}
    for key, values in arrays.items():
        np.save(output / f"{key}.npy", values)
        train = values[train_mask]
        statistics[key] = {"mean": train.mean(0).tolist(), "std": train.std(0).tolist(),
                           "min": train.min(0).tolist(), "max": train.max(0).tolist(),
                           "q01": np.quantile(train, .01, axis=0).tolist(),
                           "q99": np.quantile(train, .99, axis=0).tolist(), "mask": [True] * 14}
    statistics.update(num_trajectories=sum(r["split"] == "train" for r in records),
                      num_transitions=int(train_mask.sum()))
    np.save(output / "timestamps.npy", np.concatenate(timestamps))
    write_json(output / "dataset_statistics.json", {"arx_x5": statistics})
    write_json(output / "subtasks.json", languages if args.language_mode == "subtask" else [])
    if subtask_starts:
        np.save(output / "subtask_starts.npy", np.asarray(subtask_starts, dtype=np.float64))
    if args.include_vqa:
        if not vqa_records:
            raise ValueError("No supported VQA pairs found in annotations")
        with (output / "vqa.jsonl").open("w") as handle:
            for record in vqa_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Store the source root in the manifest; --source-override permits moving servers.
    manifest = {"spec": spec, "fps": info["fps"], "episodes": records,
                "frames": offset, "train_frames": int(train_mask.sum()), "val_frames": int((~train_mask).sum()),
                "action_order": ["left_arm_0:6", "left_gripper_6", "right_arm_7:13", "right_gripper_13"]}
    if subtask_starts:
        manifest["subtask_segments"] = sum(
            1 + int(np.count_nonzero(np.diff(subtask_starts[r["offset"]:r["offset"] + r["length"]])))
            for r in records)
    if args.include_vqa:
        manifest["vqa"] = {"train_pairs": sum(r["split"] == "train" for r in vqa_records),
                           "val_pairs": sum(r["split"] == "val" for r in vqa_records),
                           "excluded_by_type": vqa_excluded,
                           "coordinate_policy": "bbox/keypoint excluded: annotation coordinate space requires review"}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "episodes"}, indent=2), flush=True)


class RoboDojoDataset(Dataset):
    def __init__(self, root, split="train", horizon=50, image_size=224, source_override=None):
        cv2.setNumThreads(1)
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.source = Path(source_override or self.manifest["spec"]["source"])
        self.episodes = [r for r in self.manifest["episodes"] if r["split"] == split]
        self.ends = np.cumsum([r["length"] for r in self.episodes])
        self.action = np.load(self.root / "action.npy", mmap_mode="r")
        self.timestamps = np.load(self.root / "timestamps.npy", mmap_mode="r")
        self.stats = json.loads((self.root / "dataset_statistics.json").read_text())["arx_x5"]
        self.subtasks = json.loads((self.root / "subtasks.json").read_text())
        self.subtask_starts = (np.load(self.root / "subtask_starts.npy", mmap_mode="r")
                               if (self.root / "subtask_starts.npy").exists() else None)
        self.horizon, self.image_size = horizon, image_size

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def target(self, index):
        ep_idx = int(np.searchsorted(self.ends, index, side="right"))
        ep = self.episodes[ep_idx]
        frame = index - (int(self.ends[ep_idx - 1]) if ep_idx else 0)
        absolute = ep["offset"] + frame
        remaining = ep["length"] - frame
        lang = ep["task"]
        if self.subtasks:
            label = self.subtasks[absolute]
            lang += "\nCurrent subtask: " + label
            for future in range(1, min(remaining, self.horizon)):
                changed = (self.subtask_starts[absolute + future] != self.subtask_starts[absolute]
                           if self.subtask_starts is not None else self.subtasks[absolute + future] != label)
                if changed:
                    remaining = future
                    break
        valid = np.arange(self.horizon) < remaining
        indices = absolute + np.minimum(np.arange(self.horizon), remaining - 1)
        return ep, absolute, lang, normalize(self.action[indices], self.stats["action"]), valid

    def read_image(self, ep, absolute, camera):
        video = ep["videos"][camera]
        timestamp = video["from_timestamp"] + float(self.timestamps[absolute])
        rgb = read_rgb_frame(self.source / video["path"], timestamp)
        return Image.fromarray(cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA))

    def __getitem__(self, index):
        ep, absolute, lang, action, valid = self.target(index)
        images = [self.read_image(ep, absolute, camera) for camera in CAMERAS]
        return {"image": images, "lang": lang, "action": action, "action_valid_mask": valid,
                "robot_type": "arx_x5", "source_episode_index": ep["source_episode_index"],
                "original_episode_index": ep.get("original_episode_index", ep["source_episode_index"]),
                "task_index": ep["task_index"], "frame_index": absolute - ep["offset"],
                "timestamp": float(self.timestamps[absolute]),
                "current_subtask": self.subtasks[absolute] if self.subtasks else None,
                "subtask_start_timestamp": float(self.subtask_starts[absolute]) if self.subtask_starts is not None else None}


def collate(examples):
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes-per-task", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--language-mode", choices=["task", "subtask"], default="task")
    parser.add_argument("--split-unit", choices=["episode", "task"], default="episode")
    parser.add_argument("--include-vqa", action="store_true")
    quality = parser.add_mutually_exclusive_group()
    quality.add_argument("--annotations-reviewed", action="store_true")
    quality.add_argument("--allow-unreviewed-smoke", action="store_true")
    quality.add_argument("--allow-unreviewed-training", action="store_true",
                         help="Explicitly permit full pseudo-label training; does not mark annotations reviewed")
    prepare(parser.parse_args())
