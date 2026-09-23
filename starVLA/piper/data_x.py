# Adapted from kai0 (Apache-2.0); see NOTICE.md and LICENSE.kai0.
"""Task-balanced joint/EEF chunks from recorded Piper-X trajectories and read-only videos."""

import argparse
import hashlib
import json
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from starVLA.piper import geometry as g
from starVLA.piper import normalize
from starVLA.piper.data_piper import read_image
from starVLA.piper.prepare_piper import CAMERAS, atomic_json, make_split, validate_split
from starVLA.piper.transfer import PiperXTransfer

DEFAULT_TASKS = tuple(
    f"piper_x_{name}"
    for name in ("hang_mugs", "push_t", "stack_blocks", "stack_bowls", "swap_t", "sweep_blocks", "sweep_the_trash")
)
RUNGS = ("s0", "s2", "s2_pair")


def eef_poses(values):
    """Own-base flange poses and openings from recorded [xyz, quat xyzw, opening] rows."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 16 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite recorded EEF values with shape (T, 16)")
    poses = []
    for start in (0, 8):
        quat = values[:, start + 3 : start + 7]
        if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-3):
            raise ValueError("Recorded xyzw quaternions must have unit norm")
        pose = np.broadcast_to(np.eye(4), (len(values), 4, 4)).copy()
        pose[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
        pose[:, :3, 3] = values[:, start : start + 3]
        poses.append(pose)
    return poses[0], poses[1], values[:, [7, 15]]


def eef_states(values, *, rung="s2", alignment=None):
    """[xyz, quaternion xyzw, opening] per arm -> S2/S2_pair state, in SI units.

    Without ``alignment`` the pose reference stays the recording robot's own
    zero-offset flange, which must match deployment. With a ``PiperXTransfer``
    every recorded pose is mapped to the physically corresponding Piper flange
    (``T_P = inverse(A) @ T_X @ inverse(C)``), so Piper-X data, the S2_pair
    60 cm base relation, and the Piper deployment client share one tool
    convention. No source-robot FK, gripper scaling, or clipping occurs.
    """
    left, right, grippers = eef_poses(values)
    if alignment is not None:
        left, right = alignment.to_physical(left, 0), alignment.to_physical(right, 1)
    return g.states_from_poses(left, right, grippers, rung)


def _episode_path(root, task, episode_id):
    return Path(root) / "episodes" / task / f"episode_{episode_id:06d}.npz"


def _convert(job):
    source, destination, name, info, episode, prompts, rungs, alignment = job
    index = episode["episode_index"]
    fields = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
    columns = ["observation.qpos_ee", "timestamp", "task_index"]
    if "s0" in rungs:
        columns.append("observation.state")
    table = pq.read_table(source / info["data_path"].format(**fields), columns=columns).to_pydict()
    eef = np.asarray(table["observation.qpos_ee"], dtype=np.float64)
    if eef.shape != (episode["length"], 16):
        raise ValueError(f"Unexpected EEF length/width in {name}/{index}")
    states = {rung: eef_states(eef, rung=rung, alignment=alignment).astype(np.float32) for rung in rungs if rung != "s0"}
    if "s0" in rungs:
        joints = np.asarray(table["observation.state"], dtype=np.float32)
        if joints.shape != (len(eef), 14) or not np.isfinite(joints).all():
            raise ValueError(f"Expected finite joint states with shape (T, 14) in {name}/{index}")
        if not np.allclose(joints[:, [6, 13]], eef[:, [7, 15]], atol=1e-6):
            raise ValueError(f"Joint and EEF gripper openings differ in {name}/{index}")
        states["s0"] = joints
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    if (
        timestamps.shape != (len(eef),)
        or not np.isfinite(timestamps).all()
        or timestamps[0] < 0
        or not np.allclose(np.diff(timestamps), 1 / info["fps"], atol=1e-4)
    ):
        raise ValueError(f"Expected uniform 30 Hz timestamps in {name}/{index}")
    if {prompts.get(i) for i in table["task_index"]} != set(episode["tasks"]):
        raise ValueError(f"Episode prompt differs from frame task_index in {name}/{index}")
    for key in CAMERAS.values():
        video = source / info["video_path"].format(**fields, video_key=key)
        if not video.is_file():
            raise FileNotFoundError(video)
    path = _episode_path(destination, name, index)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **states, timestamps=timestamps)
    temporary.replace(path)
    return {
        **episode,
        "gripper_min_m": eef[:, [7, 15]].min(axis=0).tolist(),
        "gripper_max_m": eef[:, [7, 15]].max(axis=0).tolist(),
    }


class PiperXEEFDataSet:
    def __init__(self, root, *, rung="s2", split="train", action_horizon=50, images=True, manifest=None):
        self.root = Path(root)
        self.manifest = manifest if manifest is not None else json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("target_source") != "future_observation.qpos_ee":
            raise ValueError("Expected a recorded Piper-X EEF cache")
        available = tuple(self.manifest.get("rungs") or [self.manifest.get("rung", "s2")])
        if rung not in available:
            raise ValueError(f"Cache provides the {available} representation(s), not rung {rung!r}")
        if action_horizon != self.manifest["settings"]["horizon"]:
            raise ValueError("Action horizon differs from the cache/statistics")
        if split not in ("train", "held_out"):
            raise ValueError("Split must be train or held_out")
        alignment = self.manifest["settings"].get("alignment")
        self.alignment = None if alignment is None else PiperXTransfer.from_dict(alignment)
        self.rung, self.horizon, self.images, self.split = rung, action_horizon, images, split
        self.names = list(self.manifest["tasks"])
        self.ids, self.ends, self.episodes = {}, {}, {}
        for name, task in self.manifest["tasks"].items():
            self.ids[name] = task["split"][split]
            self.episodes[name] = {ep["episode_index"]: ep for ep in task["episodes"]}
            self.ends[name] = np.cumsum([self.episodes[name][i]["length"] - action_horizon for i in self.ids[name]])
            if not len(self.ends[name]) or self.ends[name][-1] <= 0:
                raise ValueError(f"No eligible {split} anchors for {name}")
        self.counts = np.array([self.ends[name][-1] for name in self.names])
        self.task_ends = np.cumsum(self.counts)
        self.slots = int(max(self.counts))
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.names) * self.slots if self.split == "train" else int(self.task_ends[-1])

    def locate(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.split == "train":
            slot, ordinal = divmod(int(index), len(self.names))
            # Equal virtual slots per task. Every real anchor appears, with repeat
            # counts differing by at most one within a task. Shuffle is external.
            anchor = slot * int(self.counts[ordinal]) // self.slots
        else:
            ordinal = int(np.searchsorted(self.task_ends, index, side="right"))
            anchor = int(index - (self.task_ends[ordinal - 1] if ordinal else 0))
        name = self.names[ordinal]
        episode = int(np.searchsorted(self.ends[name], anchor, side="right"))
        frame = int(anchor - (self.ends[name][episode - 1] if episode else 0))
        return name, self.ids[name][episode], frame

    def episode(self, name, index):
        key = (name, index)
        if key not in self.cache:
            with np.load(_episode_path(self.root, name, index)) as archive:
                self.cache[key] = {field: archive[field] for field in (self.rung, "timestamps")}
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def __getitem__(self, index):
        name, episode_id, frame = self.locate(index)
        episode = self.episode(name, episode_id)
        states = episode[self.rung]
        result = {
            "state": states[frame].copy(),
            "actions": g.encode_actions(states[frame], states[frame + 1 : frame + 1 + self.horizon], self.rung).astype(
                np.float32
            ),
            "prompt": self.episodes[name][episode_id]["tasks"][0],
        }
        if self.images:
            task = self.manifest["tasks"][name]
            info = task["info"]
            source = Path(task["source"])
            fields = {"episode_index": episode_id, "episode_chunk": episode_id // info["chunks_size"]}
            result["image"] = {
                key: read_image(
                    source / info["video_path"].format(**fields, video_key=video_key),
                    episode["timestamps"][frame],
                    info["fps"],
                )
                for key, video_key in CAMERAS.items()
            }
        return result


def _statistics(destination, manifest, samples, seed):
    """Every representation shares one set of anchors, episodes, and statistics seed."""
    for rung in manifest["rungs"]:
        dataset = PiperXEEFDataSet(
            destination, rung=rung, images=False, manifest=manifest, action_horizon=manifest["settings"]["horizon"]
        )
        rng = np.random.default_rng(seed)
        states, actions = [], []
        for ordinal, name in enumerate(dataset.names):
            used = set()
            # Sample the same equal-task virtual distribution used by the loader.
            indices = rng.integers(dataset.slots, size=samples) * len(dataset.names) + ordinal
            locations = sorted(dataset.locate(int(i)) for i in indices)
            for _, episode_id, frame in locations:
                used.add(episode_id)
                episode = dataset.episode(name, episode_id)[rung]
                states.append(episode[frame])
                actions.append(g.encode_actions(episode[frame], episode[frame + 1 : frame + 1 + dataset.horizon], rung))
            manifest["tasks"][name]["stats_episode_ids"] = sorted(used)
        stats = {}
        for key, chunks in (("state", states), ("actions", actions)):
            values = np.asarray(chunks, dtype=np.float64).reshape(-1, chunks[0].shape[-1])
            low, high = np.quantile(values, [0.01, 0.99], axis=0)
            stats[key] = normalize.NormStats(mean=values.mean(0), std=values.std(0), q01=low, q99=high)
        normalize.save(destination / "stats" / rung, stats)


def prepare(
    source_root,
    destination,
    *,
    tasks=DEFAULT_TASKS,
    seed=0,
    horizon=50,
    stats_samples=1024,
    workers=1,
    limit_episodes=None,
    rungs=("s2",),
    alignment=None,
):
    """Write only a separate cache; stats_samples is the anchor count PER TASK."""
    source_root, destination = Path(source_root).resolve(strict=True), Path(destination).resolve()
    if source_root == destination or source_root in destination.parents or destination in source_root.parents:
        raise ValueError("Source and output must be separate directory trees")
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(not isinstance(t, str) or not t.startswith("piper_x_") or Path(t).name != t for t in tasks)
    ):
        raise ValueError("Choose distinct Piper-X dataset directory names")
    rungs = tuple(rungs)
    if not rungs or len(set(rungs)) != len(rungs) or any(rung not in RUNGS for rung in rungs):
        raise ValueError(f"Choose distinct representations from {RUNGS}")
    if alignment is not None and not isinstance(alignment, PiperXTransfer):
        raise ValueError("alignment must be a PiperXTransfer mapping recorded poses to the Piper flange")
    if "s0" in rungs and alignment is not None:
        raise ValueError("S0 joints require native Piper X deployment without pose alignment")
    if horizon < 1 or stats_samples < 2 or workers < 1 or (limit_episodes is not None and limit_episodes < 2):
        raise ValueError("Positive horizon/workers and at least two statistics samples/smoke episodes required")
    records, fingerprints, jobs = {}, {}, []
    for name in tasks:
        source = (source_root / name).resolve(strict=True)
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError("Source and output must be separate directory trees")
        metadata = [(source / "meta" / file).read_bytes() for file in ("info.json", "episodes.jsonl", "tasks.jsonl")]
        info = json.loads(metadata[0])
        if info["codebase_version"] != "v2.1" or info["fps"] != 30 or info["chunks_size"] < 1:
            raise ValueError(f"Expected LeRobot v2.1 at 30 Hz: {name}")
        image_shapes = [info["features"][key]["shape"] for key in CAMERAS.values()]
        if info["features"]["observation.qpos_ee"]["shape"] != [16] or any(
            len(shape) != 3 or shape[2] != 3 or min(shape[:2]) <= 0 for shape in image_shapes
        ):
            raise ValueError(f"Expected 16 EEF values and three RGB camera streams: {name}")
        if "s0" in rungs and info["features"].get("observation.state", {}).get("shape") != [14]:
            raise ValueError(f"Expected 14 recorded joint/gripper values: {name}")
        episodes = [json.loads(line) for line in metadata[1].splitlines() if line.strip()]
        prompts = {
            row["task_index"]: row["task"]
            for row in (json.loads(line) for line in metadata[2].splitlines() if line.strip())
        }
        ids = [ep["episode_index"] for ep in episodes]
        if (
            len(set(ids)) != len(ids)
            or len(ids) != info["total_episodes"]
            or any(len(ep["tasks"]) != 1 or ep["tasks"][0] not in prompts.values() for ep in episodes)
        ):
            raise ValueError(f"Invalid episode IDs/count or ambiguous episode prompts: {name}")
        full_split = make_split(ids, seed)
        validate_split(full_split, ids)
        excluded = [
            {"episode_index": ep["episode_index"], "length": ep["length"]} for ep in episodes if ep["length"] <= horizon
        ]
        eligible = {ep["episode_index"] for ep in episodes if ep["length"] > horizon}
        split = {key: [i for i in full_split[key] if i in eligible] for key in ("train", "held_out")}
        if not all(split.values()):
            raise ValueError(f"No eligible train/held_out episodes in {name}")
        if limit_episodes is not None:
            split = {"train": split["train"][: limit_episodes - 1], "held_out": split["held_out"][:1]}
        selected = set(split["train"] + split["held_out"])
        episodes = [ep for ep in episodes if ep["episode_index"] in selected]
        source_files = []
        for ep in episodes:
            index = ep["episode_index"]
            path = source / info["data_path"].format(episode_index=index, episode_chunk=index // info["chunks_size"])
            stat = path.stat()
            source_files.append([index, stat.st_size, stat.st_mtime_ns])
            jobs.append((source, destination, name, info, ep, prompts, rungs, alignment))
        fingerprints[name] = hashlib.sha256(b"".join(metadata) + json.dumps(source_files).encode()).hexdigest()
        records[name] = {
            "source": str(source),
            "info": info,
            "full_split": full_split,
            "split": split,
            "excluded_episodes": excluded,
            "episodes": episodes,
        }
    settings = {
        "version": 1,
        "source_root": str(source_root),
        "tasks": list(tasks),
        "seed": seed,
        "horizon": horizon,
        "stats_samples_per_task": stats_samples,
        "limit_episodes": limit_episodes,
        "source_fingerprints": fingerprints,
        "pose_layout": "left_then_right_xyz_quat_xyzw_gripper_m",
        "pose_reference": (
            "own_base_zero_tcp_flange" if alignment is None else f"piper_flange_via_{alignment.calibration_id}"
        ),
    }
    if rungs != ("s2",) or alignment is not None:
        # Caches written before pose alignment existed keep their original key set,
        # so an unaligned S2 cache stays reusable without re-preparing it.
        settings["rungs"] = list(rungs)
        settings["alignment"] = None if alignment is None else alignment.to_dict()
    settings_path = destination / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Cache settings/source changed; use a new output directory")
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        if not settings_path.exists():
            raise ValueError("Cache manifest is missing settings")
        for rung in rungs:
            normalize.load(destination / "stats" / rung)
        return json.loads(manifest_path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(settings_path, settings)
    for name in tasks:
        (destination / "episodes" / name).mkdir(parents=True, exist_ok=True)
    if workers == 1:
        converted = list(map(_convert, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            converted = list(pool.map(_convert, jobs, chunksize=4))
    for job, episode in zip(jobs, converted, strict=True):
        name = job[2]
        records[name].setdefault("converted", []).append(episode)
    for task in records.values():
        task["episodes"] = task.pop("converted")
    manifest = {
        "settings": settings,
        "tasks": records,
        "target_source": "future_observation.qpos_ee",
        "sampling": "equal_tasks",
        "rung": rungs[0],
        "rungs": list(rungs),
        "right_base_to_left_base": g.RIGHT_BASE.tolist(),
        "tail_policy": "full_future_chunks_only",
        "smoke_only": limit_episodes is not None,
    }
    if "s0" in rungs:
        manifest["target_sources"] = {
            rung: "future_observation.state" if rung == "s0" else "future_observation.qpos_ee" for rung in rungs
        }
    _statistics(destination, manifest, stats_samples, seed)
    atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/xiaoyu_teleop_data/piper/latest"))
    parser.add_argument("--destination", type=Path, default=Path("data/piper_x_eef"))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats-samples", type=int, default=1024, help="Training anchors per task")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-episodes", type=int, help="Smoke cache: at most this many episodes per task")
    parser.add_argument("--rungs", nargs="+", choices=RUNGS, default=["s2"], help="Representations to cache")
    alignment_options = parser.add_mutually_exclusive_group()
    alignment_options.add_argument(
        "--align-to-piper",
        choices=("nominal",),
        help="Map recorded Piper-X flange poses to the Piper flange convention using the nominal tool axes",
    )
    alignment_options.add_argument(
        "--align-calibration",
        type=Path,
        help="JSON with per-arm A/C transforms using T_X = A @ T_P @ C, applied to every recorded pose",
    )
    args = vars(parser.parse_args())
    nominal, calibration = args.pop("align_to_piper"), args.pop("align_calibration")
    rungs = tuple(args.pop("rungs"))
    alignment = PiperXTransfer.nominal_axes() if nominal else None
    if calibration is not None:
        alignment = PiperXTransfer.from_file(calibration)
    manifest = prepare(**args, rungs=rungs, alignment=alignment)
    print(
        json.dumps(
            {
                "smoke_only": manifest["smoke_only"],
                "source_root": manifest["settings"]["source_root"],
                "rungs": list(manifest.get("rungs") or [manifest.get("rung", "s2")]),
                "pose_reference": manifest["settings"]["pose_reference"],
                "tasks": {
                    name: {key: len(task["split"][key]) for key in ("train", "held_out")}
                    for name, task in manifest["tasks"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
