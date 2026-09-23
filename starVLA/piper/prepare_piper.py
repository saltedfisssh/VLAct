# Adapted from kai0 (Apache-2.0); see NOTICE.md and LICENSE.kai0.
"""Prepare numeric episode caches and training-only representation statistics."""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from starVLA.piper import geometry as g
from starVLA.piper import normalize

CAMERAS = {
    "base_0_rgb": "observation.images.top_head",
    "left_wrist_0_rgb": "observation.images.hand_left",
    "right_wrist_0_rgb": "observation.images.hand_right",
}
FORMAT_VERSION = 1
RUNGS = ("s0", "s2", "s2_pair")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def make_split(ids, seed):
    order = np.random.default_rng(seed).permutation(ids).tolist()
    count = max(1, round(len(order) * 0.05))
    return {"seed": seed, "held_out_fraction": 0.05, "train": sorted(order[count:]), "held_out": sorted(order[:count])}


def validate_split(split, ids):
    train, held = split["train"], split["held_out"]
    if not train or not held or len(set(train + held)) != len(train + held) or set(train + held) != set(ids):
        raise ValueError("Split must partition every source episode exactly once with nonempty train/held_out sets")


def _convert(job):
    source, destination, info, episode, seed, xy_range, check_videos = job
    index = episode["episode_index"]
    fields = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
    path = source / info["data_path"].format(**fields)
    table = pq.read_table(path, columns=["observation.state", "action", "timestamp"]).to_pydict()
    joints, actions = np.asarray(table["observation.state"]), np.asarray(table["action"])
    if joints.shape != (episode["length"], 14) or not np.isfinite(joints).all() or not np.isfinite(actions).all():
        raise ValueError(f"Invalid state/action values in episode {index}")
    error = float(np.max(np.abs(actions[:-1] - joints[1:])))
    mismatch_fraction = float(np.mean(np.any(np.abs(actions[:-1] - joints[1:]) > 1e-6, axis=-1)))
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    if not np.allclose(np.diff(timestamps), 1 / info["fps"], atol=1e-4):
        raise ValueError(f"Episode {index} does not have uniform timestamps at the declared FPS")
    for key in CAMERAS.values():
        video = source / info["video_path"].format(**fields, video_key=key)
        if check_videos and not video.is_file():
            raise FileNotFoundError(video)
    transform = g.episode_frame(index, seed, xy_range)
    payload = {rung: g.build_states(joints, rung, transform).astype(np.float32) for rung in RUNGS}
    payload.update(joints=joints.astype(np.float32), timestamps=timestamps, episode_transform=transform)
    path_out = destination / "episodes" / f"episode_{index:06d}.npz"
    temporary = path_out.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path_out)
    return {
        **episode,
        "action_shift_max_error": error,
        "action_shift_mismatch_fraction": mismatch_fraction,
        "source_size": path.stat().st_size,
    }


def _statistics(destination, episodes, split, horizon, samples, seed):
    lengths = {ep["episode_index"]: ep["length"] - horizon for ep in episodes}
    ids = split["train"]
    ends = np.cumsum([lengths[i] for i in ids])
    starts = np.r_[0, ends[:-1]]
    # Uniform over eligible training anchors, as in the training sampler. All rungs
    # use the exact same anchors, including all k=1..horizon targets for each.
    anchors = np.random.default_rng(seed).integers(0, ends[-1], size=samples)
    owners = np.searchsorted(ends, anchors, side="right")
    state_samples = {r: [] for r in RUNGS}
    action_samples = {r: [] for r in RUNGS}
    used = []
    for ordinal in np.unique(owners):
        episode_id = ids[ordinal]
        frames = anchors[owners == ordinal] - starts[ordinal]
        used.append(episode_id)
        with np.load(destination / "episodes" / f"episode_{episode_id:06d}.npz") as data:
            future_index = frames[:, None] + np.arange(1, horizon + 1)
            for rung in RUNGS:
                state = data[rung]
                state_samples[rung].append(state[frames])
                action_samples[rung].append(g.encode_actions(state[frames, None], state[future_index], rung))
    for rung in RUNGS:
        stats = {}
        for name, chunks in (("state", state_samples[rung]), ("actions", action_samples[rung])):
            values = np.concatenate(chunks).reshape(-1, chunks[0].shape[-1]).astype(np.float64)
            low, high = np.quantile(values, [0.01, 0.99], axis=0)
            stats[name] = normalize.NormStats(mean=values.mean(0), std=values.std(0), q01=low, q99=high)
        normalize.save(destination / "stats" / rung, stats)
    return used


def prepare(
    source,
    destination,
    *,
    split_path=None,
    seed=0,
    xy_range=0.5,
    horizon=50,
    stats_samples=8192,
    workers=1,
    limit_episodes=None,
    check_videos=True,
):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and output must be separate directory trees")
    if horizon < 1 or stats_samples < 2 or workers < 1:
        raise ValueError("Positive horizon/workers and at least two statistics samples are required")
    g.episode_frame(0, seed, xy_range)
    info_bytes = (source / "meta/info.json").read_bytes()
    episodes_bytes = (source / "meta/episodes.jsonl").read_bytes()
    info = json.loads(info_bytes)
    if info["codebase_version"] != "v2.1" or info["fps"] != 30:
        raise ValueError("Expected LeRobot v2.1 data at 30 FPS")
    episodes = [json.loads(line) for line in episodes_bytes.splitlines()]
    ids = [ep["episode_index"] for ep in episodes]
    if len(set(ids)) != len(ids) or len(ids) != info["total_episodes"]:
        raise ValueError("Episode manifest does not match total_episodes")
    split = json.loads(Path(split_path).read_text()) if split_path else make_split(ids, seed)
    validate_split(split, ids)
    full_split = split
    excluded = [
        {"episode_index": ep["episode_index"], "length": ep["length"], "reason": "no_full_future_action_horizon"}
        for ep in episodes
        if ep["length"] <= horizon
    ]
    if excluded:
        logging.warning(
            "Excluding %d episodes with no full future action horizon (length <= horizon=%d): %s",
            len(excluded),
            horizon,
            [ep["episode_index"] for ep in excluded],
        )
    episodes = [ep for ep in episodes if ep["length"] > horizon]
    eligible_ids = {ep["episode_index"] for ep in episodes}
    # Preserve original membership and ordering; filter before choosing smoke episodes.
    split = {**full_split}
    for name in ("train", "held_out"):
        split[name] = [i for i in full_split[name] if i in eligible_ids]
        if not split[name]:
            raise ValueError(f"No eligible episodes in {name}: full future chunks require length > horizon={horizon}")
    if limit_episodes is not None:
        if limit_episodes < 2:
            raise ValueError("Smoke caches need at least one training and one held-out episode")
        selected = set(split["train"][: limit_episodes - 1] + split["held_out"][:1])
        episodes = [ep for ep in episodes if ep["episode_index"] in selected]
        split = {
            **split,
            "train": [i for i in split["train"] if i in selected],
            "held_out": [i for i in split["held_out"] if i in selected],
        }
    settings = {
        "version": FORMAT_VERSION,
        "source": str(source),
        "seed": seed,
        "xy_range": xy_range,
        "horizon": horizon,
        "stats_samples": stats_samples,
        "limit_episodes": limit_episodes,
        "source_metadata_sha256": hashlib.sha256(info_bytes + episodes_bytes).hexdigest(),
        "full_split": full_split,
        "check_videos": check_videos,
    }
    settings_path = destination / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Cache settings/manifest changed; use a new output directory")
    if (destination / "manifest.json").exists():
        if not settings_path.exists():
            raise ValueError("Cache manifest is missing its settings")
        return json.loads((destination / "manifest.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "episodes").mkdir(exist_ok=True)
    atomic_json(settings_path, settings)
    atomic_json(destination / "split.json", full_split)
    jobs = [(source, destination, info, ep, seed, xy_range, check_videos) for ep in episodes]
    if workers == 1:
        records = list(map(_convert, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_convert, jobs, chunksize=4))
    used = _statistics(destination, records, split, horizon, stats_samples, seed)
    manifest = {
        "settings": settings,
        "info": info,
        "episodes": records,
        "excluded_episodes": excluded,
        "split": split,
        "stats_episode_ids": used,
        "rungs": list(RUNGS),
        "smoke_only": limit_episodes is not None,
        "right_base_to_left_base": g.RIGHT_BASE.tolist(),
        "tail_policy": "full_future_chunks_only",
        "target_source": "future_observation.state",
    }
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("data/piper_ablation"))
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--xy-range", type=float, default=0.5)
    parser.add_argument("--stats-samples", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-episodes", type=int)
    manifest = prepare(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "episodes": len(manifest["episodes"]),
                "excluded_episodes": len(manifest.get("excluded_episodes", [])),
                "split_counts": {k: len(manifest["split"][k]) for k in ("train", "held_out")},
                "smoke_only": manifest["smoke_only"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
