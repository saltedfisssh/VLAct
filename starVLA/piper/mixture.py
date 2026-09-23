"""Weighted mixtures of compatible Piper caches with joint training-only statistics."""

import hashlib
import json
from pathlib import Path

import numpy as np

from starVLA.piper import normalize
from starVLA.piper.prepare_piper import atomic_json


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_partitions(manifest):
    """Identify observations by resolved source directory and episode ID."""
    tasks = manifest.get("tasks") or {"piper": {"source": manifest["settings"]["source"], "split": manifest["split"]}}
    return {
        split: {
            (str(Path(task["source"]).resolve()), episode) for task in tasks.values() for episode in task["split"][split]
        }
        for split in ("train", "held_out")
    }


def validate_members(members, rung):
    from starVLA.piper.dataset import contract

    if not members or len({m["name"] for m in members}) != len(members):
        raise ValueError("Mixture needs distinct, nonempty member names")
    weights = np.asarray([m["weight"] for m in members], dtype=np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Mixture weights must be finite and positive")
    contracts, seen = [], {"train": set(), "held_out": set()}
    fields = (
        "state_dim",
        "action_dim",
        "action_horizon",
        "action_start_offset",
        "fps",
        "action_encoding",
        "training_pose_frame",
        "robot_model",
        "tcp_offset_m",
        "pair_base_spacing_m",
        "smoke_only",
    )
    for member in members:
        cache = Path(member["cache"])
        manifest = json.loads((cache / "manifest.json").read_text())
        if manifest.get("kind") == "mixture":
            raise ValueError("Nested mixtures are not supported; list the original caches")
        meta = contract(cache, rung)
        if contracts:
            mismatch = [key for key in fields if meta[key] != contracts[0][key]]
            if mismatch:
                raise ValueError(f"Incompatible mixture member {member['name']}: {mismatch}; align EEF frames first")
        partitions = episode_partitions(manifest)
        if partitions["train"] & seen["held_out"] or partitions["held_out"] & seen["train"]:
            raise ValueError("Mixture would leak episodes between train and held_out")
        if partitions["train"] & seen["train"] or partitions["held_out"] & seen["held_out"]:
            raise ValueError("Overlapping source episodes in mixture; adjust weights instead of duplicating a cache")
        for split in seen:
            seen[split].update(partitions[split])
        contracts.append(meta)
    return contracts


class MixtureDataset:
    """Virtual slots preserve requested weights and cover each member's anchors.

    DDP shuffles the virtual indices. Within a member each real anchor appears
    at least once, and repetition counts differ by at most one. Held-out data
    is concatenated exactly once without applying training weights.
    """

    def __init__(self, root, rung, *, split="train", images=True, manifest=None):
        from starVLA.piper.dataset import raw_dataset

        self.root = Path(root)
        self.manifest = manifest or json.loads((self.root / "manifest.json").read_text())
        if rung not in self.manifest["rungs"] or split not in ("train", "held_out"):
            raise ValueError("Unavailable mixture rung or split")
        self.members = self.manifest["members"]
        self.names = [member["name"] for member in self.members]
        for member in self.members:
            if fingerprint(Path(member["cache"]) / "manifest.json") != member["manifest_sha256"]:
                raise ValueError(f"Mixture source cache changed: {member['name']}")
        self.datasets = [raw_dataset(m["cache"], rung, split=split, images=images) for m in self.members]
        self.counts = np.array([len(dataset) for dataset in self.datasets], dtype=np.int64)
        weights = np.asarray([m["weight"] for m in self.members], dtype=np.float64)
        if self.manifest["settings"].get("balance_dataset_weights", False):
            weights *= self.counts
        probabilities = weights / weights.sum()
        if split == "train":
            slots = np.ceil(np.max(self.counts / probabilities) * probabilities)
            if not np.isfinite(slots).all() or slots.sum() > 100_000_000:
                raise ValueError("Mixture weights create over 100M virtual slots; use less extreme weights")
            self.slots = slots.astype(np.int64)
        else:
            self.slots = self.counts.copy()
        self.task_ends = np.cumsum(self.slots)
        self.starts = np.r_[0, self.task_ends[:-1]]

    def __len__(self):
        return int(self.task_ends[-1])

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        member = int(np.searchsorted(self.task_ends, index, side="right"))
        slot = int(index - self.starts[member])
        anchor = slot * int(self.counts[member]) // int(self.slots[member])
        return member, anchor

    def __getitem__(self, index):
        member, anchor = self.locate(index)
        return self.datasets[member][anchor]


def prepare(
    entries,
    destination,
    *,
    data_mix,
    balance_dataset_weights=False,
    rungs=("s2", "s2_pair"),
    stats_samples=65536,
    seed=0,
    smoke=False,
):
    from starVLA.piper.dataset import RUNGS

    destination = Path(destination).resolve()
    if not rungs or len(set(rungs)) != len(rungs) or any(r not in RUNGS for r in rungs) or stats_samples < 2:
        raise ValueError("Choose distinct supported rungs and at least two statistics samples")
    members = []
    for entry in entries:
        if set(entry) != {"name", "cache", "weight"} or not isinstance(entry["name"], str) or not entry["name"].strip():
            raise ValueError("Each member requires name, cache and weight")
        cache = Path(entry["cache"]).resolve(strict=True)
        if cache == destination or cache in destination.parents or destination in cache.parents:
            raise ValueError("Mixture output and member caches must be separate trees")
        source_manifest = json.loads((cache / "manifest.json").read_text())
        if source_manifest.get("kind") == "mixture":
            raise ValueError("Nested mixtures are not supported")
        for source, _ in episode_partitions(source_manifest)["train"]:
            path = Path(source)
            if path == destination or path in destination.parents or destination in path.parents:
                raise ValueError("Mixture output must be separate from read-only source data")
        members.append(
            {
                "name": entry["name"],
                "cache": str(cache),
                "weight": float(entry["weight"]),
                "manifest_sha256": fingerprint(cache / "manifest.json"),
            }
        )
    contracts = {rung: validate_members(members, rung) for rung in rungs}
    if any(meta["smoke_only"] != smoke for metas in contracts.values() for meta in metas):
        raise ValueError("Smoke/full mode must match every member cache")
    settings = {
        "version": 1,
        "members": members,
        "rungs": list(rungs),
        "stats_samples": stats_samples,
        "seed": seed,
        "data_mix": data_mix,
        "balance_dataset_weights": bool(balance_dataset_weights),
        "smoke_only": smoke,
    }
    settings_path = destination / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Mixture settings changed; use a new output directory")
    if (destination / "manifest.json").exists():
        if not settings_path.exists():
            raise ValueError("Mixture settings file is missing")
        for rung in rungs:
            normalize.load(destination / "stats" / rung)
        return json.loads((destination / "manifest.json").read_text())
    manifest = {
        "kind": "mixture",
        "settings": settings,
        "members": members,
        "contracts": contracts,
        "rungs": list(rungs),
        "smoke_only": smoke,
        "sampling": "weighted_virtual_slots",
        "tail_policy": "full_future_chunks_only",
    }
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(settings_path, settings)
    for rung in rungs:
        dataset = MixtureDataset(destination, rung, images=False, manifest=manifest)
        rng = np.random.default_rng(seed)
        # Identical sampled virtual indices across rungs, matching the actual loader.
        indices = np.sort(rng.integers(len(dataset), size=stats_samples))
        state, actions = [], []
        for index in indices:
            sample = dataset[int(index)]
            state.append(sample["state"])
            actions.append(sample["actions"])
        stats = {}
        for key, values in (("state", state), ("actions", actions)):
            values = np.asarray(values, dtype=np.float64).reshape(-1, values[0].shape[-1])
            lo, hi = np.quantile(values, [0.01, 0.99], axis=0)
            stats[key] = normalize.NormStats(values.mean(0), values.std(0), lo, hi)
        normalize.save(destination / "stats" / rung, stats)
        manifest["virtual_slots"] = dict(zip(dataset.names, dataset.slots.tolist(), strict=False))
        manifest["effective_probabilities"] = dict(
            zip(dataset.names, (dataset.slots / len(dataset)).tolist(), strict=False)
        )
        manifest["stats_sample_counts"] = {
            name: int(np.sum((indices >= start) & (indices < end)))
            for name, start, end in zip(dataset.names, dataset.starts, dataset.task_ends, strict=False)
        }
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def mixture_contract(manifest, rung, image_size):
    metas = manifest["contracts"][rung]
    result = dict(metas[0])
    sources = sorted({meta["training_state_source"] for meta in metas})
    alignments = [meta["alignment"] for meta in metas if meta["alignment"] is not None]
    if len(sources) > 1:
        result["training_state_source"] = "mixed"
        result["target_source"] = "mixed_future_observations"
    result.update(
        image_size=list(image_size),
        source_robot_model=sorted({m["source_robot_model"] for m in metas}),
        pose_alignment="mixture_aligned" if alignments else None,
        alignment=alignments or None,
        mixture=manifest["members"],
        effective_probabilities=manifest["effective_probabilities"],
    )
    return result
