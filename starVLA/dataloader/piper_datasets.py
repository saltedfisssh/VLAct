"""Registered Piper datasets with future-state targets and read-only source videos."""

import argparse
import json
import math
from pathlib import Path

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.piper import mixture
from starVLA.piper.data_x import prepare as prepare_x
from starVLA.piper.dataset import TrainingDataset, contract
from starVLA.piper.prepare_piper import prepare as prepare_piper
from starVLA.piper.transfer import PiperXTransfer

PIPER_ROBOT_CONFIGS = {
    "piper": {"rungs": ("s0", "s2", "s2_pair"), "alignment": None},
    "piper_x": {"rungs": ("s0", "s2", "s2_pair"), "alignment": None},
    "piper_x_aligned": {"rungs": ("s2", "s2_pair"), "alignment": "nominal"},
}


def resolve_mixture(data_mix):
    if data_mix not in DATASET_NAMED_MIXTURES:
        raise ValueError(f"Unknown registered data_mix: {data_mix}")
    entries = DATASET_NAMED_MIXTURES[data_mix]
    seen = set()
    for name, weight, robot_type in entries:
        if not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError(f"{name}: sampling weight must be positive and finite")
        if robot_type not in PIPER_ROBOT_CONFIGS:
            raise ValueError(f"{name}: robot type {robot_type!r} is not a Piper data configuration")
        if not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid dataset name: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate dataset in {data_mix}: {name}")
        seen.add(name)
    if not entries:
        raise ValueError("The registered mixture is empty")
    return entries


def prepare(
    data_root_dir,
    cache_root,
    data_mix,
    *,
    smoke=False,
    workers=4,
    stats_samples=8192,
    balance_dataset_weights=False,
    alignment="nominal",
):
    """Prepare each registered task once, then shared statistics for the mixture."""
    root, cache_root = Path(data_root_dir).resolve(strict=True), Path(cache_root).resolve()
    if root == cache_root or root in cache_root.parents or cache_root in root.parents:
        raise ValueError("Source and cache roots must be separate trees")
    entries = resolve_mixture(data_mix)
    rungs = tuple(
        r for r in ("s0", "s2", "s2_pair") if all(r in PIPER_ROBOT_CONFIGS[robot]["rungs"] for _, _, robot in entries)
    )
    if len({robot == "piper_x" for _, _, robot in entries}) > 1:
        raise ValueError("Native Piper-X cannot share a policy frame with Piper; register aligned X data instead")
    members = []
    for name, weight, robot in entries:
        cache = cache_root / "datasets" / robot / name
        options = dict(workers=workers, stats_samples=stats_samples, limit_episodes=3 if smoke else None)
        if robot == "piper":
            prepare_piper(root / name, cache, **options)
        else:
            transfer = None
            if robot == "piper_x_aligned":
                transfer = (
                    PiperXTransfer.nominal_axes() if alignment == "nominal" else PiperXTransfer.from_file(alignment)
                )
            prepare_x(
                root, cache, tasks=(name,), rungs=PIPER_ROBOT_CONFIGS[robot]["rungs"], alignment=transfer, **options
            )
        members.append({"name": name, "cache": str(cache), "weight": weight})
    destination = cache_root / "mixtures" / data_mix
    manifest = mixture.prepare(
        members,
        destination,
        data_mix=data_mix,
        balance_dataset_weights=balance_dataset_weights,
        rungs=rungs,
        stats_samples=stats_samples * len(members),
        smoke=smoke,
    )
    return destination, manifest


def get_vla_dataset(data_cfg, mode="train"):
    """Use the normal data_mix config while retaining the audited Piper labels."""
    cache = Path(data_cfg.cache_dir)
    manifest = json.loads((cache / "manifest.json").read_text())
    expected = [(name, float(weight)) for name, weight, _ in resolve_mixture(data_cfg.data_mix)]
    actual = [(member["name"], float(member["weight"])) for member in manifest["members"]]
    if expected != actual or manifest["settings"]["data_mix"] != data_cfg.data_mix:
        raise ValueError("Cache and registered data_mix weights differ; prepare a new cache")
    for (_, _, robot), member in zip(resolve_mixture(data_cfg.data_mix), manifest["members"], strict=True):
        meta = contract(member["cache"], data_cfg.representation)
        frame = "piper_x_flange" if robot == "piper_x" else "piper_flange"
        source_robot = "piper" if robot == "piper" else "piper_x"
        if meta["training_pose_frame"] != frame or meta["source_robot_model"] != source_robot:
            raise ValueError("Cache and registered robot_type transforms differ")
    if bool(data_cfg.get("balance_dataset_weights", False)) != manifest["settings"]["balance_dataset_weights"]:
        raise ValueError("Cache and balance_dataset_weights differ")
    return TrainingDataset(
        cache,
        data_cfg.representation,
        data_cfg.get("image_size", (224, 224)),
        split="train" if mode == "train" else "held_out",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root-dir", default="/mnt/xiaoyu_teleop_data/piper/latest")
    p.add_argument("--cache-root", required=True)
    p.add_argument("--data-mix", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--stats-samples", type=int, default=8192, help="Statistics anchors per registered dataset")
    p.add_argument("--balance-dataset-weights", action="store_true")
    p.add_argument("--alignment", default="nominal")
    path, manifest = prepare(**vars(p.parse_args()))
    print(
        json.dumps(
            {
                "cache": str(path),
                "data_mix": manifest["settings"]["data_mix"],
                "rungs": manifest["rungs"],
                "dataset_weights": {m["name"]: m["weight"] for m in manifest["members"]},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
