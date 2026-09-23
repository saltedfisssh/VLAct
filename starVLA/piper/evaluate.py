"""Held-out physical-space errors over real observations through the policy server."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from starVLA.piper import geometry as g
from starVLA.piper.client import PiperWebsocketPolicy
from starVLA.piper.dataset import contract, raw_dataset


def physical_errors(state, predicted, target, rung):
    if rung == "s0":
        # Native X joints must never be run through Piper FK for evaluation.
        error = np.abs(predicted - target)
        return np.stack(
            [
                np.stack([np.rad2deg(error[:, start : start + 6]).mean(-1), error[:, start + 6] * 1000], -1)
                for start in (0, 7)
            ],
            1,
        )
    pred_pose, pred_grip = g.decode_actions(state, predicted, rung)
    true_pose, true_grip = g.decode_actions(state, target, rung)
    position = np.linalg.norm(pred_pose[..., :3, 3] - true_pose[..., :3, 3], axis=-1) * 1000
    relative = np.swapaxes(pred_pose[..., :3, :3], -1, -2) @ true_pose[..., :3, :3]
    angle = np.rad2deg(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()).reshape(-1, 2)
    return np.stack([position, angle, np.abs(pred_grip - true_grip) * 1000], -1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--rung", choices=("s0", "s2", "s2_pair"), required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--samples-per-task", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.samples_per_task < 1:
        p.error("samples-per-task must be positive")
    dataset = raw_dataset(args.cache, args.rung, split="held_out")
    expected = contract(args.cache, args.rung)
    policy = PiperWebsocketPolicy(args.host, args.port, timeout=300)
    try:
        metadata = policy.get_server_metadata()
        for key in ("piper_rung", "training_pose_frame", "alignment", "manifest_sha256"):
            if metadata.get(key) != expected[key]:
                raise ValueError(f"Evaluation dataset and policy disagree: {key}")
        rng = np.random.default_rng(args.seed)
        tasks = list(dataset.names) if hasattr(dataset, "names") else ["piper_fold_cloth_in_place"]
        ends = dataset.task_ends if hasattr(dataset, "task_ends") else np.array([len(dataset)])
        starts = np.r_[0, ends[:-1]]
        results = {}
        for task, start, end in zip(tasks, starts, ends, strict=False):
            errors, baseline, anchors = [], [], []
            for index in rng.integers(start, end, size=args.samples_per_task):
                sample = dataset[int(index)]
                prediction = policy.infer({key: sample[key] for key in ("state", "image", "prompt")})["actions"]
                if np.asarray(prediction).shape != sample["actions"].shape or not np.isfinite(prediction).all():
                    raise ValueError("Invalid policy actions")
                stationary = g.encode_actions(sample["state"], np.tile(sample["state"], (50, 1)), args.rung)
                errors.append(physical_errors(sample["state"], prediction, sample["actions"], args.rung))
                baseline.append(physical_errors(sample["state"], stationary, sample["actions"], args.rung))
                anchors.append([str(x) for x in dataset.locate(int(index))])

            def summary(values):
                mean = np.mean(values, axis=0)
                return {
                    "k10": mean[9].tolist(),
                    "k25": mean[24].tolist(),
                    "k50": mean[49].tolist(),
                    "chunk_mean": mean.mean(0).tolist(),
                }

            results[task] = {"errors": summary(errors), "stationary_errors": summary(baseline), "anchors": anchors}
        report = {
            "policy": metadata,
            "seed": args.seed,
            "split": "held_out",
            "tasks": results,
            "units_per_arm": ["mean_joint_deg", "gripper_mm"]
            if args.rung == "s0"
            else ["translation_mm", "rotation_deg", "gripper_mm"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote held-out metrics to {args.output}")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
