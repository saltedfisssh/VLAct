"""Offline action validation conditioned on held-out subtask labels; never a rollout score."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


class ActionErrors:
    def __init__(self):
        self.absolute_sum = np.zeros(14, dtype=np.float64)
        self.valid_steps = 0
        self.frames = 0

    def add(self, prediction, target, mask):
        prediction, target = np.asarray(prediction), np.asarray(target)
        mask = np.asarray(mask, dtype=bool)
        if prediction.shape != target.shape or prediction.shape != (len(mask), 14) or not mask.any():
            raise ValueError("Invalid action shape or empty atomic segment")
        if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
            raise ValueError("Non-finite action prediction/target")
        self.absolute_sum += np.abs(prediction[mask].astype(np.float64) - target[mask]).sum(0)
        self.valid_steps += int(mask.sum())
        self.frames += 1

    def report(self):
        if not self.valid_steps:
            raise ValueError("No evaluated action steps")
        dimension_errors = self.absolute_sum / self.valid_steps
        return {"frames": self.frames, "valid_action_steps": self.valid_steps,
                "normalized_l1": float(dimension_errors.mean()),
                "normalized_l1_per_dimension": dimension_errors.tolist()}


def check_split(manifest, training_manifest):
    train = {e.get("original_episode_index", e["source_episode_index"])
             for e in training_manifest["episodes"] if e["split"] == "train"}
    validation = {e.get("original_episode_index", e["source_episode_index"])
                  for e in manifest["episodes"] if e["split"] == "val"}
    if train & validation:
        raise ValueError("Validation episodes overlap checkpoint training episodes")
    if manifest["spec"]["source_sha256"] != training_manifest["spec"]["source_sha256"]:
        raise ValueError("Validation cache differs from checkpoint data; use a separately audited evaluation protocol")
    if manifest["spec"]["language_mode"] != "subtask":
        raise ValueError("Expected an atomic-subtask cache")


def selected_indices(dataset, stride, max_episodes, max_frames):
    if stride < 1 or max_episodes < 0 or max_frames < 0:
        raise ValueError("stride must be positive; limits must be nonnegative")
    indices, base = [], 0
    for episode in dataset.episodes[:max_episodes or None]:
        frames = list(range(0, episode["length"], stride))
        if max_frames:
            frames = frames[:max_frames]
        indices.extend(base + frame for frame in frames)
        base += episode["length"]
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-override")
    parser.add_argument("--base-vlm")
    parser.add_argument("--stride", type=int, default=25,
                        help="Evaluate one observation every N frames; 1 evaluates every held-out frame")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-frames-per-episode", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from examples.RoboDojo.data import RoboDojoDataset, collate
    from examples.RoboDojo.prompt_audit import PromptAudit
    from starVLA.model.framework import build_framework

    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation report: {args.output}")
    run = args.checkpoint.resolve().parents[1]
    cfg = OmegaConf.load(run / "config.yaml")
    cfg.trainer.enable_gradient_checkpointing = False
    cfg.datasets.vqa_data.enabled = False
    if cfg.datasets.vla_data.include_state:
        raise ValueError("This evaluation recipe has no proprioception input")
    if args.base_vlm:
        cfg.framework.qwenvl.base_vlm = args.base_vlm
    dataset = RoboDojoDataset(args.cache, split="val", source_override=args.source_override)
    training_manifest = json.loads((run / "data_manifest.json").read_text())
    check_split(dataset.manifest, training_manifest)
    indices = selected_indices(dataset, args.stride, args.max_episodes, args.max_frames_per_episode)
    if not indices:
        raise ValueError("Empty validation selection")
    cached_stats = json.loads((args.cache / "dataset_statistics.json").read_text())
    if cached_stats != json.loads((run / "dataset_statistics.json").read_text()):
        raise ValueError("Cache normalization statistics differ from checkpoint")
    model = build_framework(cfg)
    weights = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model.eval().cuda()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = PromptAudit(model.qwen_vl_interface.processor.tokenizer, args.output.with_suffix(".inputs.jsonl"))
    model.qwen_vl_interface.register_forward_pre_hook(audit, with_kwargs=True)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        num_workers=0, collate_fn=collate, shuffle=False)
    total, by_task = ActionErrors(), defaultdict(ActionErrors)
    episodes = set()
    for examples in tqdm(loader, desc="Held-out subtask-conditioned actions", unit="batch"):
        if audit.batches < 2:
            audit.arm(examples, phase="offline_validation")
        with torch.inference_mode():
            result = model.predict_action(examples=[{"image": e["image"], "lang": e["lang"]} for e in examples])
        predictions = np.asarray(result["normalized_actions"], dtype=np.float32)
        if predictions.shape != (len(examples), 50, 14):
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
        for prediction, example in zip(predictions, examples, strict=True):
            total.add(prediction, example["action"], example["action_valid_mask"])
            by_task[str(example["task_index"])].add(prediction, example["action"], example["action_valid_mask"])
            episodes.add(example["original_episode_index"])
    per_task = {key: value.report() for key, value in sorted(by_task.items())}
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "cache": str(args.cache.resolve()),
        "conditioning": "reviewed_subtask_labels" if dataset.manifest["spec"]["annotations_reviewed"]
                        else "unreviewed_pseudo_subtask_labels",
        "simulator_evaluation": False, "success_rate": None,
        "split": "validation", "stride": args.stride,
        "max_episodes": args.max_episodes, "max_frames_per_episode": args.max_frames_per_episode,
        "complete_held_out_frames": args.stride == 1 and args.max_episodes == 0 and args.max_frames_per_episode == 0,
        "episodes": len(episodes), "tasks": len(per_task), "micro": total.report(),
        "task_macro_normalized_l1": float(np.mean([v["normalized_l1"] for v in per_task.values()])),
        "per_task": per_task, "strict_load": True,
        "action_mask": "stop at current atomic-subtask and episode boundaries",
        "normalization": "checkpoint train-only q01/q99; target clipped to [-1,1]",
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
