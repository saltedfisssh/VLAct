"""QwenPI Piper training with deterministic DDP sampling and resumable checkpoints."""

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler

from starVLA.piper.dataset import RUNGS, TrainingDataset, collate, contract
from starVLA.piper.prepare_piper import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--rung", choices=RUNGS, required=True)
    p.add_argument("--data-mix")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", default="examples/Piper/qwenpi.yaml")
    p.add_argument("--vlm")
    p.add_argument("--expert")
    p.add_argument("--pretrained", type=Path)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=5e-6)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--weights-only", action="store_true", help="Smoke-only: omit optimizer/RNG state")
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.accumulation, args.save_every, args.log_every) < 1:
        p.error("steps, batches, accumulation, save/log intervals must be positive")
    if args.workers < 0 or args.lr <= 0 or not 0 <= args.min_lr <= args.lr or args.warmup < 0:
        p.error("Invalid workers or learning-rate schedule")
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.accumulation,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError("Training requires CUDA")
    if args.weights_only and not args.smoke:
        raise ValueError("Full training must save optimizer and RNG state")
    if not args.smoke and args.pretrained is None and args.resume is None:
        raise ValueError("Full training requires --pretrained VLAct_Qwen3_Pretrain checkpoint")
    if args.pretrained is not None and not args.pretrained.is_file():
        raise FileNotFoundError(f"Pretrained weights are not complete: {args.pretrained}")
    set_seed(args.seed)
    cfg = OmegaConf.load(args.config)
    if args.vlm:
        cfg.framework.qwenvl.base_vlm = args.vlm
    if args.expert:
        cfg.framework.action_model.expert_model_path = args.expert
    meta = contract(args.cache, args.rung, cfg.datasets.vla_data.image_size)
    if meta["smoke_only"] != args.smoke:
        raise ValueError("Use --smoke only with a smoke cache; full training requires full statistics")
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if args.data_mix:
        cfg.datasets.vla_data.dataset_py = "piper_datasets"
        cfg.datasets.vla_data.data_mix = args.data_mix
        cfg.datasets.vla_data.cache_dir = str(args.cache.resolve())
        cfg.datasets.vla_data.representation = args.rung
        cfg.datasets.vla_data.balance_dataset_weights = manifest["settings"]["balance_dataset_weights"]
    cfg.framework.action_model.action_dim = meta["action_dim"]
    cfg.framework.action_model.state_dim = meta["state_dim"]
    # Resolve local model paths once, before saving the portable run config.
    for node, key in ((cfg.framework.qwenvl, "base_vlm"), (cfg.framework.action_model, "expert_model_path")):
        path = Path(node[key])
        if path.exists():
            node[key] = str(path.resolve())
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("resume", "output")}
    settings["world_size"] = accelerator.num_processes
    settings["global_batch_size"] = args.batch_size * args.accumulation * accelerator.num_processes
    settings["config"] = OmegaConf.to_container(cfg, resolve=True)
    stats_path = args.cache / "stats" / args.rung / "norm_stats.json"
    meta["statistics_sha256"] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    if args.resume:
        if args.weights_only or not (args.resume / "optimizer.bin").is_file():
            raise ValueError("Cannot resume a weights-only smoke checkpoint")
        if not (args.resume / "complete.json").is_file():
            raise ValueError("Only complete checkpoints can be resumed")
        if json.loads((args.resume / "contract.json").read_text()) != meta:
            raise ValueError("Resume cache, representation, calibration or statistics changed")
        if json.loads((args.resume / "training.json").read_text()) != settings:
            raise ValueError("Resume training settings changed; use the original command")
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite run {args.output}; use a new name or --resume")
    if accelerator.is_main_process:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "contract.json", meta)
        atomic_json(args.output / "training.json", settings)
        OmegaConf.save(cfg, args.output / "config.yaml")
        shutil.copyfile(stats_path, args.output / "norm_stats.json")
        shutil.copyfile(args.cache / "manifest.json", args.output / "data_manifest.json")
    accelerator.wait_for_everyone()

    from starVLA.model.framework.QwenPI_v4 import Qwen_PI_v4

    model = Qwen_PI_v4(cfg).to(dtype=torch.bfloat16)
    if args.pretrained and not args.resume:
        from starVLA.piper.pretrained import load_backbone

        initialization = load_backbone(model, args.pretrained)
    elif args.resume:
        initialization = json.loads((args.resume / "initialization.json").read_text())
    else:
        initialization = {"source": "base Qwen3 models; pipeline smoke only"}
    if accelerator.is_main_process:
        atomic_json(args.output / "initialization.json", initialization)
    for name, param in model.named_parameters():
        if name.startswith("qwen_vl_interface.model.model.visual.") or ".lm_head." in name:
            param.requires_grad_(False)
    model.qwen_vl_interface.model.config.use_cache = False
    trainable = [param for param in model.parameters() if param.requires_grad]
    accelerator.print(
        json.dumps(
            {
                "trainable_parameters": sum(p.numel() for p in trainable),
                **meta,
                "global_batch_size": settings["global_batch_size"],
            }
        )
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)

    def schedule(step):
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        phase = min(1, (step - args.warmup) / max(1, args.steps - args.warmup))
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * phase))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(scheduler)
    if args.data_mix:
        from starVLA.dataloader.piper_datasets import get_vla_dataset

        dataset = get_vla_dataset(cfg.datasets.vla_data)
    else:
        dataset = TrainingDataset(args.cache, args.rung, cfg.datasets.vla_data.image_size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed + accelerator.process_index),
    )
    if not len(loader):
        raise ValueError("Dataset is smaller than one per-rank batch")
    set_seed(args.seed + accelerator.process_index)
    step, epoch, batch_offset = 0, 0, 0
    if args.resume:
        accelerator.load_state(str(args.resume))
        progress = json.loads((args.resume / "complete.json").read_text())
        step, epoch, batch_offset = progress["step"], progress["epoch"], progress["batch_offset"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    started = time.monotonic()
    while step < args.steps:
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if batch_index < batch_offset:
                continue
            with accelerator.accumulate(model):
                output = model(batch)
                loss = output["action_loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                accumulated_loss += float(loss.detach()) / args.accumulation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(trainable, 1.0)
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError(f"Nonfinite gradient at step {step}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            scheduler.step()
            step += 1
            if step % args.log_every == 0 or step == args.steps:
                mean_loss = accelerator.reduce(torch.tensor(accumulated_loss, device=accelerator.device), "mean")
                record = {
                    "step": step,
                    "loss": float(mean_loss),
                    "grad_norm": float(grad_norm),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
                if accelerator.is_main_process:
                    with (args.output / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
            accumulated_loss = 0.0
            if step % args.save_every == 0 or step == args.steps:
                target = args.output / f"step_{step:06d}"
                pending = args.output / f".step_{step:06d}.incomplete"
                if target.exists() or pending.exists():
                    raise FileExistsError(f"Checkpoint path already exists: {target} / {pending}")
                accelerator.wait_for_everyone()
                if args.weights_only:
                    accelerator.save_model(model, str(pending), max_shard_size="100GB", safe_serialization=True)
                else:
                    accelerator.save_state(str(pending), safe_serialization=True)
                # Every rank must finish its RNG file before rank zero publishes the directory.
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    for name in (
                        "contract.json",
                        "config.yaml",
                        "training.json",
                        "norm_stats.json",
                        "data_manifest.json",
                        "initialization.json",
                    ):
                        shutil.copyfile(args.output / name, pending / name)
                    atomic_json(
                        pending / "complete.json", {"step": step, "epoch": epoch, "batch_offset": batch_index + 1}
                    )
                    pending.rename(target)
                    print(f"Saved complete checkpoint: {target}", flush=True)
                accelerator.wait_for_everyone()
            if step >= args.steps:
                break
        epoch += 1
        batch_offset = 0
    accelerator.end_training()


if __name__ == "__main__":
    main()
