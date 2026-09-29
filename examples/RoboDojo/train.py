"""Full VLAct OFT finetuning, using Accelerate + DeepSpeed ZeRO-2 on 1 or 8 GPUs."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from itertools import islice
from pathlib import Path

import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import GradientAccumulationPlugin, broadcast_object_list, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import get_scheduler

from examples.RoboDojo.data import RoboDojoDataset, collate, write_json
from examples.RoboDojo.prompt_audit import PromptAudit
from examples.RoboDojo.validation import balanced_validation_subset
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups


def evaluate(model, loader, accelerator, batches, vqa=False):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for examples in islice(loader, batches):
            loss = (model(vqa_examples=examples)["vqa_loss"] if vqa else model(examples)["action_loss"]).float()
            total += loss * len(examples)
            count += len(examples)
    model.train()
    values = accelerator.reduce(torch.tensor([float(total), count], device=accelerator.device), reduction="sum")
    return float(values[0] / values[1]) if values[1] else None


def save_weights(model, accelerator, output, step):
    # ZeRO-2 keeps model weights replicated. Call get_state_dict on all ranks.
    state = accelerator.get_state_dict(model)
    target = output / "checkpoints" / f"steps_{step}_pytorch_model.pt"
    if accelerator.is_main_process:
        temporary = target.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(target)
    accelerator.wait_for_everyone()
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/RoboDojo/train.yaml")
    parser.add_argument("--resume", help="Accelerate training_state directory, including optimizer and RNG")
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    batch = int(cfg.datasets.vla_data.per_device_batch_size)
    accumulation = int(cfg.trainer.gradient_accumulation_steps)
    ds_config = {"bf16": {"enabled": True}, "fp16": {"enabled": False},
                 "train_micro_batch_size_per_gpu": batch, "gradient_accumulation_steps": accumulation,
                 "gradient_clipping": float(cfg.trainer.gradient_clipping), "steps_per_print": 1000000,
                 "zero_optimization": {"stage": 2, "overlap_comm": True, "contiguous_gradients": True,
                                       "reduce_bucket_size": 50000000, "allgather_bucket_size": 50000000}}
    if cfg.trainer.get("optimizer_offload", False):
        ds_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        ds_config["zero_force_ds_cpu_optimizer"] = False
    # ZeRO-2 partitions gradients on every microbatch; its no_sync() is unsupported.
    grad_plugin = GradientAccumulationPlugin(num_steps=accumulation, sync_each_batch=True, sync_with_dataloader=False)
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_plugin=grad_plugin,
                              deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds_config))
    accelerator.dataloader_config.dispatch_batches = False
    accelerator.dataloader_config.use_seedable_sampler = True
    accelerator.dataloader_config.data_seed = int(cfg.seed)
    set_seed(int(cfg.seed))
    output = Path(cfg.run_root_dir) / cfg.run_id
    # Rank 0 alone checks the path before creation; otherwise other ranks can race mkdir.
    error = [f"Use a fresh run_id or --resume; refusing to overwrite {output}"
             if accelerator.is_main_process and output.exists() and not args.resume else None]
    broadcast_object_list(error)
    if error[0]:
        raise FileExistsError(error[0])
    if accelerator.is_main_process:
        (output / "checkpoints").mkdir(parents=True, exist_ok=bool(args.resume))
    accelerator.wait_for_everyone()
    data_cfg = cfg.datasets.vla_data
    data = RoboDojoDataset(data_cfg.cache_dir, source_override=data_cfg.source_override)
    val_data = RoboDojoDataset(data_cfg.cache_dir, split="val", source_override=data_cfg.source_override)
    language_mode = data.manifest["spec"]["language_mode"]
    expected = data_cfg.get("expected_language_mode", language_mode)
    if language_mode != expected:
        raise ValueError(f"Expected {expected} conditioning, got a {language_mode} cache")
    if data.manifest["spec"].get("smoke_only", False) and int(cfg.trainer.max_train_steps) > 10:
        raise ValueError("This cache contains unreviewed labels and is marked smoke_only (at most 10 optimizer steps)")
    cfg.datasets.vla_data.language_mode = language_mode
    if len(data) < batch * accelerator.num_processes:
        raise ValueError("Training split is smaller than one global microbatch")
    loader = DataLoader(data, batch_size=batch, shuffle=True, collate_fn=collate,
                        num_workers=int(data_cfg.num_workers), pin_memory=False, drop_last=True,
                        persistent_workers=int(data_cfg.num_workers) > 0)
    val_subset, action_validation = balanced_validation_subset(
        val_data, int(cfg.trainer.eval_batches) * batch * accelerator.num_processes, int(cfg.seed))
    val_loader = DataLoader(val_subset, batch_size=batch, shuffle=False, collate_fn=collate,
                            num_workers=0) if len(val_subset) else None
    vqa_validation = None
    vqa_enabled = bool(cfg.datasets.get("vqa_data", {}).get("enabled", False))
    vqa_loader = vqa_val_loader = None
    if vqa_enabled:
        from examples.RoboDojo.vqa import AnnotationVQADataset
        vqa_data = AnnotationVQADataset(data_cfg.cache_dir, source_override=data_cfg.source_override)
        vqa_val = AnnotationVQADataset(data_cfg.cache_dir, split="val", source_override=data_cfg.source_override)
        vqa_batch = int(cfg.datasets.vqa_data.per_device_batch_size)
        if len(vqa_data) < vqa_batch * accelerator.num_processes:
            raise ValueError("Too few training VQA pairs for a global microbatch")
        vqa_loader = DataLoader(vqa_data, batch_size=vqa_batch, shuffle=True, collate_fn=collate,
                               num_workers=int(cfg.datasets.vqa_data.num_workers), drop_last=True)
        if len(vqa_val):
            vqa_subset, vqa_validation = balanced_validation_subset(
                vqa_val, int(cfg.trainer.eval_batches) * vqa_batch * accelerator.num_processes, int(cfg.seed))
            vqa_val_loader = DataLoader(vqa_subset, batch_size=vqa_batch, shuffle=False,
                                       collate_fn=collate, num_workers=0)
    accelerator.print("Building QwenOFT from local VLAct base model", flush=True)
    if vqa_enabled:
        from examples.RoboDojo.cotrain import VQACoTrainingOFT
        model = VQACoTrainingOFT(cfg)
        if accelerator.is_main_process:
            model.vqa_audit_path = output / "vqa_training_inputs.jsonl"
            model.vqa_audit_batches = int(cfg.trainer.get("audit_batches", 2))
    else:
        model = build_framework(cfg)
    if not args.resume:
        checkpoint = torch.load(cfg.trainer.pretrained_checkpoint, map_location="cpu", mmap=True, weights_only=True)
        prefix = "qwen_vl_interface."
        backbone = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
        if not backbone:
            raise ValueError("No qwen_vl_interface weights in the pretrained checkpoint")
        model.qwen_vl_interface.load_state_dict(backbone, strict=True)
        accelerator.print(f"Loaded {len(backbone)} backbone tensors strictly; fresh 14D OFT head", flush=True)
        del backbone, checkpoint
    model.qwen_vl_interface.model.config.use_cache = False
    audit = None
    if accelerator.is_main_process and int(cfg.trainer.get("audit_batches", 0)):
        audit = PromptAudit(model.qwen_vl_interface.processor.tokenizer, output / "training_inputs.jsonl")
        model.qwen_vl_interface.register_forward_pre_hook(audit, with_kwargs=True)
    probe_names = [next(name for name, _ in model.named_parameters()
                        if "language_model.layers.27.mlp.down_proj.weight" in name),
                   "action_model.model.fc2.weight"]
    if vqa_enabled:
        output_weight = model.qwen_vl_interface.model.get_output_embeddings().weight
        probe_names.append(next(name for name, p in model.named_parameters() if p is output_weight))
    initial_probes = {name: dict(model.named_parameters())[name].detach().flatten()[:65536].float().cpu().clone()
                      for name in probe_names}
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable == total, "This recipe requires full backbone + action head finetuning"
    optimizer = torch.optim.AdamW(build_param_lr_groups(model, cfg), lr=cfg.trainer.learning_rate.base,
                                  betas=tuple(cfg.trainer.optimizer.betas),
                                  eps=cfg.trainer.optimizer.eps, weight_decay=cfg.trainer.optimizer.weight_decay)
    scheduler = get_scheduler(cfg.trainer.lr_scheduler_type, optimizer,
                              num_warmup_steps=cfg.trainer.num_warmup_steps,
                              num_training_steps=cfg.trainer.max_train_steps,
                              scheduler_specific_kwargs=dict(cfg.trainer.scheduler_specific_kwargs))
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if vqa_loader is not None:
        vqa_loader = accelerator.prepare(vqa_loader)
    if vqa_val_loader is not None:
        vqa_val_loader = accelerator.prepare(vqa_val_loader)
    accelerator.register_for_checkpointing(scheduler)
    # A small persistent buffer records progress in Accelerator training states.
    progress = {"step": 0, "epoch": 0, "microbatch_in_epoch": 0}
    if args.resume:
        accelerator.load_state(args.resume)
        progress = json.loads((Path(args.resume) / "progress.json").read_text())
        restored = dict(accelerator.unwrap_model(model).named_parameters())
        initial_probes = {name: restored[name].detach().flatten()[:65536].float().cpu().clone()
                          for name in probe_names}
    if vqa_loader is not None:
        progress.setdefault("vqa_epoch", 0)
        progress.setdefault("vqa_microbatch_in_epoch", 0)
        vqa_loader.set_epoch(progress["vqa_epoch"])
        vqa_iter = iter(accelerator.skip_first_batches(vqa_loader, progress["vqa_microbatch_in_epoch"]))
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output / "config.yaml")
        OmegaConf.save(cfg, output / "config.full.yaml")
        shutil.copy2(Path(data_cfg.cache_dir) / "dataset_statistics.json", output / "dataset_statistics.json")
        shutil.copy2(Path(data_cfg.cache_dir) / "manifest.json", output / "data_manifest.json")
        write_json(output / "training_setup.json", {"trainable_parameters": trainable,
            "total_parameters": total, "world_size": accelerator.num_processes,
            "global_batch_size": batch * accumulation * accelerator.num_processes,
            "train_frames": len(data), "val_frames": len(val_data), "zero_stage": 2,
            "optimizer_offload": bool(cfg.trainer.get("optimizer_offload", False)),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
            "init": "resumed complete model/optimizer state" if args.resume else "VLAct pretrain backbone only; new OFT head",
            "resume_state": str(Path(args.resume).resolve()) if args.resume else None,
            "language_mode": data.manifest["spec"]["language_mode"]})
        write_json(output / "validation_sampling.json", {
            "action": action_validation, "vqa": vqa_validation,
            "metric_scope": "fixed sampled validation; not simulator performance"})
        write_json(output / "annotation_training.json", {
            "language_mode": language_mode, "annotations_reviewed": data.manifest["spec"]["annotations_reviewed"],
            "smoke_only": data.manifest["spec"].get("smoke_only", False),
            "allow_unreviewed_training": data.manifest["spec"].get("allow_unreviewed_training", False),
            "annotation_quality": data.manifest["spec"].get("annotation_quality"),
            "split_unit": data.manifest["spec"].get("split_unit", "episode"),
            "subtask_segments": data.manifest.get("subtask_segments", 0),
            "objective": "actions conditioned on task and current subtask" if language_mode == "subtask" else "actions conditioned on task",
            "subtask_text_generation_trained": False, "vqa_cotraining": vqa_enabled,
            "vqa_loss_weight": float(cfg.datasets.vqa_data.loss_weight) if vqa_enabled else 0.,
            "vqa_pairs": data.manifest.get("vqa", {})})
    accelerator.print(f"Training {trainable:,} parameters; {len(data)} frames; global batch {batch * accumulation * accelerator.num_processes}", flush=True)
    model.train()
    resume_start_step = progress["step"]
    started = time.monotonic()
    running = []
    running_vqa, running_total = [], []
    while progress["step"] < cfg.trainer.max_train_steps:
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(progress["epoch"])
        skip = progress["microbatch_in_epoch"]
        active_loader = accelerator.skip_first_batches(loader, skip) if skip else loader
        for index, examples in enumerate(active_loader, start=skip):
            vqa_examples = None
            if vqa_loader is not None:
                try:
                    vqa_examples = next(vqa_iter)
                except StopIteration:
                    progress["vqa_epoch"] += 1
                    progress["vqa_microbatch_in_epoch"] = 0
                    vqa_loader.set_epoch(progress["vqa_epoch"])
                    vqa_iter = iter(vqa_loader)
                    vqa_examples = next(vqa_iter)
                progress["vqa_microbatch_in_epoch"] += 1
            if audit is not None and audit.batches < int(cfg.trainer.audit_batches):
                audit.arm(examples, optimizer_step=progress["step"] + 1, microbatch_in_epoch=index, phase="train")
            with accelerator.accumulate(model):
                result = model(examples, vqa_examples=vqa_examples) if vqa_enabled else model(examples)
                loss = result["loss"] if vqa_enabled else result["action_loss"]
                finite = accelerator.reduce(torch.isfinite(loss.detach()).int(), reduction="sum")
                if int(finite) != accelerator.num_processes:
                    raise FloatingPointError("Non-finite loss on at least one rank")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            running.append(result["action_loss"].detach().float())
            if vqa_enabled:
                running_vqa.append(result["vqa_loss"].detach().float())
                running_total.append(loss.detach().float())
            progress["microbatch_in_epoch"] = index + 1
            if not accelerator.sync_gradients:
                continue
            scheduler.step()
            progress["step"] += 1
            step = progress["step"]
            metric = {"step": step, "resume_from_step": resume_start_step, "train_l1": float(accelerator.reduce(torch.stack(running).mean(), reduction="mean")),
                      "lr": scheduler.get_last_lr(), "elapsed_s": time.monotonic() - started,
                      "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
            running.clear()
            if vqa_enabled:
                metric["vqa_ce"] = float(accelerator.reduce(torch.stack(running_vqa).mean(), reduction="mean"))
                metric["total_loss"] = float(accelerator.reduce(torch.stack(running_total).mean(), reduction="mean"))
                metric["vqa_weight"] = float(cfg.datasets.vqa_data.loss_weight)
                running_vqa.clear()
                running_total.clear()
            if val_loader is not None and step % cfg.trainer.eval_interval == 0:
                metric["val_l1"] = evaluate(model, val_loader, accelerator, cfg.trainer.eval_batches)
            if vqa_val_loader is not None and step % cfg.trainer.eval_interval == 0:
                metric["val_vqa_ce"] = evaluate(model, vqa_val_loader, accelerator, cfg.trainer.eval_batches, vqa=True)
            if accelerator.is_main_process:
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metric) + "\n")
                if step % cfg.trainer.logging_frequency == 0 or step == 1:
                    print(json.dumps(metric), flush=True)
            final = step >= cfg.trainer.max_train_steps
            if final or step % cfg.trainer.save_interval == 0:
                target = save_weights(model, accelerator, output, step)
                accelerator.print(f"Saved inference checkpoint: {target}", flush=True)
                # Full optimizer/RNG state is optional for short smoke runs (large on disk).
                if cfg.trainer.get("save_training_state", True):
                    state_dir = output / "training_states" / f"step_{step}"
                    accelerator.save_state(str(state_dir))
                    # Accelerate saves RNG state separately on each rank. Publish the
                    # recovery marker only after every rank has finished writing.
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        marker = state_dir / "progress.json.tmp"
                        write_json(marker, progress)
                        marker.replace(state_dir / "progress.json")
                    accelerator.wait_for_everyone()
            if final:
                break
        if progress["step"] >= cfg.trainer.max_train_steps:
            break
        progress["epoch"] += 1
        progress["microbatch_in_epoch"] = 0
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        updates = {}
        for name, before in initial_probes.items():
            after = dict(unwrapped.named_parameters())[name].detach().flatten()[:65536].float().cpu()
            updates[name] = {"changed_elements": int((after != before).sum()),
                             "sampled_elements": before.numel(), "max_abs_delta": float((after - before).abs().max())}
        write_json(output / "parameter_updates.json", updates)
        if not all(value["changed_elements"] for value in updates.values()):
            raise RuntimeError("No observable parameter update in backbone/action-head probes")
        write_json(output / "completed.json", {**progress, "elapsed_s": time.monotonic() - started,
                   "checkpoint": str(target.resolve()), "finite_loss": True})
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
