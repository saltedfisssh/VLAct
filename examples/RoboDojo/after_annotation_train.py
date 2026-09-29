"""Wait for a particular annotation run, validate it, then smoke and fully train VLAct.

Run this module inside tmux on the training host. No annotation process is killed.
A failed prerequisite stops the chain and leaves status.json plus command logs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def process_start(pid):
    try:
        # comm may contain spaces and parentheses; fields after the final ')' start at field 3.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def read_tail(path, limit=128 * 1024):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode(errors="replace")


def ready_report(source):
    report = json.loads((source / "annotation_report.json").read_text())
    info = json.loads((source / "meta/info.json").read_text())
    if report["status"] != "passed" or report["errors"]:
        raise RuntimeError("Annotation verification failed")
    if report["episodes"] != 3500 or info["total_episodes"] != 3500:
        raise RuntimeError("Expected all 3500 RoboDojo episodes")
    if report["frames"] != info["total_frames"]:
        raise RuntimeError("Annotation frame count mismatch")
    if len(report["per_episode"]) != 3500:
        raise RuntimeError("Incomplete per-episode annotation report")
    return {key: value for key, value in report.items() if key != "per_episode"}


def verify_cache(cache):
    manifest = json.loads((cache / "manifest.json").read_text())
    tasks = {}
    sets = {"train": set(), "val": set()}
    for episode in manifest["episodes"]:
        split = episode["split"]
        sets[split].add(episode["original_episode_index"])
        counts = tasks.setdefault(str(episode["task_index"]), {"train": 0, "val": 0})
        counts[split] += 1
    if len(tasks) != 35 or any(v != {"train": 90, "val": 10} for v in tasks.values()):
        raise RuntimeError(f"Expected 35 tasks each with 90 train / 10 val: {tasks}")
    if sets["train"] & sets["val"] or tuple(map(len, sets.values())) != (3150, 350):
        raise RuntimeError("Episode leakage or missing episodes")
    spec = manifest["spec"]
    if spec["annotations_reviewed"] or not spec.get("allow_unreviewed_training"):
        raise RuntimeError("Incorrect pseudo-label provenance")
    if spec["language_mode"] != "subtask" or not spec.get("include_vqa"):
        raise RuntimeError("Expected atomic-subtask and VQA cache")
    if not all(manifest["vqa"][key] > 0 for key in ("train_pairs", "val_pairs")):
        raise RuntimeError("Missing supported VQA pairs")
    return {"tasks": tasks, "train_episodes": 3150, "val_episodes": 350,
            "frames": manifest["frames"], "vqa": manifest["vqa"], "spec": spec}


def verify_training(run, expected_steps):
    completed = json.loads((run / "completed.json").read_text())
    setup = json.loads((run / "training_setup.json").read_text())
    updates = json.loads((run / "parameter_updates.json").read_text())
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    if completed["step"] != expected_steps or setup["world_size"] != 8:
        raise RuntimeError("Training did not complete expected steps on 8 GPUs")
    if setup["global_batch_size"] != 128 or setup["optimizer_offload"]:
        raise RuntimeError("Training recipe differs from the requested 8-GPU configuration")
    if not all(value["changed_elements"] > 0 for value in updates.values()):
        raise RuntimeError("Backbone/action/VQA parameter probes did not all update")
    for metric in metrics:
        for key in ("train_l1", "vqa_ce", "total_loss"):
            if not math.isfinite(metric[key]):
                raise RuntimeError(f"Non-finite {key}")
    if metrics[-1]["step"] != expected_steps:
        raise RuntimeError("Missing final metrics")
    inputs = [json.loads(line) for line in (run / "training_inputs.jsonl").read_text().splitlines()]
    if not inputs or not all(row["current_subtask_in_model_prompt"] for row in inputs):
        raise RuntimeError("Atomic instructions missing from actual model inputs")
    return {"completed": completed, "setup": setup, "last_metric": metrics[-1],
            "parameter_updates": updates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotation-log", type=Path, required=True)
    parser.add_argument("--annotation-pid", type=int, required=True)
    parser.add_argument("--annotation-start-ticks", required=True)
    parser.add_argument("--boot-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--accept-verified-annotation", action="store_true",
                        help="Accept an already repaired and verified source without a launcher success marker")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"host": socket.gethostname(), "source": str(args.source),
                  "output": str(output), "pid": os.getpid(), "steps": args.steps,
                  "annotations_reviewed": False, "vqa_weight": 0.1}

        def update(phase, **details):
            status.update(phase=phase, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details)
            write_json(output / "status.json", status)
            print(json.dumps(status, ensure_ascii=False), flush=True)

        def run(name, command, env=None):
            update(name, command=command)
            write_json(output / f"{name}.command.json", command)
            with (output / f"{name}.log").open("a") as stream:
                subprocess.run(command, cwd=repo, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)

        try:
            if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != args.boot_id:
                raise RuntimeError("Wrong host boot identity; do not run on the other shared-disk host")
            update("waiting_annotation")
            while process_start(args.annotation_pid) == args.annotation_start_ticks:
                time.sleep(args.poll_seconds)
            if not args.accept_verified_annotation and f"Finished. Dataset: {args.source}" not in read_tail(args.annotation_log):
                raise RuntimeError("Annotation launcher exited without its successful completion marker")
            summary = ready_report(args.source)
            write_json(output / "annotation_gate.json", summary)

            # Keep the source immutable through cache preparation and video-backed training.
            with (args.source / ".robodojo_annotation.lock").open("a") as source_lock, (
                args.source / ".annotate_staging/.lock"
            ).open("a") as staging_lock:
                for stream in (source_lock, staging_lock):
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Recheck under both source locks, including separately repaired annotations.
                ready_report(args.source)
                update("waiting_gpu_release")
                deadline = time.monotonic() + 900
                while True:
                    values = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
                    ], text=True).split()
                    if len(values) == 8 and all(int(value) < 4096 for value in values):
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("GPUs not released after annotation; no processes were killed")
                    time.sleep(args.poll_seconds)
                source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(__file__).parent.glob("*.py"))}
                write_json(output / "code_sha256.json", source_hashes)
                cache = output / "data_full_subtasks_vqa"
                prefix = ["uv", "run", "--no-sync", "python"]
                run("prepare_cache", [*prefix, "-m", "examples.RoboDojo.data",
                    "--source", str(args.source), "--output", str(cache),
                    "--language-mode", "subtask", "--include-vqa",
                    "--allow-unreviewed-training", "--validation-fraction", "0.1"])
                write_json(output / "cache_gate.json", verify_cache(cache))
                env = {**os.environ, "SOURCE": str(args.source), "CACHE_DIR": str(cache),
                       "RUN_ROOT": str(output), "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                       "NPROC": "8", "BATCH_SIZE": "8", "GRAD_ACCUM": "2", "MAX_STEPS": "4",
                       "RUN_ID": "capacity_smoke", "TOKENIZERS_PARALLELISM": "false",
                       "OMP_NUM_THREADS": "8"}
                train = ["bash", "examples/RoboDojo/train_subtasks_8xh100.sh",
                         "datasets.vqa_data.enabled=true", "datasets.vqa_data.loss_weight=0.1",
                         "datasets.vqa_data.per_device_batch_size=1"]
                run("capacity_smoke", [*train, "trainer.num_warmup_steps=0",
                    "trainer.eval_interval=4", "trainer.eval_batches=1", "trainer.save_interval=4",
                    "trainer.logging_frequency=1", "trainer.audit_batches=2",
                    "trainer.save_training_state=true", "datasets.vla_data.num_workers=2"], env)
                smoke = output / "capacity_smoke"
                write_json(output / "smoke_gate.json", verify_training(smoke, 4))
                if not (smoke / "training_states/step_4/progress.json").is_file():
                    raise RuntimeError("Smoke did not save optimizer/RNG/progress for recovery")
                verify_env = {**env, "CUDA_VISIBLE_DEVICES": "0"}
                for name, extra in (
                    ("verify_native_smoke", []),
                    ("verify_robodojo_smoke", ["--policy-source",
                     str(repo.parent / "RoboDojo/XPolicyLab/policy/starVLA/source_starvla")]),
                ):
                    run(name, [*prefix, "-m", "examples.RoboDojo.verify_checkpoint",
                        "--checkpoint", str(smoke / "checkpoints/steps_4_pytorch_model.pt"),
                        "--cache", str(cache), *extra, "--output", str(output / f"{name}.json")],
                        verify_env)
                # Exercise four more real updates after ZeRO optimizer/RNG reload (diagnostic schedule only).
                run("resume_smoke", [*train, "trainer.num_warmup_steps=0",
                    "trainer.eval_interval=4", "trainer.eval_batches=1", "trainer.save_interval=4",
                    "trainer.logging_frequency=1", "trainer.audit_batches=2",
                    "trainer.save_training_state=true", "datasets.vla_data.num_workers=2",
                    "--resume", str(smoke / "training_states/step_4")], {**env, "MAX_STEPS": "8"})
                write_json(output / "resume_gate.json", verify_training(smoke, 8))
                full_env = {**env, "RUN_ID": "full_train", "MAX_STEPS": str(args.steps)}
                run("full_train", [*train, "trainer.save_interval=1000",
                    "trainer.eval_interval=500", "trainer.eval_batches=8"], full_env)
                full = output / "full_train"
                write_json(output / "full_training_gate.json", verify_training(full, args.steps))
                run("verify_full_checkpoint", [*prefix, "-m", "examples.RoboDojo.verify_checkpoint",
                    "--checkpoint", str(full / f"checkpoints/steps_{args.steps}_pytorch_model.pt"),
                    "--cache", str(cache), "--policy-source",
                    str(repo.parent / "RoboDojo/XPolicyLab/policy/starVLA/source_starvla"),
                    "--output", str(output / "verify_full_checkpoint.json")], verify_env)
                update("completed", checkpoint=str(full / f"checkpoints/steps_{args.steps}_pytorch_model.pt"))
        except BaseException as error:
            update("failed", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
