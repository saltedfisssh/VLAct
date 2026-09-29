"""Continue a full RoboDojo run from complete checkpoints, without another smoke run."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import traceback
import zipfile

from examples.RoboDojo.after_annotation_train import (
    read_tail, ready_report, verify_cache, verify_training, write_json,
)


def checkpoint_step(path):
    try:
        step = int(path.name.removeprefix("step_"))
        if json.loads((path / "progress.json").read_text())["step"] != step:
            return None
        tag = (path / "latest").read_text().strip()
        if tag != "pytorch_model":
            return None
        files = [path / tag / "mp_rank_00_model_states.pt", path / "custom_checkpoint_0.pkl"]
        for rank in range(8):
            files.extend([path / f"random_states_{rank}.pkl",
                          path / tag / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"])
        if any(not zipfile.is_zipfile(file) for file in files):
            return None
        return step
    except (OSError, ValueError, KeyError, TypeError):
        return None


def latest_checkpoint(run):
    candidates = [(step, path) for path in (run / "training_states").glob("step_*")
                  if (step := checkpoint_step(path)) is not None]
    if not candidates:
        raise RuntimeError("No complete 8-rank training checkpoint")
    return max(candidates, key=lambda item: item[0])


def archive_uncommitted_metrics(run, step, archive):
    path = run / "metrics.jsonl"
    if not path.exists():
        return
    shutil.copy2(path, archive / "metrics_before_resume.jsonl")
    keep = [line for line in path.read_text().splitlines() if json.loads(line)["step"] <= step]
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("\n".join(keep) + ("\n" if keep else ""))
    temporary.replace(path)


def resource_snapshot(group):
    processes = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[2]) != group:
                continue
            fields = dict(line.split(":", 1) for line in (path / "status").read_text().splitlines())
            processes.append({"pid": int(path.name), "threads": int(fields["Threads"]),
                              "rss_kib": int(fields.get("VmRSS", "0 kB").split()[0])})
        except (OSError, ValueError, KeyError, ProcessLookupError):
            continue
    return {"processes": processes, "total_threads": sum(p["threads"] for p in processes),
            "max_process_rss_kib": max((p["rss_kib"] for p in processes), default=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boot-id", required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--max-restarts", type=int, default=3)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = args.output.resolve()
    run = root / "full_train"
    cache = root / "data_full_subtasks_vqa"
    source = Path(json.loads((cache / "manifest.json").read_text())["spec"]["source"])
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    recovery = root / f"recovery_{stamp}"
    status = {"host": socket.gethostname(), "source": str(source), "output": str(root),
              "pid": os.getpid(), "steps": args.steps, "vqa_weight": 0.1,
              "annotations_reviewed": False, "recovery_directory": str(recovery)}

    def update(phase, **fields):
        status.update(phase=phase, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields)
        write_json(root / "status.json", status)
        print(json.dumps(status), flush=True)

    with ExitStack() as locks:
        for path in (root / "runner.lock", source / ".robodojo_annotation.lock",
                     source / ".annotate_staging/.lock"):
            handle = locks.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recovery.mkdir()
        if (root / "status.json").exists():
            shutil.copy2(root / "status.json", recovery / "previous_status.json")
        try:
            if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != args.boot_id:
                raise RuntimeError("Wrong training host boot identity")
            ready_report(source)
            verify_cache(cache)
            if json.loads((root / "decoder_fix_verification.json").read_text())["status"] != "passed":
                raise RuntimeError("Real-video decoder verification has not passed")
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path(__file__).parent.glob("*.py")}
            write_json(recovery / "code_sha256.json", hashes)
            env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "8",
                   "TOKENIZERS_PARALLELISM": "false"}
            env.pop("VIRTUAL_ENV", None)
            for attempt in range(args.max_restarts + 1):
                step, checkpoint = latest_checkpoint(run)
                if step >= args.steps:
                    break
                attempt_dir = recovery / f"attempt_{attempt + 1:02d}"
                attempt_dir.mkdir()
                for name in ("training_setup.json", "config.full.yaml", "annotation_training.json"):
                    if (run / name).exists():
                        shutil.copy2(run / name, attempt_dir / name)
                archive_uncommitted_metrics(run, step, attempt_dir)
                command = ["bash", str(root / "train_full.sh"), "--resume", str(checkpoint),
                           f"trainer.max_train_steps={args.steps}"]
                write_json(attempt_dir / "command.json", command)
                log = attempt_dir / "train.log"
                # Preserve the initial failed attempt, and keep the public log path current.
                public = root / "full_train.log"
                if public.exists() and not public.is_symlink():
                    public.replace(recovery / "initial_full_train.log")
                link = root / "full_train.log.tmp"
                link.symlink_to(log)
                link.replace(public)
                deadline = time.monotonic() + 900
                while True:
                    memory = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
                    ], text=True).split()
                    if len(memory) == 8 and all(int(n) < 4096 for n in memory):
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("GPUs are occupied; no existing processes were stopped")
                    time.sleep(10)
                with log.open("w") as stream:
                    process = subprocess.Popen(command, cwd=repo, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    update("full_train", resume_from_step=step, current_step=step, attempt=attempt + 1,
                           training_pid=process.pid, training_log=str(log), command=command, error=None)
                    while process.poll() is None:
                        metric = {}
                        try:
                            lines = read_tail(run / "metrics.jsonl", 8192).splitlines()
                            if lines:
                                metric = json.loads(lines[-1])
                        except (OSError, ValueError):
                            pass
                        resources = resource_snapshot(process.pid)
                        with (attempt_dir / "resources.jsonl").open("a") as report:
                            report.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                     "step": metric.get("step", step), **resources}) + "\n")
                        update("full_train", current_step=metric.get("step", step), latest_metric=metric,
                               total_process_threads=resources["total_threads"],
                               max_process_rss_kib=resources["max_process_rss_kib"])
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            pass
                if process.returncode == 0:
                    break
                text = read_tail(log)
                retryable = any(s in text for s in (
                    "av.error.MemoryError", "Stale file handle", "[Errno 5] Input/output error"))
                if not retryable or attempt >= args.max_restarts:
                    raise RuntimeError(f"Training exited {process.returncode}; see {log}")
                update("recovering", error=f"Transient decoder/storage failure; return code {process.returncode}")
                time.sleep(10)
            write_json(root / "full_training_gate.json", verify_training(run, args.steps))
            checkpoint = run / f"checkpoints/steps_{args.steps}_pytorch_model.pt"
            update("verify_full_checkpoint", checkpoint=str(checkpoint))
            command = ["uv", "run", "--no-sync", "python", "-m", "examples.RoboDojo.verify_checkpoint",
                       "--checkpoint", str(checkpoint), "--cache", str(cache), "--policy-source",
                       str(repo.parent / "RoboDojo/XPolicyLab/policy/starVLA/source_starvla"),
                       "--output", str(root / "verify_full_checkpoint.json")]
            with (root / "verify_full_checkpoint.log").open("a") as log:
                subprocess.run(command, cwd=repo, env={**env, "CUDA_VISIBLE_DEVICES": "0"},
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            update("completed", current_step=args.steps, checkpoint=str(checkpoint), error=None)
        except BaseException as error:
            update("failed", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
