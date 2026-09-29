"""Read-only monitoring of the entire torchrun process tree across rank sessions."""
import argparse
import json
from pathlib import Path
import time

from examples.RoboDojo.after_annotation_train import write_json


def snapshot(root_pid):
    all_processes = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            fields = dict(line.split(":", 1) for line in (directory / "status").read_text().splitlines())
            pid = int(directory.name)
            all_processes[pid] = {"pid": pid, "ppid": int(stat[1]), "pgrp": int(stat[2]),
                                 "threads": int(fields["Threads"]),
                                 "rss_kib": int(fields.get("VmRSS", "0 kB").split()[0])}
        except (OSError, ValueError, KeyError):
            continue
    selected = {root_pid}
    while True:
        descendants = {pid for pid, p in all_processes.items() if p["ppid"] in selected}
        if descendants <= selected:
            break
        selected |= descendants
    processes = [all_processes[pid] for pid in selected if pid in all_processes]
    ranks = [p for p in processes if p["pgrp"] == p["pid"] and p["pid"] != root_pid]
    return {"scope": "all descendants including independent torchrun rank sessions",
            "process_count": len(processes), "total_threads": sum(p["threads"] for p in processes),
            "rank_processes": ranks,
            "max_process_rss_kib": max((p["rss_kib"] for p in processes), default=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    while True:
        status = json.loads((args.output / "status.json").read_text())
        report = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "phase": status["phase"], "step": status.get("current_step"),
                  **snapshot(status.get("training_pid", -1))}
        write_json(args.output / "runtime_resources.json", report)
        with (args.output / "runtime_resources.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "rank_processes"}), flush=True)
        if status["phase"] in ("completed", "failed"):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
