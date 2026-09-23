"""Check checkpoint reload, real held-out RGB inference and WebSocket transport without hardware."""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
from websockets.sync.server import serve

from starVLA.piper.client import PiperWebsocketPolicy
from starVLA.piper.dataset import contract, raw_dataset
from starVLA.piper.evaluate import physical_errors
from starVLA.piper.serve import Policy, make_handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--rung", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    policy = Policy(args.checkpoint)
    expected = contract(args.cache, args.rung)
    for key in ("manifest_sha256", "piper_rung", "training_pose_frame", "alignment"):
        if policy.metadata[key] != expected[key]:
            raise ValueError(f"Checkpoint/cache mismatch: {key}")
    dataset = raw_dataset(args.cache, args.rung, split="held_out")
    report = {"checkpoint": str(args.checkpoint.resolve()), "contract": policy.metadata, "samples": []}
    with serve(make_handler(policy), "127.0.0.1", 0, compression=None, max_size=32 * 1024 * 1024) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = PiperWebsocketPolicy("127.0.0.1", server.socket.getsockname()[1], timeout=300)
        try:
            if client.get_server_metadata() != policy.metadata:
                raise AssertionError("WebSocket handshake changed metadata")
            indices = (
                dataset.starts
                if hasattr(dataset, "starts")
                else (np.r_[0, dataset.task_ends[:-1]] if hasattr(dataset, "task_ends") else [0])
            )
            for index in indices:
                sample = dataset[int(index)]
                response = client.infer({key: sample[key] for key in ("state", "image", "prompt")})
                actions = np.asarray(response["actions"])
                if actions.shape != sample["actions"].shape or not np.isfinite(actions).all():
                    raise AssertionError("Invalid served actions")
                errors = physical_errors(sample["state"], actions, sample["actions"], args.rung)
                report["samples"].append(
                    {
                        "index": int(index),
                        "shape": list(actions.shape),
                        "inference_seconds": response["inference_seconds"],
                        "chunk_mean_error": errors.mean(0).tolist(),
                    }
                )
            # The network path must reject malformed state rather than infer with a mismatched representation.
            try:
                client.infer({"state": np.zeros(1), "image": sample["image"], "prompt": sample["prompt"]})
            except RuntimeError as exc:
                if "Invalid state" not in str(exc):
                    raise
                report["invalid_state_rejected"] = True
            else:
                raise AssertionError("Malformed state was accepted")
        finally:
            client.close()
            server.shutdown()
            thread.join(timeout=10)
    report["total_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
