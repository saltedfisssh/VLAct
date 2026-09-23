"""Serve a complete Piper checkpoint using the CPU client's MessagePack protocol."""

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np

from starVLA.piper import msgpack_numpy, normalize
from starVLA.piper.dataset import prepare_images


class Policy:
    def __init__(self, checkpoint, *, vlm=None, expert=None):
        import torch
        from accelerate import PartialState
        from omegaconf import OmegaConf
        from safetensors.torch import load_model

        from starVLA.model.framework.QwenPI_v4 import Qwen_PI_v4

        PartialState()
        checkpoint = Path(checkpoint)
        if not (checkpoint / "complete.json").is_file():
            raise ValueError("Checkpoint is incomplete")
        self.metadata = json.loads((checkpoint / "contract.json").read_text())
        if (
            hashlib.sha256((checkpoint / "norm_stats.json").read_bytes()).hexdigest()
            != self.metadata["statistics_sha256"]
        ):
            raise ValueError("Checkpoint normalization fingerprint mismatch")
        cfg = OmegaConf.load(checkpoint / "config.yaml")
        if vlm:
            cfg.framework.qwenvl.base_vlm = vlm
        if expert:
            cfg.framework.action_model.expert_model_path = expert
        cfg.trainer.enable_gradient_checkpointing = False
        if (
            cfg.framework.action_model.action_dim != self.metadata["action_dim"]
            or cfg.framework.action_model.state_dim != self.metadata["state_dim"]
            or cfg.framework.action_model.future_action_window_size + 1 != self.metadata["action_horizon"]
        ):
            raise ValueError("Model config and representation contract disagree")
        self.stats = normalize.load(checkpoint)
        self.model = Qwen_PI_v4(cfg).to(dtype=torch.bfloat16)
        load_model(self.model, str(checkpoint / "model.safetensors"), strict=True)
        self.model = self.model.to("cuda").eval()
        self.lock = threading.Lock()

    def infer(self, payload):
        import torch

        if set(payload) != {"state", "image", "prompt"}:
            raise ValueError("Expected raw state, RGB image dict and prompt; RTC inputs are unsupported")
        state = np.asarray(payload["state"])
        if state.shape != (self.metadata["state_dim"],) or not np.isfinite(state).all():
            raise ValueError("Invalid state shape or nonfinite values")
        if not isinstance(payload["prompt"], str) or not payload["prompt"].strip():
            raise ValueError("A nonempty task prompt is required")
        images = prepare_images(payload["image"], self.metadata["image_size"])
        sample = {
            "image": images,
            "lang": payload["prompt"],
            "state": normalize.transform(state, self.stats["state"], clip=True)[None],
        }
        started = time.monotonic()
        with self.lock, torch.inference_mode():
            prediction = self.model.predict_action([sample])["normalized_actions"][0]
        if prediction.shape != (50, self.metadata["action_dim"]):
            raise ValueError("Invalid model action shape")
        return {
            "actions": normalize.transform(prediction, self.stats["actions"], inverse=True),
            "inference_seconds": time.monotonic() - started,
        }


def make_handler(policy):
    from websockets.exceptions import ConnectionClosed

    def handle(connection):
        connection.send(msgpack_numpy.packb(policy.metadata))
        try:
            for message in connection:
                try:
                    response = policy.infer(msgpack_numpy.unpackb(message))
                    connection.send(msgpack_numpy.packb(response))
                except Exception as exc:
                    connection.send(f"{type(exc).__name__}: {exc}")
        except ConnectionClosed:
            pass

    return handle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--vlm")
    p.add_argument("--expert")
    args = p.parse_args()
    policy = Policy(args.checkpoint, vlm=args.vlm, expert=args.expert)
    from websockets.sync.server import serve

    handle = make_handler(policy)

    with serve(handle, args.host, args.port, compression=None, max_size=32 * 1024 * 1024, ping_timeout=None) as server:
        print(f"Piper policy ready on {args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
