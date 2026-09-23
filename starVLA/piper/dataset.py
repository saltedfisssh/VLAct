"""VLAct samples and checkpoint contracts for the Piper experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from starVLA.piper import geometry as g
from starVLA.piper import normalize

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
RUNGS = ("s0", "s2", "s2_pair")


def prepare_images(images, image_size):
    """Use the same resize for variable-resolution source videos and robot RGB."""
    if set(images) != set(CAMERAS):
        raise ValueError("Expected the three configured RGB views")
    result = []
    for key in CAMERAS:
        value = np.asarray(images[key])
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
            raise ValueError(f"{key}: expected a nonempty RGB uint8 image")
        result.append(Image.fromarray(value).resize(tuple(image_size), Image.Resampling.BILINEAR))
    return result


def raw_dataset(cache, rung, *, split="train", images=True):
    manifest = json.loads((Path(cache) / "manifest.json").read_text())
    if rung not in RUNGS or rung not in manifest["rungs"]:
        raise ValueError(f"Representation {rung!r} is not available")
    if manifest.get("kind") == "mixture":
        from starVLA.piper.mixture import MixtureDataset

        return MixtureDataset(cache, rung, split=split, images=images)
    if "tasks" in manifest:
        from starVLA.piper.data_x import PiperXEEFDataSet

        return PiperXEEFDataSet(cache, rung=rung, split=split, images=images)
    from starVLA.piper.data_piper import PiperDataset

    return PiperDataset(cache, rung, split=split, images=images)


def contract(cache, rung, image_size=(224, 224)):
    manifest_path = Path(cache) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if rung not in RUNGS or rung not in manifest["rungs"]:
        raise ValueError("Rung is absent from the cache")
    if manifest.get("kind") == "mixture":
        from starVLA.piper.mixture import mixture_contract

        result = mixture_contract(manifest, rung, image_size)
        result["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return result
    source_x = "tasks" in manifest
    alignment = manifest["settings"].get("alignment")
    native_x = source_x and alignment is None
    state_source = "observation.qpos_ee" if source_x and rung != "s0" else "observation.state"
    return {
        "format_version": 1,
        "framework": "QwenPI_v4",
        "piper_rung": rung,
        "state_dim": 23 if rung == "s2_pair" else 14,
        "action_dim": g.action_dim(rung),
        "action_horizon": 50,
        "action_start_offset": 1,
        "fps": 30,
        "action_encoding": "anchor_relative_absolute_gripper",
        "tcp_offset_m": 0.0,
        "robot_model": "piper_x" if native_x else "piper",
        "source_robot_model": "piper_x" if source_x else "piper",
        "training_pose_frame": "piper_x_flange" if native_x else "piper_flange",
        "training_state_source": state_source,
        "target_source": f"future_{state_source}",
        "pose_alignment": alignment["calibration_id"] if alignment else None,
        "alignment": alignment,
        "pair_base_spacing_m": 0.6,
        "rtc_mode": "none",
        "raw_image_resolution": [640, 480],
        "image_size": list(image_size),
        "camera_order": list(CAMERAS),
        "normalization": "q01_q99_centered_constant_scale_one_v1",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "smoke_only": manifest["smoke_only"],
    }


class TrainingDataset:
    def __init__(self, cache, rung, image_size=(224, 224), split="train"):
        self.raw = raw_dataset(cache, rung, split=split)
        self.stats = normalize.load(Path(cache) / "stats" / rung)
        self.image_size = tuple(image_size)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        sample = self.raw[index]
        return {
            "image": prepare_images(sample["image"], self.image_size),
            "lang": sample["prompt"],
            "state": normalize.transform(sample["state"], self.stats["state"], clip=True)[None],
            "action": normalize.transform(sample["actions"], self.stats["actions"]),
        }


def collate(samples):
    return samples
