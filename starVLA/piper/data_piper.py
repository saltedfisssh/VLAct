# Adapted from kai0 (Apache-2.0); see NOTICE.md and LICENSE.kai0.
"""Episode-isolated chunks over small numeric caches and read-only source videos."""

import json
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np

from starVLA.piper import geometry as g
from starVLA.piper.prepare_piper import CAMERAS


def read_image(path, timestamp, fps):
    """Decode the nearest presentation timestamp after a keyframe seek."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 1
        container.seek(max(0, int(timestamp / stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is not None and abs(float(frame.pts * stream.time_base) - timestamp) <= 0.51 / fps:
                return frame.to_ndarray(format="rgb24")
    raise ValueError(f"No video frame within half a frame of {timestamp:.6f}s in {path}")


class PiperDataset:
    def __init__(self, root, rung, *, split="train", action_horizon=50, images=True):
        g.action_dim(rung)
        self.root, self.rung, self.images = Path(root), rung, images
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if action_horizon != self.manifest["settings"]["horizon"]:
            raise ValueError("Action horizon differs from the normalization/cache manifest")
        if split not in ("train", "held_out"):
            raise ValueError("Split must be train or held_out")
        self.source = Path(self.manifest["settings"]["source"])
        self.episode_ids = self.manifest["split"][split]
        self.episodes = {ep["episode_index"]: ep for ep in self.manifest["episodes"]}
        self.horizon = action_horizon
        self.ends = np.cumsum([self.episodes[i]["length"] - action_horizon for i in self.episode_ids])
        if not len(self.ends) or self.ends[-1] <= 0:
            raise ValueError("No eligible action chunks")
        self.cache = OrderedDict()

    def __len__(self):
        return int(self.ends[-1])

    def locate(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ordinal = int(np.searchsorted(self.ends, index, side="right"))
        frame = index - (self.ends[ordinal - 1] if ordinal else 0)
        return self.episode_ids[ordinal], int(frame)

    def episode(self, index):
        if index not in self.cache:
            with np.load(self.root / "episodes" / f"episode_{index:06d}.npz") as archive:
                self.cache[index] = {
                    key: archive[key] for key in (self.rung, "joints", "timestamps", "episode_transform")
                }
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self, index):
        episode_id, frame = self.locate(index)
        episode = self.episode(episode_id)
        states = episode[self.rung]
        state = states[frame]
        actions = g.encode_actions(state, states[frame + 1 : frame + 1 + self.horizon], self.rung)
        result = {
            "state": state.copy(),
            "actions": actions.astype(np.float32),
            "prompt": self.episodes[episode_id]["tasks"][0],
        }
        if self.images:
            info = self.manifest["info"]
            fields = {"episode_index": episode_id, "episode_chunk": episode_id // info["chunks_size"]}
            result["image"] = {
                name: read_image(
                    self.source / info["video_path"].format(**fields, video_key=key),
                    episode["timestamps"][frame],
                    info["fps"],
                )
                for name, key in CAMERAS.items()
            }
        return result
