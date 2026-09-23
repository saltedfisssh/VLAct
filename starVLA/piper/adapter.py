# Adapted from kai0 (Apache-2.0); see NOTICE.md and LICENSE.kai0.
"""Synchronous Piper execution using the exact training representation conventions.

Only NumPy and SciPy are required here. No JAX, Torch, ROS, or robot SDK is
imported. Commands use radians and metres, in each physical arm's own base.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from starVLA.piper import geometry as g
from starVLA.piper.transfer import PiperXTransfer

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
HORIZON = 50
FPS = 30


def finite_array(value, shape):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite values with shape {shape}, got {result.shape}")
    return result


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class Observation:
    joints: np.ndarray
    images: dict[str, np.ndarray]
    received_at: float
    source_stamp: float | None = None
    captured_at: float | None = None
    flange_poses: np.ndarray | None = None


@dataclass(frozen=True)
class Request:
    rung: str
    generation: int
    state: np.ndarray
    joints: np.ndarray
    images: dict[str, np.ndarray]
    prompt: str
    received_at: float
    source_stamp: float | None
    episode_transform: np.ndarray | None
    transfer: PiperXTransfer | None = None

    @property
    def payload(self):
        # The original anchor stays independent of serialization or policy transforms.
        return {
            "state": self.state.astype(np.float32),
            "image": {k: v.copy() for k, v in self.images.items()},
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class CommandChunk:
    mode: str
    values: np.ndarray
    clipped_gripper_values: int


def _continuous_euler(matrices, reference):
    """Choose equivalent XYZ Euler branches nearest the previous physical command."""
    result = Rotation.from_matrix(matrices).as_euler("xyz")
    previous = np.asarray(reference)
    for i, value in enumerate(result):
        alternate = np.array([value[0] + np.pi, np.pi - value[1], value[2] + np.pi])
        candidates = np.stack([value, alternate])
        candidates += 2 * np.pi * np.round((previous - candidates) / (2 * np.pi))
        result[i] = candidates[np.argmin(np.linalg.norm(candidates - previous, axis=-1))]
        previous = result[i]
    return result


class PiperAdapter:
    pose_frame = "piper_flange"

    def __init__(
        self,
        rung,
        *,
        prompt,
        episode_id=0,
        seed=0,
        xy_range=0.5,
        gripper_max=0.08,
        episode_transform=None,
        transfer: PiperXTransfer | None = None,
    ):
        self.width = g.action_dim(rung)
        if transfer is not None and rung != "s2":
            raise ValueError("Piper-X transfer requires the S2 representation (--rung s2)")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty training-compatible prompt is required")
        if not np.isfinite(gripper_max) or gripper_max <= 0:
            raise ValueError("gripper_max must be positive and finite")
        self.rung, self.prompt, self.seed, self.xy_range = rung, prompt, seed, xy_range
        self.gripper_max = gripper_max
        self.transfer = transfer
        self.generation = 0
        self.reset(episode_id, episode_transform=episode_transform)

    def reset(self, episode_id, *, episode_transform=None):
        self.generation += 1
        self.episode_id = int(episode_id)
        self.previous_state = None
        self.episode_transform = None
        if self.rung == "s4":
            h = (
                g.episode_frame(self.episode_id, self.seed, self.xy_range)
                if episode_transform is None
                else finite_array(episode_transform, (4, 4))
            )
            if (
                not np.allclose(h[3], [0, 0, 0, 1])
                or not np.allclose(h[:3, :3] @ h[:3, :3].T, np.eye(3))
                or not np.isclose(np.linalg.det(h[:3, :3]), 1)
                or not np.allclose(h[:3, 2], [0, 0, 1])
                or not np.isclose(h[2, 3], 0)
            ):
                raise ValueError("S4 requires a proper gravity-aligned transform with fixed height origin")
            self.episode_transform = _readonly(h)
        elif episode_transform is not None:
            raise ValueError("An episode transform is only valid for S4")

    def validate_metadata(self, metadata):
        expected = {
            "piper_rung": self.rung,
            "action_dim": self.width,
            "action_encoding": "anchor_relative_absolute_gripper",
            "tcp_offset_m": 0.0,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Server {key} mismatch: expected {value!r}, got {metadata.get(key)!r}")
        if metadata.get("action_horizon", HORIZON) != HORIZON:
            raise ValueError("Server action horizon must be 50")
        if tuple(metadata.get("raw_image_resolution", (640, 480))) != (640, 480):
            raise ValueError("Server image resolution differs from training")
        frame = metadata.get("training_pose_frame", "piper_flange")
        if frame != self.pose_frame and self.transfer is None:
            raise ValueError(
                f"Server was trained in the {frame!r} tool frame; supply a Piper-X calibration or serve an "
                "alignment-trained policy"
            )
        if self.transfer is not None:
            self.transfer.validate_metadata(metadata)

    def _build_state(self, observation, joints):
        return g.build_states(joints[None], self.rung, self.episode_transform)[0]

    def _reference_euler(self, request, arm):
        start = arm * 7
        return Rotation.from_matrix(g.fk(request.joints[start : start + 6])[:3, :3]).as_euler("xyz")

    def prepare(self, observation):
        joints = finite_array(observation.joints, (14,))
        if not np.isfinite(observation.received_at):
            raise ValueError("Observation time must be finite")
        if set(observation.images) != set(CAMERAS):
            raise ValueError(f"Expected camera keys {CAMERAS}")
        images = {}
        for name, raw_image in observation.images.items():
            image = np.asarray(raw_image)
            if image.dtype != np.uint8 or image.shape != (480, 640, 3):
                raise ValueError(f"{name}: expected RGB uint8 image with shape (480, 640, 3)")
            images[name] = _readonly(image)
        state = self._build_state(observation, joints)
        if self.transfer is not None:
            for arm, start in enumerate((0, 7)):
                pose = self.transfer.to_policy(g.fk(joints[start : start + 6]), arm)
                state[start : start + 3] = pose[:3, 3]
                state[start + 3 : start + 6] = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        if self.rung != "s0" and self.previous_state is not None:
            for start in (3, 10):
                vector = state[start : start + 3]
                theta = np.linalg.norm(vector)
                if theta > 1e-10:
                    axis = vector / theta
                    winding = round((np.dot(self.previous_state[start : start + 3], axis) - theta) / (2 * np.pi))
                    state[start : start + 3] += 2 * np.pi * winding * axis
        self.previous_state = state.copy()
        return Request(
            self.rung,
            self.generation,
            _readonly(state),
            _readonly(joints),
            images,
            self.prompt,
            observation.received_at,
            observation.source_stamp,
            self.episode_transform,
            self.transfer,
        )

    def decode(self, request, actions, *, previous_command=None):
        if request.rung != self.rung or request.generation != self.generation:
            raise ValueError("Prediction belongs to another rung or episode")
        if request.transfer is not self.transfer:
            raise ValueError("Prediction belongs to another pose calibration")
        actions = finite_array(actions, (HORIZON, self.width))
        if self.width == 20:
            for start in (3, 13):
                a, b = actions[:, start : start + 3], actions[:, start + 3 : start + 6]
                # Offline metrics can project arbitrary outputs; execution must reject
                # undefined orientations instead of inventing an identity rotation.
                if (np.linalg.norm(a, axis=-1) < 1e-8).any() or (np.linalg.norm(np.cross(a, b), axis=-1) < 1e-8).any():
                    raise ValueError("Degenerate rotation6d prediction")
        commands = np.empty((HORIZON, 14))
        if self.rung == "s0":
            commands[:] = actions
            for start in (0, 7):
                commands[:, start : start + 6] += request.joints[start : start + 6]
            mode = "joint"
        else:
            # SciPy versions used on robot hosts may require a writable rotation
            # buffer. Decode a copy while keeping the request anchor immutable.
            poses, grippers = g.decode_actions(request.state.copy(), actions, self.rung, request.episode_transform)
            if previous_command is not None:
                previous_command = finite_array(previous_command, (14,))
            for arm, start in enumerate((0, 7)):
                if request.transfer is not None:
                    poses[:, arm] = request.transfer.to_physical(poses[:, arm], arm)
                reference = (
                    self._reference_euler(request, arm)
                    if previous_command is None
                    else previous_command[start + 3 : start + 6]
                )
                commands[:, start : start + 3] = poses[:, arm, :3, 3]
                commands[:, start + 3 : start + 6] = _continuous_euler(poses[:, arm, :3, :3], reference)
                commands[:, start + 6] = grippers[:, arm]
            mode = "eef"
        grippers = commands[:, [6, 13]]
        clipped = np.clip(grippers, 0, self.gripper_max)
        changed = int(np.count_nonzero(grippers != clipped))
        commands[:, [6, 13]] = clipped
        return CommandChunk(mode, _readonly(finite_array(commands, (HORIZON, 14))), changed)


class MockPolicy:
    """A stationary policy for protocol checks; not a learned policy or robot simulator."""

    def __init__(self, rung, *, rtc_mode="none", piper_x=False):
        self.rung = rung
        self.rtc_mode = rtc_mode
        self.piper_x = piper_x

    def get_server_metadata(self):
        metadata = {
            "piper_rung": self.rung,
            "action_dim": g.action_dim(self.rung),
            "action_encoding": "anchor_relative_absolute_gripper",
            "tcp_offset_m": 0.0,
            "config_name": f"pi05_piper_{self.rung}",
            "action_horizon": HORIZON,
            "action_start_offset": 1,
            "rtc_mode": self.rtc_mode,
            "rtc_max_delay": 12,
        }
        if self.piper_x:
            metadata.update(
                config_name="pi05_piper_x_eef_multitask" + ("_train_rtc" if self.rtc_mode == "train_rtc" else ""),
                training_pose_source="observation.qpos_ee",
                training_pose_frame="piper_x_flange",
            )
        return metadata

    def infer(self, payload):
        state = np.asarray(payload["state"])
        actions = g.encode_actions(state, np.tile(state, (HORIZON, 1)), self.rung)
        if self.rtc_mode == "train_rtc" and "prev_action_chunk" in payload:
            delay = int(payload["inference_delay"])
            actions[:delay] = payload["prev_action_chunk"][:delay]
        return {"actions": actions}

    def close(self):
        pass


class MockBackend:
    def __init__(self):
        self.joints = np.array([0, 0.4, -0.5, 0, 0.3, 0, 0.04] * 2, dtype=np.float64)
        self.images = {key: np.zeros((480, 640, 3), dtype=np.uint8) for key in CAMERAS}

    def capture(self):
        return Observation(self.joints.copy(), self.images, time.monotonic())

    def send(self, mode, values):
        if mode == "joint":
            self.joints = values.copy()

    def close(self):
        pass


class SyncRunner:
    """Hold while inferring, then play a prefix at native 30 Hz without RTC.

    Source.capture() returns Observation; sink.send(mode, values) accepts a
    left-first vector of 14 absolute SI-unit commands. No enable/home is sent.
    """

    def __init__(
        self,
        adapter,
        source,
        sink,
        policy,
        *,
        execute_steps=10,
        max_observation_age=0.5,
        clock: Callable = time.monotonic,
        sleep: Callable = time.sleep,
        should_stop: Callable = lambda: False,
    ):
        if not 1 <= execute_steps <= HORIZON:
            raise ValueError("execute_steps must be between 1 and 50")
        if not np.isfinite(max_observation_age) or max_observation_age <= 0:
            raise ValueError("max_observation_age must be positive")
        self.adapter, self.source, self.sink, self.policy = adapter, source, sink, policy
        self.execute_steps, self.max_age = execute_steps, max_observation_age
        self.clock, self.sleep, self.should_stop = clock, sleep, should_stop
        self.previous_command = None
        self.adapter.validate_metadata(policy.get_server_metadata())

    def cycle(self):
        observation = self.source.capture()
        age = self.clock() - observation.received_at
        if age < -0.001 or age > self.max_age:
            raise ValueError("Observation is stale or uses a different clock")
        request = self.adapter.prepare(observation)
        started = self.clock()
        response = self.policy.infer(request.payload)
        received = self.clock()
        chunk = self.adapter.decode(request, response["actions"], previous_command=self.previous_command)
        deadline = self.clock() + 1 / FPS
        executed = 0
        for command in chunk.values[: self.execute_steps]:
            if self.should_stop():
                break
            self.sleep(max(0, deadline - self.clock()))
            if self.should_stop():
                break
            self.sink.send(chunk.mode, command.copy())
            self.previous_command = command.copy()
            executed += 1
            # An overrun must not cause an immediate burst of old commands.
            deadline = max(deadline + 1 / FPS, self.clock() + 1 / FPS)
        return {
            "rung": self.adapter.rung,
            "episode_id": self.adapter.episode_id,
            "rtc": False,
            "mode": chunk.mode,
            "fps": FPS,
            "executed_steps": executed,
            "observation_received_at": observation.received_at,
            "source_stamp": observation.source_stamp,
            "inference_seconds": received - started,
            "state": request.state.tolist(),
            "joints": request.joints.tolist(),
            "actions": np.asarray(response["actions"]).tolist(),
            "commands": chunk.values.tolist(),
            "clipped_gripper_values": chunk.clipped_gripper_values,
            "episode_transform": None if request.episode_transform is None else request.episode_transform.tolist(),
            "pose_transfer": None if request.transfer is None else request.transfer.to_dict(),
        }
