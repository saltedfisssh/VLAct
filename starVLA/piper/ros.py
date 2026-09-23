# Adapted from kai0 (Apache-2.0); see NOTICE.md and LICENSE.kai0.
"""Optional ROS 1 transport for Piper JointState and PosCmd interfaces.

ROS imports are deferred until RosBackend is constructed. Image decoding and
message conversion remain testable without ROS or cv_bridge.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from starVLA.piper.adapter import CAMERAS, Observation, finite_array

JOINT_NAMES = tuple(f"joint{i}" for i in range(7))


def image_to_rgb(message):
    if message.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"Unsupported image encoding: {message.encoding}; expected rgb8 or bgr8")
    height, width, step = int(message.height), int(message.width), int(message.step)
    if height <= 0 or width <= 0 or step < width * 3:
        raise ValueError("Invalid ROS image dimensions or row stride")
    data = np.frombuffer(message.data, dtype=np.uint8)
    if data.size != height * step:
        raise ValueError("ROS image buffer size does not match its row stride")
    rgb = data.reshape(height, step)[:, : width * 3].reshape(height, width, 3)
    return (rgb[..., ::-1] if message.encoding == "bgr8" else rgb).copy()


def joint_positions(message):
    values = finite_array(message.position, (7,))
    if message.name:
        if len(message.name) != 7 or set(message.name) != set(JOINT_NAMES):
            raise ValueError(f"Expected joint feedback names {JOINT_NAMES}")
        values = values[[message.name.index(name) for name in JOINT_NAMES]]
    return values


def command_messages(mode, values, *, joint_factory, pose_factory, stamp):
    values = finite_array(values, (14,))
    if mode not in ("joint", "eef"):
        raise ValueError("Unknown command mode")
    messages = []
    for start in (0, 7):
        vector = values[start : start + 7].tolist()
        if mode == "joint":
            message = joint_factory()
            message.header.stamp = stamp
            message.name = list(JOINT_NAMES)
            message.position = vector
        else:
            message = pose_factory()
            # Plans use continuous Euler branches internally. The Piper SDK
            # clamps roll/yaw to +/-pi and pitch to +/-pi/2, so send an equivalent
            # principal representation at the wire boundary, never clipped angles.
            vector[3:6] = Rotation.from_euler("xyz", vector[3:6]).as_euler("xyz").tolist()
            message.x, message.y, message.z, message.roll, message.pitch, message.yaw, message.gripper = vector
            message.mode1 = 0
            message.mode2 = 0
        messages.append(message)
    return tuple(messages)


class ObservationBuffer:
    def __init__(self, *, max_age=0.5, max_skew=0.04):
        if not np.isfinite([max_age, max_skew]).all() or min(max_age, max_skew) <= 0:
            raise ValueError("Observation age/skew thresholds must be positive")
        self.max_age, self.max_skew = max_age, max_skew
        self.buffers = {key: deque(maxlen=60) for key in (*CAMERAS, "left", "right")}
        self.condition = threading.Condition()

    def push(self, key, value, *, stamp, received_at):
        if not np.isfinite([stamp, received_at]).all() or stamp <= 0:
            raise ValueError("ROS feedback requires positive, finite header timestamps")
        with self.condition:
            buffer = self.buffers[key]
            if buffer and stamp <= buffer[-1][0]:
                return
            buffer.append((stamp, received_at, np.array(value, copy=True)))
            self.condition.notify_all()

    def snapshot(self, *, now, source_now):
        with self.condition:
            if any(not rows for rows in self.buffers.values()):
                raise ValueError("Missing camera or joint feedback")
            anchor = min(self.buffers[key][-1][0] for key in CAMERAS)
            selected = {key: min(rows, key=lambda row: abs(row[0] - anchor)) for key, rows in self.buffers.items()}
            stamps = [row[0] for row in selected.values()]
            if max(stamps) - min(stamps) > self.max_skew:
                raise ValueError("Observation timestamp skew exceeds max_skew")
            for stamp, arrival, _ in selected.values():
                if now - arrival > self.max_age or source_now - stamp > self.max_age:
                    raise ValueError("Observation feedback is stale")
                if stamp - source_now > self.max_skew or arrival > now + 0.001:
                    raise ValueError("Observation timestamp is in the future")
            joints = np.r_[selected["left"][2], selected["right"][2]]
            images = {key: selected[key][2].copy() for key in CAMERAS}
            return Observation(
                joints,
                images,
                min(row[1] for row in selected.values()),
                anchor,
                captured_at=now - (source_now - anchor),
            )


@dataclass(frozen=True)
class RosTopics:
    head: str = "/camera_f/color/image_raw"
    left_camera: str = "/camera_l/color/image_raw"
    right_camera: str = "/camera_r/color/image_raw"
    left_state: str = "/puppet/joint_left"
    right_state: str = "/puppet/joint_right"
    left_joint_command: str = "/master/joint_left"
    right_joint_command: str = "/master/joint_right"
    left_pose_command: str = "/pos_cmd_left"
    right_pose_command: str = "/pos_cmd_right"


class RosBackend:
    def __init__(self, *, execute=False, topics=None, max_age=0.5, max_skew=0.04, observation_timeout=5.0):
        try:
            import rospy
            from piper_msgs.msg import PosCmd
            from sensor_msgs.msg import Image, JointState
        except ImportError as exc:
            raise RuntimeError(
                "ROS backend requires rospy, sensor_msgs, and piper_msgs in the robot environment"
            ) from exc
        if observation_timeout <= 0 or not np.isfinite(observation_timeout):
            raise ValueError("observation_timeout must be positive")
        self.rospy, self.joint_factory, self.pose_factory = rospy, JointState, PosCmd
        self.execute, self.timeout = execute, observation_timeout
        self.buffer = ObservationBuffer(max_age=max_age, max_skew=max_skew)
        self.error = None
        self.publishers = {}
        self.subscribers = []
        topics = topics or RosTopics()
        rospy.init_node("piper_ablation_sync", anonymous=True, disable_signals=True)
        try:
            for key, topic in zip(CAMERAS, (topics.head, topics.left_camera, topics.right_camera), strict=True):
                self.subscribers.append(
                    rospy.Subscriber(
                        topic,
                        Image,
                        self._callback,
                        callback_args=key,
                        queue_size=2,
                        buff_size=4 * 1024 * 1024,
                        tcp_nodelay=True,
                    )
                )
            for key, topic in (("left", topics.left_state), ("right", topics.right_state)):
                self.subscribers.append(
                    rospy.Subscriber(
                        topic, JointState, self._callback, callback_args=key, queue_size=10, tcp_nodelay=True
                    )
                )
            # Dry-run only subscribes. It cannot publish an enable or motion message.
            if execute:
                for mode, names, factory in (
                    ("joint", (topics.left_joint_command, topics.right_joint_command), JointState),
                    ("eef", (topics.left_pose_command, topics.right_pose_command), PosCmd),
                ):
                    self.publishers[mode] = [rospy.Publisher(name, factory, queue_size=1) for name in names]
        except BaseException:
            self.close()
            raise

    def _callback(self, message, key):
        try:
            value = image_to_rgb(message) if key in CAMERAS else joint_positions(message)
            self.buffer.push(key, value, stamp=message.header.stamp.to_sec(), received_at=time.monotonic())
        except Exception as exc:
            self.error = exc

    def capture(self):
        deadline = time.monotonic() + self.timeout
        while not self.rospy.is_shutdown():
            if self.error is not None:
                raise RuntimeError(f"Invalid ROS feedback: {self.error}") from self.error
            try:
                return self.buffer.snapshot(now=time.monotonic(), source_now=self.rospy.Time.now().to_sec())
            except ValueError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"No synchronized observation: {exc}") from exc
                with self.buffer.condition:
                    self.buffer.condition.wait(timeout=0.02)
        raise RuntimeError("ROS is shutting down")

    def send(self, mode, values):
        if not self.execute:
            return
        if self.rospy.is_shutdown():
            raise RuntimeError("ROS shut down before command publication")
        publishers = self.publishers[mode]
        if any(publisher.get_num_connections() == 0 for publisher in publishers):
            raise RuntimeError(f"No subscriber on one or both {mode} command topics")
        messages = command_messages(
            mode, values, joint_factory=self.joint_factory, pose_factory=self.pose_factory, stamp=self.rospy.Time.now()
        )
        for publisher, message in zip(publishers, messages, strict=True):
            publisher.publish(message)

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()
        for publishers in self.publishers.values():
            for publisher in publishers:
                publisher.unregister()
