# Adapted from kai0 (Apache-2.0); see NOTICE.md and LICENSE.kai0.
"""Native Piper X feedback, commands, and RealSense RGB capture without ROS.

Hardware dependencies are imported only when the backend is constructed.
All poses refer to each arm's own-base, zero-offset flange, in metres/radians.
"""

import threading
import time
from contextlib import ExitStack

import numpy as np
from scipy.spatial.transform import Rotation

from starVLA.piper import geometry as g
from starVLA.piper.adapter import CAMERAS, Observation, PiperAdapter, finite_array


class PiperXAdapter(PiperAdapter):
    pose_frame = "piper_x_flange"

    def __init__(self, rung, *, prompt, gripper_max=0.10):
        if rung not in ("s0", "s2", "s2_pair"):
            raise ValueError("Native Piper X supports s0, s2 and s2_pair")
        super().__init__(rung, prompt=prompt, gripper_max=gripper_max)

    def validate_metadata(self, metadata):
        super().validate_metadata(metadata)
        source = "observation.state" if self.rung == "s0" else "observation.qpos_ee"
        expected = {
            "robot_model": "piper_x",
            "training_pose_frame": self.pose_frame,
            "training_state_source": source,
            "target_source": f"future_{source}",
            "action_start_offset": 1,
        }
        if self.rung == "s2_pair":
            expected["pair_base_spacing_m"] = float(-g.RIGHT_BASE[1, 3])
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Native Piper X server {key} mismatch: expected {value!r}")
        if metadata.get("pose_alignment") is not None:
            raise ValueError("Native Piper X deployment cannot use Piper-aligned checkpoints")

    def _build_state(self, observation, joints):
        if self.rung == "s0":
            return joints.copy()
        poses = finite_array(observation.flange_poses, (2, 4, 4))
        return g.states_from_poses(poses[0:1], poses[1:2], joints[None, [6, 13]], self.rung)[0]

    def _reference_euler(self, request, arm):
        start = arm * 7 + 3
        return Rotation.from_rotvec(request.state[start : start + 3].copy()).as_euler("xyz")


class RealSenseImages:
    """Keep the latest RGB image from each explicitly selected camera."""

    def __init__(self, serials):
        import pyrealsense2 as rs

        if set(serials) != set(CAMERAS) or len(set(serials.values())) != 3 or not all(serials.values()):
            raise ValueError(f"Supply three distinct RealSense serials keyed by {CAMERAS}")
        self.lock = threading.Lock()
        self.latest = {}
        self.resources = ExitStack()
        try:
            for key in CAMERAS:
                pipeline, config = rs.pipeline(), rs.config()
                config.enable_device(str(serials[key]))
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                profile = config.resolve(rs.pipeline_wrapper(pipeline))
                for sensor in profile.get_device().query_sensors():
                    if sensor.supports(rs.option.global_time_enabled):
                        sensor.set_option(rs.option.global_time_enabled, 1)

                def receive(frame, name=key):
                    if not frame.is_frameset():
                        return
                    color = frame.as_frameset().get_color_frame()
                    if not color or color.get_frame_timestamp_domain() == rs.timestamp_domain.hardware_clock:
                        return
                    # Global/system timestamps are milliseconds since the Unix epoch.
                    stamp = color.get_timestamp() / 1000
                    received = time.monotonic() - (time.time() - stamp)
                    image = np.asanyarray(color.get_data()).copy()
                    with self.lock:
                        self.latest[name] = (image, received)

                pipeline.start(config, receive)
                self.resources.callback(pipeline.stop)
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        with self.lock:
            return {key: (value.copy(), stamp) for key, (value, stamp) in self.latest.items()}

    def close(self):
        self.resources.close()


class PiperXBackend:
    """Dual SocketCAN Piper X arms and three RGB cameras; read-only by default."""

    def __init__(
        self,
        *,
        left_can,
        right_can,
        camera_serials,
        firmware="default",
        execute=False,
        speed_percent=30,
        gripper_force=1.0,
        gripper_max=0.10,
        max_age=0.5,
        max_skew=0.04,
        observation_timeout=5.0,
    ):
        from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config

        if left_can == right_can or not left_can or not right_can:
            raise ValueError("Use distinct CAN interfaces for the left and right arms")
        if not 1 <= speed_percent <= 100 or not 0 <= gripper_force <= 3:
            raise ValueError("Speed must be 1..100 percent and gripper force 0..3 N")
        if not all(np.isfinite(v) and v > 0 for v in (max_age, max_skew, observation_timeout, gripper_max)):
            raise ValueError("Observation limits and gripper maximum must be positive and finite")
        self.execute, self.ready = execute, False
        self.speed_percent, self.gripper_force, self.gripper_max = speed_percent, gripper_force, gripper_max
        self.max_age, self.max_skew, self.timeout = max_age, max_skew, observation_timeout
        self.resources = ExitStack()
        self.arms, self.grippers = [], []
        try:
            for channel in (left_can, right_can):
                cfg = create_agx_arm_config(
                    robot=ArmModel.PIPER_X, firmeware_version=firmware, channel=channel, interface="socketcan"
                )
                arm = AgxArmFactory.create_arm(cfg)
                self.resources.callback(arm.disconnect)
                self.grippers.append(arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER))
                arm.connect()
                self.arms.append(arm)
            self.cameras = RealSenseImages(camera_serials)
            self.resources.callback(self.cameras.close)
        except BaseException:
            self.close()
            raise

    def _feedback(self):
        joints, poses, stamps = [], [], []
        for arm, gripper in zip(self.arms, self.grippers, strict=True):
            if not arm.is_ok() or arm.has_comm_error():
                return None
            joint, pose, grip = arm.get_joint_angles(), arm.get_flange_pose(), gripper.get_gripper_status()
            if any(message is None for message in (joint, pose, grip)):
                return None
            if grip.msg.mode != "width":
                raise ValueError("Recorded grippers use metres; SDK feedback must be in width mode")
            joints.extend([*finite_array(joint.msg, (6,)), float(grip.msg.value)])
            poses.append(finite_array(pose.msg, (6,)))
            # SocketCAN receive timestamps used by pyAgxArm are Unix seconds.
            stamps.extend(float(message.timestamp) for message in (joint, pose, grip))
        now, wall = time.monotonic(), time.time()
        return finite_array(joints, (14,)), np.asarray(poses), now - (wall - np.asarray(stamps))

    def capture(self):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            feedback, images = self._feedback(), self.cameras.snapshot()
            if feedback is not None and set(images) == set(CAMERAS):
                joints, euler_poses, stamps = feedback
                stamps = np.r_[stamps, [images[name][1] for name in CAMERAS]]
                now = time.monotonic()
                if (
                    np.isfinite(stamps).all()
                    and now - stamps.max() >= -0.001
                    and now - stamps.min() <= self.max_age
                    and np.ptp(stamps) <= self.max_skew
                ):
                    poses = np.tile(np.eye(4), (2, 1, 1))
                    poses[:, :3, 3] = euler_poses[:, :3]
                    poses[:, :3, :3] = Rotation.from_euler("xyz", euler_poses[:, 3:]).as_matrix()
                    return Observation(
                        joints=joints,
                        images={name: images[name][0] for name in CAMERAS},
                        received_at=float(stamps.min()),
                        captured_at=float(stamps.min()),
                        flange_poses=poses,
                    )
            time.sleep(0.002)
        raise TimeoutError("No fresh, synchronized Piper X feedback and RGB images")

    def start_execution(self):
        if not self.execute:
            return
        # Require a complete observation before enabling; never send a home target.
        self.capture()
        deadline = time.monotonic() + self.timeout
        while True:
            enabled = [arm.enable() for arm in self.arms]
            if all(enabled):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Piper X enable timed out")
            time.sleep(0.01)
        for arm in self.arms:
            arm.set_speed_percent(self.speed_percent)
        self.ready = True

    def send(self, mode, values):
        values = finite_array(values, (14,))
        if mode not in ("joint", "eef"):
            raise ValueError("Command mode must be joint or eef")
        if np.any(values[[6, 13]] < 0) or np.any(values[[6, 13]] > self.gripper_max):
            raise ValueError("Gripper command exceeds the configured physical range")
        if not self.execute:
            return
        if not self.ready:
            raise RuntimeError("Call start_execution before sending motion")
        # Check both arms before sending either command.
        for arm in self.arms:
            status = arm.get_arm_status()
            if (
                not arm.is_ok()
                or arm.has_comm_error()
                or status is None
                or not 0 <= time.time() - status.timestamp <= self.max_age
                or status.msg.arm_status != 0
            ):
                raise RuntimeError("Piper X communication/status is not ready for motion")
        for arm, gripper, start in zip(self.arms, self.grippers, (0, 7), strict=True):
            target = values[start : start + 6].copy()
            if mode == "joint":
                arm.move_j(target.tolist())
            else:
                # RTC retains continuous Euler branches; the SDK expects canonical ranges.
                target[3:] = Rotation.from_euler("xyz", target[3:]).as_euler("xyz")
                arm.move_p(target.tolist())
            gripper.move_gripper_m(value=float(values[start + 6]), force=self.gripper_force)

    def close(self):
        self.ready = False
        self.resources.close()
