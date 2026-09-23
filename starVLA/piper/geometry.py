# Adapted from kai0 (Apache-2.0); see NOTICE.md and LICENSE.kai0.
"""SI-unit Piper flange kinematics and invertible S0--S4 representations.

MDH constants match pyAgxArm's ``piper`` preset (including the two-degree
offset). No robot SDK or OpenDM installation is needed at runtime.
"""

import hashlib

import numpy as np
from scipy.spatial.transform import Rotation

RUNGS = ("s0", "s1", "s2", "s2_pair", "s3a", "s3", "s4")
MDH = (
    (0.123, 0, 0, 0),
    (0, 0, -np.pi / 2, -3.0058060377846343),
    (0, 0.28503, 0, -1.793849405199772),
    (0.25075, -0.02198, np.pi / 2, 0),
    (0, 0, -np.pi / 2, 0),
    (0.091, 0, np.pi / 2, 0),
)
RIGHT_BASE = np.eye(4)
RIGHT_BASE[1, 3] = -0.60


def action_dim(rung):
    if rung not in RUNGS:
        raise ValueError(f"Unknown representation: {rung}")
    return 14 if rung in ("s0", "s1") else 20


def fk(joints):
    q = np.asarray(joints, dtype=np.float64)
    if q.shape[-1] != 6 or not np.isfinite(q).all():
        raise ValueError("Expected finite six-dimensional joint angles")
    result = np.broadcast_to(np.eye(4), (*q.shape[:-1], 4, 4)).copy()
    for i, (d, a, alpha, offset) in enumerate(MDH):
        c, s = np.cos(q[..., i] + offset), np.sin(q[..., i] + offset)
        ca, sa = np.cos(alpha), np.sin(alpha)
        link = np.zeros_like(result)
        link[..., 0, 0], link[..., 0, 1], link[..., 0, 3] = c, -s, a
        link[..., 1, 0], link[..., 1, 1], link[..., 1, 2], link[..., 1, 3] = s * ca, c * ca, -sa, -sa * d
        link[..., 2, 0], link[..., 2, 1], link[..., 2, 2], link[..., 2, 3] = s * sa, c * sa, ca, ca * d
        link[..., 3, 3] = 1
        result = result @ link
    return result


def inverse(pose):
    pose = np.asarray(pose)
    result = np.zeros_like(pose)
    result[..., :3, :3] = np.swapaxes(pose[..., :3, :3], -1, -2)
    result[..., :3, 3] = -np.einsum("...ij,...j->...i", result[..., :3, :3], pose[..., :3, 3])
    result[..., 3, 3] = 1
    return result


def rot6d(matrix):
    matrix = np.asarray(matrix)
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def from_rot6d(value):
    """Project predicted row vectors onto SO(3), including degenerate outputs."""
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite rotation prediction")
    a, b = value[..., :3], value[..., 3:6]
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    a = np.where(a_norm > 1e-10, a / np.maximum(a_norm, 1e-10), [1.0, 0.0, 0.0])
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    alternate = np.eye(3)[np.argmin(np.abs(a), axis=-1)]
    alternate -= np.sum(alternate * a, axis=-1, keepdims=True) * a
    alternate /= np.linalg.norm(alternate, axis=-1, keepdims=True)
    b = np.where(b_norm > 1e-10, b / np.maximum(b_norm, 1e-10), alternate)
    return np.stack([a, b, np.cross(a, b)], axis=-2)


def unwrap_rotations(matrices):
    """Use a continuous episode chart instead of the principal pi branch."""
    vectors = Rotation.from_matrix(matrices).as_rotvec()
    for t in range(1, len(vectors)):
        theta = np.linalg.norm(vectors[t])
        if theta < 1e-10:
            continue
        axis = vectors[t] / theta
        winding = round((np.dot(vectors[t - 1], axis) - theta) / (2 * np.pi))
        vectors[t] += 2 * np.pi * winding * axis
    return vectors


def pose_from_vector(vector):
    vector = np.asarray(vector)
    result = np.broadcast_to(np.eye(4), (*vector.shape[:-1], 4, 4)).copy()
    result[..., :3, :3] = (
        Rotation.from_rotvec(vector[..., 3:6].reshape(-1, 3).copy()).as_matrix().reshape(*vector.shape[:-1], 3, 3)
    )
    result[..., :3, 3] = vector[..., :3]
    return result


def episode_frame(episode_id, seed=0, xy_range=0.5):
    if not np.isfinite(xy_range) or xy_range < 0:
        raise ValueError("xy_range must be finite and nonnegative")
    # Match OpenDM's episode filename based seed, independent of worker ordering.
    digest = hashlib.sha256(f"{seed}:episode_{int(episode_id):06d}.jsonl".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:16], "little"))
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi)).as_matrix()
    result[:2, 3] = rng.uniform(-xy_range, xy_range, 2)
    return result


def states_from_poses(left, right, grippers, rung):
    """S2/S2_pair states from own-base flange poses, without any robot-specific FK.

    Both arms are assumed parallel and forward facing, with RIGHT_BASE relating
    the right base to the left base. Callers must supply poses that already use
    one common flange/tool convention for every source robot.
    """
    if rung not in ("s2", "s2_pair"):
        raise ValueError(f"Pose-based states require the s2 or s2_pair representation, got {rung!r}")
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    grippers = np.asarray(grippers, dtype=np.float64)
    if left.ndim != 3 or left.shape[1:] != (4, 4) or right.shape != left.shape or not len(left):
        raise ValueError("Expected matching nonempty episode pose sequences with shape (T, 4, 4)")
    if grippers.shape != (len(left), 2):
        raise ValueError("Expected two gripper openings per frame, ordered left then right")
    if not (np.isfinite(left).all() and np.isfinite(right).all() and np.isfinite(grippers).all()):
        raise ValueError("Expected finite poses and gripper openings")
    blocks = []
    for pose, index in ((left, 0), (right, 1)):
        blocks.extend([pose[:, :3, 3], unwrap_rotations(pose[:, :3, :3]), grippers[:, index : index + 1]])
    if rung == "s2_pair":
        pair = inverse(left) @ RIGHT_BASE @ right
        blocks.extend([pair[:, :3, 3], rot6d(pair[:, :3, :3])])
    return np.concatenate(blocks, axis=-1)


def build_states(joints, rung, episode_transform=None):
    action_dim(rung)
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 14 or not np.isfinite(joints).all():
        raise ValueError("Expected finite episode joint states with shape (T, 14)")
    if rung == "s0":
        return joints.copy()
    left, right = fk(joints[:, :6]), fk(joints[:, 7:13])
    if rung in ("s2", "s2_pair"):
        return states_from_poses(left, right, joints[:, [6, 13]], rung)
    # The pair block always relates the two own-base flanges, before rebasing.
    pair = inverse(left) @ RIGHT_BASE @ right if rung == "s3" else None
    if rung in ("s3a", "s3", "s4"):
        right = RIGHT_BASE @ right
    if rung == "s4":
        if episode_transform is None:
            raise ValueError("S4 requires the fixed episode transform")
        left, right = episode_transform @ left, episode_transform @ right
    blocks = []
    for pose, grip in ((left, 6), (right, 13)):
        blocks.extend([pose[:, :3, 3], unwrap_rotations(pose[:, :3, :3]), joints[:, grip : grip + 1]])
    if rung == "s3":
        blocks.extend([pair[:, :3, 3], rot6d(pair[:, :3, :3])])
    return np.concatenate(blocks, axis=-1)


def encode_actions(state, future, rung):
    """All future steps share the current observation anchor; grippers stay absolute."""
    action_dim(rung)
    state, future = np.asarray(state), np.asarray(future)
    if rung in ("s0", "s1"):
        result = future[..., :14].copy()
        for start in (0, 7):
            result[..., start : start + 6] -= state[..., start : start + 6]
        return result
    blocks = []
    for start in (0, 7):
        delta = inverse(pose_from_vector(state[..., start : start + 6])) @ pose_from_vector(
            future[..., start : start + 6]
        )
        blocks.extend([delta[..., :3, 3], rot6d(delta[..., :3, :3]), future[..., start + 6 : start + 7]])
    return np.concatenate(blocks, axis=-1)


def decode_actions(state, actions, rung, episode_transform=None):
    """Decode to each physical arm's own base flange poses and grippers."""
    width = action_dim(rung)
    actions = np.asarray(actions)[..., :width]
    state = np.asarray(state)
    poses, grippers = [], []
    for arm, start in enumerate((0, 7)):
        if rung in ("s0", "s1"):
            vector = actions[..., start : start + 6] + state[..., start : start + 6]
            pose = fk(vector) if rung == "s0" else pose_from_vector(vector)
            grip = actions[..., start + 6]
        else:
            block = actions[..., arm * 10 : (arm + 1) * 10]
            delta = np.broadcast_to(np.eye(4), (*block.shape[:-1], 4, 4)).copy()
            delta[..., :3, :3], delta[..., :3, 3] = from_rot6d(block[..., 3:9]), block[..., :3]
            pose = pose_from_vector(state[..., start : start + 6]) @ delta
            grip = block[..., 9]
        if rung == "s4":
            if episode_transform is None:
                raise ValueError("S4 decoding requires the episode transform")
            pose = inverse(episode_transform) @ pose
        if arm == 1 and rung in ("s3a", "s3", "s4"):
            pose = inverse(RIGHT_BASE) @ pose
        poses.append(pose)
        grippers.append(grip)
    return np.stack(poses, axis=-3), np.stack(grippers, axis=-1)
