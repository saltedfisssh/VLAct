# Adapted from kai0 (Apache-2.0); see NOTICE.md and LICENSE.kai0.
"""CPU-only mapping between physical Piper flanges and a Piper-X S2 policy.

For each arm, T_X = A @ T_P @ C. A maps the base/workspace convention;
C maps the tool convention, including any measured TCP translation.
This module does not change joints, images, gripper units, or normalization.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from starVLA.piper import geometry as g

CONVENTION = "T_X = A @ T_P @ C"
POLICY_CONFIGS = ("pi05_piper_x_eef_multitask", "pi05_piper_x_eef_multitask_train_rtc")
NOMINAL_CALIBRATION_ID = "piper_to_piper_x_nominal_axes_v1"


def _transforms(value, name):
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected two finite numeric 4x4 transforms") from exc
    if result.shape != (2, 4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected two finite 4x4 transforms, ordered left then right")
    rotation = result[:, :3, :3]
    if (
        not np.allclose(result[:, 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-6, rtol=0)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name}: expected proper rigid SE(3) transforms, without scale or reflection")
    # Back with immutable bytes so a caller cannot re-enable array writes while
    # an asynchronous request still relies on this calibration.
    return np.frombuffer(result.tobytes(), dtype=np.float64).reshape(2, 4, 4)


@dataclass(frozen=True, eq=False)
class PiperXTransfer:
    calibration_id: str
    base_transforms: np.ndarray
    tool_transforms: np.ndarray
    nominal: bool = False

    def __post_init__(self):
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise ValueError("A nonempty calibration_id is required")
        if not isinstance(self.nominal, bool):
            raise ValueError("nominal must be a boolean")
        object.__setattr__(self, "base_transforms", _transforms(self.base_transforms, "A"))
        object.__setattr__(self, "tool_transforms", _transforms(self.tool_transforms, "C"))

    @classmethod
    def nominal_axes(cls):
        """Official mounting-axis candidate; identity bases and no TCP translation.

        The Piper gripper mount has zero yaw and Piper-X has +90 degrees.
        Thus C = M_P @ inverse(M_X) has local-z -90 degrees. This preset is
        not a measured calibration of the installed cameras, bases, or fingers.
        """
        bases = np.tile(np.eye(4), (2, 1, 1))
        tools = bases.copy()
        tools[:, :3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        return cls(NOMINAL_CALIBRATION_ID, bases, tools, nominal=True)

    @classmethod
    def from_file(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data):
        keys = {"version", "convention", "calibration_id", "nominal", "left", "right"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError(f"Calibration must contain exactly {sorted(keys)}")
        if type(data["version"]) is not int or data["version"] != 1 or data["convention"] != CONVENTION:
            raise ValueError(f"Expected calibration version 1 and convention {CONVENTION!r}")
        for arm in ("left", "right"):
            if not isinstance(data[arm], dict) or set(data[arm]) != {"A", "C"}:
                raise ValueError(f"{arm}: expected exactly A and C transforms")
        return cls(
            data["calibration_id"],
            [data[arm]["A"] for arm in ("left", "right")],
            [data[arm]["C"] for arm in ("left", "right")],
            nominal=data["nominal"],
        )

    def to_dict(self):
        return {
            "version": 1,
            "convention": CONVENTION,
            "calibration_id": self.calibration_id,
            "nominal": self.nominal,
            **{
                arm: {"A": self.base_transforms[index].tolist(), "C": self.tool_transforms[index].tolist()}
                for index, arm in enumerate(("left", "right"))
            },
        }

    def to_policy(self, poses, arm):
        return self.base_transforms[arm] @ poses @ self.tool_transforms[arm]

    def to_physical(self, poses, arm):
        return g.inverse(self.base_transforms[arm]) @ poses @ g.inverse(self.tool_transforms[arm])

    def validate_metadata(self, metadata):
        if (
            metadata.get("config_name") not in POLICY_CONFIGS
            or metadata.get("training_pose_source") != "observation.qpos_ee"
        ):
            raise ValueError("Piper-X transfer requires a Piper-X recorded-EEF multitask policy server")
