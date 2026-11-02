"""Map a 14-dim bimanual action onto 16 joint targets. No Isaac Sim imports.

Physical joints are not the action dimension. Each arm contributes six arm
controls and one gripper control. The gripper control is written to both
prismatic finger joints. The source URDF uses the same coordinate on both
fingers: 0 is closed and a negative value opens them. The station URDF flips
that sign; ``finger_open_m`` / ``finger_closed_m`` carry whichever convention
the loaded USDA actually has.
"""

from __future__ import annotations

from dataclasses import dataclass

from my_package.scoring.config import BatteryInsertionConfig
from my_package.scoring.math3d import clamp

ACTION_DIM = 14


@dataclass(frozen=True)
class ArmBinding:
    side: str
    asset_name: str
    arm_joints: tuple[str, ...]
    finger_joints: tuple[str, ...]
    gripper_body: str


def arm_bindings(cfg: BatteryInsertionConfig) -> tuple[ArmBinding, ArmBinding]:
    layout = cfg.robot.arm_layout
    if layout == "duplicated_single_arm":
        # Confirmed names are the single-arm URDF (joint1..joint8). The USDA is
        # spawned twice, once per arm. joint7 and joint8 are the two fingers.
        left = ArmBinding(
            side="left",
            asset_name="robot",
            arm_joints=tuple(f"joint{i}" for i in range(1, 7)),
            finger_joints=("joint7", "joint8"),
            gripper_body="gripper",
        )
        right = ArmBinding(
            side="right",
            asset_name="robot_right",
            arm_joints=tuple(f"joint{i}" for i in range(1, 7)),
            finger_joints=("joint7", "joint8"),
            gripper_body="gripper",
        )
        return left, right
    if layout == "combined":
        # Unconfirmed alternate: one articulation with the station prefixes.
        # Switching is YAML-only. Finger sign may also need to flip.
        left = ArmBinding(
            side="left",
            asset_name="robot",
            arm_joints=tuple(f"left_joint{i}" for i in range(1, 7)),
            finger_joints=("left_joint7", "left_joint8"),
            gripper_body="left_gripper",
        )
        right = ArmBinding(
            side="right",
            asset_name="robot",
            arm_joints=tuple(f"right_joint{i}" for i in range(1, 7)),
            finger_joints=("right_joint7", "right_joint8"),
            gripper_body="right_gripper",
        )
        return left, right
    raise ValueError(f"unknown arm_layout {layout!r}")


def finger_position(open_fraction: float, finger_closed_m: float, finger_open_m: float) -> float:
    fraction = clamp(open_fraction, 0.0, 1.0)
    return finger_closed_m + (finger_open_m - finger_closed_m) * fraction


def open_fraction_from_finger(position_m: float, finger_closed_m: float, finger_open_m: float) -> float:
    span = finger_open_m - finger_closed_m
    if abs(span) < 1e-12:
        raise ValueError("finger open and closed positions must differ")
    return clamp((position_m - finger_closed_m) / span, 0.0, 1.0)


def map_bimanual_action(
    action: tuple[float, ...] | list[float],
    bindings: tuple[ArmBinding, ArmBinding],
    home_arm: tuple[float, ...],
    finger_closed_m: float,
    finger_open_m: float,
    home_open_fraction: float,
    arm_scale: float,
    gripper_scale: float,
) -> dict[tuple[str, str], float]:
    """Return absolute joint targets keyed by ``(asset_name, joint_name)``.

    A zero action holds the home arm and the home gripper opening. It does not
    integrate, so ``zero_action`` cannot collapse the arm into the table.
    """
    if len(action) != ACTION_DIM:
        raise ValueError(f"bimanual action must have {ACTION_DIM} values, got {len(action)}")
    if len(home_arm) != 6:
        raise ValueError("home_arm must have six revolute targets")
    targets: dict[tuple[str, str], float] = {}
    for index, binding in enumerate(bindings):
        base = index * 7
        arm_delta = action[base : base + 6]
        gripper_delta = float(action[base + 6])
        for joint, home, delta in zip(binding.arm_joints, home_arm, arm_delta):
            targets[(binding.asset_name, joint)] = float(home) + arm_scale * float(delta)
        opening = clamp(home_open_fraction + gripper_scale * gripper_delta, 0.0, 1.0)
        finger = finger_position(opening, finger_closed_m, finger_open_m)
        for joint in binding.finger_joints:
            targets[(binding.asset_name, joint)] = finger
    if len(targets) != 16:
        raise RuntimeError(f"expected 16 joint targets, produced {len(targets)}")
    return targets


def map_config_action(cfg: BatteryInsertionConfig, action: tuple[float, ...] | list[float]) -> dict[tuple[str, str], float]:
    robot = cfg.robot
    return map_bimanual_action(
        action,
        arm_bindings(cfg),
        robot.home_arm_rad,
        robot.finger_closed_m,
        robot.finger_open_m,
        robot.home_open_fraction,
        robot.arm_action_scale,
        robot.gripper_action_scale,
    )
