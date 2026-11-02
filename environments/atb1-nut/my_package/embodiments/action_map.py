"""Map a 14-D bimanual command onto YAM joint targets. No Isaac Sim imports.

Zero keeps each arm at its home pose and both fingers open. One gripper scalar
per arm is written to both finger joints. Clockwise is not encoded here.
"""

from __future__ import annotations

ACTION_DIM = 14
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
FINGER_JOINTS = ("joint7", "joint8")


class JointResolveError(RuntimeError):
    def __init__(self, missing: list[str], live: list[str]):
        self.missing = tuple(missing)
        self.live = tuple(live)
        super().__init__(
            "YAM joint resolve failed. Missing "
            + ", ".join(missing)
            + ". Live joints: "
            + ", ".join(str(name) for name in live)
        )


def expected_joint_names(layout: str) -> tuple[str, ...]:
    """Joint names one articulation must expose for this layout."""
    if layout == "duplicated_single_arm":
        return ARM_JOINTS + FINGER_JOINTS
    if layout == "combined":
        return tuple(f"left_{name}" for name in ARM_JOINTS + FINGER_JOINTS) + tuple(
            f"right_{name}" for name in ARM_JOINTS + FINGER_JOINTS
        )
    raise ValueError(f"unknown YAM layout: {layout}")


def resolve_joint_names(live_joint_names, required_names) -> tuple[str, ...]:
    live = [str(name) for name in live_joint_names]
    missing = [name for name in required_names if name not in live]
    if missing:
        raise JointResolveError(missing, live)
    return tuple(required_names)


def finger_command(grip: float, finger_open_m: float, finger_closed_m: float) -> float:
    return finger_open_m + float(grip) * (finger_closed_m - finger_open_m)


def _arm_targets(offsets, grip: float, prefix: str, home_arm_rad, finger_open_m: float, finger_closed_m: float):
    targets = {
        f"{prefix}{name}": float(home_arm_rad[index]) + float(offsets[index]) for index, name in enumerate(ARM_JOINTS)
    }
    finger = finger_command(grip, finger_open_m, finger_closed_m)
    for name in FINGER_JOINTS:
        targets[f"{prefix}{name}"] = finger
    return targets


def map_bimanual_action(
    action,
    *,
    home_arm_rad,
    finger_open_m: float,
    finger_closed_m: float,
    layout: str,
) -> dict[str, dict[str, float]]:
    """Return joint targets keyed by articulation (left/right, or robot)."""
    values = tuple(float(item) for item in action)
    if len(values) != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} actions, got {len(values)}")
    if layout == "duplicated_single_arm":
        return {
            "left": _arm_targets(values[0:6], values[6], "", home_arm_rad, finger_open_m, finger_closed_m),
            "right": _arm_targets(values[7:13], values[13], "", home_arm_rad, finger_open_m, finger_closed_m),
        }
    if layout == "combined":
        merged = {}
        merged.update(_arm_targets(values[0:6], values[6], "left_", home_arm_rad, finger_open_m, finger_closed_m))
        merged.update(_arm_targets(values[7:13], values[13], "right_", home_arm_rad, finger_open_m, finger_closed_m))
        return {"robot": merged}
    raise ValueError(f"unknown YAM layout: {layout}")


def initial_joint_positions(home_arm_rad, finger_open_m: float, layout: str, side: str) -> dict[str, float]:
    prefix = "" if layout == "duplicated_single_arm" else f"{side}_"
    if layout == "combined" and side not in {"left", "right"}:
        raise ValueError(f"combined layout needs a left or right side, got {side}")
    positions = {f"{prefix}{name}": float(home_arm_rad[index]) for index, name in enumerate(ARM_JOINTS)}
    for name in FINGER_JOINTS:
        positions[f"{prefix}{name}"] = float(finger_open_m)
    return positions


def actuator_joint_expressions(layout: str, side: str) -> dict[str, list[str]]:
    prefix = "" if layout == "duplicated_single_arm" else f"{side}_"
    return {
        "proximal": [f"{prefix}joint[1-3]"],
        "distal": [f"{prefix}joint[4-6]"],
        "finger": [f"{prefix}joint[7-8]"],
    }
