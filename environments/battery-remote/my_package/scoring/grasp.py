"""Decide whether a gripper is still holding a battery. No Isaac Sim imports."""

from __future__ import annotations

from my_package.scoring.action_map import open_fraction_from_finger
from my_package.scoring.config import BatteryInsertionConfig
from my_package.scoring.math3d import norm, sub


def gripper_is_holding(
    cfg: BatteryInsertionConfig,
    battery_position_xyz: tuple[float, float, float],
    gripper_positions_xyz: tuple[tuple[float, float, float] | None, ...],
    finger_positions_m: tuple[tuple[float, float] | None, ...],
) -> bool | None:
    """GEOMETRIC PROXY: True when either arm's gripper body is near and its fingers are shut.

    This is not a contact test. If any arm cannot be observed (missing gripper
    body or finger joints) and no observed arm is holding, the result is None:
    the hold is unknown, and the scorer treats unknown as not released so a
    naming mismatch cannot grant insertion credit.
    """
    robot = cfg.robot
    holding = False
    unknown = False
    for position, fingers in zip(gripper_positions_xyz, finger_positions_m):
        if position is None or fingers is None:
            unknown = True
            continue
        opening = 0.5 * (
            open_fraction_from_finger(fingers[0], robot.finger_closed_m, robot.finger_open_m)
            + open_fraction_from_finger(fingers[1], robot.finger_closed_m, robot.finger_open_m)
        )
        close_enough = norm(sub(battery_position_xyz, position)) <= cfg.scoring.grasp_radius_m
        closed = opening <= cfg.scoring.hold_open_fraction_max
        holding = holding or (close_enough and closed)
    if holding:
        return True
    return None if unknown else False
