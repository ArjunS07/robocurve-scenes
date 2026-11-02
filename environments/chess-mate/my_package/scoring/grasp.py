"""Decide whether a gripper is still holding a chess piece. No Isaac Sim imports."""

from __future__ import annotations

from my_package.scoring.action_map import open_fraction_from_finger
from my_package.scoring.config import ChessMateConfig
from my_package.scoring.math3d import norm, sub


def gripper_hold_known(
    gripper_positions_xyz: tuple[tuple[float, float, float] | None, ...],
    finger_positions_m: tuple[tuple[float, float] | None, ...],
) -> bool:
    """Holding can only be ruled out when every arm's gripper body and fingers were read."""
    return all(position is not None and fingers is not None for position, fingers in zip(gripper_positions_xyz, finger_positions_m))


def gripper_is_holding(
    cfg: ChessMateConfig,
    piece_position_xyz: tuple[float, float, float],
    gripper_positions_xyz: tuple[tuple[float, float, float] | None, ...],
    finger_positions_m: tuple[tuple[float, float] | None, ...],
) -> bool:
    """True when either arm is closed on the piece.

    A missing gripper body or finger joint does not count as a hold here; the
    caller must also check ``gripper_hold_known`` so a missing frame cannot
    grant release credit. This is a distance and finger-opening proxy, not a
    contact check: closed fingers near a piece count as holding it.
    """
    robot = cfg.robot
    holding = False
    for position, fingers in zip(gripper_positions_xyz, finger_positions_m):
        if position is None or fingers is None:
            continue
        opening = 0.5 * (
            open_fraction_from_finger(fingers[0], robot.finger_closed_m, robot.finger_open_m)
            + open_fraction_from_finger(fingers[1], robot.finger_closed_m, robot.finger_open_m)
        )
        close_enough = norm(sub(piece_position_xyz, position)) <= cfg.scoring.grasp_radius_m
        closed = opening <= cfg.scoring.hold_open_fraction_max
        holding = holding or (close_enough and closed)
    return holding
