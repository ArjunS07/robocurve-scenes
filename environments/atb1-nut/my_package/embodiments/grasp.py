"""Decide whether a gripper is holding the nut. No Isaac Sim imports."""

from __future__ import annotations


def finger_open_fraction(finger_pos_m: float, finger_open_m: float, finger_closed_m: float) -> float:
    span = finger_open_m - finger_closed_m
    if span == 0.0:
        raise ValueError("finger open and closed positions must differ")
    return (float(finger_pos_m) - finger_closed_m) / span


def mean_open_fraction(finger_positions, finger_open_m: float, finger_closed_m: float) -> float | None:
    if not finger_positions:
        return None
    fractions = [
        finger_open_fraction(position, finger_open_m, finger_closed_m) for position in finger_positions
    ]
    return sum(fractions) / len(fractions)


def is_holding(
    *,
    distance_m: float | None,
    open_fraction: float | None,
    radius_m: float,
    hold_open_fraction_max: float,
    hold_open_fraction_min: float = 0.0,
) -> bool:
    """A missing distance or finger reading is not a hold.

    Fingers shut on nothing read near zero open, so they fall below the min and
    do not count as holding.
    """
    if distance_m is None or open_fraction is None:
        return False
    fraction = float(open_fraction)
    return float(distance_m) <= radius_m and hold_open_fraction_min < fraction <= hold_open_fraction_max


def any_arm_holding(
    arms: list[dict | None],
    *,
    radius_m: float,
    hold_open_fraction_max: float,
    hold_open_fraction_min: float = 0.0,
) -> bool:
    return any(
        arm is not None
        and is_holding(
            distance_m=arm.get("distance_m"),
            open_fraction=arm.get("open_fraction"),
            radius_m=radius_m,
            hold_open_fraction_max=hold_open_fraction_max,
            hold_open_fraction_min=hold_open_fraction_min,
        )
        for arm in arms
    )


def hold_known(arms: list[dict | None]) -> bool:
    """Holding can only be ruled out when every arm's gripper body and fingers were read."""
    return all(
        arm is not None and arm.get("distance_m") is not None and arm.get("open_fraction") is not None
        for arm in arms
    )
