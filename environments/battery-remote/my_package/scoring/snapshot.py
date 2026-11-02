"""Observation snapshot the scorer reads. No Isaac Sim imports.

The simulator adapter fills one of these from physics. The pure tests build
them directly. Memory fields (ever-lifted, retention timers, latches) travel
inside the snapshot so a later step can see the whole episode.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BatterySnapshot:
    name: str
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    positive_axis_xyz: tuple[float, float, float]
    seating_depth_m: float
    radial_offset_m: float
    polarity_aligned: bool
    gripper_holding: bool
    released: bool
    spring_contact: bool
    off_table: bool
    peak_force_n: float
    ever_lifted: bool = False
    retention_timer_s: float = 0.0


@dataclass(frozen=True)
class SceneSnapshot:
    batteries: tuple[BatterySnapshot, ...]
    remote_position_xyz: tuple[float, float, float]
    remote_rotation_xyzw: tuple[float, float, float, float]
    remote_displacement_m: float
    remote_yaw_error_deg: float
    damage_force_n: float
    ever_off_table: bool = False
    ever_remote_moved: bool = False
    ever_damage: bool = False


def battery_by_name(snapshot: SceneSnapshot, name: str) -> BatterySnapshot:
    for battery in snapshot.batteries:
        if battery.name == name:
            return battery
    raise KeyError(name)


def update_battery(snapshot: SceneSnapshot, name: str, **changes) -> SceneSnapshot:
    found = False
    batteries = []
    for battery in snapshot.batteries:
        if battery.name == name:
            batteries.append(replace(battery, **changes))
            found = True
        else:
            batteries.append(battery)
    if not found:
        raise KeyError(name)
    return replace(snapshot, batteries=tuple(batteries))


def clear_episode_state(snapshot: SceneSnapshot) -> SceneSnapshot:
    """Drop ever-lifted, score latches, and retention timers. Poses stay as they are.

    The physics reset is separate: it restores root poses and joint targets and
    does not weld anything. This only clears the task's episode memory.
    """
    batteries = tuple(
        replace(battery, ever_lifted=False, retention_timer_s=0.0) for battery in snapshot.batteries
    )
    return replace(
        snapshot,
        batteries=batteries,
        ever_off_table=False,
        ever_remote_moved=False,
        ever_damage=False,
    )
