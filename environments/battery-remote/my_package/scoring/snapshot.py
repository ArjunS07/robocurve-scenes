"""Observation snapshot the scorer reads. No Isaac Sim imports.

The simulator adapter fills one of these from physics. The pure tests build
them directly. Memory fields (ever-lifted, per-slot retention, latches) travel
inside the snapshot so a later step can see the whole episode.

Cells are interchangeable. Each cell carries one ``SlotFit`` per active slot,
and the scorer decides which cell, if any, occupies each slot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SlotFit:
    """One cell measured against one slot."""

    slot: str
    axial_offset_m: float
    radial_offset_m: float
    polarity_aligned: bool
    # None means the spring plunger's joint could not be read. That is never contact.
    spring_contact: bool | None


@dataclass(frozen=True)
class SlotRetention:
    slot: str
    cell: str | None
    timer_s: float


@dataclass(frozen=True)
class BatterySnapshot:
    name: str
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    positive_axis_xyz: tuple[float, float, float]
    fits: tuple[SlotFit, ...]
    # None means the hold could not be observed (missing gripper names). That is not a release.
    gripper_holding: bool | None
    released: bool
    off_table: bool
    peak_force_n: float
    ever_lifted: bool = False


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
    retention: tuple[SlotRetention, ...] = ()


def battery_by_name(snapshot: SceneSnapshot, name: str) -> BatterySnapshot:
    for battery in snapshot.batteries:
        if battery.name == name:
            return battery
    raise KeyError(name)


def fit_for(battery: BatterySnapshot, slot: str) -> SlotFit | None:
    for fit in battery.fits:
        if fit.slot == slot:
            return fit
    return None


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
    batteries = tuple(replace(battery, ever_lifted=False) for battery in snapshot.batteries)
    return replace(
        snapshot,
        batteries=batteries,
        ever_off_table=False,
        ever_remote_moved=False,
        ever_damage=False,
        retention=(),
    )
