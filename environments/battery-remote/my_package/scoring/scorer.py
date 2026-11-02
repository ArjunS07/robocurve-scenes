"""Score one control step from an observation snapshot. No Isaac Sim imports.

Success is not "the battery center is inside the remote's bounding box".
The two cells are interchangeable. A slot counts as filled only when one cell,
for the whole retention window, is:

- within ``axial_tolerance_m`` of the seat along the slot, on either side
- within ``seat_radial_max_m`` of the slot's seated axis
- pointing its positive terminal the way that slot requires
- touching that slot's spring plunger (geometric test against the live joint pose;
  an unreadable plunger joint is unknown, and unknown is never contact)
- released, with the hold actually observed as false

Each cell fills at most one slot and each slot holds at most one cell.

Hard failures latch for the episode and force the score to 0. ``never_lifted``
also forces 0, but it is not an early termination: a cell may still be lifted
later. Visible damage is not modeled. ``damage_force_proxy_n`` is only a
contact-force threshold.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

from my_package.scoring.config import BatteryInsertionConfig, SlotSpec, active_slots
from my_package.scoring.math3d import angle_deg, rotate
from my_package.scoring.snapshot import BatterySnapshot, SceneSnapshot, SlotFit, SlotRetention, fit_for


@dataclass(frozen=True)
class ScoreReport:
    score: int
    hard_failure: bool
    hard_reasons: tuple[str, ...]
    inserted: dict[str, bool]
    occupant: dict[str, str | None]
    ever_lifted: dict[str, bool]
    retention_timer_s: dict[str, float]
    hold_unknown: tuple[str, ...]
    spring_unknown: tuple[str, ...]
    ever_off_table: bool
    ever_remote_moved: bool
    ever_damage: bool


def battery_lowest_z(cfg: BatteryInsertionConfig, battery: BatterySnapshot) -> float:
    """Lowest point of a flat-capped cylinder at any orientation."""
    axis = rotate(battery.rotation_xyzw, (0.0, 0.0, 1.0))
    vertical = min(1.0, abs(axis[2]))
    radius = cfg.battery.diameter_m / 2.0
    half_length = cfg.battery.length_m / 2.0
    extent = radius * math.sqrt(max(0.0, 1.0 - vertical * vertical)) + half_length * vertical
    return battery.position_xyz[2] - extent


def battery_is_lifted(cfg: BatteryInsertionConfig, battery: BatterySnapshot) -> bool:
    """The lowest point of the cell is clear of the table by the configured gap."""
    return battery_lowest_z(cfg, battery) > cfg.table.top_z_m + cfg.scoring.lift_clearance_m


def battery_pose_off_table(cfg: BatteryInsertionConfig, position_xyz: tuple[float, float, float]) -> bool:
    center_x, center_y = cfg.table.center_xy
    half_x, half_y = cfg.table.half_extents_xy
    outside_xy = (
        position_xyz[0] < center_x - half_x
        or position_xyz[0] > center_x + half_x
        or position_xyz[1] < center_y - half_y
        or position_xyz[1] > center_y + half_y
    )
    fell = position_xyz[2] < cfg.table.top_z_m - cfg.table.fall_below_m
    return outside_xy or fell


def polarity_matches(cfg: BatteryInsertionConfig, snapshot: SceneSnapshot, battery: BatterySnapshot, slot: SlotSpec, fit: SlotFit) -> bool:
    """Both the geometric axis and the measured flag have to agree.

    A true flag cannot grant credit for a reversed axis. A false flag fails
    even when the axis itself is right.
    """
    required = rotate(snapshot.remote_rotation_xyzw, slot.required_axis_xyz)
    angle = angle_deg(battery.positive_axis_xyz, required)
    return angle <= cfg.scoring.polarity_angle_deg and fit.polarity_aligned


def is_released(battery: BatterySnapshot) -> bool:
    """Released only when the hold was observed and is false. Unknown is not released."""
    return battery.released and battery.gripper_holding is False


def is_fully_seated(cfg: BatteryInsertionConfig, snapshot: SceneSnapshot, battery: BatterySnapshot, slot: SlotSpec) -> bool:
    fit = fit_for(battery, slot.name)
    if fit is None:
        return False
    return (
        abs(fit.axial_offset_m) <= cfg.scoring.axial_tolerance_m
        and fit.radial_offset_m <= cfg.scoring.seat_radial_max_m
        and fit.spring_contact is True
        and polarity_matches(cfg, snapshot, battery, slot, fit)
        and is_released(battery)
    )


def assign_cells(
    seated: set[tuple[str, str]],
    cells: tuple[str, ...],
    slots: tuple[str, ...],
    previous: dict[str, str | None],
) -> dict[str, str | None]:
    """Pick the one-to-one cell-to-slot match with the most seated pairs.

    Ties keep the previous occupants so a retention window is not restarted.
    """
    best: dict[str, str | None] = {slot: None for slot in slots}
    best_key = (-1, -1)
    options = list(cells) + [None] * len(slots)
    for chosen in itertools.permutations(options, len(slots)):
        mapping = dict(zip(slots, chosen))
        if any(cell is not None and (cell, slot) not in seated for slot, cell in mapping.items()):
            continue
        count = sum(1 for cell in mapping.values() if cell is not None)
        kept = sum(1 for slot, cell in mapping.items() if cell is not None and previous.get(slot) == cell)
        if (count, kept) > best_key:
            best_key = (count, kept)
            best = mapping
    return best


def carry_episode_memory(previous: SceneSnapshot | None, observed: SceneSnapshot) -> SceneSnapshot:
    """Keep ever-lifted, retention timers, and latches from the previous step."""
    if previous is None:
        return observed
    previous_batteries = {battery.name: battery for battery in previous.batteries}
    batteries = []
    for battery in observed.batteries:
        prior = previous_batteries.get(battery.name)
        batteries.append(battery if prior is None else replace(battery, ever_lifted=prior.ever_lifted))
    return replace(
        observed,
        batteries=tuple(batteries),
        ever_off_table=previous.ever_off_table,
        ever_remote_moved=previous.ever_remote_moved,
        ever_damage=previous.ever_damage,
        retention=previous.retention,
    )


def step(cfg: BatteryInsertionConfig, snapshot: SceneSnapshot, dt: float) -> tuple[SceneSnapshot, ScoreReport]:
    """Advance retention timers and latches by one control step."""
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    active = active_slots(cfg)
    if len(snapshot.batteries) < len(active):
        raise KeyError(f"snapshot has {len(snapshot.batteries)} cells for {len(active)} active slots")

    remote_moved_now = (
        snapshot.remote_displacement_m > cfg.scoring.remote_motion_tolerance_m
        or snapshot.remote_yaw_error_deg > cfg.scoring.remote_yaw_tolerance_deg
    )
    damage_now = snapshot.damage_force_n > cfg.scoring.damage_force_proxy_n
    off_now = False
    updated: list[BatterySnapshot] = []
    seated: set[tuple[str, str]] = set()
    for battery in snapshot.batteries:
        lifted = battery.ever_lifted or battery_is_lifted(cfg, battery)
        if battery.off_table or battery_pose_off_table(cfg, battery.position_xyz):
            off_now = True
        if battery.peak_force_n > cfg.scoring.damage_force_proxy_n:
            damage_now = True
        for slot in active:
            if is_fully_seated(cfg, snapshot, battery, slot):
                seated.add((battery.name, slot.name))
        updated.append(replace(battery, ever_lifted=lifted))

    prior = {entry.slot: entry for entry in snapshot.retention}
    slot_names = tuple(slot.name for slot in active)
    occupant = assign_cells(
        seated,
        tuple(battery.name for battery in snapshot.batteries),
        slot_names,
        {name: (prior[name].cell if name in prior else None) for name in slot_names},
    )
    window = cfg.scoring.retention_window_s
    retention: list[SlotRetention] = []
    inserted: dict[str, bool] = {}
    for name in slot_names:
        cell = occupant[name]
        before = prior.get(name)
        if cell is None:
            timer = 0.0
        else:
            carried = before.timer_s if before is not None and before.cell == cell else 0.0
            timer = min(window, carried + dt)
        retention.append(SlotRetention(slot=name, cell=cell, timer_s=timer))
        inserted[name] = cell is not None and timer >= window - 1e-9

    ever_off = snapshot.ever_off_table or off_now
    ever_remote = snapshot.ever_remote_moved or remote_moved_now
    ever_damage = snapshot.ever_damage or damage_now
    new_snapshot = replace(
        snapshot,
        batteries=tuple(updated),
        ever_off_table=ever_off,
        ever_remote_moved=ever_remote,
        ever_damage=ever_damage,
        retention=tuple(retention),
    )

    reasons: list[str] = []
    if ever_off:
        reasons.append("off_table")
    if ever_remote:
        reasons.append("remote_moved")
    if ever_damage:
        reasons.append("damage_proxy")
    any_lifted = any(battery.ever_lifted for battery in new_snapshot.batteries)
    if not any_lifted:
        reasons.append("never_lifted")
    inserted_count = sum(1 for value in inserted.values() if value)
    active_count = len(active)
    if reasons:
        score = 0
    elif inserted_count == active_count:
        score = 100
    elif inserted_count == 1 and active_count == 2:
        score = 50
    else:
        score = 25

    report = ScoreReport(
        score=score,
        hard_failure=ever_off or ever_remote or ever_damage,
        hard_reasons=tuple(reasons),
        inserted=inserted,
        occupant=dict(occupant),
        ever_lifted={battery.name: battery.ever_lifted for battery in new_snapshot.batteries},
        retention_timer_s={entry.slot: entry.timer_s for entry in retention},
        hold_unknown=tuple(battery.name for battery in snapshot.batteries if battery.gripper_holding is None),
        spring_unknown=tuple(
            sorted({fit.slot for battery in snapshot.batteries for fit in battery.fits if fit.spring_contact is None})
        ),
        ever_off_table=ever_off,
        ever_remote_moved=ever_remote,
        ever_damage=ever_damage,
    )
    return new_snapshot, report
