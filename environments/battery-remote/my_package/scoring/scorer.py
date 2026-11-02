"""Score one control step from an observation snapshot. No Isaac Sim imports.

Success is not "the battery center is inside the remote's bounding box".
A battery counts as inserted only when all of these are true together:

- its center has passed the full-seat depth along the slot axis
- its center is close to that slot's centerline
- its positive terminal matches that slot's required direction
- the caller reports contact with that slot's spring plunger
- the gripper has released it
- it stays that way for the retention window

Hard failures latch for the episode and force the score to 0. ``never_lifted``
also forces 0, but it is not an early termination: the battery may still be
lifted later. Visible damage is not modeled. ``damage_force_proxy_n`` is only
a force threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from my_package.scoring.config import BatteryInsertionConfig, active_slots
from my_package.scoring.math3d import angle_deg, rotate
from my_package.scoring.snapshot import BatterySnapshot, SceneSnapshot


@dataclass(frozen=True)
class ScoreReport:
    score: int
    hard_failure: bool
    hard_reasons: tuple[str, ...]
    inserted: dict[str, bool]
    ever_lifted: dict[str, bool]
    retention_timer_s: dict[str, float]
    ever_off_table: bool
    ever_remote_moved: bool
    ever_damage: bool


def full_seat_depth_m(cfg: BatteryInsertionConfig) -> float:
    """Distance from the mouth plane to the seated battery center."""
    compartment = cfg.compartment
    return compartment.mouth_x - compartment.seat_center_x


def battery_radius_m(cfg: BatteryInsertionConfig) -> float:
    return cfg.battery.diameter_m / 2.0


def battery_is_lifted(cfg: BatteryInsertionConfig, battery: BatterySnapshot) -> bool:
    """The lowest point of the cylinder is clear of the table by the configured gap."""
    lowest = battery.position_xyz[2] - battery_radius_m(cfg)
    return lowest > cfg.table.top_z_m + cfg.scoring.lift_clearance_m


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


def _required_axis_world(snapshot: SceneSnapshot, required_local: tuple[float, float, float]) -> tuple[float, float, float]:
    return rotate(snapshot.remote_rotation_xyzw, required_local)


def polarity_matches(cfg: BatteryInsertionConfig, snapshot: SceneSnapshot, battery: BatterySnapshot) -> bool:
    """Both the geometric axis and the snapshot flag have to agree.

    A true flag cannot grant credit for a reversed axis. A false flag fails
    even when the axis itself is right.
    """
    slot = next(slot for slot in cfg.compartment.slots if slot.name == battery.name)
    required = _required_axis_world(snapshot, slot.required_axis_xyz)
    angle = angle_deg(battery.positive_axis_xyz, required)
    return angle <= cfg.scoring.polarity_angle_deg and battery.polarity_aligned


def is_released(battery: BatterySnapshot) -> bool:
    """A held battery is not released, even if the released flag was set."""
    return battery.released and not battery.gripper_holding


def is_fully_seated(cfg: BatteryInsertionConfig, snapshot: SceneSnapshot, battery: BatterySnapshot) -> bool:
    depth_ready = battery.seating_depth_m >= full_seat_depth_m(cfg) - cfg.scoring.axial_tolerance_m
    radial_ready = battery.radial_offset_m <= cfg.scoring.seat_radial_max_m
    return (
        depth_ready
        and radial_ready
        and battery.spring_contact
        and polarity_matches(cfg, snapshot, battery)
        and is_released(battery)
    )


def carry_episode_memory(previous: SceneSnapshot | None, observed: SceneSnapshot) -> SceneSnapshot:
    """Keep ever-lifted, retention timers, and latches from the previous step."""
    if previous is None:
        return observed
    previous_batteries = {battery.name: battery for battery in previous.batteries}
    batteries = []
    for battery in observed.batteries:
        prior = previous_batteries.get(battery.name)
        if prior is None:
            batteries.append(battery)
        else:
            batteries.append(
                replace(
                    battery,
                    ever_lifted=prior.ever_lifted,
                    retention_timer_s=prior.retention_timer_s,
                )
            )
    return replace(
        observed,
        batteries=tuple(batteries),
        ever_off_table=previous.ever_off_table,
        ever_remote_moved=previous.ever_remote_moved,
        ever_damage=previous.ever_damage,
    )


def step(cfg: BatteryInsertionConfig, snapshot: SceneSnapshot, dt: float) -> tuple[SceneSnapshot, ScoreReport]:
    """Advance retention timers and latches by one control step."""
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    active = active_slots(cfg)
    present = {battery.name: battery for battery in snapshot.batteries}
    missing = [slot.name for slot in active if slot.name not in present]
    if missing:
        raise KeyError(f"snapshot is missing active batteries: {missing}")

    remote_moved_now = (
        snapshot.remote_displacement_m > cfg.scoring.remote_motion_tolerance_m
        or snapshot.remote_yaw_error_deg > cfg.scoring.remote_yaw_tolerance_deg
    )
    damage_now = snapshot.damage_force_n > cfg.scoring.damage_force_proxy_n
    off_now = False
    updated: list[BatterySnapshot] = []
    inserted: dict[str, bool] = {}
    for battery in snapshot.batteries:
        lifted = battery.ever_lifted or battery_is_lifted(cfg, battery)
        if battery.off_table or battery_pose_off_table(cfg, battery.position_xyz):
            off_now = True
        if battery.peak_force_n > cfg.scoring.damage_force_proxy_n:
            damage_now = True
        is_active = any(slot.name == battery.name for slot in active)
        if is_active and is_fully_seated(cfg, snapshot, battery):
            timer = min(cfg.scoring.retention_window_s, battery.retention_timer_s + dt)
        elif is_active:
            timer = 0.0
        else:
            timer = battery.retention_timer_s
        retained = is_active and timer >= cfg.scoring.retention_window_s - 1e-9
        if is_active:
            inserted[battery.name] = retained
        updated.append(replace(battery, ever_lifted=lifted, retention_timer_s=timer))

    ever_off = snapshot.ever_off_table or off_now
    ever_remote = snapshot.ever_remote_moved or remote_moved_now
    ever_damage = snapshot.ever_damage or damage_now
    new_snapshot = replace(
        snapshot,
        batteries=tuple(updated),
        ever_off_table=ever_off,
        ever_remote_moved=ever_remote,
        ever_damage=ever_damage,
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
    inserted_count = sum(1 for name in inserted if inserted[name])
    active_count = len(active)
    if reasons:
        score = 0
    elif inserted_count == active_count:
        score = 100
    elif inserted_count == 1 and active_count == 2:
        score = 50
    elif any_lifted:
        score = 25
    else:
        score = 0

    report = ScoreReport(
        score=score,
        hard_failure=ever_off or ever_remote or ever_damage,
        hard_reasons=tuple(reasons),
        inserted=inserted,
        ever_lifted={battery.name: battery.ever_lifted for battery in new_snapshot.batteries},
        retention_timer_s={battery.name: battery.retention_timer_s for battery in new_snapshot.batteries},
        ever_off_table=ever_off,
        ever_remote_moved=ever_remote,
        ever_damage=ever_damage,
    )
    return new_snapshot, report
