"""Score one control step of the nut-threading task. No Isaac Sim imports.

The stud carries a physical helical ridge and the nut three pins, so PhysX
contacts couple rotation and descent (see assets/geometry.py). The scorer does
not enforce that coupling; it only checks, on every step, that the observed
pose is consistent with it while the nut is coaxial. If descent and turns ever disagree by more than the tolerance while
the nut is engaged, the nut is cross-threaded until it is backed off above the
first thread or taken off the stud. A nut sitting on the tip, or tilted onto
the stud, is not threaded. Visible damage is not modeled.

A hex nut is symmetric, so an upside-down nut is treated the same as an upright
one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from my_package.scoring.config import NutThreadingConfig
from my_package.scoring.math3d import (
    angle_deg,
    dot,
    norm,
    quat_inv,
    quat_mul,
    rotate,
    rotation_angle_deg,
    scale,
    sub,
)
from my_package.scoring.snapshot import NutSnapshot, SceneSnapshot


@dataclass(frozen=True)
class ThreadState:
    tilt_deg: float
    radial_m: float
    gap_m: float
    descent_m: float
    travel_fraction: float
    thread_turns: float
    aligned: bool
    engaged: bool
    turn_matched: bool
    cross_threaded: bool
    threaded: bool
    lifted_now: bool


@dataclass(frozen=True)
class ScoreReport:
    score: int
    hard_failure: bool
    hard_reasons: tuple[str, ...]
    thread: ThreadState
    ever_lifted: bool
    ever_off_table: bool
    ever_fixture_moved: bool


def required_travel_m(cfg: NutThreadingConfig) -> float:
    return cfg.fastener.turns * cfg.fastener.pitch_m


def _released(nut: NutSnapshot) -> bool:
    """An unreadable gripper is never treated as released."""
    return nut.hold_known and nut.released and not nut.gripper_holding


def _unwrap(delta: float) -> float:
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _relative_yaw(nut_q, fixture_q) -> float:
    nut_in_fixture = quat_mul(quat_inv(fixture_q), nut_q)
    x_axis = rotate(nut_in_fixture, (1.0, 0.0, 0.0))
    return math.atan2(x_axis[1], x_axis[0])


def measure(cfg: NutThreadingConfig, snapshot: SceneSnapshot) -> tuple[NutSnapshot, ThreadState]:
    fastener = cfg.fastener
    scoring = cfg.scoring
    nut = snapshot.nut
    bolt_axis = rotate(snapshot.fixture_rotation_xyzw, (0.0, 0.0, 1.0))
    nut_axis = rotate(nut.rotation_xyzw, (0.0, 0.0, 1.0))
    # Either face of a hex nut can go down. Use the nut axis that points up the bolt.
    if dot(nut_axis, bolt_axis) < 0.0:
        nut_axis = scale(nut_axis, -1.0)
    tilt = angle_deg(nut_axis, bolt_axis)
    local = rotate(quat_inv(snapshot.fixture_rotation_xyzw), sub(nut.position_xyz, snapshot.fixture_position_xyz))
    radial = math.hypot(local[0], local[1])
    lower = sub(nut.position_xyz, scale(nut_axis, fastener.nut_height_m / 2.0))
    gap = dot(sub(lower, snapshot.fixture_position_xyz), bolt_axis)
    travel = required_travel_m(cfg)
    descent = travel - gap
    fraction = descent / travel
    expected = descent / fastener.pitch_m

    near = (
        radial <= scoring.track_radial_m
        and tilt <= scoring.track_tilt_deg
        and -fastener.nut_height_m < gap < travel + fastener.pitch_m
    )
    turns = nut.thread_turns
    prev = nut.prev_yaw_rad
    cross = nut.cross_threaded
    yaw = _relative_yaw(nut.rotation_xyzw, snapshot.fixture_rotation_xyzw)
    if not near:
        # Off the stud: nothing is engaged, so the next attempt starts clean.
        turns, prev, cross = 0.0, None, False
    else:
        if prev is not None:
            # Right-hand thread: clockwise when viewed from above screws the nut down.
            turns += -_unwrap(yaw - prev) / (2.0 * math.pi)
        prev = yaw
        if gap >= travel:
            # Above the first thread. Spinning here does not advance the nut.
            turns, cross = 0.0, False
        elif abs(turns - expected) > scoring.turn_tolerance:
            # Dropped faster than it turned, or turned without descending.
            cross = True

    aligned = tilt <= scoring.wedge_angle_deg and radial <= scoring.radial_max_m
    engaged = descent >= scoring.min_engagement_turns * fastener.pitch_m
    turn_matched = abs(turns - expected) <= scoring.turn_tolerance
    threaded = aligned and engaged and turn_matched and not cross and gap < travel
    corner = fastener.across_flats_m / math.sqrt(3.0)
    lowest = (
        nut.position_xyz[2]
        - abs(dot(nut_axis, (0.0, 0.0, 1.0))) * fastener.nut_height_m / 2.0
        - corner * math.sin(math.radians(min(angle_deg(nut_axis, (0.0, 0.0, 1.0)), 90.0)))
    )
    lifted_now = lowest > cfg.table.top_z_m + scoring.lift_clearance_m
    updated = replace(
        nut,
        thread_turns=turns,
        prev_yaw_rad=prev,
        cross_threaded=cross,
        ever_lifted=nut.ever_lifted or lifted_now,
    )
    state = ThreadState(
        tilt_deg=tilt,
        radial_m=radial,
        gap_m=gap,
        descent_m=descent,
        travel_fraction=fraction,
        thread_turns=turns,
        aligned=aligned,
        engaged=engaged,
        turn_matched=turn_matched,
        cross_threaded=cross,
        threaded=threaded,
        lifted_now=lifted_now,
    )
    return updated, state


def _off_table(cfg: NutThreadingConfig, position) -> bool:
    center_x, center_y = cfg.table.center_xy
    half_x, half_y = cfg.table.half_extents_xy
    outside = (
        position[0] < center_x - half_x
        or position[0] > center_x + half_x
        or position[1] < center_y - half_y
        or position[1] > center_y + half_y
    )
    fell = position[2] < cfg.table.top_z_m - cfg.table.fall_below_m
    return outside or fell


def carry_episode_memory(previous: SceneSnapshot | None, observed: SceneSnapshot) -> SceneSnapshot:
    if previous is None:
        return observed
    nut = replace(
        observed.nut,
        ever_lifted=previous.nut.ever_lifted,
        thread_turns=previous.nut.thread_turns,
        prev_yaw_rad=previous.nut.prev_yaw_rad,
        cross_threaded=previous.nut.cross_threaded,
    )
    return replace(
        observed,
        nut=nut,
        ever_off_table=previous.ever_off_table,
        ever_fixture_moved=previous.ever_fixture_moved,
    )


def step(cfg: NutThreadingConfig, snapshot: SceneSnapshot, dt: float) -> tuple[SceneSnapshot, ScoreReport]:
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    nut, state = measure(cfg, snapshot)
    shift = norm(sub(snapshot.fixture_position_xyz, cfg.fixture_pose.position_xyz))
    turned = rotation_angle_deg(snapshot.fixture_rotation_xyzw, cfg.fixture_pose.rotation_xyzw)
    moved_now = (
        shift > cfg.scoring.fixture_motion_tolerance_m or turned > cfg.scoring.fixture_rotation_tolerance_deg
    )
    off_now = nut.off_table or _off_table(cfg, nut.position_xyz)
    ever_off = snapshot.ever_off_table or off_now
    ever_moved = snapshot.ever_fixture_moved or moved_now
    new_snapshot = replace(snapshot, nut=nut, ever_off_table=ever_off, ever_fixture_moved=ever_moved)

    reasons: list[str] = []
    if ever_off:
        reasons.append("off_table")
    if ever_moved:
        reasons.append("fixture_moved")
    if not nut.ever_lifted:
        reasons.append("never_lifted")
    if state.cross_threaded:
        reasons.append("cross_threaded")

    seated = abs(state.gap_m) <= cfg.scoring.seat_gap_m
    deep = state.travel_fraction >= cfg.scoring.deep_fraction
    released = _released(nut)
    if ever_off or ever_moved or not nut.ever_lifted:
        score = 0
    elif state.threaded and released and seated:
        score = 100
    elif state.threaded and deep:
        score = 75
    elif state.threaded and released:
        score = 50
    else:
        # Lifted but not threaded, or threaded short of 75% and still held. The 50 tier needs a release.
        score = 25
        if state.threaded:
            reasons.append("held_short_of_75")

    report = ScoreReport(
        score=score,
        hard_failure=ever_off or ever_moved,
        hard_reasons=tuple(reasons),
        thread=state,
        ever_lifted=nut.ever_lifted,
        ever_off_table=ever_off,
        ever_fixture_moved=ever_moved,
    )
    return new_snapshot, report
