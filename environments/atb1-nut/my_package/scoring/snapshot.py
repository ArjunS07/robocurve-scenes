"""Observation snapshot for the nut-threading scorer. No Isaac Sim imports."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class NutSnapshot:
    # The nut's geometric centre (scorer.nut_centre_xyz), not the Factory body origin.
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    gripper_holding: bool
    released: bool
    off_table: bool
    # False when a gripper body or finger joint could not be read. Unknown is not released.
    hold_known: bool = True
    ever_lifted: bool = False
    thread_turns: float = 0.0
    prev_yaw_rad: float | None = None
    cross_threaded: bool = False


@dataclass(frozen=True)
class SceneSnapshot:
    nut: NutSnapshot
    fixture_position_xyz: tuple[float, float, float]
    fixture_rotation_xyzw: tuple[float, float, float, float]
    ever_off_table: bool = False
    ever_fixture_moved: bool = False


def clear_episode_state(snapshot: SceneSnapshot) -> SceneSnapshot:
    nut = replace(snapshot.nut, ever_lifted=False, thread_turns=0.0, prev_yaw_rad=None, cross_threaded=False)
    return replace(snapshot, nut=nut, ever_off_table=False, ever_fixture_moved=False)
