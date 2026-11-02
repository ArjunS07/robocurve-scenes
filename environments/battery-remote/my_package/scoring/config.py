"""Load the provisional battery-insertion parameters. No Isaac Sim imports."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class PoseSpec:
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class BatteryGeometry:
    """AAA cell. Every length is provisional; this is not a measured CAD model."""

    diameter_m: float
    length_m: float
    mass_kg: float
    nub_radius_m: float
    nub_length_m: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    # Light rolling damping so a cell lying on its side does not roll away from a nudge.
    angular_damping: float
    # Each start mark is a flat static mat this thick; cells rest on it, not on the table mesh.
    start_mat_thickness_m: float


@dataclass(frozen=True)
class SlotSpec:
    """One open-top compartment slot. Either physical cell may be scored in it."""

    name: str
    center_y_m: float
    required_axis_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class CompartmentGeometry:
    """Remote body and the open battery bay. Numbers are provisional.

    The cover is off, so each slot is open from above. A cell is pressed down
    into its slot: its negative end compresses a spring plunger and its positive
    end seats against a fixed contact. Polarity therefore decides which end of
    the slot carries the spring.
    """

    size_xyz: tuple[float, float, float]
    silhouette_length_m: float
    shell_nose_half_width_m: float
    shell_edge_fillet_m: float
    shell_end_radius_m: float
    mass_kg: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    rear_face_x: float
    front_face_x: float
    seat_center_x: float
    floor_z_m: float
    channel_y_half: float
    spring_preload_m: float
    plunger_size_xyz: tuple[float, float, float]
    plunger_travel_m: float
    plunger_end_gap_m: float
    plunger_mass_kg: float
    plunger_stiffness: float
    plunger_damping: float
    plunger_effort_limit_n: float
    plunger_velocity_limit_m_s: float
    spring_contact_distance_m: float
    contact_offset_m: float
    solver_position_iterations: int
    solver_velocity_iterations: int
    slots: tuple[SlotSpec, ...]


@dataclass(frozen=True)
class ScoringSpec:
    retention_window_s: float
    lift_clearance_m: float
    remote_motion_tolerance_m: float
    remote_yaw_tolerance_deg: float
    damage_force_proxy_n: float
    grasp_radius_m: float
    hold_open_fraction_max: float
    polarity_angle_deg: float
    axial_tolerance_m: float
    seat_radial_max_m: float
    episode_length_s: float
    enable_contact_sensors: bool


@dataclass(frozen=True)
class TableSpec:
    top_z_m: float
    center_xy: tuple[float, float]
    half_extents_xy: tuple[float, float]
    fall_below_m: float


@dataclass(frozen=True)
class CameraSpec:
    name: str
    eye_xyz: tuple[float, float, float]
    target_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    width: int
    height: int
    focal_length: float
    focus_distance: float
    horizontal_aperture: float
    vertical_aperture: float


@dataclass(frozen=True)
class RobotSpec:
    usda_path: str
    arm_layout: str
    home_arm_rad: tuple[float, float, float, float, float, float]
    finger_closed_m: float
    finger_open_m: float
    # Centre of each finger's geometry (pad) in that finger body's frame: tip_left, tip_right.
    finger_pad_offsets_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    home_open_fraction: float
    arm_action_scale: float
    gripper_action_scale: float
    proximal_stiffness: float
    proximal_damping: float
    distal_stiffness: float
    distal_damping: float
    finger_stiffness: float
    finger_damping: float
    arm_effort_limit: float
    arm_velocity_limit: float
    finger_effort_limit: float
    finger_velocity_limit: float
    left_pose: PoseSpec
    right_pose: PoseSpec


@dataclass(frozen=True)
class BatteryInsertionConfig:
    single_slot: bool
    light_intensity: float
    battery: BatteryGeometry
    compartment: CompartmentGeometry
    scoring: ScoringSpec
    table: TableSpec
    remote_pose: PoseSpec
    battery_a_pose: PoseSpec
    battery_b_pose: PoseSpec
    cameras: tuple[CameraSpec, ...]
    robot: RobotSpec


CELL_NAMES = ("battery_a", "battery_b")


def active_slots(cfg: BatteryInsertionConfig) -> tuple[SlotSpec, ...]:
    """Slots scored and cut into the compartment. Single-slot keeps the same package."""
    if cfg.single_slot:
        return cfg.compartment.slots[:1]
    return cfg.compartment.slots


def cell_names(cfg: BatteryInsertionConfig) -> tuple[str, ...]:
    """Physical cells spawned. They are interchangeable: neither is tied to a slot."""
    return CELL_NAMES[: len(active_slots(cfg))]


def plunger_joint_name(slot: SlotSpec) -> str:
    return f"{slot.name}_plunger"


def plunger_link_name(slot: SlotSpec) -> str:
    return f"{slot.name}_plunger_link"


def plunger_link_path(slot: SlotSpec) -> str:
    """Path under the remote prim. Plunger links nest under the chassis, like YAM's links."""
    return f"chassis/{plunger_link_name(slot)}"


def battery_pose(cfg: BatteryInsertionConfig, slot_name: str) -> PoseSpec:
    if slot_name == "battery_a":
        return cfg.battery_a_pose
    if slot_name == "battery_b":
        return cfg.battery_b_pose
    raise KeyError(slot_name)


def _convert(hint: Any, value: Any) -> Any:
    origin = get_origin(hint)
    if origin is tuple:
        args = get_args(hint)
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"expected a list for {hint}, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], item) for item in value)
        if len(value) != len(args):
            raise ValueError(f"expected {len(args)} values for {hint}, got {len(value)}")
        return tuple(_convert(arg, item) for arg, item in zip(args, value))
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"expected a number, got {value!r}")
        return float(value)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"expected an int, got {value!r}")
        return int(value)
    if hint is bool:
        if not isinstance(value, bool):
            raise TypeError(f"expected a bool, got {value!r}")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise TypeError(f"expected a string, got {value!r}")
        return value
    if isinstance(hint, type) and is_dataclass(hint):
        if not isinstance(value, dict):
            raise TypeError(f"expected a mapping for {hint.__name__}")
        return _build(hint, value)
    raise TypeError(f"unsupported config type {hint}")


def _build(cls: type, data: dict) -> Any:
    hints = get_type_hints(cls)
    known = set(hints)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__} has unknown keys: {sorted(unknown)}")
    missing = [name for name in hints if name not in data]
    if missing:
        raise ValueError(f"{cls.__name__} is missing keys: {missing}")
    return cls(**{name: _convert(hints[name], data[name]) for name in hints})


def load_config(path: str | Path, *, validate_config: bool = True) -> BatteryInsertionConfig:
    """Read YAML. Unknown keys are an error so a typo cannot silently drop a parameter."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    cfg = _build(BatteryInsertionConfig, raw)
    if validate_config:
        validate(cfg)
    return cfg


def with_single_slot(cfg: BatteryInsertionConfig, enabled: bool) -> BatteryInsertionConfig:
    updated = replace(cfg, single_slot=enabled)
    validate(updated)
    return updated


def validate(cfg: BatteryInsertionConfig) -> None:
    """Reject a config whose provisional numbers disagree with the retention geometry."""
    problems: list[str] = []
    if cfg.scoring.episode_length_s != 90.0:
        problems.append(f"episode_length_s must be 90, got {cfg.scoring.episode_length_s}")
    if len(cfg.robot.home_arm_rad) != 6:
        problems.append("home_arm_rad must contain six joint angles")
    if cfg.robot.finger_open_m == cfg.robot.finger_closed_m:
        problems.append("finger_open_m and finger_closed_m must differ")
    if not 0.0 <= cfg.robot.home_open_fraction <= 1.0:
        problems.append("home_open_fraction must be in [0, 1]")
    if cfg.robot.arm_layout not in {"duplicated_single_arm", "combined"}:
        problems.append(f"unknown arm_layout {cfg.robot.arm_layout!r}")
    if len(cfg.compartment.slots) != 2:
        problems.append("the config must define two slots; single_slot only hides the second")
    names = [slot.name for slot in cfg.compartment.slots]
    if names != ["slot_a", "slot_b"]:
        problems.append(f"slots must be slot_a then slot_b, got {names}")
    for slot in cfg.compartment.slots:
        axis = slot.required_axis_xyz
        if abs(abs(axis[0]) - 1.0) > 1e-9 or abs(axis[1]) > 1e-9 or abs(axis[2]) > 1e-9:
            problems.append(f"{slot.name} required_axis_xyz must be +X or -X, got {axis}")
    radius = cfg.battery.diameter_m / 2.0
    rest_z = cfg.table.top_z_m + cfg.battery.start_mat_thickness_m + radius
    if abs(cfg.battery_a_pose.position_xyz[2] - rest_z) > 1e-6:
        problems.append("battery_a start z must be table top plus the start mat plus the battery radius")
    if abs(cfg.battery_b_pose.position_xyz[2] - rest_z) > 1e-6:
        problems.append("battery_b start z must be table top plus the start mat plus the battery radius")
    if not 0.0 < cfg.battery.start_mat_thickness_m <= 0.003:
        problems.append("start_mat_thickness_m must be in (0, 3 mm]")
    if cfg.battery.angular_damping < 0.0:
        problems.append("angular_damping must be non-negative")
    half_height = cfg.compartment.size_xyz[2] / 2.0
    if abs(cfg.remote_pose.position_xyz[2] - (cfg.table.top_z_m + half_height)) > 1e-6:
        problems.append("remote start z must be table top plus half the remote height")
    problems.extend(mounting_problems(cfg))
    camera_names = [camera.name for camera in cfg.cameras]
    if camera_names != ["overhead_cam", "front_cam", "side_cam"]:
        problems.append(f"cameras must be overhead_cam, front_cam, side_cam; got {camera_names}")
    if cfg.scoring.retention_window_s <= 0.0:
        problems.append("retention_window_s must be positive")
    if cfg.scoring.damage_force_proxy_n <= 0.0:
        problems.append("damage_force_proxy_n must be positive")
    if cfg.compartment.contact_offset_m <= 0.0:
        problems.append("contact_offset_m must be positive")
    if cfg.compartment.plunger_travel_m <= 0.0:
        problems.append("plunger_travel_m must be positive")

    from my_package.scoring.math3d import look_at_ros_xyzw, sub

    for camera in cfg.cameras:
        expected = look_at_ros_xyzw(camera.eye_xyz, camera.target_xyz)
        delta = sub(camera.rotation_xyzw[:3], expected[:3])
        # Compare the full quaternion, including w, with a sign-invariant check.
        same = sum((a - b) ** 2 for a, b in zip(camera.rotation_xyzw, expected))
        flipped = sum((a + b) ** 2 for a, b in zip(camera.rotation_xyzw, expected))
        if min(same, flipped) > 1e-12:
            problems.append(
                f"{camera.name} rotation does not match look_at_ros_xyzw "
                f"(stored {camera.rotation_xyzw}, expected {expected})"
            )
        del delta

    # Imported lazily so geometry can import this module for types.
    from my_package.assets.geometry import design_report

    report = design_report(cfg)
    problems.extend(report["problems"])
    if problems:
        raise ValueError("battery insertion config is inconsistent:\n- " + "\n- ".join(problems))


# A robot base must sit at least this far inside the table edge. The YAM base-link
# footprint is not measured yet (environments/tools/probe_runtime.py reports it).
BASE_EDGE_MARGIN_M = 0.05
# Horizontal distance from the nearer robot base to each object's grasp point.
REACH_RANGE_M = (0.20, 0.45)


def _inside_table(cfg: BatteryInsertionConfig, xy: tuple[float, float], margin: float) -> bool:
    cx, cy = cfg.table.center_xy
    hx, hy = cfg.table.half_extents_xy
    return abs(xy[0] - cx) <= hx - margin and abs(xy[1] - cy) <= hy - margin


def mounting_problems(cfg: BatteryInsertionConfig) -> list[str]:
    """Bases on the tabletop (z and XY), and objects within a plausible reach of a base."""
    problems: list[str] = []
    bases = []
    for side, pose in (("left", cfg.robot.left_pose), ("right", cfg.robot.right_pose)):
        x, y, z = pose.position_xyz
        if abs(z - cfg.table.top_z_m) > 1e-6:
            problems.append(f"robot {side} base z {z} must equal the table top {cfg.table.top_z_m}")
        if not _inside_table(cfg, (x, y), BASE_EDGE_MARGIN_M):
            problems.append(
                f"robot {side} base XY ({x}, {y}) is not on the tabletop with a {BASE_EDGE_MARGIN_M} m margin"
            )
        bases.append((x, y))
    seat = (cfg.remote_pose.position_xyz[0] + cfg.compartment.seat_center_x, cfg.remote_pose.position_xyz[1])
    targets = {"battery bay": seat}
    for name in cell_names(cfg):
        pose = battery_pose(cfg, name)
        targets[name] = (pose.position_xyz[0], pose.position_xyz[1])
    low, high = REACH_RANGE_M
    for name, (x, y) in targets.items():
        nearest = min(((x - bx) ** 2 + (y - by) ** 2) ** 0.5 for bx, by in bases)
        if not low <= nearest <= high:
            problems.append(f"{name} is {nearest:.3f} m from the nearer robot base; expected {low}..{high} m")
        if not _inside_table(cfg, (x, y), 0.02):
            problems.append(f"{name} is not on the tabletop")
    return problems


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "battery_insertion.yaml"


def field_names(cfg: BatteryInsertionConfig) -> list[str]:
    """Flat field names, used by the contract test to show the saved parameters exist."""
    found: list[str] = []

    def walk(value: Any, prefix: str) -> None:
        if is_dataclass(value):
            for field in fields(value):
                walk(getattr(value, field.name), f"{prefix}{field.name}")
        elif isinstance(value, tuple) and value and is_dataclass(value[0]):
            for index, item in enumerate(value):
                walk(item, f"{prefix}[{index}].")
        else:
            found.append(prefix.rstrip("."))

    walk(cfg, "")
    return found
