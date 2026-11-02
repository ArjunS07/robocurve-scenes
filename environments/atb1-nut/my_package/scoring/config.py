"""Load and check the nut-threading YAML. No Isaac Sim imports."""

from __future__ import annotations

import math

from dataclasses import dataclass
from pathlib import Path

import yaml

from my_package.scoring.math3d import look_at_ros_xyzw

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "nut_threading.yaml"
TASK_DESCRIPTION = "Tighten the nut onto the bolt."
IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class PoseSpec:
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class FastenerSpec:
    nominal: str
    pitch_m: float
    turns: float
    core_radius_m: float
    major_radius_m: float
    hole_radius_m: float
    ridge_thickness_m: float
    pin_tip_radius_m: float
    pin_thickness_m: float
    pin_width_m: float
    pin_count: int
    segments_per_turn: int
    thread_contact_offset_m: float
    across_flats_m: float
    nut_height_m: float
    nut_mass_kg: float
    lead_in_m: float
    static_friction: float
    dynamic_friction: float


@dataclass(frozen=True)
class TableSpec:
    center_xy: tuple[float, float]
    half_extents_xy: tuple[float, float]
    top_z_m: float
    fall_below_m: float


@dataclass(frozen=True)
class FixtureBodySpec:
    plate_thickness_m: float
    collar_height_m: float
    mass_kg: float


@dataclass(frozen=True)
class ScoringSpec:
    lift_clearance_m: float
    fixture_motion_tolerance_m: float
    fixture_rotation_tolerance_deg: float
    wedge_angle_deg: float
    radial_max_m: float
    track_radial_m: float
    track_tilt_deg: float
    min_engagement_turns: float
    turn_tolerance: float
    seat_gap_m: float
    deep_fraction: float


@dataclass(frozen=True)
class GraspSpec:
    radius_m: float
    hold_open_fraction_min: float
    hold_open_fraction_max: float


@dataclass(frozen=True)
class PhysicsSpec:
    contact_offset_m: float
    solver_position_iterations: int
    solver_velocity_iterations: int
    # Caps how fast PhysX pushes overlapping bodies apart. Uncapped, a 5 g nut was flung
    # ~0.7 m when driven at 5 mN m (server sweep at c715f9a).
    max_depenetration_velocity_m_s: float
    nut_speculative_ccd: bool
    nut_max_angular_velocity_deg_s: float


@dataclass(frozen=True)
class ActuatorSpec:
    stiffness: float
    damping: float
    effort: float
    velocity: float


@dataclass(frozen=True)
class RobotSpec:
    usd_path: str
    layout: str
    home_arm_rad: tuple[float, float, float, float, float, float]
    finger_open_m: float
    finger_closed_m: float
    left_base_xyz: tuple[float, float, float]
    right_base_xyz: tuple[float, float, float]
    gripper_body_name: str
    proximal: ActuatorSpec
    distal: ActuatorSpec
    finger: ActuatorSpec


@dataclass(frozen=True)
class CameraSpec:
    name: str
    eye_xyz: tuple[float, float, float]
    target_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class NutThreadingConfig:
    task_description: str
    episode_length_s: float
    fastener: FastenerSpec
    table: TableSpec
    fixture_pose: PoseSpec
    nut_pose: PoseSpec
    fixture_body: FixtureBodySpec
    scoring: ScoringSpec
    grasp: GraspSpec
    physics: PhysicsSpec
    robot: RobotSpec
    cameras: tuple[CameraSpec, ...]


def _mapping(value, path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _exact_keys(value: dict, allowed: tuple[str, ...], path: str) -> None:
    missing = [key for key in allowed if key not in value]
    unknown = sorted(set(value) - set(allowed))
    if missing or unknown:
        raise ValueError(f"{path} keys mismatch. missing={missing} unknown={unknown}")


def _number(value, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    return float(value)


def _flag(value, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be true or false, got {value!r}")
    return value


def _integer(value, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _text(value, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _vector(value, path: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{path} must have {length} numbers")
    return tuple(_number(item, f"{path}[{index}]") for index, item in enumerate(value))


def _close(actual: float, expected: float, path: str, tol: float = 1e-6) -> None:
    if abs(actual - expected) > tol:
        raise ValueError(f"{path} must be {expected}, got {actual}")


def _pose(value, path: str) -> PoseSpec:
    data = _mapping(value, path)
    _exact_keys(data, ("position_xyz", "rotation_xyzw"), path)
    position = _vector(data["position_xyz"], f"{path}.position_xyz", 3)
    rotation = _vector(data["rotation_xyzw"], f"{path}.rotation_xyzw", 4)
    return PoseSpec(position, rotation)


def _actuator(value, path: str) -> ActuatorSpec:
    data = _mapping(value, path)
    _exact_keys(data, ("stiffness", "damping", "effort", "velocity"), path)
    return ActuatorSpec(
        stiffness=_number(data["stiffness"], f"{path}.stiffness"),
        damping=_number(data["damping"], f"{path}.damping"),
        effort=_number(data["effort"], f"{path}.effort"),
        velocity=_number(data["velocity"], f"{path}.velocity"),
    )


def _camera(value, index: int) -> CameraSpec:
    path = f"cameras[{index}]"
    data = _mapping(value, path)
    _exact_keys(data, ("name", "eye_xyz", "target_xyz", "rotation_xyzw"), path)
    return CameraSpec(
        name=_text(data["name"], f"{path}.name"),
        eye_xyz=_vector(data["eye_xyz"], f"{path}.eye_xyz", 3),
        target_xyz=_vector(data["target_xyz"], f"{path}.target_xyz", 3),
        rotation_xyzw=_vector(data["rotation_xyzw"], f"{path}.rotation_xyzw", 4),
    )


def parse_config(data: dict) -> NutThreadingConfig:
    """Build a config from a decoded mapping and reject unknown keys."""
    root = _mapping(data, "config")
    _exact_keys(
        root,
        (
            "task_description",
            "episode_length_s",
            "fastener",
            "table",
            "fixture_pose",
            "nut_pose",
            "fixture_body",
            "scoring",
            "grasp",
            "physics",
            "robot",
            "cameras",
        ),
        "config",
    )
    fastener_data = _mapping(root["fastener"], "fastener")
    _exact_keys(
        fastener_data,
        (
            "nominal",
            "pitch_m",
            "turns",
            "core_radius_m",
            "major_radius_m",
            "hole_radius_m",
            "ridge_thickness_m",
            "pin_tip_radius_m",
            "pin_thickness_m",
            "pin_width_m",
            "pin_count",
            "segments_per_turn",
            "thread_contact_offset_m",
            "across_flats_m",
            "nut_height_m",
            "nut_mass_kg",
            "lead_in_m",
            "static_friction",
            "dynamic_friction",
        ),
        "fastener",
    )
    table_data = _mapping(root["table"], "table")
    _exact_keys(table_data, ("center_xy", "half_extents_xy", "top_z_m", "fall_below_m"), "table")
    body_data = _mapping(root["fixture_body"], "fixture_body")
    _exact_keys(body_data, ("plate_thickness_m", "collar_height_m", "mass_kg"), "fixture_body")
    scoring_data = _mapping(root["scoring"], "scoring")
    _exact_keys(
        scoring_data,
        (
            "lift_clearance_m",
            "fixture_motion_tolerance_m",
            "fixture_rotation_tolerance_deg",
            "wedge_angle_deg",
            "radial_max_m",
            "track_radial_m",
            "track_tilt_deg",
            "min_engagement_turns",
            "turn_tolerance",
            "seat_gap_m",
            "deep_fraction",
        ),
        "scoring",
    )
    grasp_data = _mapping(root["grasp"], "grasp")
    _exact_keys(grasp_data, ("radius_m", "hold_open_fraction_min", "hold_open_fraction_max"), "grasp")
    physics_data = _mapping(root["physics"], "physics")
    _exact_keys(
        physics_data,
        (
            "contact_offset_m",
            "solver_position_iterations",
            "solver_velocity_iterations",
            "max_depenetration_velocity_m_s",
            "nut_speculative_ccd",
            "nut_max_angular_velocity_deg_s",
        ),
        "physics",
    )
    robot_data = _mapping(root["robot"], "robot")
    _exact_keys(
        robot_data,
        (
            "usd_path",
            "layout",
            "home_arm_rad",
            "finger_open_m",
            "finger_closed_m",
            "left_base_xyz",
            "right_base_xyz",
            "gripper_body_name",
            "actuators",
        ),
        "robot",
    )
    actuator_data = _mapping(robot_data["actuators"], "robot.actuators")
    _exact_keys(actuator_data, ("proximal", "distal", "finger"), "robot.actuators")
    cameras_data = root["cameras"]
    if not isinstance(cameras_data, list) or not cameras_data:
        raise ValueError("cameras must be a non-empty list")

    config = NutThreadingConfig(
        task_description=_text(root["task_description"], "task_description"),
        episode_length_s=_number(root["episode_length_s"], "episode_length_s"),
        fastener=FastenerSpec(
            nominal=_text(fastener_data["nominal"], "fastener.nominal"),
            pitch_m=_number(fastener_data["pitch_m"], "fastener.pitch_m"),
            turns=_number(fastener_data["turns"], "fastener.turns"),
            core_radius_m=_number(fastener_data["core_radius_m"], "fastener.core_radius_m"),
            major_radius_m=_number(fastener_data["major_radius_m"], "fastener.major_radius_m"),
            hole_radius_m=_number(fastener_data["hole_radius_m"], "fastener.hole_radius_m"),
            ridge_thickness_m=_number(fastener_data["ridge_thickness_m"], "fastener.ridge_thickness_m"),
            pin_tip_radius_m=_number(fastener_data["pin_tip_radius_m"], "fastener.pin_tip_radius_m"),
            pin_thickness_m=_number(fastener_data["pin_thickness_m"], "fastener.pin_thickness_m"),
            pin_width_m=_number(fastener_data["pin_width_m"], "fastener.pin_width_m"),
            pin_count=_integer(fastener_data["pin_count"], "fastener.pin_count"),
            segments_per_turn=_integer(fastener_data["segments_per_turn"], "fastener.segments_per_turn"),
            thread_contact_offset_m=_number(
                fastener_data["thread_contact_offset_m"], "fastener.thread_contact_offset_m"
            ),
            across_flats_m=_number(fastener_data["across_flats_m"], "fastener.across_flats_m"),
            nut_height_m=_number(fastener_data["nut_height_m"], "fastener.nut_height_m"),
            nut_mass_kg=_number(fastener_data["nut_mass_kg"], "fastener.nut_mass_kg"),
            lead_in_m=_number(fastener_data["lead_in_m"], "fastener.lead_in_m"),
            static_friction=_number(fastener_data["static_friction"], "fastener.static_friction"),
            dynamic_friction=_number(fastener_data["dynamic_friction"], "fastener.dynamic_friction"),
        ),
        table=TableSpec(
            center_xy=_vector(table_data["center_xy"], "table.center_xy", 2),
            half_extents_xy=_vector(table_data["half_extents_xy"], "table.half_extents_xy", 2),
            top_z_m=_number(table_data["top_z_m"], "table.top_z_m"),
            fall_below_m=_number(table_data["fall_below_m"], "table.fall_below_m"),
        ),
        fixture_pose=_pose(root["fixture_pose"], "fixture_pose"),
        nut_pose=_pose(root["nut_pose"], "nut_pose"),
        fixture_body=FixtureBodySpec(
            plate_thickness_m=_number(body_data["plate_thickness_m"], "fixture_body.plate_thickness_m"),
            collar_height_m=_number(body_data["collar_height_m"], "fixture_body.collar_height_m"),
            mass_kg=_number(body_data["mass_kg"], "fixture_body.mass_kg"),
        ),
        scoring=ScoringSpec(
            lift_clearance_m=_number(scoring_data["lift_clearance_m"], "scoring.lift_clearance_m"),
            fixture_motion_tolerance_m=_number(
                scoring_data["fixture_motion_tolerance_m"], "scoring.fixture_motion_tolerance_m"
            ),
            fixture_rotation_tolerance_deg=_number(
                scoring_data["fixture_rotation_tolerance_deg"], "scoring.fixture_rotation_tolerance_deg"
            ),
            wedge_angle_deg=_number(scoring_data["wedge_angle_deg"], "scoring.wedge_angle_deg"),
            radial_max_m=_number(scoring_data["radial_max_m"], "scoring.radial_max_m"),
            track_radial_m=_number(scoring_data["track_radial_m"], "scoring.track_radial_m"),
            track_tilt_deg=_number(scoring_data["track_tilt_deg"], "scoring.track_tilt_deg"),
            min_engagement_turns=_number(scoring_data["min_engagement_turns"], "scoring.min_engagement_turns"),
            turn_tolerance=_number(scoring_data["turn_tolerance"], "scoring.turn_tolerance"),
            seat_gap_m=_number(scoring_data["seat_gap_m"], "scoring.seat_gap_m"),
            deep_fraction=_number(scoring_data["deep_fraction"], "scoring.deep_fraction"),
        ),
        grasp=GraspSpec(
            radius_m=_number(grasp_data["radius_m"], "grasp.radius_m"),
            hold_open_fraction_min=_number(grasp_data["hold_open_fraction_min"], "grasp.hold_open_fraction_min"),
            hold_open_fraction_max=_number(grasp_data["hold_open_fraction_max"], "grasp.hold_open_fraction_max"),
        ),
        physics=PhysicsSpec(
            contact_offset_m=_number(physics_data["contact_offset_m"], "physics.contact_offset_m"),
            solver_position_iterations=int(_number(physics_data["solver_position_iterations"], "physics.solver_position_iterations")),
            solver_velocity_iterations=int(
                _number(physics_data["solver_velocity_iterations"], "physics.solver_velocity_iterations")
            ),
            max_depenetration_velocity_m_s=_number(
                physics_data["max_depenetration_velocity_m_s"], "physics.max_depenetration_velocity_m_s"
            ),
            nut_speculative_ccd=_flag(physics_data["nut_speculative_ccd"], "physics.nut_speculative_ccd"),
            nut_max_angular_velocity_deg_s=_number(
                physics_data["nut_max_angular_velocity_deg_s"], "physics.nut_max_angular_velocity_deg_s"
            ),
        ),
        robot=RobotSpec(
            usd_path=_text(robot_data["usd_path"], "robot.usd_path"),
            layout=_text(robot_data["layout"], "robot.layout"),
            home_arm_rad=_vector(robot_data["home_arm_rad"], "robot.home_arm_rad", 6),
            finger_open_m=_number(robot_data["finger_open_m"], "robot.finger_open_m"),
            finger_closed_m=_number(robot_data["finger_closed_m"], "robot.finger_closed_m"),
            left_base_xyz=_vector(robot_data["left_base_xyz"], "robot.left_base_xyz", 3),
            right_base_xyz=_vector(robot_data["right_base_xyz"], "robot.right_base_xyz", 3),
            gripper_body_name=_text(robot_data["gripper_body_name"], "robot.gripper_body_name"),
            proximal=_actuator(actuator_data["proximal"], "robot.actuators.proximal"),
            distal=_actuator(actuator_data["distal"], "robot.actuators.distal"),
            finger=_actuator(actuator_data["finger"], "robot.actuators.finger"),
        ),
        cameras=tuple(_camera(item, index) for index, item in enumerate(cameras_data)),
    )
    validate_config(config)
    return config


def threaded_tilt_deg(fastener: FastenerSpec) -> float:
    """Largest tilt of a nut whose pins sit in the groove: axial play across the pin circle."""
    import math

    play = fastener.pitch_m - fastener.ridge_thickness_m - fastener.pin_thickness_m
    pin_radius = 0.5 * (fastener.pin_tip_radius_m + fastener.hole_radius_m)
    # Three pins at 120 deg: the shortest lever between a pin and the line through the other two is 1.5 r.
    return math.degrees(math.atan2(play, 1.5 * pin_radius))


def jam_tilt_deg(fastener: FastenerSpec) -> float:
    """Largest tilt a nut can reach while its bore still surrounds the threaded stud.

    Solves h*sin(t) + 2*r_major*cos(t) = 2*r_hole, the tilted bore just spanning the ridge.
    """
    import math

    h = fastener.nut_height_m
    d = 2.0 * fastener.major_radius_m
    bore = 2.0 * fastener.hole_radius_m
    amplitude = math.hypot(h, d)
    return math.degrees(math.asin(min(1.0, bore / amplitude)) - math.atan2(d, h))


def validate_config(config: NutThreadingConfig) -> None:
    """Check the written protocol against the loaded numbers."""
    if config.task_description != TASK_DESCRIPTION:
        raise ValueError(f"task_description must be {TASK_DESCRIPTION!r}")
    _close(config.episode_length_s, 120.0, "episode_length_s")
    fastener = config.fastener
    if fastener.nominal != "M8":
        raise ValueError("fastener.nominal must be M8")
    if not 6.0 < fastener.turns < 7.0:
        raise ValueError(f"fastener.turns must be between 6 and 7, got {fastener.turns}")
    _close(fastener.pitch_m, 0.00125, "fastener.pitch_m")
    _validate_thread(fastener)
    if fastener.nut_height_m <= 0.0 or fastener.lead_in_m <= 0.0:
        raise ValueError("nut height and lead-in must be positive")
    if not 0.0 < fastener.dynamic_friction <= fastener.static_friction:
        raise ValueError("friction must be positive with dynamic <= static")
    play = fastener.pin_tip_radius_m - fastener.core_radius_m
    if config.scoring.radial_max_m < play:
        raise ValueError("scoring.radial_max_m must cover the pins' radial play on the core")
    if config.scoring.wedge_angle_deg >= jam_tilt_deg(fastener):
        raise ValueError(
            f"scoring.wedge_angle_deg must be below the jam tilt {jam_tilt_deg(fastener):.2f} deg, "
            "or a nut jammed on the stud reads as aligned"
        )
    if config.scoring.wedge_angle_deg <= threaded_tilt_deg(fastener):
        raise ValueError(
            f"scoring.wedge_angle_deg must exceed the threaded tilt {threaded_tilt_deg(fastener):.2f} deg, "
            "or a correctly threaded nut reads as wedged"
        )
    _validate_robot_mount(config)
    if not 0.0 <= config.grasp.hold_open_fraction_min < config.grasp.hold_open_fraction_max <= 1.0:
        raise ValueError("grasp open fractions must satisfy 0 <= min < max <= 1")
    _close(config.nut_pose.position_xyz[2], config.table.top_z_m + fastener.nut_height_m / 2.0, "nut z")
    _close(
        config.fixture_pose.position_xyz[2],
        config.table.top_z_m + config.fixture_body.plate_thickness_m + config.fixture_body.collar_height_m,
        "fixture z",
    )
    for pose, label in ((config.nut_pose, "nut"), (config.fixture_pose, "fixture")):
        for actual, expected in zip(pose.rotation_xyzw, IDENTITY_XYZW, strict=True):
            _close(actual, expected, f"{label} rotation")
    dx = config.nut_pose.position_xyz[0] - config.fixture_pose.position_xyz[0]
    dy = config.nut_pose.position_xyz[1] - config.fixture_pose.position_xyz[1]
    _close((dx * dx + dy * dy) ** 0.5, 0.15, "nut distance from the bolt")
    _close(config.scoring.deep_fraction, 0.75, "scoring.deep_fraction")
    _close(config.scoring.seat_gap_m, 0.001, "scoring.seat_gap_m")
    _close(config.scoring.min_engagement_turns, 0.5, "scoring.min_engagement_turns")
    _close(config.scoring.turn_tolerance, 0.5, "scoring.turn_tolerance")
    if config.robot.layout not in {"duplicated_single_arm", "combined"}:
        raise ValueError(f"unknown robot layout {config.robot.layout}")
    if not config.robot.usd_path.endswith(".usda"):
        raise ValueError("robot.usd_path must point at a USDA file")
    if not Path(config.robot.usd_path).is_absolute():
        raise ValueError("robot.usd_path must be absolute so launching does not depend on the working directory")
    if config.robot.finger_open_m >= 0.0 or config.robot.finger_closed_m != 0.0:
        raise ValueError("finger open is negative and finger closed is zero")
    if config.physics.solver_position_iterations < 16 or config.physics.solver_velocity_iterations < 1:
        raise ValueError("solver iterations must be at least 16 position and 1 velocity")
    if not 0.0 < config.physics.max_depenetration_velocity_m_s <= 1.0:
        raise ValueError("max_depenetration_velocity_m_s must be in (0, 1] m/s")
    if not 360.0 <= config.physics.nut_max_angular_velocity_deg_s <= 7200.0:
        raise ValueError("nut_max_angular_velocity_deg_s must be 1-20 turns/s (360-7200 deg/s)")
    names = [camera.name for camera in config.cameras]
    if names != ["overhead", "front", "side"]:
        raise ValueError(f"cameras must be overhead, front, side in that order, got {names}")
    for camera in config.cameras:
        expected = look_at_ros_xyzw(camera.eye_xyz, camera.target_xyz)
        for actual, wanted in zip(camera.rotation_xyzw, expected, strict=True):
            _close(actual, wanted, f"camera {camera.name} rotation", tol=1e-5)


def _validate_thread(fastener: FastenerSpec) -> None:
    """Clearances of the physical thread, and a contact offset large enough to stop tunnelling.

    The contact offset is a detection margin (restOffset is 0), not a gap, so it is not
    compared with the clearances. It must be at least the ridge thickness: otherwise a pin
    can pass the ridge midline before contact starts and get pushed through (server 942249a).
    """
    offset = fastener.thread_contact_offset_m
    gaps = {
        "bore wall to ridge": fastener.hole_radius_m - fastener.major_radius_m,
        "pin tip to core": fastener.pin_tip_radius_m - fastener.core_radius_m,
        "axial groove": (fastener.pitch_m - fastener.ridge_thickness_m - fastener.pin_thickness_m) / 2.0,
    }
    for name, gap in gaps.items():
        if gap <= 0.0:
            raise ValueError(f"thread clearance {name} is {gap * 1000:.3f} mm; it must be positive")
    if offset < fastener.ridge_thickness_m:
        raise ValueError(
            f"thread contact offset {offset * 1000:.3f} mm is below the ridge thickness "
            f"{fastener.ridge_thickness_m * 1000:.3f} mm; pins would tunnel through the ridge"
        )
    overlap = fastener.major_radius_m - fastener.pin_tip_radius_m
    if overlap <= 2.0 * (fastener.pin_tip_radius_m - fastener.core_radius_m):
        raise ValueError("pins must overlap the ridge by more than twice their radial play, or they can slip past it")
    if not fastener.core_radius_m < fastener.pin_tip_radius_m < fastener.major_radius_m < fastener.hole_radius_m:
        raise ValueError("thread radii must satisfy core < pin tip < major < bore")
    if fastener.hole_radius_m >= fastener.across_flats_m / 2.0:
        raise ValueError("nut bore must be inside the hex flats")
    if fastener.pin_count < 3:
        raise ValueError("pin_count must be at least 3 to support the nut")
    if fastener.segments_per_turn < 12:
        raise ValueError("segments_per_turn must be at least 12")
    if math.gcd(fastener.segments_per_turn, fastener.pin_count) != 1:
        raise ValueError(
            "segments_per_turn must share no factor with pin_count, or several pins cross a ridge seam at once"
        )
    span = fastener.pitch_m * (fastener.pin_count - 1) / fastener.pin_count + fastener.pin_thickness_m
    if span >= fastener.nut_height_m / 2.0:
        raise ValueError("the pin stack must fit in the lower half of the nut")


# The YAM base footprint is not measured yet; keep the base origin this far inside the table edge.
BASE_TABLE_MARGIN_M = 0.04
REACH_MIN_M = 0.25
REACH_MAX_M = 0.40


def _validate_robot_mount(config: NutThreadingConfig) -> None:
    table = config.table
    cx, cy = table.center_xy
    hx, hy = table.half_extents_xy
    fixture = config.fixture_pose.position_xyz
    for side, base in (("left", config.robot.left_base_xyz), ("right", config.robot.right_base_xyz)):
        inside = (
            cx - hx + BASE_TABLE_MARGIN_M <= base[0] <= cx + hx - BASE_TABLE_MARGIN_M
            and cy - hy + BASE_TABLE_MARGIN_M <= base[1] <= cy + hy - BASE_TABLE_MARGIN_M
        )
        if not inside:
            raise ValueError(f"robot {side} base {base[:2]} is not on the tabletop footprint")
        _close(base[2], table.top_z_m, f"robot {side} base z (table top)")
        reach = ((fixture[0] - base[0]) ** 2 + (fixture[1] - base[1]) ** 2) ** 0.5
        if not REACH_MIN_M <= reach <= REACH_MAX_M:
            raise ValueError(f"fixture is {reach:.3f} m from the {side} base; keep it within {REACH_MIN_M}-{REACH_MAX_M} m")


def load_config(path: str | Path | None = None) -> NutThreadingConfig:
    config_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return parse_config(data)
