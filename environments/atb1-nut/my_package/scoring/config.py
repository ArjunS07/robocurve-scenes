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
FASTENER_KEYS = (
    "nominal",
    "pitch_m",
    "handedness",
    "nut_usd",
    "bolt_usd",
    "bolt_length_m",
    "bolt_head_height_m",
    "bolt_head_corner_radius_m",
    "shank_major_radius_m",
    "shank_minor_radius_m",
    "nut_height_m",
    "nut_across_flats_m",
    "nut_origin_to_lower_face_m",
    "nut_bore_minor_radius_m",
    "nut_mass_kg",
)


@dataclass(frozen=True)
class PoseSpec:
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class FastenerSpec:
    """Isaac Lab's Factory M16 nut and bolt, with dimensions measured from their meshes."""

    nominal: str
    pitch_m: float
    handedness: str
    nut_usd: str
    bolt_usd: str
    # Bolt frame: origin at the bottom of the hex head, +z up the shank.
    bolt_length_m: float
    bolt_head_height_m: float
    bolt_head_corner_radius_m: float
    shank_major_radius_m: float
    shank_minor_radius_m: float
    # Nut frame: origin on the axis, nut_origin_to_lower_face_m below the lower face.
    nut_height_m: float
    nut_across_flats_m: float
    nut_origin_to_lower_face_m: float
    nut_bore_minor_radius_m: float
    nut_mass_kg: float


@dataclass(frozen=True)
class TableSpec:
    center_xy: tuple[float, float]
    half_extents_xy: tuple[float, float]
    top_z_m: float
    fall_below_m: float


@dataclass(frozen=True)
class FixtureBodySpec:
    plate_thickness_m: float
    # Square clearance hole around the shank (upper plate) and pocket around the head (lower layer).
    hole_half_m: float
    pocket_half_m: float
    block_half_xy_m: tuple[float, float]
    mass_kg: float
    static_friction: float
    dynamic_friction: float


@dataclass(frozen=True)
class StartMatSpec:
    thickness_m: float
    half_xy_m: tuple[float, float]


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
    # Top of the exposed thread where the nut spins or drops onto the first full crest without
    # the thread holding it (chamfered tip): turns are re-anchored to the descent here.
    thread_lead_in_m: float
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
    # Authored on the Factory nut and bolt, matching Isaac Lab's Factory task.
    fastener_solver_position_iterations: int
    fastener_solver_velocity_iterations: int
    fastener_max_depenetration_velocity_m_s: float
    fastener_enable_gyroscopic_forces: bool
    fastener_max_contact_impulse: float
    fastener_contact_offset_m: float
    fastener_rest_offset_m: float
    fastener_friction: float
    fixture_solver_position_iterations: int
    fixture_solver_velocity_iterations: int


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
    # Centre of each finger's geometry (pad) in that finger body's frame: tip_left, tip_right.
    finger_pad_offsets_m: tuple[tuple[float, float, float], tuple[float, float, float]]
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
    start_mat: StartMatSpec
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
            "start_mat",
            "scoring",
            "grasp",
            "physics",
            "robot",
            "cameras",
        ),
        "config",
    )
    fastener_data = _mapping(root["fastener"], "fastener")
    _exact_keys(fastener_data, FASTENER_KEYS, "fastener")
    table_data = _mapping(root["table"], "table")
    _exact_keys(table_data, ("center_xy", "half_extents_xy", "top_z_m", "fall_below_m"), "table")
    body_data = _mapping(root["fixture_body"], "fixture_body")
    _exact_keys(
        body_data,
        (
            "plate_thickness_m",
            "hole_half_m",
            "pocket_half_m",
            "block_half_xy_m",
            "mass_kg",
            "static_friction",
            "dynamic_friction",
        ),
        "fixture_body",
    )
    mat_data = _mapping(root["start_mat"], "start_mat")
    _exact_keys(mat_data, ("thickness_m", "half_xy_m"), "start_mat")
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
            "thread_lead_in_m",
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
            "fastener_solver_position_iterations",
            "fastener_solver_velocity_iterations",
            "fastener_max_depenetration_velocity_m_s",
            "fastener_enable_gyroscopic_forces",
            "fastener_max_contact_impulse",
            "fastener_contact_offset_m",
            "fastener_rest_offset_m",
            "fastener_friction",
            "fixture_solver_position_iterations",
            "fixture_solver_velocity_iterations",
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
            "finger_pad_offsets_m",
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
            handedness=_text(fastener_data["handedness"], "fastener.handedness"),
            nut_usd=_text(fastener_data["nut_usd"], "fastener.nut_usd"),
            bolt_usd=_text(fastener_data["bolt_usd"], "fastener.bolt_usd"),
            **{key: _number(fastener_data[key], f"fastener.{key}") for key in FASTENER_KEYS[5:]},
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
            hole_half_m=_number(body_data["hole_half_m"], "fixture_body.hole_half_m"),
            pocket_half_m=_number(body_data["pocket_half_m"], "fixture_body.pocket_half_m"),
            block_half_xy_m=_vector(body_data["block_half_xy_m"], "fixture_body.block_half_xy_m", 2),
            mass_kg=_number(body_data["mass_kg"], "fixture_body.mass_kg"),
            static_friction=_number(body_data["static_friction"], "fixture_body.static_friction"),
            dynamic_friction=_number(body_data["dynamic_friction"], "fixture_body.dynamic_friction"),
        ),
        start_mat=StartMatSpec(
            thickness_m=_number(mat_data["thickness_m"], "start_mat.thickness_m"),
            half_xy_m=_vector(mat_data["half_xy_m"], "start_mat.half_xy_m", 2),
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
            thread_lead_in_m=_number(scoring_data["thread_lead_in_m"], "scoring.thread_lead_in_m"),
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
            fastener_solver_position_iterations=_integer(
                physics_data["fastener_solver_position_iterations"], "physics.fastener_solver_position_iterations"
            ),
            fastener_solver_velocity_iterations=_integer(
                physics_data["fastener_solver_velocity_iterations"], "physics.fastener_solver_velocity_iterations"
            ),
            fastener_max_depenetration_velocity_m_s=_number(
                physics_data["fastener_max_depenetration_velocity_m_s"],
                "physics.fastener_max_depenetration_velocity_m_s",
            ),
            fastener_enable_gyroscopic_forces=_flag(
                physics_data["fastener_enable_gyroscopic_forces"], "physics.fastener_enable_gyroscopic_forces"
            ),
            fastener_max_contact_impulse=_number(
                physics_data["fastener_max_contact_impulse"], "physics.fastener_max_contact_impulse"
            ),
            fastener_contact_offset_m=_number(
                physics_data["fastener_contact_offset_m"], "physics.fastener_contact_offset_m"
            ),
            fastener_rest_offset_m=_number(physics_data["fastener_rest_offset_m"], "physics.fastener_rest_offset_m"),
            fastener_friction=_number(physics_data["fastener_friction"], "physics.fastener_friction"),
            fixture_solver_position_iterations=_integer(
                physics_data["fixture_solver_position_iterations"], "physics.fixture_solver_position_iterations"
            ),
            fixture_solver_velocity_iterations=_integer(
                physics_data["fixture_solver_velocity_iterations"], "physics.fixture_solver_velocity_iterations"
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
            finger_pad_offsets_m=tuple(
                _vector(offset, f"robot.finger_pad_offsets_m[{k}]", 3)
                for k, offset in enumerate(robot_data["finger_pad_offsets_m"])
            ),
            proximal=_actuator(actuator_data["proximal"], "robot.actuators.proximal"),
            distal=_actuator(actuator_data["distal"], "robot.actuators.distal"),
            finger=_actuator(actuator_data["finger"], "robot.actuators.finger"),
        ),
        cameras=tuple(_camera(item, index) for index, item in enumerate(cameras_data)),
    )
    validate_config(config)
    return config


def exposed_thread_m(config: NutThreadingConfig) -> float:
    """Shank above the fixture plate: how far the nut's lower face travels from the tip to the seat."""
    fastener = config.fastener
    return fastener.bolt_length_m - fastener.bolt_head_height_m - config.fixture_body.plate_thickness_m


def turns_to_seat(config: NutThreadingConfig) -> float:
    return exposed_thread_m(config) / config.fastener.pitch_m


def bolt_position_xyz(config: NutThreadingConfig) -> tuple[float, float, float]:
    """The bolt origin (bottom of the head) sits on the table, under the fixture's seat face."""
    x, y, _ = config.fixture_pose.position_xyz
    return (x, y, config.table.top_z_m)


def start_mat_top_z(config: NutThreadingConfig) -> float:
    return config.table.top_z_m + config.start_mat.thickness_m


def nut_spawn_z(config: NutThreadingConfig) -> float:
    """The nut origin is below its lower face, so it spawns below the mat top it rests on."""
    return start_mat_top_z(config) - config.fastener.nut_origin_to_lower_face_m


def threaded_tilt_bound_deg(fastener: FastenerSpec) -> float:
    """Upper bound on the tilt of a nut whose threads interleave the bolt's.

    The diametral clearance between the nut's minor diameter and the bolt's minor diameter,
    across the nut height. Thread flanks hold the nut tighter than this.
    """
    clearance = 2.0 * (fastener.nut_bore_minor_radius_m - fastener.shank_minor_radius_m)
    return math.degrees(math.atan2(clearance, fastener.nut_height_m))


def validate_config(config: NutThreadingConfig) -> None:
    """Check the written protocol against the loaded numbers."""
    if config.task_description != TASK_DESCRIPTION:
        raise ValueError(f"task_description must be {TASK_DESCRIPTION!r}")
    _close(config.episode_length_s, 120.0, "episode_length_s")
    fastener = config.fastener
    if fastener.nominal != "M16":
        raise ValueError("fastener.nominal must be M16")
    _close(fastener.pitch_m, 0.002, "fastener.pitch_m")
    if fastener.handedness != "right":
        raise ValueError("fastener.handedness must be right (measured from the Factory bolt mesh)")
    for key in ("nut_usd", "bolt_usd"):
        url = getattr(fastener, key)
        if not (url.startswith("https://") or Path(url).is_absolute()) or not url.endswith(".usd"):
            raise ValueError(f"fastener.{key} must be an https URL or absolute path to a .usd file, got {url!r}")
    _validate_fastener_fit(config)
    _validate_fixture(config)
    if config.scoring.radial_max_m < fastener.nut_bore_minor_radius_m - fastener.shank_minor_radius_m:
        raise ValueError("scoring.radial_max_m must cover the nut's radial play on the bolt")
    if config.scoring.wedge_angle_deg <= threaded_tilt_bound_deg(fastener):
        raise ValueError(
            f"scoring.wedge_angle_deg must exceed {threaded_tilt_bound_deg(fastener):.2f} deg, "
            "the most a nut can tilt on the thread, or a threaded nut reads as wedged"
        )
    _validate_robot_mount(config)
    if not 0.0 <= config.grasp.hold_open_fraction_min < config.grasp.hold_open_fraction_max <= 1.0:
        raise ValueError("grasp open fractions must satisfy 0 <= min < max <= 1")
    _close(config.nut_pose.position_xyz[2], nut_spawn_z(config), "nut z (origin below the mat top)")
    _close(
        config.fixture_pose.position_xyz[2],
        config.table.top_z_m + fastener.bolt_head_height_m + config.fixture_body.plate_thickness_m,
        "fixture z (seat face: table + bolt head + plate)",
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
    if not 0.0 <= config.scoring.thread_lead_in_m <= 2.0 * config.fastener.pitch_m:
        raise ValueError("thread_lead_in_m must be between 0 and two pitches")
    if config.robot.layout not in {"duplicated_single_arm", "combined"}:
        raise ValueError(f"unknown robot layout {config.robot.layout}")
    if not config.robot.usd_path.endswith(".usda"):
        raise ValueError("robot.usd_path must point at a USDA file")
    if not Path(config.robot.usd_path).is_absolute():
        raise ValueError("robot.usd_path must be absolute so launching does not depend on the working directory")
    if config.robot.finger_open_m >= 0.0 or config.robot.finger_closed_m != 0.0:
        raise ValueError("finger open is negative and finger closed is zero")
    physics = config.physics
    if physics.fastener_solver_position_iterations < 32 or physics.fastener_solver_velocity_iterations < 1:
        raise ValueError("fastener solver iterations must be at least 32 position and 1 velocity")
    if physics.fixture_solver_position_iterations < 16 or physics.fixture_solver_velocity_iterations < 1:
        raise ValueError("fixture solver iterations must be at least 16 position and 1 velocity")
    if physics.fastener_max_depenetration_velocity_m_s <= 0.0 or physics.fastener_max_contact_impulse <= 0.0:
        raise ValueError("fastener depenetration velocity and contact impulse must be positive")
    if not 0.0 <= physics.fastener_rest_offset_m < physics.fastener_contact_offset_m:
        raise ValueError("fastener offsets must satisfy 0 <= rest < contact")
    if physics.fastener_friction <= 0.0:
        raise ValueError("fastener friction must be positive")
    names = [camera.name for camera in config.cameras]
    if names != ["overhead", "front", "side"]:
        raise ValueError(f"cameras must be overhead, front, side in that order, got {names}")
    for camera in config.cameras:
        expected = look_at_ros_xyzw(camera.eye_xyz, camera.target_xyz)
        for actual, wanted in zip(camera.rotation_xyzw, expected, strict=True):
            _close(actual, wanted, f"camera {camera.name} rotation", tol=1e-5)


def _validate_fastener_fit(config: NutThreadingConfig) -> None:
    fastener = config.fastener
    for key in FASTENER_KEYS[5:]:
        if getattr(fastener, key) <= 0.0:
            raise ValueError(f"fastener.{key} must be positive")
    if not fastener.shank_minor_radius_m < fastener.nut_bore_minor_radius_m < fastener.shank_major_radius_m:
        raise ValueError("the nut's minor radius must lie between the bolt's minor and major radii (threads interleave)")
    if fastener.shank_major_radius_m >= fastener.bolt_head_corner_radius_m:
        raise ValueError("the bolt head must be wider than the shank")
    turns = turns_to_seat(config)
    if not 6.0 <= turns <= 10.0:
        raise ValueError(f"the exposed thread must take 6-10 turns to seat, got {turns:.2f}")
    if exposed_thread_m(config) <= fastener.nut_height_m:
        raise ValueError("the exposed thread must be longer than the nut")


def _validate_fixture(config: NutThreadingConfig) -> None:
    """The fixture holds the bolt without touching it, and the start mat stays clear of the bolt."""
    fastener = config.fastener
    body = config.fixture_body
    mat = config.start_mat
    if body.plate_thickness_m <= 0.0 or mat.thickness_m <= 0.0:
        raise ValueError("plate and mat thickness must be positive")
    if body.hole_half_m <= fastener.shank_major_radius_m:
        raise ValueError("the plate hole must clear the shank's thread crests")
    if body.hole_half_m >= fastener.nut_across_flats_m / 2.0:
        raise ValueError("the plate hole must be narrower than the nut's flats so the plate top is the seat face")
    if body.pocket_half_m <= fastener.bolt_head_corner_radius_m:
        raise ValueError("the head pocket must clear the bolt head's corners")
    if min(body.block_half_xy_m) <= body.pocket_half_m:
        raise ValueError("the block must be wider than the head pocket")
    if not 0.0 < body.dynamic_friction <= body.static_friction:
        raise ValueError("fixture friction must be positive with dynamic <= static")
    if mat.half_xy_m[0] < fastener.nut_across_flats_m / 2.0 or mat.half_xy_m[1] < fastener.nut_across_flats_m / 2.0:
        raise ValueError("the start mat must be at least as wide as the nut")
    mat_x = config.nut_pose.position_xyz[0] - config.fixture_pose.position_xyz[0]
    mat_y = config.nut_pose.position_xyz[1] - config.fixture_pose.position_xyz[1]
    clear_x = abs(mat_x) >= mat.half_xy_m[0] + body.block_half_xy_m[0]
    clear_y = abs(mat_y) >= mat.half_xy_m[1] + body.block_half_xy_m[1]
    if not (clear_x or clear_y):
        raise ValueError("the start mat under the nut must not overlap the fixture block")


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
