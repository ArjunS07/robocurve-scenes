"""Load the provisional chess-mate parameters. No Isaac Sim imports."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from my_package.scoring.puzzles import Puzzle, build_puzzles


@dataclass(frozen=True)
class PoseSpec:
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class PieceProfile:
    """One Staunton-like role. Heights are a provisional likeness, not catalog CAD."""

    role: str
    height_m: float
    base_radius_m: float
    mass_kg: float


@dataclass(frozen=True)
class BoardSpec:
    square_m: float
    border_m: float
    thickness_m: float
    mass_kg: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    contact_offset_m: float
    solver_position_iterations: int
    solver_velocity_iterations: int


@dataclass(frozen=True)
class ScoringSpec:
    lift_clearance_m: float
    board_motion_tolerance_m: float
    board_yaw_tolerance_deg: float
    upright_angle_deg: float
    rest_z_tolerance_m: float
    center_tolerance_m: float
    grasp_radius_m: float
    hold_open_fraction_max: float
    episode_length_s: float


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
class ChessMateConfig:
    light_intensity: float
    fen_sequence: tuple[str, ...]
    white_rgb: tuple[float, float, float]
    black_rgb: tuple[float, float, float]
    frame_rgb: tuple[float, float, float]
    light_square_rgb: tuple[float, float, float]
    dark_square_rgb: tuple[float, float, float]
    board: BoardSpec
    pieces: tuple[PieceProfile, ...]
    scoring: ScoringSpec
    table: TableSpec
    board_pose: PoseSpec
    cameras: tuple[CameraSpec, ...]
    robot: RobotSpec


def puzzles_for(cfg: ChessMateConfig) -> tuple[Puzzle, ...]:
    return build_puzzles(cfg.fen_sequence)


def profile_for(cfg: ChessMateConfig, role: str) -> PieceProfile:
    found = [piece for piece in cfg.pieces if piece.role == role]
    if len(found) != 1:
        raise KeyError(role)
    return found[0]


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
    unknown = set(data) - set(hints)
    if unknown:
        raise ValueError(f"{cls.__name__} has unknown keys: {sorted(unknown)}")
    missing = [name for name in hints if name not in data]
    if missing:
        raise ValueError(f"{cls.__name__} is missing keys: {missing}")
    return cls(**{name: _convert(hints[name], data[name]) for name in hints})


def load_config(path: str | Path, *, validate_config: bool = True) -> ChessMateConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    cfg = _build(ChessMateConfig, raw)
    if validate_config:
        validate(cfg)
    return cfg


def validate(cfg: ChessMateConfig) -> None:
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
    roles = [piece.role for piece in cfg.pieces]
    if roles != ["king", "queen", "bishop", "knight", "rook", "pawn"]:
        problems.append(f"piece profiles must be the six roles in standard order, got {roles}")
    half_square = cfg.board.square_m / 2.0
    for piece in cfg.pieces:
        margin = half_square - piece.base_radius_m
        if margin < 0.004:
            problems.append(f"{piece.role} base leaves only {margin:.4f} m inside the square; need at least 4 mm")
        if piece.height_m <= 0.0 or piece.mass_kg <= 0.0:
            problems.append(f"{piece.role} height and mass must be positive")
    if cfg.board.contact_offset_m <= 0.0 or cfg.board.contact_offset_m > 0.001:
        problems.append("contact_offset_m must be in (0, 0.001]")
    if not 0.0 < cfg.scoring.upright_angle_deg <= 20.0:
        problems.append("upright_angle_deg must be in (0, 20]")
    if cfg.scoring.lift_clearance_m <= 0.0:
        problems.append("lift_clearance_m must be positive")
    if cfg.scoring.center_tolerance_m <= 0.0:
        problems.append("center_tolerance_m must be positive")
    expected_board_z = cfg.table.top_z_m + cfg.board.thickness_m / 2.0
    if abs(cfg.board_pose.position_xyz[2] - expected_board_z) > 1e-9:
        problems.append("board pose z must be the table top plus half the board thickness")
    camera_names = [camera.name for camera in cfg.cameras]
    if camera_names != ["overhead_cam", "front_cam", "side_cam"]:
        problems.append(f"cameras must be overhead_cam, front_cam, side_cam; got {camera_names}")

    from my_package.scoring.math3d import look_at_ros_xyzw

    for camera in cfg.cameras:
        expected = look_at_ros_xyzw(camera.eye_xyz, camera.target_xyz)
        same = sum((a - b) ** 2 for a, b in zip(camera.rotation_xyzw, expected))
        flipped = sum((a + b) ** 2 for a, b in zip(camera.rotation_xyzw, expected))
        if min(same, flipped) > 1e-12:
            problems.append(f"{camera.name} rotation does not match look_at_ros_xyzw")

    try:
        puzzles = puzzles_for(cfg)
    except ValueError as exc:
        problems.append(str(exc))
        puzzles = ()
    if puzzles and problems:
        pass
    if not problems and puzzles:
        from my_package.assets.geometry import design_report

        problems.extend(design_report(cfg)["problems"])
    if problems:
        raise ValueError("chess mate config is inconsistent:\n- " + "\n- ".join(problems))


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "chess_mate.yaml"


def with_fens(cfg: ChessMateConfig, fens: tuple[str, ...]) -> ChessMateConfig:
    updated = replace(cfg, fen_sequence=fens)
    validate(updated)
    return updated
