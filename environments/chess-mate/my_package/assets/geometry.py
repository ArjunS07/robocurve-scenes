"""Procedural board and Staunton-like pieces. No Isaac Sim imports.

The board is one rigid slab with visual squares on top. Pieces are separate
rigid bodies. Nothing welds a piece to the board and nothing teleports a piece
onto the mating square. Dimensions are a provisional likeness of a light-wood
3-in-1 set with dark squares, not a measured product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from my_package.scoring.config import ChessMateConfig, profile_for, puzzles_for
from my_package.scoring.math3d import local_to_world, quat_mul
from my_package.scoring.puzzles import Puzzle
from my_package.scoring.scorer import square_center_local

_BLACK_YAW = (0.0, 0.0, 1.0, 0.0)


@dataclass(frozen=True)
class PiecePose:
    name: str
    color: str
    role: str
    square: str
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


def board_half_extent(cfg: ChessMateConfig) -> float:
    return 4.0 * cfg.board.square_m + cfg.board.border_m


def board_top_local_z(cfg: ChessMateConfig) -> float:
    return cfg.board.thickness_m / 2.0


def piece_pose(cfg: ChessMateConfig, puzzle: Puzzle, name: str) -> PiecePose:
    square = puzzle.homes[name]
    color, role, _square = name.split("_", 2)
    center = square_center_local(cfg, square)
    local = (center[0], center[1], board_top_local_z(cfg))
    position = local_to_world(cfg.board_pose.position_xyz, cfg.board_pose.rotation_xyzw, local)
    rotation = cfg.board_pose.rotation_xyzw
    if color == "black":
        rotation = quat_mul(rotation, _BLACK_YAW)
    return PiecePose(name, color, role, square, position, rotation)


def piece_poses(cfg: ChessMateConfig, puzzle: Puzzle) -> tuple[PiecePose, ...]:
    return tuple(piece_pose(cfg, puzzle, name) for name in sorted(puzzle.homes))


def design_report(cfg: ChessMateConfig) -> dict:
    problems: list[str] = []
    puzzles = puzzles_for(cfg)
    half = board_half_extent(cfg)
    center_x, center_y = cfg.board_pose.position_xyz[0], cfg.board_pose.position_xyz[1]
    table_x, table_y = cfg.table.center_xy
    table_hx, table_hy = cfg.table.half_extents_xy
    if center_x - half < table_x - table_hx or center_x + half > table_x + table_hx:
        problems.append("the board hangs off the table in x")
    if center_y - half < table_y - table_hy or center_y + half > table_y + table_hy:
        problems.append("the board hangs off the table in y")
    poses = piece_poses(cfg, puzzles[0])
    if len(poses) != 32:
        problems.append(f"expected 32 pieces, got {len(poses)}")
    by_square = {}
    for pose in poses:
        if pose.square in by_square:
            problems.append(f"two pieces claim {pose.square}")
        by_square[pose.square] = pose
        if abs(pose.position_xyz[2] - (cfg.table.top_z_m + cfg.board.thickness_m)) > 1e-9:
            problems.append(f"{pose.name} does not sit on the board top")
        profile = profile_for(cfg, pose.role)
        if profile.base_radius_m >= cfg.board.square_m / 2.0:
            problems.append(f"{pose.role} base does not fit in a square")
    for index, first in enumerate(poses):
        for second in poses[index + 1 :]:
            gap = math.hypot(first.position_xyz[0] - second.position_xyz[0], first.position_xyz[1] - second.position_xyz[1])
            need = profile_for(cfg, first.role).base_radius_m + profile_for(cfg, second.role).base_radius_m + 0.001
            if gap < need:
                problems.append(f"{first.name} overlaps {second.name}")
    white_king = next(pose for pose in poses if pose.role == "king" and pose.color == "white")
    black_king = next(pose for pose in poses if pose.role == "king" and pose.color == "black")
    if white_king.position_xyz[0] >= black_king.position_xyz[0]:
        problems.append("white pieces are not nearer the robot than black")
    dest = puzzles[0].move.destination_square
    if dest in by_square:
        problems.append(f"mating square {dest} is occupied at the start")
    return {"ok": not problems, "problems": problems, "piece_count": len(poses)}


def write_geometry(cfg: ChessMateConfig, directory: str | Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = design_report(cfg)
    if not report["ok"]:
        raise ValueError("refusing to write geometry:\n- " + "\n- ".join(report["problems"]))
    puzzles = puzzles_for(cfg)
    board_path = directory / "board.usda"
    marks_path = directory / "table_marks.usda"
    board_path.write_text(_board_usda(cfg))
    marks_path.write_text(_marks_usda(cfg))
    paths = {"board": board_path, "marks": marks_path}
    for color, rgb in (("white", cfg.white_rgb), ("black", cfg.black_rgb)):
        for profile in cfg.pieces:
            path = directory / f"{color}_{profile.role}.usda"
            path.write_text(_piece_usda(cfg, profile.role, rgb))
            paths[f"{color}_{profile.role}"] = path
    diagram_path = directory / "setup_diagram.txt"
    record_path = directory / "puzzle_record.txt"
    diagram_path.write_text("\n\n".join(f"puzzle {puzzle.index}\n{puzzle.diagram()}" for puzzle in puzzles) + "\n")
    record_path.write_text(_record_text(puzzles))
    paths["diagram"] = diagram_path
    paths["record"] = record_path
    return paths


def _record_text(puzzles: tuple[Puzzle, ...]) -> str:
    blocks = []
    for puzzle in puzzles:
        move = puzzle.move
        blocks.append(
            "\n".join(
                [
                    f"puzzle {puzzle.index}",
                    f"fen {puzzle.fen}",
                    f"piece {move.piece_name}",
                    f"start {move.start_square}",
                    f"destination {move.destination_square}",
                    f"uci {move.uci}",
                    puzzle.diagram(),
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def _fmt(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _rgb(color: tuple[float, float, float]) -> str:
    return ", ".join(_fmt(channel) for channel in color)


def _cube(name: str, center: tuple[float, float, float], size: tuple[float, float, float], color, collision: bool) -> str:
    schemas = '\n        prepend apiSchemas = ["PhysicsCollisionAPI"]' if collision else ""
    header = f'    def Cube "{name}" ({schemas}\n    )' if collision else f'    def Cube "{name}"'
    contact = f"\n        float physxCollision:contactOffset = {_fmt(0.0003)}" if collision else ""
    return f"""{header}
    {{
        double size = 1
        double3 xformOp:translate = ({_fmt(center[0])}, {_fmt(center[1])}, {_fmt(center[2])})
        float3 xformOp:scale = ({_fmt(size[0])}, {_fmt(size[1])}, {_fmt(size[2])})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [({_rgb(color)})]{contact}
    }}"""


def _cylinder(name: str, radius: float, height: float, center_z: float, color, collision: bool, center_xy=(0.0, 0.0)) -> str:
    schemas = '\n        prepend apiSchemas = ["PhysicsCollisionAPI"]' if collision else ""
    header = f'    def Cylinder "{name}" ({schemas}\n    )' if collision else f'    def Cylinder "{name}"'
    contact = f"\n        float physxCollision:contactOffset = {_fmt(0.0003)}" if collision else ""
    return f"""{header}
    {{
        uniform token axis = "Z"
        double height = {_fmt(height)}
        double radius = {_fmt(radius)}
        double3 xformOp:translate = ({_fmt(center_xy[0])}, {_fmt(center_xy[1])}, {_fmt(center_z)})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [({_rgb(color)})]{contact}
    }}"""


def _board_usda(cfg: ChessMateConfig) -> str:
    half = board_half_extent(cfg)
    overall = half * 2.0
    thickness = cfg.board.thickness_m
    cubes = [
        _cube(
            "slab",
            (0.0, 0.0, 0.0),
            (overall, overall, thickness),
            cfg.frame_rgb,
            True,
        )
    ]
    inset = cfg.board.square_m * 0.96
    for file_index in range(8):
        for rank_index in range(8):
            square = "abcdefgh"[file_index] + str(rank_index + 1)
            center = square_center_local(cfg, square)
            dark = (file_index + rank_index) % 2 == 0
            color = cfg.dark_square_rgb if dark else cfg.light_square_rgb
            cubes.append(
                _cube(
                    f"square_{square}",
                    (center[0], center[1], thickness / 2.0 - 0.0002),
                    (inset, inset, 0.0004),
                    color,
                    False,
                )
            )
    body = "\n".join(cubes)
    return f"""#usda 1.0
(
    defaultPrim = "board"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
    doc = "Provisional wooden chessboard. Squares are visual. The slab is the collider."
)

def Xform "board" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)
{{
    float physics:mass = {_fmt(cfg.board.mass_kg)}
    float physics:staticFriction = {_fmt(cfg.board.static_friction)}
    float physics:dynamicFriction = {_fmt(cfg.board.dynamic_friction)}
    float physics:restitution = {_fmt(cfg.board.restitution)}
    int physxRigidBody:solverPositionIterationCount = {cfg.board.solver_position_iterations}
    int physxRigidBody:solverVelocityIterationCount = {cfg.board.solver_velocity_iterations}
{body}
}}
"""


def _piece_usda(cfg: ChessMateConfig, role: str, color: tuple[float, float, float]) -> str:
    profile = profile_for(cfg, role)
    radius = profile.base_radius_m
    height = profile.height_m
    base_h = height * 0.18
    stem_h = height * 0.42
    head_h = height * 0.32
    shapes = [
        _cylinder("base", radius, base_h, base_h / 2.0, color, True),
        _cylinder("stem", radius * 0.42, stem_h, base_h + stem_h / 2.0, color, True),
    ]
    head_z = base_h + stem_h + head_h / 2.0
    if role == "knight":
        head = radius * 0.7
        shapes.append(
            _cube(
                "head",
                (radius * 0.25, 0.0, head_z),
                (head, radius * 0.55, head_h),
                color,
                True,
            )
        )
    else:
        shapes.append(_cylinder("head", radius * 0.72, head_h, head_z, color, True))
    if role == "king":
        arm = radius * 0.5
        top = base_h + stem_h + head_h
        shapes.append(_cube("cross_up", (0.0, 0.0, top + arm * 0.35), (arm * 0.22, arm * 0.22, arm), color, False))
        shapes.append(_cube("cross_bar", (0.0, 0.0, top + arm * 0.45), (arm, arm * 0.22, arm * 0.22), color, False))
    body = "\n".join(shapes)
    return f"""#usda 1.0
(
    defaultPrim = "piece"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
    doc = "Provisional {role}. Origin is the center of the bottom face. The mesh stays inside the base radius."
)

def Xform "piece" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)
{{
    float physics:mass = {_fmt(profile.mass_kg)}
    float physics:staticFriction = {_fmt(cfg.board.static_friction)}
    float physics:dynamicFriction = {_fmt(cfg.board.dynamic_friction)}
    float physics:restitution = {_fmt(cfg.board.restitution)}
    int physxRigidBody:solverPositionIterationCount = {cfg.board.solver_position_iterations}
    int physxRigidBody:solverVelocityIterationCount = {cfg.board.solver_velocity_iterations}
{body}
}}
"""


def _marks_usda(cfg: ChessMateConfig) -> str:
    half = board_half_extent(cfg) + 0.004
    top = cfg.table.top_z_m + 0.0004
    mark = _cube(
        "board_mark",
        (cfg.board_pose.position_xyz[0], cfg.board_pose.position_xyz[1], top),
        (half * 2.0, half * 2.0, 0.0004),
        (0.15, 0.45, 0.22),
        False,
    )
    return f"""#usda 1.0
(
    defaultPrim = "marks"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "Visual outline of the board mark. No physics and no mating-square marker."
)

def Xform "marks"
{{
{mark}
}}
"""
