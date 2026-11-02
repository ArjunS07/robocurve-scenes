"""Score one control step of the mate-in-one task. No Isaac Sim imports.

100 requires the recorded piece upright, released, and resting with its base
entirely inside the mating square, and every other piece still on its own
square. 75 is that pose while a gripper is still holding it. 50 is a lift of
the recorded piece. 25 is a lift of some other piece. Off the table, a board
that leaves its mark, and two pieces each fully on a new square force 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from my_package.scoring.chess_rules import parse_square
from my_package.scoring.config import ChessMateConfig, profile_for
from my_package.scoring.math3d import add, angle_deg, norm, rotate, sub, world_to_local, yaw_error_deg
from my_package.scoring.puzzles import Puzzle
from my_package.scoring.snapshot import PieceSnapshot, SceneSnapshot


@dataclass(frozen=True)
class ScoreReport:
    score: int
    hard_failure: bool
    hard_reasons: tuple[str, ...]
    correct_piece: str
    destination_square: str
    transferred: tuple[str, ...]
    ever_lifted: dict[str, bool]
    ever_off_table: bool
    ever_board_moved: bool


def square_center_local(cfg: ChessMateConfig, square: str) -> tuple[float, float]:
    """Board-frame center. +X points toward rank 8, +Y toward the a-file."""
    file_index, rank_index = parse_square(square)
    size = cfg.board.square_m
    return ((rank_index - 3.5) * size, (3.5 - file_index) * size)


def base_inside_square(local_xy: tuple[float, float], center_xy: tuple[float, float], radius: float, square: float) -> bool:
    limit = square / 2.0 - radius
    if limit <= 0.0:
        return False
    return abs(local_xy[0] - center_xy[0]) <= limit + 1e-9 and abs(local_xy[1] - center_xy[1]) <= limit + 1e-9


def _tilt_deg(rotation_xyzw: tuple[float, float, float, float]) -> float:
    return angle_deg(rotate(rotation_xyzw, (0.0, 0.0, 1.0)), (0.0, 0.0, 1.0))


def _lowest_z(position_xyz: tuple[float, float, float], rotation_xyzw, radius: float) -> float:
    tilt = math.radians(_tilt_deg(rotation_xyzw))
    return position_xyz[2] - radius * math.sin(tilt)


def _board_top(snapshot: SceneSnapshot, cfg: ChessMateConfig) -> tuple[float, float, float]:
    return add(snapshot.board_position_xyz, rotate(snapshot.board_rotation_xyzw, (0.0, 0.0, cfg.board.thickness_m / 2.0)))


def _off_table(cfg: ChessMateConfig, position_xyz: tuple[float, float, float]) -> bool:
    center_x, center_y = cfg.table.center_xy
    half_x, half_y = cfg.table.half_extents_xy
    outside = (
        position_xyz[0] < center_x - half_x
        or position_xyz[0] > center_x + half_x
        or position_xyz[1] < center_y - half_y
        or position_xyz[1] > center_y + half_y
    )
    fell = position_xyz[2] < cfg.table.top_z_m - cfg.table.fall_below_m
    return outside or fell


def _square_of(cfg: ChessMateConfig, local_xy: tuple[float, float], radius: float) -> str | None:
    files = "abcdefgh"
    found = None
    for file_index in range(8):
        for rank_index in range(8):
            square = files[file_index] + str(rank_index + 1)
            center = square_center_local(cfg, square)
            if base_inside_square(local_xy, center, radius, cfg.board.square_m):
                if found is not None:
                    return None
                found = square
    return found


def is_released(piece: PieceSnapshot) -> bool:
    return piece.released and not piece.gripper_holding


def carry_episode_memory(previous: SceneSnapshot | None, observed: SceneSnapshot) -> SceneSnapshot:
    if previous is None:
        return observed
    prior = {piece.name: piece for piece in previous.pieces}
    pieces = []
    for piece in observed.pieces:
        old = prior.get(piece.name)
        if old is None:
            pieces.append(piece)
        else:
            pieces.append(replace(piece, ever_lifted=old.ever_lifted))
    return replace(
        observed,
        pieces=tuple(pieces),
        ever_off_table=previous.ever_off_table,
        ever_board_moved=previous.ever_board_moved,
    )


def arrangement_errors(cfg: ChessMateConfig, snapshot: SceneSnapshot, puzzle: Puzzle) -> dict[str, float]:
    """Horizontal distance from each piece origin to its recorded square center."""
    errors = {}
    for piece in snapshot.pieces:
        local = world_to_local(snapshot.board_position_xyz, snapshot.board_rotation_xyzw, piece.position_xyz)
        center = square_center_local(cfg, puzzle.homes[piece.name])
        errors[piece.name] = math.hypot(local[0] - center[0], local[1] - center[1])
    return errors


def arrangement_centered(cfg: ChessMateConfig, snapshot: SceneSnapshot, puzzle: Puzzle) -> bool:
    limit = cfg.scoring.center_tolerance_m
    return all(error <= limit for error in arrangement_errors(cfg, snapshot, puzzle).values())


def step(cfg: ChessMateConfig, snapshot: SceneSnapshot, dt: float, puzzle: Puzzle) -> tuple[SceneSnapshot, ScoreReport]:
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    present = {piece.name for piece in snapshot.pieces}
    missing = sorted(set(puzzle.homes) - present)
    extra = sorted(present - set(puzzle.homes))
    if missing or extra:
        raise KeyError(f"snapshot pieces do not match the puzzle (missing {missing}, extra {extra})")

    top = _board_top(snapshot, cfg)
    board_shift = norm(sub(snapshot.board_position_xyz, cfg.board_pose.position_xyz))
    board_yaw = yaw_error_deg(snapshot.board_rotation_xyzw, cfg.board_pose.rotation_xyzw)
    board_moved_now = (
        board_shift > cfg.scoring.board_motion_tolerance_m or board_yaw > cfg.scoring.board_yaw_tolerance_deg
    )
    off_now = False
    updated: list[PieceSnapshot] = []
    transferred: list[str] = []
    states: dict[str, dict] = {}
    for piece in snapshot.pieces:
        profile = profile_for(cfg, puzzle.move.role if piece.name == puzzle.move.piece_name else _role_of(piece.name))
        local = world_to_local(snapshot.board_position_xyz, snapshot.board_rotation_xyzw, piece.position_xyz)
        lowest = _lowest_z(piece.position_xyz, piece.rotation_xyzw, profile.base_radius_m)
        lifted_now = lowest > top[2] + cfg.scoring.lift_clearance_m
        ever = piece.ever_lifted or lifted_now
        if piece.off_table or _off_table(cfg, piece.position_xyz):
            off_now = True
        square = _square_of(cfg, (local[0], local[1]), profile.base_radius_m)
        home = puzzle.homes[piece.name]
        if square is not None and square != home:
            transferred.append(piece.name)
        upright = _tilt_deg(piece.rotation_xyzw) <= cfg.scoring.upright_angle_deg
        rested = abs(local[2] - cfg.board.thickness_m / 2.0) <= cfg.scoring.rest_z_tolerance_m
        on_home = square == home and not lifted_now and rested
        states[piece.name] = {
            "square": square,
            "upright": upright,
            "rested": rested,
            "lifted_now": lifted_now,
            "on_home": on_home,
        }
        updated.append(replace(piece, ever_lifted=ever))

    ever_off = snapshot.ever_off_table or off_now
    ever_board = snapshot.ever_board_moved or board_moved_now
    new_snapshot = replace(snapshot, pieces=tuple(updated), ever_off_table=ever_off, ever_board_moved=ever_board)
    lifted_flags = {piece.name: piece.ever_lifted for piece in new_snapshot.pieces}
    correct = puzzle.move.piece_name
    correct_state = states[correct]
    others_remain = all(states[name]["on_home"] for name in states if name != correct)
    on_dest = (
        correct_state["square"] == puzzle.move.destination_square
        and correct_state["upright"]
    )
    released = is_released(piece_by_name_local(new_snapshot, correct))
    correct_lifted = lifted_flags[correct]
    other_lifted = any(flag for name, flag in lifted_flags.items() if name != correct)

    reasons: list[str] = []
    if ever_off:
        reasons.append("off_table")
    if ever_board:
        reasons.append("board_moved")
    if len(transferred) >= 2:
        reasons.append("multiple_transferred")
    if not any(lifted_flags.values()):
        reasons.append("never_lifted")

    nobody_lifted = not any(lifted_flags.values())
    if ever_off or ever_board or len(transferred) >= 2 or nobody_lifted:
        score = 0
    elif on_dest and released and correct_state["rested"] and others_remain:
        score = 100
    elif on_dest and not released:
        score = 75
    elif correct_lifted:
        score = 50
    elif other_lifted:
        score = 25
    else:
        score = 0

    report = ScoreReport(
        score=score,
        hard_failure=ever_off or ever_board or len(transferred) >= 2,
        hard_reasons=tuple(reasons),
        correct_piece=correct,
        destination_square=puzzle.move.destination_square,
        transferred=tuple(transferred),
        ever_lifted=lifted_flags,
        ever_off_table=ever_off,
        ever_board_moved=ever_board,
    )
    return new_snapshot, report


def _role_of(piece_name: str) -> str:
    return piece_name.split("_")[1]


def piece_by_name_local(snapshot: SceneSnapshot, name: str) -> PieceSnapshot:
    for piece in snapshot.pieces:
        if piece.name == name:
            return piece
    raise KeyError(name)
