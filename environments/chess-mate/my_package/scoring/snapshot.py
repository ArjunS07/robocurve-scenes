"""Observation snapshot the chess scorer reads. No Isaac Sim imports."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PieceSnapshot:
    name: str
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    gripper_holding: bool
    released: bool
    off_table: bool
    ever_lifted: bool = False


@dataclass(frozen=True)
class SceneSnapshot:
    pieces: tuple[PieceSnapshot, ...]
    board_position_xyz: tuple[float, float, float]
    board_rotation_xyzw: tuple[float, float, float, float]
    ever_off_table: bool = False
    ever_board_moved: bool = False


def piece_by_name(snapshot: SceneSnapshot, name: str) -> PieceSnapshot:
    for piece in snapshot.pieces:
        if piece.name == name:
            return piece
    raise KeyError(name)


def clear_episode_state(snapshot: SceneSnapshot) -> SceneSnapshot:
    """Drop ever-lifted and the hard-failure latches. Poses stay as they are."""
    pieces = tuple(replace(piece, ever_lifted=False) for piece in snapshot.pieces)
    return replace(snapshot, pieces=pieces, ever_off_table=False, ever_board_moved=False)
