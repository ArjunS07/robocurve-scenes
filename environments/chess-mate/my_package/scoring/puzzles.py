"""Evaluator record for the mate-in-one sequence. No Isaac Sim imports.

Object names are fixed from the first puzzle so a later puzzle with the same
army can be posed onto the same rigid bodies. The mating square is not part of
the robot task string.
"""

from __future__ import annotations

from dataclasses import dataclass

from my_package.scoring.chess_rules import (
    parse_fen,
    piece_signature,
    square_name,
    unique_mating_move,
)


@dataclass(frozen=True)
class MatingMove:
    piece_name: str
    color: str
    role: str
    start_square: str
    destination_square: str
    uci: str


@dataclass(frozen=True)
class Puzzle:
    index: int
    fen: str
    homes: dict[str, str]
    move: MatingMove

    def diagram(self) -> str:
        from my_package.scoring.chess_rules import diagram

        return diagram(self.fen)


def index_for_reset(reset_ordinal: int, length: int) -> int:
    """Puzzle index for a reset. The first reset, ordinal 0, stays on puzzle 0."""
    if length <= 0:
        raise ValueError("the puzzle sequence is empty")
    if reset_ordinal < 0:
        raise ValueError(f"reset ordinal must be >= 0, got {reset_ordinal}")
    return reset_ordinal % length


def _names_from_first(fen: str) -> dict[tuple[str, str], tuple[str, ...]]:
    grouped: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for square, color, role in parse_fen(fen).board:
        grouped.setdefault((color, role), []).append((square_name(square), f"{color}_{role}_{square_name(square)}"))
    names: dict[tuple[str, str], tuple[str, ...]] = {}
    for key, pairs in grouped.items():
        pairs.sort()
        names[key] = tuple(name for _square, name in pairs)
    return names


def _homes_for(fen: str, names: dict[tuple[str, str], tuple[str, ...]]) -> dict[str, str]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for square, color, role in parse_fen(fen).board:
        grouped.setdefault((color, role), []).append(square_name(square))
    homes: dict[str, str] = {}
    for key, squares in grouped.items():
        squares.sort()
        labels = names.get(key)
        if labels is None or len(labels) != len(squares):
            raise ValueError(f"{fen!r} does not have the same pieces as the first puzzle ({key[0]} {key[1]})")
        for name, square in zip(labels, squares):
            homes[name] = square
    if len(homes) != sum(len(labels) for labels in names.values()):
        raise ValueError(f"{fen!r} is missing a piece group present in the first puzzle")
    return homes


def _require_white_single_piece_move(index: int, fen: str, move) -> None:
    """The task has White mate by moving one piece while every other piece stays put.

    A capture removes a second piece, castling moves a rook, and a promotion
    swaps the piece, so none of those fit the physical scorer.
    """
    placement = parse_fen(fen)
    if placement.side != "white":
        raise ValueError(f"puzzle {index} has {placement.side} to move; the task requires White to deliver mate")
    if move.castle is not None:
        raise ValueError(f"puzzle {index} mates by castling ({move.uci}), which moves two pieces")
    if move.promotion is not None:
        raise ValueError(f"puzzle {index} mates by promotion ({move.uci}), which replaces the piece")
    if move.dest in placement.occupants():
        raise ValueError(f"puzzle {index} mates by capture ({move.uci}), which removes a second piece")
    if move.role == "pawn" and move.start[0] != move.dest[0]:
        raise ValueError(f"puzzle {index} mates en passant ({move.uci}), which removes a second piece")


def build_puzzles(fens: tuple[str, ...] | list[str]) -> tuple[Puzzle, ...]:
    if not fens:
        raise ValueError("fen_sequence must contain at least one puzzle")
    signature = piece_signature(fens[0])
    names = _names_from_first(fens[0])
    puzzles = []
    for index, fen in enumerate(fens):
        if piece_signature(fen) != signature:
            raise ValueError(f"puzzle {index} does not use the same set of pieces as puzzle 0")
        homes = _homes_for(fen, names)
        move = unique_mating_move(fen)
        _require_white_single_piece_move(index, fen, move)
        start = square_name(move.start)
        dest = square_name(move.dest)
        owners = [name for name, square in homes.items() if square == start]
        if len(owners) != 1:
            raise ValueError(f"puzzle {index} start square {start} maps to {owners}")
        piece_name = owners[0]
        _color, role, _square = piece_name.split("_", 2)
        # piece_name is "{color}_{role}_{square}". Roles are a single word.
        color = piece_name.split("_")[0]
        role = piece_name.split("_")[1]
        if (color, role) != (parse_fen(fen).occupants()[move.start]):
            raise ValueError(f"{piece_name} does not match the piece on {start}")
        puzzles.append(
            Puzzle(
                index=index,
                fen=fen,
                homes=homes,
                move=MatingMove(
                    piece_name=piece_name,
                    color=color,
                    role=role,
                    start_square=start,
                    destination_square=dest,
                    uci=move.uci,
                ),
            )
        )
    return tuple(puzzles)
