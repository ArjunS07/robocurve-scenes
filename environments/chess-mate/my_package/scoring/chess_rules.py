"""Chess rules for the mate-in-one record. No Isaac Sim imports.

The evaluator uses this to find the one legal checkmating move. The robot task
string does not receive that move.
"""

from __future__ import annotations

from dataclasses import dataclass


FILES = "abcdefgh"
RANKS = "12345678"
ROLES = ("king", "queen", "rook", "bishop", "knight", "pawn")
ROLE_LETTER = {"king": "k", "queen": "q", "rook": "r", "bishop": "b", "knight": "n", "pawn": "p"}
LETTER_ROLE = {letter: role for role, letter in ROLE_LETTER.items()}

_SLIDERS = {
    "rook": ((1, 0), (-1, 0), (0, 1), (0, -1)),
    "bishop": ((1, 1), (1, -1), (-1, 1), (-1, -1)),
    "queen": ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)),
}
_KNIGHT = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
_KING = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass(frozen=True)
class Placement:
    """One parsed FEN. Squares are (file, rank) with a1 = (0, 0)."""

    board: tuple[tuple[tuple[int, int], str, str], ...]
    side: str
    castling: str
    en_passant: tuple[int, int] | None
    fen: str

    def occupants(self) -> dict[tuple[int, int], tuple[str, str]]:
        return {square: (color, role) for square, color, role in self.board}


@dataclass(frozen=True)
class Move:
    start: tuple[int, int]
    dest: tuple[int, int]
    role: str
    promotion: str | None = None
    castle: str | None = None

    @property
    def uci(self) -> str:
        text = square_name(self.start) + square_name(self.dest)
        if self.promotion:
            text += ROLE_LETTER[self.promotion]
        return text


def square_name(square: tuple[int, int]) -> str:
    return FILES[square[0]] + RANKS[square[1]]


def parse_square(name: str) -> tuple[int, int]:
    if len(name) != 2 or name[0] not in FILES or name[1] not in RANKS:
        raise ValueError(f"not a square: {name!r}")
    return FILES.index(name[0]), RANKS.index(name[1])


def parse_fen(fen: str) -> Placement:
    parts = fen.split()
    if len(parts) != 6:
        raise ValueError(f"FEN must have 6 fields, got {len(parts)}: {fen!r}")
    ranks = parts[0].split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN board must have 8 ranks: {fen!r}")
    board: list[tuple[tuple[int, int], str, str]] = []
    for rank_index, rank in enumerate(ranks):
        file_index = 0
        for char in rank:
            if char.isdigit():
                file_index += int(char)
                continue
            color = "white" if char.isupper() else "black"
            role = LETTER_ROLE.get(char.lower())
            if role is None:
                raise ValueError(f"bad FEN piece {char!r}")
            square = (file_index, 7 - rank_index)
            board.append((square, color, role))
            file_index += 1
        if file_index != 8:
            raise ValueError(f"FEN rank does not fill 8 files: {rank!r}")
    side = {"w": "white", "b": "black"}.get(parts[1])
    if side is None:
        raise ValueError(f"bad side to move: {parts[1]!r}")
    castling = parts[2]
    if any(char not in "KQkq-" for char in castling):
        raise ValueError(f"bad castling field: {castling!r}")
    en_passant = None if parts[3] == "-" else parse_square(parts[3])
    return Placement(board=tuple(sorted(board)), side=side, castling=castling, en_passant=en_passant, fen=fen)


def diagram(fen: str) -> str:
    """Setup diagram. Rank 8 is the far rank. This is the evaluator's picture, not a robot prompt."""
    placement = parse_fen(fen)
    occupants = placement.occupants()
    lines = []
    for rank in range(7, -1, -1):
        cells = []
        for file_index in range(8):
            piece = occupants.get((file_index, rank))
            if piece is None:
                cells.append(".")
            else:
                letter = ROLE_LETTER[piece[1]]
                cells.append(letter.upper() if piece[0] == "white" else letter)
        lines.append(f"{RANKS[rank]} " + " ".join(cells))
    lines.append("  " + " ".join(FILES))
    lines.append(f"side {placement.side}")
    return "\n".join(lines)


def _on_board(square: tuple[int, int]) -> bool:
    return 0 <= square[0] < 8 and 0 <= square[1] < 8


def _king_square(occupants: dict[tuple[int, int], tuple[str, str]], color: str) -> tuple[int, int]:
    found = [square for square, piece in occupants.items() if piece == (color, "king")]
    if len(found) != 1:
        raise ValueError(f"{color} king count is {len(found)}")
    return found[0]


def is_attacked(occupants: dict[tuple[int, int], tuple[str, str]], square: tuple[int, int], by_color: str) -> bool:
    """True when a piece of ``by_color`` attacks ``square``."""
    for origin, (color, role) in occupants.items():
        if color != by_color:
            continue
        if role in _SLIDERS:
            for step in _SLIDERS[role]:
                cursor = (origin[0] + step[0], origin[1] + step[1])
                while _on_board(cursor):
                    if cursor == square:
                        return True
                    if cursor in occupants:
                        break
                    cursor = (cursor[0] + step[0], cursor[1] + step[1])
        elif role == "knight":
            for step in _KNIGHT:
                if (origin[0] + step[0], origin[1] + step[1]) == square:
                    return True
        elif role == "king":
            for step in _KING:
                if (origin[0] + step[0], origin[1] + step[1]) == square:
                    return True
        elif role == "pawn":
            direction = 1 if color == "white" else -1
            for file_step in (-1, 1):
                if (origin[0] + file_step, origin[1] + direction) == square:
                    return True
    return False


def _apply(occupants: dict[tuple[int, int], tuple[str, str]], move: Move, placement: Placement) -> dict:
    board = dict(occupants)
    color, role = board.pop(move.start)
    if move.castle == "K":
        board.pop((7, 0))
        board[(6, 0)] = (color, "king")
        board[(5, 0)] = (color, "rook")
        return board
    if move.castle == "Q":
        board.pop((0, 0))
        board[(2, 0)] = (color, "king")
        board[(3, 0)] = (color, "rook")
        return board
    if move.castle == "k":
        board.pop((7, 7))
        board[(6, 7)] = (color, "king")
        board[(5, 7)] = (color, "rook")
        return board
    if move.castle == "q":
        board.pop((0, 7))
        board[(2, 7)] = (color, "king")
        board[(3, 7)] = (color, "rook")
        return board
    captured = move.dest
    if role == "pawn" and placement.en_passant is not None and move.dest == placement.en_passant:
        captured_rank = move.dest[1] - (1 if color == "white" else -1)
        captured = (move.dest[0], captured_rank)
    board.pop(captured, None)
    placed = move.promotion or role
    board[move.dest] = (color, placed)
    return board


def _add_slider(moves: list[Move], occupants, origin, color, role, steps) -> None:
    for step in steps:
        cursor = (origin[0] + step[0], origin[1] + step[1])
        while _on_board(cursor):
            occupant = occupants.get(cursor)
            if occupant is None:
                moves.append(Move(origin, cursor, role))
            elif occupant[0] != color:
                moves.append(Move(origin, cursor, role))
                break
            else:
                break
            cursor = (cursor[0] + step[0], cursor[1] + step[1])


def _pseudo(placement: Placement) -> list[Move]:
    occupants = placement.occupants()
    color = placement.side
    moves: list[Move] = []
    for origin, (piece_color, role) in occupants.items():
        if piece_color != color:
            continue
        if role in _SLIDERS:
            _add_slider(moves, occupants, origin, color, role, _SLIDERS[role])
        elif role == "knight":
            for step in _KNIGHT:
                dest = (origin[0] + step[0], origin[1] + step[1])
                occupant = occupants.get(dest) if _on_board(dest) else (color, "blocked")
                if _on_board(dest) and (occupant is None or occupant[0] != color):
                    moves.append(Move(origin, dest, role))
        elif role == "king":
            for step in _KING:
                dest = (origin[0] + step[0], origin[1] + step[1])
                if not _on_board(dest):
                    continue
                occupant = occupants.get(dest)
                if occupant is None or occupant[0] != color:
                    moves.append(Move(origin, dest, role))
            moves.extend(_castles(placement, occupants, color))
        elif role == "pawn":
            moves.extend(_pawn_moves(placement, occupants, origin, color))
    return moves


def _pawn_moves(placement: Placement, occupants, origin, color) -> list[Move]:
    direction = 1 if color == "white" else -1
    start_rank = 1 if color == "white" else 6
    last_rank = 7 if color == "white" else 0
    moves: list[Move] = []
    one = (origin[0], origin[1] + direction)
    if _on_board(one) and one not in occupants:
        moves.extend(_promotions(origin, one, last_rank))
        two = (origin[0], origin[1] + 2 * direction)
        if origin[1] == start_rank and two not in occupants:
            moves.append(Move(origin, two, "pawn"))
    for file_step in (-1, 1):
        dest = (origin[0] + file_step, origin[1] + direction)
        if not _on_board(dest):
            continue
        occupant = occupants.get(dest)
        if occupant is not None and occupant[0] != color:
            moves.extend(_promotions(origin, dest, last_rank))
        elif placement.en_passant is not None and dest == placement.en_passant:
            moves.append(Move(origin, dest, "pawn"))
    return moves


def _promotions(origin, dest, last_rank) -> list[Move]:
    if dest[1] != last_rank:
        return [Move(origin, dest, "pawn")]
    return [Move(origin, dest, "pawn", promotion=role) for role in ("queen", "rook", "bishop", "knight")]


def _castles(placement: Placement, occupants, color) -> list[Move]:
    rights = placement.castling
    if color == "white":
        king = (4, 0)
        specs = (("K", (6, 0), (5, 0), (7, 0), ((5, 0), (6, 0))), ("Q", (2, 0), (3, 0), (0, 0), ((1, 0), (2, 0), (3, 0))))
    else:
        king = (4, 7)
        specs = (("k", (6, 7), (5, 7), (7, 7), ((5, 7), (6, 7))), ("q", (2, 7), (3, 7), (0, 7), ((1, 7), (2, 7), (3, 7))))
    if occupants.get(king) != (color, "king"):
        return []
    opponent = "black" if color == "white" else "white"
    if is_attacked(occupants, king, opponent):
        return []
    found = []
    for right, dest, _rook_dest, rook_home, between in specs:
        if right not in rights:
            continue
        if occupants.get(rook_home) != (color, "rook"):
            continue
        if any(square in occupants for square in between):
            continue
        crossed = between[-2:]
        if any(is_attacked(occupants, square, opponent) for square in crossed):
            continue
        found.append(Move(king, dest, "king", castle=right))
    return found


def legal_moves(placement: Placement) -> tuple[Move, ...]:
    occupants = placement.occupants()
    king = _king_square(occupants, placement.side)
    opponent = "black" if placement.side == "white" else "white"
    # The side that just moved must not have left its own king in check.
    other = _king_square(occupants, opponent)
    if is_attacked(occupants, other, placement.side):
        raise ValueError(f"illegal FEN, {opponent} king is in check while {placement.side} is to move: {placement.fen}")
    legal = []
    for move in _pseudo(placement):
        nxt = _apply(occupants, move, placement)
        if not is_attacked(nxt, _king_square(nxt, placement.side), opponent):
            legal.append(move)
    del king
    return tuple(legal)


def is_checkmate(placement: Placement) -> bool:
    occupants = placement.occupants()
    opponent = "black" if placement.side == "white" else "white"
    king = _king_square(occupants, placement.side)
    if not is_attacked(occupants, king, opponent):
        return False
    return len(legal_moves(placement)) == 0


def _after(placement: Placement, move: Move) -> Placement:
    occupants = _apply(placement.occupants(), move, placement)
    board = tuple(sorted((square, color, role) for square, (color, role) in occupants.items()))
    side = "black" if placement.side == "white" else "white"
    return Placement(board=board, side=side, castling="-", en_passant=None, fen=placement.fen + " after " + move.uci)


def mating_moves(placement: Placement) -> tuple[Move, ...]:
    found = []
    for move in legal_moves(placement):
        if is_checkmate(_after(placement, move)):
            found.append(move)
    return tuple(found)


def unique_mating_move(fen: str) -> Move:
    placement = parse_fen(fen)
    mates = mating_moves(placement)
    if len(mates) != 1:
        rendered = ", ".join(move.uci for move in mates) or "none"
        raise ValueError(f"{fen!r} has {len(mates)} mating moves ({rendered}), expected one")
    return mates[0]


def piece_signature(fen: str) -> tuple[tuple[str, str, int], ...]:
    counts: dict[tuple[str, str], int] = {}
    for _square, color, role in parse_fen(fen).board:
        counts[(color, role)] = counts.get((color, role), 0) + 1
    return tuple(sorted((color, role, count) for (color, role), count in counts.items()))
