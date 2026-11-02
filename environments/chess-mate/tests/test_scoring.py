"""Score the recorded mate. These tests do not import Isaac Sim."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

from my_package.assets.geometry import board_top_local_z, piece_poses
from my_package.scoring.chess_rules import mating_moves, parse_fen
from my_package.scoring.config import load_config, puzzles_for, with_fens, default_config_path
from my_package.scoring.math3d import local_to_world, quat_from_axis_angle, quat_mul
from my_package.scoring.runtime_state import clear_all, note_reset
from my_package.scoring.scorer import arrangement_centered, square_center_local, step
from my_package.scoring.snapshot import PieceSnapshot, SceneSnapshot

FEN = "r1bqrk2/pppp1p2/3nb1p1/4pn1p/6P1/5P1P/PPPPP3/RNBQKBNR b KQ - 0 2"
QUEEN = "black_queen_d8"


def _cfg():
    return load_config(default_config_path())


def _home(cfg, puzzle, **changes) -> SceneSnapshot:
    poses = {pose.name: pose for pose in piece_poses(cfg, puzzle)}
    pieces = []
    for name, pose in poses.items():
        values = {
            "name": name,
            "position_xyz": pose.position_xyz,
            "rotation_xyzw": pose.rotation_xyzw,
            "gripper_holding": False,
            "released": True,
            "off_table": False,
        }
        values.update(changes.get(name, {}))
        pieces.append(PieceSnapshot(**values))
    return SceneSnapshot(
        pieces=tuple(pieces),
        board_position_xyz=cfg.board_pose.position_xyz,
        board_rotation_xyzw=cfg.board_pose.rotation_xyzw,
    )


def _on_square(cfg, square: str, color: str, z: float | None = None):
    center = square_center_local(cfg, square)
    local_z = board_top_local_z(cfg) if z is None else z
    position = local_to_world(cfg.board_pose.position_xyz, cfg.board_pose.rotation_xyzw, (center[0], center[1], local_z))
    rotation = cfg.board_pose.rotation_xyzw
    if color == "black":
        rotation = quat_mul(rotation, (0.0, 0.0, 1.0, 0.0))
    return position, rotation


class MateRecordTests(unittest.TestCase):
    def test_only_black_queen_d8_h4_mates(self):
        cfg = _cfg()
        puzzle = puzzles_for(cfg)[0]
        self.assertEqual(puzzle.move.uci, "d8h4")
        self.assertEqual(puzzle.move.piece_name, QUEEN)
        self.assertEqual(puzzle.move.start_square, "d8")
        self.assertEqual(puzzle.move.destination_square, "h4")
        self.assertEqual(puzzle.move.color, "black")
        mates = mating_moves(parse_fen(FEN))
        self.assertEqual([move.uci for move in mates], ["d8h4"])

    def test_white_to_move_has_no_mate(self):
        parts = FEN.split()
        parts[1] = "w"
        mates = mating_moves(parse_fen(" ".join(parts)))
        self.assertEqual(list(mates), [])

    def test_scholars_mate_is_queen_h5_f7(self):
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4"
        self.assertEqual([move.uci for move in mating_moves(parse_fen(fen))], ["h5f7"])

    def test_second_reset_advances_and_wraps(self):
        clear_all()
        self.assertEqual(note_reset(0, 1), 0)
        self.assertEqual(note_reset(0, 1), 0)
        clear_all()
        self.assertEqual(note_reset(3, 2), 0)
        self.assertEqual(note_reset(3, 2), 1)
        self.assertEqual(note_reset(3, 2), 0)
        cfg = with_fens(_cfg(), (FEN, FEN))
        puzzles = puzzles_for(cfg)
        self.assertEqual(puzzles[1].move.uci, puzzles[0].move.uci)
        self.assertEqual(puzzles[1].homes, puzzles[0].homes)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg = _cfg()
        self.puzzle = puzzles_for(self.cfg)[0]

    def _queen(self, snapshot: SceneSnapshot) -> PieceSnapshot:
        return next(piece for piece in snapshot.pieces if piece.name == QUEEN)

    def test_start_scores_zero(self):
        _updated, report = step(self.cfg, _home(self.cfg, self.puzzle), 0.1, self.puzzle)
        self.assertEqual(report.score, 0)
        self.assertIn("never_lifted", report.hard_reasons)
        self.assertFalse(report.hard_failure)
        self.assertTrue(arrangement_centered(self.cfg, _home(self.cfg, self.puzzle), self.puzzle))

    def test_full_success_after_a_lift(self):
        position, rotation = _on_square(self.cfg, "h4", "black", board_top_local_z(self.cfg) + 0.04)
        lifted = _home(self.cfg, self.puzzle, **{QUEEN: {"position_xyz": position, "rotation_xyzw": rotation}})
        carried, report = step(self.cfg, lifted, 0.1, self.puzzle)
        self.assertEqual(report.score, 50)
        rested_position, _rotation = _on_square(self.cfg, "h4", "black")
        placed = replace(
            carried,
            pieces=tuple(
                replace(piece, position_xyz=rested_position, rotation_xyzw=rotation, gripper_holding=False, released=True)
                if piece.name == QUEEN
                else piece
                for piece in carried.pieces
            ),
        )
        _final, report = step(self.cfg, placed, 0.1, self.puzzle)
        self.assertEqual(report.score, 100)
        self.assertFalse(report.hard_failure)
        self.assertEqual(report.transferred, (QUEEN,))

    def test_held_on_the_square_scores_75(self):
        air, rotation = _on_square(self.cfg, "h4", "black", board_top_local_z(self.cfg) + 0.04)
        lifted = _home(
            self.cfg,
            self.puzzle,
            **{QUEEN: {"position_xyz": air, "rotation_xyzw": rotation, "gripper_holding": True, "released": False}},
        )
        carried, report = step(self.cfg, lifted, 0.1, self.puzzle)
        self.assertEqual(report.score, 75)
        rested, _rotation = _on_square(self.cfg, "h4", "black")
        lowered = replace(
            carried,
            pieces=tuple(
                replace(piece, position_xyz=rested, gripper_holding=True, released=False) if piece.name == QUEEN else piece
                for piece in carried.pieces
            ),
        )
        _final, report = step(self.cfg, lowered, 0.1, self.puzzle)
        self.assertEqual(report.score, 75)

    def test_lift_without_the_square_scores_50(self):
        home = next(pose for pose in piece_poses(self.cfg, self.puzzle) if pose.name == QUEEN)
        lifted_z = (home.position_xyz[0], home.position_xyz[1], home.position_xyz[2] + 0.03)
        snapshot = _home(self.cfg, self.puzzle, **{QUEEN: {"position_xyz": lifted_z}})
        _updated, report = step(self.cfg, snapshot, 0.1, self.puzzle)
        self.assertEqual(report.score, 50)

    def test_other_piece_lift_scores_25(self):
        pawn = "white_pawn_a2"
        home = next(pose for pose in piece_poses(self.cfg, self.puzzle) if pose.name == pawn)
        lifted_z = (home.position_xyz[0], home.position_xyz[1], home.position_xyz[2] + 0.03)
        snapshot = _home(self.cfg, self.puzzle, **{pawn: {"position_xyz": lifted_z}})
        _updated, report = step(self.cfg, snapshot, 0.1, self.puzzle)
        self.assertEqual(report.score, 25)
        self.assertFalse(report.ever_lifted[QUEEN])
        self.assertTrue(report.ever_lifted[pawn])

    def test_two_transfers_override_a_finished_mate(self):
        queen_pos, queen_rot = _on_square(self.cfg, "h4", "black")
        pawn_pos, pawn_rot = _on_square(self.cfg, "a3", "white")
        lifted = _home(
            self.cfg,
            self.puzzle,
            **{QUEEN: {"position_xyz": (queen_pos[0], queen_pos[1], queen_pos[2] + 0.03), "rotation_xyzw": queen_rot}},
        )
        carried, _report = step(self.cfg, lifted, 0.1, self.puzzle)
        moved = []
        for piece in carried.pieces:
            if piece.name == QUEEN:
                moved.append(replace(piece, position_xyz=queen_pos, rotation_xyzw=queen_rot))
            elif piece.name == "white_pawn_a2":
                moved.append(replace(piece, position_xyz=pawn_pos, rotation_xyzw=pawn_rot))
            else:
                moved.append(piece)
        _final, report = step(self.cfg, replace(carried, pieces=tuple(moved)), 0.1, self.puzzle)
        self.assertEqual(report.score, 0)
        self.assertIn("multiple_transferred", report.hard_reasons)
        self.assertTrue(report.hard_failure)

    def test_off_table_and_board_motion_latch(self):
        fallen = _home(self.cfg, self.puzzle, **{QUEEN: {"position_xyz": (-1.0, 0.0, 0.2), "off_table": True}})
        carried, report = step(self.cfg, fallen, 0.1, self.puzzle)
        self.assertEqual(report.score, 0)
        self.assertTrue(report.ever_off_table)
        put_back = replace(carried, pieces=_home(self.cfg, self.puzzle).pieces)
        _final, report = step(self.cfg, put_back, 0.1, self.puzzle)
        self.assertEqual(report.score, 0)
        self.assertTrue(report.ever_off_table)

        yaw = quat_from_axis_angle((0.0, 0.0, 1.0), math.radians(10.0))
        shifted = replace(_home(self.cfg, self.puzzle), board_rotation_xyzw=yaw)
        carried, report = step(self.cfg, shifted, 0.1, self.puzzle)
        self.assertIn("board_moved", report.hard_reasons)
        pieces = _home(self.cfg, self.puzzle).pieces
        back = replace(
            carried,
            pieces=pieces,
            board_position_xyz=self.cfg.board_pose.position_xyz,
            board_rotation_xyzw=self.cfg.board_pose.rotation_xyzw,
        )
        _final, report = step(self.cfg, back, 0.1, self.puzzle)
        self.assertEqual(report.score, 0)
        self.assertTrue(report.ever_board_moved)

    def test_tilt_and_a_straddle_are_not_success(self):
        position, rotation = _on_square(self.cfg, "h4", "black")
        tilt = quat_mul(rotation, quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(40.0)))
        lifted = _home(
            self.cfg,
            self.puzzle,
            **{QUEEN: {"position_xyz": (position[0], position[1], position[2] + 0.04), "rotation_xyzw": rotation}},
        )
        carried, _report = step(self.cfg, lifted, 0.1, self.puzzle)
        tipped = []
        for piece in carried.pieces:
            if piece.name == QUEEN:
                tipped.append(replace(piece, position_xyz=position, rotation_xyzw=tilt))
            else:
                tipped.append(piece)
        _final, report = step(self.cfg, replace(carried, pieces=tuple(tipped)), 0.1, self.puzzle)
        self.assertNotIn(report.score, (100, 75))
        self.assertEqual(report.score, 50)

        offset = (position[0] + 0.008, position[1], position[2])
        straddling = []
        for piece in carried.pieces:
            if piece.name == QUEEN:
                straddling.append(replace(piece, position_xyz=offset, rotation_xyzw=rotation))
            else:
                straddling.append(piece)
        _final, report = step(self.cfg, replace(carried, pieces=tuple(straddling)), 0.1, self.puzzle)
        self.assertEqual(report.score, 50)
        self.assertNotIn(QUEEN, report.transferred)


if __name__ == "__main__":
    unittest.main()
