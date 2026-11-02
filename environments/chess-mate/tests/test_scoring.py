"""Score the recorded mate. These tests do not import Isaac Sim."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

from my_package.assets.geometry import board_top_local_z, piece_poses
from my_package.scoring.chess_rules import mating_moves, parse_fen, parse_square
from my_package.scoring.puzzles import build_puzzles
from my_package.scoring.config import default_config_path, load_config, profile_for, puzzles_for, with_fens
from my_package.tasks.state_values import env_indices, quaternion_xyzw, sim_quaternion
from my_package.scoring.math3d import local_to_world, quat_from_axis_angle, quat_mul
from my_package.scoring.runtime_state import clear_all, note_reset
from my_package.scoring.scorer import arrangement_centered, square_center_local, step
from my_package.scoring.snapshot import PieceSnapshot, SceneSnapshot

FEN = "rnbqkbnr/ppppp3/5p1p/6p1/4PN1P/3NB1P1/PPPP1P2/R1BQRK2 w kq - 0 2"
BLACK_TO_MOVE_FEN = "r1bqrk2/pppp1p2/3nb1p1/4pn1p/6P1/5P1P/PPPPP3/RNBQKBNR b KQ - 0 2"
QUEEN = "white_queen_d1"
PAWN = "white_pawn_a2"


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
    def test_only_white_queen_d1_h5_mates(self):
        cfg = _cfg()
        puzzle = puzzles_for(cfg)[0]
        self.assertEqual(puzzle.move.uci, "d1h5")
        self.assertEqual(puzzle.move.piece_name, QUEEN)
        self.assertEqual(puzzle.move.start_square, "d1")
        self.assertEqual(puzzle.move.destination_square, "h5")
        self.assertEqual(puzzle.move.color, "white")
        placement = parse_fen(FEN)
        self.assertEqual(placement.side, "white")
        mates = mating_moves(placement)
        self.assertEqual([move.uci for move in mates], ["d1h5"])
        self.assertNotIn(parse_square("h5"), placement.occupants())

    def test_black_to_move_has_no_mate(self):
        parts = FEN.split()
        parts[1] = "b"
        parts[2] = "-"
        mates = mating_moves(parse_fen(" ".join(parts)))
        self.assertEqual(list(mates), [])

    def test_puzzle_validation_rejects_wrong_side_and_captures(self):
        with self.assertRaisesRegex(ValueError, "black to move"):
            build_puzzles([BLACK_TO_MOVE_FEN])
        scholars = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4"
        with self.assertRaisesRegex(ValueError, "capture"):
            build_puzzles([scholars])

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
        position, rotation = _on_square(self.cfg, "h5", "white", board_top_local_z(self.cfg) + 0.04)
        lifted = _home(self.cfg, self.puzzle, **{QUEEN: {"position_xyz": position, "rotation_xyzw": rotation}})
        carried, report = step(self.cfg, lifted, 0.1, self.puzzle)
        self.assertEqual(report.score, 50)
        rested_position, _rotation = _on_square(self.cfg, "h5", "white")
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
        air, rotation = _on_square(self.cfg, "h5", "white", board_top_local_z(self.cfg) + 0.04)
        lifted = _home(
            self.cfg,
            self.puzzle,
            **{QUEEN: {"position_xyz": air, "rotation_xyzw": rotation, "gripper_holding": True, "released": False}},
        )
        carried, report = step(self.cfg, lifted, 0.1, self.puzzle)
        self.assertEqual(report.score, 75)
        rested, _rotation = _on_square(self.cfg, "h5", "white")
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
        pawn = PAWN
        home = next(pose for pose in piece_poses(self.cfg, self.puzzle) if pose.name == pawn)
        lifted_z = (home.position_xyz[0], home.position_xyz[1], home.position_xyz[2] + 0.03)
        snapshot = _home(self.cfg, self.puzzle, **{pawn: {"position_xyz": lifted_z}})
        _updated, report = step(self.cfg, snapshot, 0.1, self.puzzle)
        self.assertEqual(report.score, 25)
        self.assertFalse(report.ever_lifted[QUEEN])
        self.assertTrue(report.ever_lifted[pawn])

    def test_two_transfers_override_a_finished_mate(self):
        queen_pos, queen_rot = _on_square(self.cfg, "h5", "white")
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
            elif piece.name == PAWN:
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
        position, rotation = _on_square(self.cfg, "h5", "white")
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

    def _lift_then_place(self, **queen_changes):
        position, rotation = _on_square(self.cfg, "h5", "white", board_top_local_z(self.cfg) + 0.04)
        lifted = _home(self.cfg, self.puzzle, **{QUEEN: {"position_xyz": position, "rotation_xyzw": rotation}})
        carried, _report = step(self.cfg, lifted, 0.1, self.puzzle)
        rested, _rotation = _on_square(self.cfg, "h5", "white")
        values = {"position_xyz": rested, "rotation_xyzw": rotation, "gripper_holding": False, "released": True}
        values.update(queen_changes)
        return replace(
            carried,
            pieces=tuple(replace(piece, **values) if piece.name == QUEEN else piece for piece in carried.pieces),
        )

    def test_sliding_the_queen_after_lifting_a_pawn_is_not_success(self):
        pawn_home = next(pose for pose in piece_poses(self.cfg, self.puzzle) if pose.name == PAWN)
        raised = (pawn_home.position_xyz[0], pawn_home.position_xyz[1], pawn_home.position_xyz[2] + 0.03)
        carried, report = step(self.cfg, _home(self.cfg, self.puzzle, **{PAWN: {"position_xyz": raised}}), 0.1, self.puzzle)
        self.assertEqual(report.score, 25)
        slid, rotation = _on_square(self.cfg, "h5", "white")
        moved = replace(
            carried,
            pieces=tuple(
                replace(piece, position_xyz=slid, rotation_xyzw=rotation)
                if piece.name == QUEEN
                else replace(piece, position_xyz=pawn_home.position_xyz)
                if piece.name == PAWN
                else piece
                for piece in carried.pieces
            ),
        )
        _final, report = step(self.cfg, moved, 0.1, self.puzzle)
        self.assertFalse(report.ever_lifted[QUEEN])
        self.assertEqual(report.score, 25)
        held = replace(moved, pieces=tuple(replace(p, gripper_holding=True, released=False) if p.name == QUEEN else p for p in moved.pieces))
        _final, report = step(self.cfg, held, 0.1, self.puzzle)
        self.assertEqual(report.score, 25)

    def test_lifted_and_replaced_other_piece_does_not_block_success(self):
        placed = self._lift_then_place()
        placed = replace(placed, pieces=tuple(replace(p, ever_lifted=True) if p.name == PAWN else p for p in placed.pieces))
        _final, report = step(self.cfg, placed, 0.1, self.puzzle)
        self.assertTrue(report.ever_lifted[PAWN])
        self.assertEqual(report.score, 100)

    def test_unknown_gripper_state_never_scores_100(self):
        placed = self._lift_then_place(hold_known=False, released=False)
        _final, report = step(self.cfg, placed, 0.1, self.puzzle)
        self.assertNotEqual(report.score, 100)
        self.assertEqual(report.score, 75)
        # A stale released flag cannot override an unknown hold.
        placed = self._lift_then_place(hold_known=False, released=True)
        _final, report = step(self.cfg, placed, 0.1, self.puzzle)
        self.assertNotEqual(report.score, 100)

    def test_toppled_neighbour_in_its_own_square_blocks_success(self):
        placed = self._lift_then_place()
        pawn = next(p for p in placed.pieces if p.name == PAWN)
        knocked = quat_mul(pawn.rotation_xyzw, quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(60.0)))
        placed = replace(placed, pieces=tuple(replace(p, rotation_xyzw=knocked) if p.name == PAWN else p for p in placed.pieces))
        _final, report = step(self.cfg, placed, 0.1, self.puzzle)
        self.assertEqual(report.score, 50)

    def test_lowest_point_uses_cylinder_height_when_inverted(self):
        home = next(pose for pose in piece_poses(self.cfg, self.puzzle) if pose.name == QUEEN)
        height = profile_for(self.cfg, "queen").height_m
        # Upside down with the base 2 cm above the board: the head is still on the board.
        flipped = quat_mul(home.rotation_xyzw, quat_from_axis_angle((1.0, 0.0, 0.0), math.pi))
        raised = (home.position_xyz[0], home.position_xyz[1], home.position_xyz[2] + height)
        _updated, report = step(
            self.cfg,
            _home(self.cfg, self.puzzle, **{QUEEN: {"position_xyz": raised, "rotation_xyzw": flipped}}),
            0.1,
            self.puzzle,
        )
        self.assertFalse(report.ever_lifted[QUEEN])


class StateValueTests(unittest.TestCase):
    def test_env_indices_accept_lists_tuples_none_and_slices(self):
        self.assertEqual(env_indices([2, 0], 4), [2, 0])
        self.assertEqual(env_indices((1,), 4), [1])
        self.assertEqual(env_indices(None, 3), [0, 1, 2])
        self.assertEqual(env_indices(slice(None), 2), [0, 1])

    def test_env_indices_accept_tensor_like_values(self):
        class FakeTensor:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return list(self.values)

        self.assertEqual(env_indices(FakeTensor([3, 1]), 4), [3, 1])

    def test_quaternion_order_round_trips(self):
        xyzw = (0.1, 0.2, 0.3, 0.9)
        self.assertEqual(quaternion_xyzw(sim_quaternion(xyzw, "xyzw"), "xyzw"), xyzw)
        self.assertEqual(sim_quaternion(xyzw, "wxyz"), (0.9, 0.1, 0.2, 0.3))
        self.assertEqual(quaternion_xyzw((0.9, 0.1, 0.2, 0.3), "wxyz"), xyzw)
        self.assertEqual(quaternion_xyzw([0.0, 0.0, 0.0, 1.0]), (0.0, 0.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            quaternion_xyzw([1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
