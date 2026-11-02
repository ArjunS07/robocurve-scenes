"""Geometry, action mapping, and launch-script checks. No Isaac Sim."""

from __future__ import annotations

import unittest
from pathlib import Path

from my_package.assets.geometry import design_report, write_geometry
from my_package.scoring.action_map import map_config_action
from my_package.scoring.config import default_config_path, load_config


class GeometryTests(unittest.TestCase):
    def test_board_fits_and_white_is_nearer(self):
        cfg = load_config(default_config_path())
        report = design_report(cfg)
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["piece_count"], 32)

    def test_usda_has_squares_and_no_weld_or_answer_mark(self):
        cfg = load_config(default_config_path())
        root = Path(__file__).resolve().parents[1]
        paths = write_geometry(cfg, root / "my_package" / "assets" / "generated")
        board = paths["board"].read_text()
        self.assertEqual(board.count('def Cube "square_'), 64)
        self.assertNotIn("PhysicsFixedJoint", board)
        self.assertNotIn("d8h4", board)
        marks = paths["marks"].read_text()
        self.assertNotIn("d8h4", marks)
        self.assertNotIn("square_h4", marks)
        record = paths["record"].read_text()
        self.assertIn("d8h4", record)
        self.assertIn("black_queen_d8", record)
        self.assertIn("destination h4", record)
        diagram = paths["diagram"].read_text()
        self.assertNotIn("d8h4", diagram)
        self.assertIn("side black", diagram)

    def test_zero_action_holds_home_and_open_fingers(self):
        cfg = load_config(default_config_path())
        targets = map_config_action(cfg, (0.0,) * 14)
        self.assertEqual(len(targets), 16)
        for joint in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
            self.assertAlmostEqual(targets[("robot", joint)], cfg.robot.home_arm_rad[int(joint[-1]) - 1])
        self.assertAlmostEqual(targets[("robot", "joint7")], cfg.robot.finger_open_m)
        self.assertAlmostEqual(targets[("robot_right", "joint8")], cfg.robot.finger_open_m)

    def test_gripper_command_closes_only_that_arm(self):
        cfg = load_config(default_config_path())
        action = [0.0] * 14
        action[6] = -1.0
        targets = map_config_action(cfg, action)
        self.assertAlmostEqual(targets[("robot", "joint7")], cfg.robot.finger_closed_m)
        self.assertAlmostEqual(targets[("robot", "joint8")], cfg.robot.finger_closed_m)
        self.assertAlmostEqual(targets[("robot_right", "joint7")], cfg.robot.finger_open_m)

    def test_pure_modules_do_not_import_isaac(self):
        root = Path(__file__).resolve().parents[1]
        files = list((root / "my_package" / "scoring").rglob("*.py"))
        files.append(root / "my_package" / "assets" / "geometry.py")
        files.extend((root / "tests").rglob("*.py"))
        banned = ("import isaaclab", "import omni", "from pxr", "import isaacsim")
        for path in files:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                for token in banned:
                    self.assertFalse(stripped.startswith(token), f"{path} contains {token}")

    def test_launch_script_and_task_hide_the_move(self):
        root = Path(__file__).resolve().parents[1]
        launch = (root / "launch_chess_mate.py").read_text()
        self.assertLess(launch.index("import torch"), launch.index("import runpy"))
        argv = launch.split("argv = [", 1)[1].split("]", 1)[0]
        self.assertIn("--visualizer", argv)
        self.assertIn('"none"', argv)
        self.assertNotIn("--headless", argv)
        self.assertIn("zero_action", launch)
        self.assertIn("--no_solve_relations", launch)
        hidden = (
            root / "launch_chess_mate.py",
            root / "my_package" / "tasks" / "chess_mate_task.py",
            root / "my_package" / "isaaclab_arena_environments" / "chess_environment.py",
            root / "my_package" / "config" / "chess_mate.yaml",
        )
        for path in hidden:
            text = path.read_text()
            self.assertNotIn("d8h4", text, path.name)
            self.assertNotIn("destination h4", text, path.name)

    def test_yaml_names_the_provisional_thresholds(self):
        text = default_config_path().read_text()
        for token in (
            "episode_length_s: 90.0",
            "lift_clearance_m",
            "board_motion_tolerance_m",
            "upright_angle_deg",
            "overhead_cam",
            "front_cam",
            "side_cam",
            "Provisional",
            "finger_open_m",
            "contact_offset_m",
        ):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
