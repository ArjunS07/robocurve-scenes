"""Geometry and action-map tests. No Isaac Sim imports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from my_package.assets.geometry import design_report, write_geometry
from my_package.scoring.action_map import ACTION_DIM, map_config_action
from my_package.scoring.config import default_config_path, load_config, with_single_slot


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(default_config_path())

    def test_dual_design_is_consistent(self):
        report = design_report(self.cfg)
        self.assertTrue(report["ok"], report["problems"])
        self.assertGreater(report["numbers"]["channel_width_m"], self.cfg.battery.diameter_m)
        self.assertLess(report["numbers"]["top_opening_m"], self.cfg.battery.diameter_m)
        self.assertGreaterEqual(report["numbers"]["back_gap_m"], 0.0008)
        self.assertAlmostEqual(report["numbers"]["full_seat_depth_m"], 0.03875)
        self.assertAlmostEqual(report["numbers"]["silhouette_length_m"], 0.235)
        self.assertLess(report["numbers"]["nose_x_m"], self.cfg.compartment.rear_face_x)

    def test_single_slot_design_and_usda(self):
        single = with_single_slot(self.cfg, True)
        report = design_report(single)
        self.assertTrue(report["ok"], report["problems"])
        with tempfile.TemporaryDirectory() as directory:
            paths = write_geometry(single, directory)
            text = paths["remote"].read_text()
        self.assertIn("slot_a_plunger", text)
        self.assertNotIn("slot_b_plunger", text)
        self.assertNotIn("PhysicsFixedJoint", text)

    def test_dual_usda_has_both_plungers_and_no_weld(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_geometry(self.cfg, directory)
            remote = paths["remote"].read_text()
            battery = paths["battery"].read_text()
            marks = paths["marks"].read_text()
        self.assertIn('def Cube "nose"', remote)
        self.assertIn('def Cube "ir_window"', remote)
        self.assertIn('def PhysicsPrismaticJoint "slot_a_plunger"', remote)
        self.assertIn('def PhysicsPrismaticJoint "slot_b_plunger"', remote)
        self.assertNotIn("PhysicsFixedJoint", remote)
        self.assertNotIn("PhysicsFixedJoint", battery)
        self.assertIn('def Cylinder "body"', battery)
        self.assertIn("PhysicsCollisionAPI", battery)
        self.assertNotIn("PhysicsCollisionAPI", battery.split('def Cylinder "positive_nub"')[1])
        self.assertNotIn("PhysicsRigidBodyAPI", marks)
        self.assertNotIn("PhysicsCollisionAPI", marks)


class ActionMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(default_config_path())

    def test_rejects_the_wrong_width(self):
        with self.assertRaises(ValueError):
            map_config_action(self.cfg, (0.0,) * 16)

    def test_zero_action_holds_home_and_open_fingers(self):
        targets = map_config_action(self.cfg, (0.0,) * ACTION_DIM)
        self.assertEqual(len(targets), 16)
        for index, home in enumerate(self.cfg.robot.home_arm_rad, start=1):
            self.assertAlmostEqual(targets[("robot", f"joint{index}")], home)
            self.assertAlmostEqual(targets[("robot_right", f"joint{index}")], home)
        for asset in ("robot", "robot_right"):
            self.assertAlmostEqual(targets[(asset, "joint7")], self.cfg.robot.finger_open_m)
            self.assertAlmostEqual(targets[(asset, "joint8")], self.cfg.robot.finger_open_m)
            self.assertEqual(targets[(asset, "joint7")], targets[(asset, "joint8")])

    def test_gripper_command_closes_only_that_arm(self):
        action = [0.0] * ACTION_DIM
        action[6] = -1.0
        targets = map_config_action(self.cfg, action)
        self.assertAlmostEqual(targets[("robot", "joint7")], self.cfg.robot.finger_closed_m)
        self.assertAlmostEqual(targets[("robot", "joint8")], self.cfg.robot.finger_closed_m)
        self.assertAlmostEqual(targets[("robot_right", "joint7")], self.cfg.robot.finger_open_m)
        self.assertAlmostEqual(targets[("robot", "joint1")], self.cfg.robot.home_arm_rad[0])


class PackageContractTests(unittest.TestCase):
    def test_pure_modules_do_not_import_isaac(self):
        root = Path(__file__).resolve().parents[1]
        pure_roots = [
            root / "my_package" / "scoring",
            root / "my_package" / "assets" / "geometry.py",
            root / "tests",
        ]
        files = []
        for path in pure_roots:
            if path.is_dir():
                files.extend(path.rglob("*.py"))
            else:
                files.append(path)
        banned = ("import isaaclab", "import omni", "from pxr", "import isaacsim")
        for path in files:
            for line in path.read_text().splitlines():
                stripped = line.strip()
                for token in banned:
                    self.assertFalse(stripped.startswith(token), f"{path} contains {token}")

    def test_launch_script_matches_this_arena_install(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "launch_battery_insertion.py").read_text()
        self.assertLess(text.index("import torch"), text.index("import runpy"))
        argv = text.split("argv = [", 1)[1].split("]", 1)[0]
        self.assertIn("--visualizer", argv)
        self.assertIn('"none"', argv)
        self.assertNotIn("--headless", argv)
        self.assertIn("zero_action", text)
        self.assertIn("--no_solve_relations", text)

    def test_yaml_names_the_provisional_thresholds(self):
        text = (default_config_path()).read_text()
        for token in (
            "episode_length_s: 90.0",
            "retention_window_s",
            "damage_force_proxy_n",
            "overhead_cam",
            "front_cam",
            "side_cam",
            "provisional",
            "finger_open_m",
            "contact_offset_m",
        ):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
