"""Geometry, action mapping, and launch-script checks. No Isaac Sim."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from my_package.assets import geometry
from my_package.assets.geometry import ASSET_DIR, board_half_extent, design_report, write_geometry
from my_package.scoring.action_map import map_config_action
from my_package.scoring.config import default_config_path, load_config


class GeometryTests(unittest.TestCase):
    def test_board_fits_and_white_is_nearer(self):
        cfg = load_config(default_config_path())
        report = design_report(cfg)
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["piece_count"], 32)

    def test_usda_has_no_weld_or_answer_mark(self):
        cfg = load_config(default_config_path())
        with tempfile.TemporaryDirectory() as directory:
            paths = write_geometry(cfg, directory)
            board = paths["board"].read_text()
            self.assertNotIn("PhysicsFixedJoint", board)
            self.assertNotIn("d1h5", board)
            marks = paths["marks"].read_text()
            self.assertNotIn("d1h5", marks)
            self.assertNotIn("square_h5", marks)
            record = paths["record"].read_text()
            self.assertIn("d1h5", record)
            self.assertIn("white_queen_d1", record)
            self.assertIn("destination h5", record)
            diagram = paths["diagram"].read_text()
            self.assertNotIn("d1h5", diagram)
            self.assertIn("side white", diagram)

    def test_board_top_is_one_textured_face_with_no_overlay(self):
        cfg = load_config(default_config_path())
        with tempfile.TemporaryDirectory() as directory:
            board = write_geometry(cfg, directory)["board"].read_text()
        # No per-square geometry: the old coplanar square cubes caused z-fighting.
        self.assertNotIn("def Cube", board)
        meshes = re.findall(r'def Mesh "(\w+)"', board)
        self.assertEqual(meshes, ["top", "edge", "bottom", "collider"])
        top_z = cfg.board.thickness_m / 2.0
        heights = {}
        for name in meshes:
            block = board.split(f'def Mesh "{name}"', 1)[1].split("def Mesh", 1)[0]
            points = re.search(r"point3f\[\] points = \[(.*?)\]", block).group(1)
            heights[name] = [float(z) for z in re.findall(r"\([^,]+, [^,]+, ([^)]+)\)", points)]
        # Only the top mesh has faces in the top plane; the edge only touches it at the rim.
        self.assertTrue(all(abs(z - top_z) < 1e-9 for z in heights["top"]))
        self.assertTrue(all(abs(z - top_z) > 1e-9 for z in heights["bottom"]))
        self.assertIn("board_top.png", board)
        self.assertIn('physics:approximation = "convexHull"', board)
        self.assertEqual(len(heights["collider"]) // 2, 4 * (geometry._COLLIDER_CORNER_SEGMENTS + 1))
        self.assertLessEqual(len(heights["collider"]), 64)
        self.assertAlmostEqual(max(heights["collider"]), top_z)

    def test_board_matches_the_listing_size(self):
        cfg = load_config(default_config_path())
        self.assertAlmostEqual(2.0 * board_half_extent(cfg), 10.5 * 0.0254, places=4)

    def test_texture_puts_a_dark_square_on_a1(self):
        from PIL import Image

        cfg = load_config(default_config_path())
        image = Image.open(ASSET_DIR / "textures" / "board_top.png").convert("L")
        half = board_half_extent(cfg)

        def brightness(square):
            from my_package.scoring.scorer import square_center_local

            x, y = square_center_local(cfg, square)
            col = int((x + half) / (2 * half) * image.width)
            row = int((1.0 - (y + half) / (2 * half)) * image.height)
            return image.getpixel((col, row))

        self.assertLess(brightness("a1"), 40)
        self.assertGreater(brightness("h1"), 120)
        self.assertLess(brightness("h8"), 40)

    def test_pieces_reference_committed_meshes_and_fitted_colliders(self):
        cfg = load_config(default_config_path())
        prisms = json.loads((ASSET_DIR / "pieces" / "collision.json").read_text())
        manifest = json.loads((ASSET_DIR / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            paths = write_geometry(cfg, directory)
            for color in ("white", "black"):
                for role in geometry.ROLES:
                    text = paths[f"{color}_{role}"].read_text()
                    reference = re.search(r"references = @([^@]+)@", text).group(1)
                    self.assertTrue(Path(reference).is_file(), reference)
                    self.assertEqual(text.count('physics:approximation = "convexHull"'), len(prisms[role]))
                    self.assertNotIn("def Cylinder", text)
        for role, bands in prisms.items():
            self.assertAlmostEqual(bands[0]["z0"], 0.0)
            self.assertAlmostEqual(bands[-1]["z1"], manifest["pieces"][role]["height_m"], places=6)
            for band in bands:
                self.assertLessEqual(2 * len(band["xy"]), 64)
                widest = max((x * x + y * y) ** 0.5 for x, y in band["xy"])
                self.assertLessEqual(widest, manifest["pieces"][role]["max_radius_m"] + 1e-6)
            # The bottom prism is the base: it must be about as wide as the measured base.
            base = max((x * x + y * y) ** 0.5 for x, y in bands[0]["xy"])
            self.assertGreater(base, 0.9 * manifest["pieces"][role]["base_radius_m"])

    def test_assets_record_source_and_license(self):
        manifest = json.loads((ASSET_DIR / "manifest.json").read_text())
        self.assertEqual(manifest["source"]["license"], "CC0 1.0")
        self.assertIn("polyhaven.com", manifest["source"]["page"])
        for entry in manifest["source"]["files"].values():
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("CC0 1.0 Universal", (ASSET_DIR / "LICENSE-CC0.txt").read_text())

    def test_missing_assets_fail_loudly(self):
        cfg = load_config(default_config_path())
        with tempfile.TemporaryDirectory() as empty, mock.patch.object(geometry, "ASSET_DIR", Path(empty)):
            with self.assertRaises(FileNotFoundError) as caught:
                geometry.require_assets()
            self.assertIn("build_chess_assets.py", str(caught.exception))
            with self.assertRaises(FileNotFoundError):
                write_geometry(cfg, Path(empty) / "out")

    def test_marks_stay_outside_the_board(self):
        cfg = load_config(default_config_path())
        with tempfile.TemporaryDirectory() as directory:
            marks = write_geometry(cfg, directory)["marks"].read_text()
        half = board_half_extent(cfg)
        cx, cy = cfg.board_pose.position_xyz[:2]
        for x, y, z in re.findall(r"\(([-0-9.e]+), ([-0-9.e]+), ([-0-9.e]+)\)", marks):
            inside = abs(float(x) - cx) < half and abs(float(y) - cy) < half
            self.assertFalse(inside)
            self.assertGreater(float(z), cfg.table.top_z_m)

    def test_robot_bases_must_sit_on_the_table(self):
        from dataclasses import replace

        from my_package.scoring.config import validate

        cfg = load_config(default_config_path())
        validate(cfg)
        # The old placement hung the bases 0.25 m in front of the table's near edge.
        off_table = replace(cfg.robot.left_pose, position_xyz=(-0.05, 0.22, cfg.table.top_z_m))
        with self.assertRaises(ValueError) as caught:
            validate(replace(cfg, robot=replace(cfg.robot, left_pose=off_table)))
        self.assertIn("table footprint", str(caught.exception))

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
            self.assertNotIn("d1h5", text, path.name)
            self.assertNotIn("destination h5", text, path.name)

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
