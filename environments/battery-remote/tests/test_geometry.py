"""Geometry and action-map tests. No Isaac Sim imports."""

from __future__ import annotations

import importlib
import importlib.util
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from my_package.assets.geometry import _aabb, design_report, plunger_box, rest_center_z, static_boxes, write_geometry
from my_package.assets.physics_props import battery_physics, remote_physics
from my_package.scoring.action_map import ACTION_DIM, map_config_action
from my_package.scoring.config import default_config_path, load_config, validate, with_single_slot


def _prim_paths(usda: str) -> set[str]:
    """Absolute prim paths from a generated USDA, by tracking def blocks and braces."""
    paths: set[str] = set()
    stack: list[tuple[str, int]] = []
    pending: str | None = None
    depth = 0
    for line in usda.splitlines():
        match = re.match(r'\s*def\s+\w+\s+"([^"]+)"', line)
        if match:
            parent = stack[-1][0] if stack else ""
            pending = f"{parent}/{match.group(1)}"
        for char in line:
            if char == "{":
                depth += 1
                if pending is not None:
                    stack.append((pending, depth))
                    paths.add(pending)
                    pending = None
            elif char == "}":
                if stack and stack[-1][1] == depth:
                    stack.pop()
                depth -= 1
    return paths


def _joint_targets(usda: str) -> dict[str, dict[str, str]]:
    joints: dict[str, dict[str, str]] = {}
    for block in usda.split('def PhysicsPrismaticJoint "')[1:]:
        name = block.split('"', 1)[0]
        rels = dict(re.findall(r"rel physics:(body[01]) = <([^>]+)>", block.split("}", 1)[0]))
        joints[name] = rels
    return joints


def _resolve(prim: str, target: str) -> str:
    if target.startswith("/"):
        return target
    parts = prim.split("/")
    for piece in target.split("/"):
        if piece == "..":
            parts.pop()
        elif piece and piece != ".":
            parts.append(piece)
    return "/".join(parts)


def _float_after(text: str, key: str) -> float:
    return float(text.split(key, 1)[1].split("=", 1)[1].split()[0])


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(default_config_path())

    def test_dual_design_is_consistent(self):
        report = design_report(self.cfg)
        self.assertTrue(report["ok"], report["problems"])
        numbers = report["numbers"]
        self.assertGreater(numbers["channel_width_m"], self.cfg.battery.diameter_m)
        self.assertGreater(numbers["drop_slack_m"], 0.002)
        self.assertGreaterEqual(numbers["seated_rim_margin_m"], 0.0)
        self.assertAlmostEqual(numbers["silhouette_length_m"], 0.235)
        self.assertLess(numbers["nose_x_m"], self.cfg.compartment.rear_face_x)
        # The extended spring is in the way: the robot has to press it to seat the cell.
        self.assertTrue(numbers["slot_a_extended_plunger_blocks_drop"])
        self.assertTrue(numbers["slot_b_extended_plunger_blocks_drop"])

    def test_bay_has_no_ceiling_and_is_open_above_each_slot(self):
        boxes = [box for box in static_boxes(self.cfg) if box.collision]
        self.assertNotIn("ceiling", [box.name for box in boxes])
        self.assertFalse(any("lip" in box.name for box in boxes))
        top = self.cfg.compartment.size_xyz[2] / 2.0
        for slot in self.cfg.compartment.slots:
            # A point just above the seated cell is inside no collider.
            point = (self.cfg.compartment.seat_center_x, slot.center_y_m, rest_center_z(self.cfg) + 0.0055)
            for box in boxes:
                lo, hi = _aabb(box)
                inside = all(lo[i] < point[i] < hi[i] for i in range(3))
                self.assertFalse(inside, f"{box.name} covers {slot.name}")
            self.assertLess(point[2], top)

    def test_spring_sits_at_each_slots_negative_end(self):
        for slot in self.cfg.compartment.slots:
            plunger_x = plunger_box(self.cfg, slot, 0.0).center[0]
            positive_dir = slot.required_axis_xyz[0]
            self.assertLess(positive_dir * (plunger_x - self.cfg.compartment.seat_center_x), 0.0, slot.name)

    def test_plunger_is_checked_in_the_insertion_path(self):
        # Preload past the travel means even a fully compressed spring blocks a seated cell.
        compartment = replace(self.cfg.compartment, spring_preload_m=0.005)
        report = design_report(replace(self.cfg, compartment=compartment))
        self.assertFalse(report["ok"])
        self.assertTrue(any("preload" in problem or "plunger" in problem for problem in report["problems"]))
        # No travel beyond the preload leaves no room to lower the cell.
        compartment = replace(self.cfg.compartment, plunger_travel_m=0.0016)
        report = design_report(replace(self.cfg, compartment=compartment))
        self.assertFalse(report["ok"])
        self.assertTrue(any("drop" in problem or "straight down" in problem for problem in report["problems"]))

    def test_single_slot_design_and_usda(self):
        single = with_single_slot(self.cfg, True)
        report = design_report(single)
        self.assertTrue(report["ok"], report["problems"])
        with tempfile.TemporaryDirectory() as directory:
            paths = write_geometry(single, directory)
            text = paths["remote"].read_text()
        self.assertIn("slot_a_plunger", text)
        self.assertNotIn("slot_b_plunger", text)
        # The only fixed joint is the fixture anchor; nothing welds a cell.
        self.assertEqual(text.count("def PhysicsFixedJoint"), 1)
        self.assertIn('def PhysicsFixedJoint "root_joint"', text)

    def test_dual_usda_has_both_plungers_and_no_weld(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_geometry(self.cfg, directory)
            remote = paths["remote"].read_text()
            battery = paths["battery"].read_text()
            marks = paths["marks"].read_text()
        self.assertIn('def Mesh "nose_collider"', remote)
        self.assertIn('def Mesh "handle_collider"', remote)
        self.assertIn('def Mesh "shell"', remote)
        self.assertIn('def Mesh "ir_window"', remote)
        self.assertNotIn('"ceiling"', remote)
        self.assertIn('def PhysicsPrismaticJoint "slot_a_plunger"', remote)
        self.assertIn('def PhysicsPrismaticJoint "slot_b_plunger"', remote)
        self.assertEqual(remote.count("def PhysicsFixedJoint"), 1)
        self.assertNotIn("PhysicsFixedJoint", battery)
        self.assertIn('def Cylinder "body"', battery)
        self.assertIn("PhysicsCollisionAPI", battery)
        self.assertNotIn("PhysicsCollisionAPI", battery.split('def Cylinder "positive_nub"')[1])
        self.assertNotIn("PhysicsRigidBodyAPI", marks)
        self.assertNotIn("PhysicsCollisionAPI", marks)
        self.assertIn("battery_a_mark", marks)
        self.assertIn("battery_b_mark", marks)

    def test_joint_bodies_resolve_to_existing_sibling_prims(self):
        with tempfile.TemporaryDirectory() as directory:
            remote = write_geometry(self.cfg, directory)["remote"]
            text = remote.read_text()
            prims = _prim_paths(text)
            joints = _joint_targets(text)
            self.assertEqual(set(joints), {"slot_a_plunger", "slot_b_plunger"})
            for name, rels in joints.items():
                joint_path = f"/remote/{name}"
                self.assertIn(joint_path, prims)
                self.assertEqual(rels["body0"], "../chassis")
                self.assertEqual(_resolve(joint_path, rels["body0"]), "/remote/chassis")
                # Plunger links nest under the chassis, like YAM's links under its base.
                self.assertEqual(_resolve(joint_path, rels["body1"]), f"/remote/chassis/{name}_link")
                for body in rels.values():
                    self.assertIn(_resolve(joint_path, body), prims, f"{name} -> {body}")
            anchor = dict(re.findall(r"rel physics:(body[01]) = <([^>]+)>", text.split('def PhysicsFixedJoint "root_joint"', 1)[1].split("}", 1)[0]))
            self.assertEqual(_resolve("/remote/root_joint", anchor["body0"]), "/remote")
            self.assertEqual(_resolve("/remote/root_joint", anchor["body1"]), "/remote/chassis")
            try:
                usd = importlib.import_module("pxr.Usd")
                physics = importlib.import_module("pxr.UsdPhysics")
            except ImportError:
                return
            stage = usd.Stage.Open(str(remote))
            for name in joints:
                joint = physics.PrismaticJoint(stage.GetPrimAtPath(f"/remote/{name}"))
                for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()):
                    targets = rel.GetTargets()
                    self.assertEqual(len(targets), 1)
                    self.assertTrue(stage.GetPrimAtPath(targets[0]).IsValid(), str(targets[0]))

    def test_usda_authors_separate_chassis_and_pad_masses(self):
        with tempfile.TemporaryDirectory() as directory:
            text = write_geometry(self.cfg, directory)["remote"].read_text()
        chassis = text.split('def Xform "chassis"', 1)[1]
        self.assertAlmostEqual(_float_after(chassis, "physics:mass"), self.cfg.compartment.mass_kg)
        for slot in self.cfg.compartment.slots:
            link = text.split(f'def Xform "{slot.name}_plunger_link"', 1)[1]
            self.assertAlmostEqual(_float_after(link, "physics:mass"), self.cfg.compartment.plunger_mass_kg)
        self.assertIn("PhysicsMaterialAPI", text)
        self.assertIn("rel material:binding:physics = </remote/physics_material>", text)

    def test_spawner_keeps_usd_masses_and_does_not_request_fix_root_link(self):
        remote = remote_physics(self.cfg)
        self.assertIsNone(remote["mass_kg"])
        self.assertNotIn("fix_root_link", remote)
        cell = battery_physics(self.cfg)
        self.assertAlmostEqual(cell["mass_kg"], self.cfg.battery.mass_kg)
        spawn = (Path(__file__).resolve().parents[1] / "my_package" / "assets" / "spawn.py").read_text()
        self.assertIn("mass_props=None if mass is None", spawn)
        # The server's Isaac Lab cannot apply fix_root_link to this hierarchy; the USDA anchors it.
        self.assertNotIn("fix_root_link=True", spawn)
        self.assertNotIn("ArticulationRootPropertiesCfg", spawn)
        self.assertNotIn("physics_material=", spawn)

    def test_fixed_base_is_authored_like_the_yam_asset(self):
        """One articulation root, on the chassis rigid body, anchored by root_joint."""
        with tempfile.TemporaryDirectory() as directory:
            text = write_geometry(self.cfg, directory)["remote"].read_text()
        self.assertEqual(text.count("PhysicsArticulationRootAPI"), 1)
        chassis_schemas = text.split('def Xform "chassis" (', 1)[1].split(")", 1)[0]
        self.assertIn("PhysicsArticulationRootAPI", chassis_schemas)
        self.assertIn("PhysicsRigidBodyAPI", chassis_schemas)
        remote_header = text.split('def Xform "remote"', 1)[1].split("{", 1)[0]
        self.assertNotIn("apiSchemas", remote_header)
        # Rigid bodies: the chassis and one link per plunger, nothing else.
        self.assertEqual(len(re.findall(r'"PhysicsRigidBodyAPI"', text)), 1 + len(self.cfg.compartment.slots))

    def test_composed_usd_resolves_when_referenced_under_an_env(self):
        """Same structure check through real USD composition, when pxr is installed."""
        try:
            usd = importlib.import_module("pxr.Usd")
            physics = importlib.import_module("pxr.UsdPhysics")
            shade = importlib.import_module("pxr.UsdShade")
        except ImportError:
            self.skipTest("pxr is not installed here; tools/physics_check.py exercises this on the server")
        with tempfile.TemporaryDirectory() as directory:
            path = write_geometry(self.cfg, directory)["remote"]
            stage = usd.Stage.CreateInMemory()
            stage.DefinePrim("/World/envs/env_0/remote").GetReferences().AddReference(str(path))
            roots = [p for p in stage.Traverse() if p.HasAPI(physics.ArticulationRootAPI)]
            self.assertEqual([str(p.GetPath()) for p in roots], ["/World/envs/env_0/remote/chassis"])
            self.assertTrue(roots[0].HasAPI(physics.RigidBodyAPI))
            for prim in stage.Traverse():
                if prim.IsA(physics.Joint):
                    joint = physics.Joint(prim)
                    for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()):
                        for target in rel.GetTargets():
                            self.assertTrue(stage.GetPrimAtPath(target).IsValid(), f"{prim.GetPath()} -> {target}")
                if prim.GetName() in ("shell", "positive_contacts", "polarity_marks", "coil"):
                    material, _ = shade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                    self.assertTrue(material, f"{prim.GetPath()} has no visual material")

    def test_tracked_examples_match_the_generator(self):
        generated = Path(__file__).resolve().parents[1] / "my_package" / "assets" / "generated"
        with tempfile.TemporaryDirectory() as directory:
            fresh = write_geometry(self.cfg, directory)
            for key in ("remote", "battery", "marks"):
                self.assertEqual(
                    (generated / fresh[key].name).read_text(),
                    fresh[key].read_text(),
                    f"regenerate my_package/assets/generated/{fresh[key].name}",
                )

    def test_robot_bases_sit_on_the_measured_table(self):
        self.assertAlmostEqual(self.cfg.table.top_z_m, 0.003)
        for pose in (self.cfg.robot.left_pose, self.cfg.robot.right_pose):
            self.assertAlmostEqual(pose.position_xyz[2], self.cfg.table.top_z_m)
        raised = replace(self.cfg.robot, left_pose=replace(self.cfg.robot.left_pose, position_xyz=(-0.05, 0.22, 0.75)))
        with self.assertRaises(ValueError):
            validate(replace(self.cfg, robot=raised))

    def test_robot_bases_must_stand_over_the_table_footprint(self):
        """Z equal to the tabletop is not enough: the old x=-0.05 bases hung off the near edge."""
        off_edge = replace(self.cfg.robot, left_pose=replace(self.cfg.robot.left_pose, position_xyz=(-0.05, 0.22, 0.003)))
        with self.assertRaisesRegex(ValueError, "not on the tabletop"):
            validate(replace(self.cfg, robot=off_edge))
        from my_package.scoring.config import mounting_problems

        self.assertEqual(mounting_problems(self.cfg), [])

    def test_objects_must_be_within_reach_of_a_base(self):
        far = replace(self.cfg.robot, left_pose=replace(self.cfg.robot.left_pose, position_xyz=(0.27, 0.45, 0.003)),
                      right_pose=replace(self.cfg.robot.right_pose, position_xyz=(0.27, 0.45, 0.003)))
        with self.assertRaisesRegex(ValueError, "from the nearer robot base"):
            validate(replace(self.cfg, robot=far))


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
            root / "my_package" / "assets" / "physics_props.py",
            root / "my_package" / "tasks" / "state_values.py",
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

    def _launcher(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("launch_battery_insertion", root / "launch_battery_insertion.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, root

    def test_launch_script_imports_torch_before_runpy(self):
        text = (Path(__file__).resolve().parents[1] / "launch_battery_insertion.py").read_text()
        self.assertLess(text.index("import torch"), text.index("import runpy"))
        self.assertNotIn("--headless", text)

    def test_smoke_argv_puts_runner_flags_before_the_subcommand(self):
        module, root = self._launcher()
        args = module.make_parser().parse_args(["--arena-root", "/arena", "--single-slot"])
        argv = module.build_argv(args, root)
        command = argv.index("battery_insertion")
        self.assertEqual(argv[argv.index("--visualizer") + 1], "none")
        self.assertEqual(argv[argv.index("--num_steps") + 1], "50")
        for flag in ("--visualizer", "--num_envs", "--policy_type", "--no_solve_relations", "--external_environment_class_path"):
            self.assertLess(argv.index(flag), command, flag)
        self.assertEqual(argv[command + 1:], ["--single_slot"])
        self.assertNotIn("--livestream", argv)
        self.assertNotIn("--headless", argv)
        self.assertIn("zero_action", argv)

    def test_stream_argv_selects_kit_and_livestream(self):
        module, root = self._launcher()
        args = module.make_parser().parse_args(["--arena-root", "/arena", "--stream", "--public-ip", "1.2.3.4", "--enable-cameras"])
        argv = module.build_argv(args, root)
        command = argv.index("battery_insertion")
        self.assertEqual(argv[argv.index("--visualizer") + 1], "kit")
        self.assertEqual(argv[argv.index("--livestream") + 1], "1")
        self.assertEqual(argv[argv.index("--num_steps") + 1], "1000000")
        self.assertLess(argv.index("--livestream"), command)
        self.assertLess(argv.index("--enable_cameras"), command)

    def test_capture_dir_needs_cameras_and_sets_the_env_var(self):
        module, _root = self._launcher()
        parser = module.make_parser()
        args = parser.parse_args(["--arena-root", "/arena", "--enable-cameras", "--capture-dir", "/tmp/caps"])
        self.assertEqual(str(args.capture_dir), "/tmp/caps")
        text = (Path(__file__).resolve().parents[1] / "launch_battery_insertion.py").read_text()
        self.assertIn('parser.error("--capture-dir needs --enable-cameras")', text)
        self.assertIn('os.environ["ROBOCURVE_CAPTURE_DIR"]', text)

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
