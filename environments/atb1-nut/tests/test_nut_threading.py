"""Pure tests for the single-station nut task."""

from __future__ import annotations

import ast
import argparse
import math
import os
import re
import struct
import tempfile
import unittest
import unittest.mock
import zlib
from pathlib import Path

import yaml

from my_package.assets.geometry import (
    ensure_generated_assets,
    fixture_prims,
    fixture_usda,
    generated_dir,
    lead_angle_rad,
    mark_z_m,
    nut_pin_boxes,
    nut_prims,
    nut_usda,
    pin_bottom_above_face_m,
    pin_layout,
    ridge_boxes,
    ridge_end_rad,
    ridge_top_z_m,
)
from my_package.assets.spawn import USD_FILE_FIELDS, resolve_robot_usd, usd_file_spawn_kwargs
from my_package.embodiments.action_map import (
    ACTION_DIM,
    JointResolveError,
    expected_joint_names,
    finger_command,
    initial_joint_positions,
    map_bimanual_action,
    resolve_joint_names,
)
from my_package.embodiments.camera_frames import camera_offset_position
from my_package.embodiments.grasp import any_arm_holding, finger_open_fraction, hold_known, is_holding
from my_package.scoring.config import DEFAULT_CONFIG_PATH, jam_tilt_deg, load_config, parse_config, threaded_tilt_deg
from my_package.scoring.math3d import quat_from_axis_angle, quat_inv, rotate
from my_package.scoring.scorer import carry_episode_memory, measure, required_travel_m, step
from my_package.scoring.snapshot import NutSnapshot, SceneSnapshot, clear_episode_state
from my_package.scoring.summary import summarize_recorded_scores
from my_package.tasks.capture import capture_steps, png_bytes
from my_package.tasks.runtime_state import EpisodeRuntime
from my_package.tasks.sim_bridge import env_local_position
from my_package.tasks.state_values import SIM_QUAT_ORDER, env_indices, quaternion_xyzw, sim_quaternion

ROOT = Path(__file__).resolve().parents[1]


def _gap_position(cfg, gap: float, radial: float = 0.0):
    fx, fy, fz = cfg.fixture_pose.position_xyz
    return (fx, fy + radial, fz + gap + cfg.fastener.nut_height_m / 2.0)


def _scene(cfg, position, *, quat=(0.0, 0.0, 0.0, 1.0), holding=False, released=None, turns=0.0, prev=None, fixture_pos=None, fixture_quat=None):
    if released is None:
        released = not holding
    nut = NutSnapshot(
        position_xyz=position,
        rotation_xyzw=quat,
        gripper_holding=holding,
        released=released,
        off_table=False,
        thread_turns=turns,
        prev_yaw_rad=prev,
    )
    return SceneSnapshot(
        nut=nut,
        fixture_position_xyz=cfg.fixture_pose.position_xyz if fixture_pos is None else fixture_pos,
        fixture_rotation_xyzw=cfg.fixture_pose.rotation_xyzw if fixture_quat is None else fixture_quat,
    )


class ConfigTests(unittest.TestCase):
    def test_protocol(self):
        cfg = load_config()
        self.assertEqual(cfg.episode_length_s, 120.0)
        self.assertEqual(cfg.task_description, "Tighten the nut onto the bolt.")
        self.assertGreater(cfg.fastener.turns, 6.0)
        self.assertLess(cfg.fastener.turns, 7.0)
        self.assertAlmostEqual(required_travel_m(cfg), cfg.fastener.turns * cfg.fastener.pitch_m)
        self.assertAlmostEqual(required_travel_m(cfg), 0.008125)
        f = cfg.fastener
        self.assertLess(f.core_radius_m, f.pin_tip_radius_m)
        self.assertLess(f.pin_tip_radius_m, f.major_radius_m)
        self.assertLess(f.major_radius_m, f.hole_radius_m)
        self.assertTrue(Path(cfg.robot.usd_path).is_absolute())
        self.assertEqual(cfg.robot.gripper_body_name, "gripper")
        self.assertLess(threaded_tilt_deg(f), cfg.scoring.wedge_angle_deg)
        self.assertLess(cfg.scoring.wedge_angle_deg, jam_tilt_deg(f))
        self.assertAlmostEqual(cfg.nut_pose.position_xyz[2], cfg.table.top_z_m + cfg.fastener.nut_height_m / 2.0)
        self.assertAlmostEqual(
            cfg.fixture_pose.position_xyz[2],
            cfg.table.top_z_m + cfg.fixture_body.plate_thickness_m + cfg.fixture_body.collar_height_m,
        )
        self.assertEqual([camera.name for camera in cfg.cameras], ["overhead", "front", "side"])

    def test_robot_bases_sit_on_the_table_footprint(self):
        cfg = load_config()
        table_min_x = cfg.table.center_xy[0] - cfg.table.half_extents_xy[0]
        table_min_y = cfg.table.center_xy[1] - cfg.table.half_extents_xy[1]
        table_max_y = cfg.table.center_xy[1] + cfg.table.half_extents_xy[1]
        for base in (cfg.robot.left_base_xyz, cfg.robot.right_base_xyz):
            self.assertGreater(base[0], table_min_x)
            self.assertTrue(table_min_y < base[1] < table_max_y)
            self.assertAlmostEqual(base[2], cfg.table.top_z_m)
            reach = math.dist(base[:2], cfg.fixture_pose.position_xyz[:2])
            self.assertTrue(0.25 <= reach <= 0.40, reach)

    def test_off_table_base_and_relative_robot_path_are_rejected(self):
        base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        for mutate, message in (
            (lambda d: d["robot"].__setitem__("left_base_xyz", [-0.05, 0.22, 0.003]), "tabletop"),
            (lambda d: d["robot"].__setitem__("usd_path", "assets/yam/yam.usda"), "absolute"),
            (lambda d: d["fastener"].__setitem__("pin_tip_radius_m", 0.0039), "overlap"),
            # 0.8 mm ridge + 0.5 mm pin exceed the 1.25 mm pitch: no axial groove left.
            (lambda d: d["fastener"].__setitem__("ridge_thickness_m", 0.0008), "clearance"),
            # A detection margin below the ridge thickness let pins tunnel on the server.
            (lambda d: d["fastener"].__setitem__("thread_contact_offset_m", 0.00004), "tunnel"),
            # 24 segments with 3 pins put every pin on a seam at once; it stalled on the server.
            (lambda d: d["fastener"].__setitem__("segments_per_turn", 24), "seam"),
        ):
            data = yaml.safe_load(yaml.safe_dump(base))
            mutate(data)
            with self.assertRaisesRegex(ValueError, message):
                parse_config(data)

    def test_unknown_key_is_rejected(self):
        data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        data["bonus"] = True
        with self.assertRaises(ValueError):
            parse_config(data)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.travel = required_travel_m(self.cfg)
        self.pitch = self.cfg.fastener.pitch_m

    def _score(self, snapshot):
        _updated, report = step(self.cfg, snapshot, 0.02)
        return report

    def test_seated_and_released_scores_100(self):
        report = self._score(_scene(self.cfg, _gap_position(self.cfg, 0.0), turns=self.cfg.fastener.turns))
        self.assertEqual(report.score, 100)
        self.assertTrue(report.thread.threaded)
        self.assertFalse(report.hard_failure)

    def test_held_at_full_seat_scores_75(self):
        report = self._score(
            _scene(self.cfg, _gap_position(self.cfg, 0.0), turns=self.cfg.fastener.turns, holding=True, released=False)
        )
        self.assertEqual(report.score, 75)
        self.assertFalse(report.hard_failure)

    def test_released_below_75_percent_scores_50(self):
        descent = 0.5 * self.pitch
        gap = self.travel - descent
        report = self._score(_scene(self.cfg, _gap_position(self.cfg, gap), turns=descent / self.pitch))
        self.assertEqual(report.score, 50)
        self.assertLess(report.thread.travel_fraction, 0.75)

    def test_lifted_but_not_threaded_scores_25(self):
        report = self._score(_scene(self.cfg, (0.48, 0.15, 0.82)))
        self.assertEqual(report.score, 25)
        self.assertFalse(report.thread.threaded)

    def test_never_lifted_scores_0_without_ending_early(self):
        report = self._score(_scene(self.cfg, self.cfg.nut_pose.position_xyz))
        self.assertEqual(report.score, 0)
        self.assertIn("never_lifted", report.hard_reasons)
        self.assertFalse(report.hard_failure)

    def test_held_short_of_75_scores_25_and_continues(self):
        descent = 0.2 * self.travel
        report = self._score(
            _scene(
                self.cfg,
                _gap_position(self.cfg, self.travel - descent),
                turns=descent / self.pitch,
                holding=True,
                released=False,
            )
        )
        self.assertEqual(report.score, 25)
        self.assertIn("held_short_of_75", report.hard_reasons)
        self.assertFalse(report.hard_failure)
        self.assertTrue(report.thread.threaded)

    def test_tip_rest_is_not_threaded(self):
        gap = self.travel + self.cfg.fastener.lead_in_m
        report = self._score(_scene(self.cfg, _gap_position(self.cfg, gap), turns=20.0))
        self.assertFalse(report.thread.threaded)
        self.assertEqual(report.score, 25)

    def test_wedge_is_not_threaded(self):
        quat = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(30.0))
        report = self._score(_scene(self.cfg, (0.48, 0.0, 0.84), quat=quat, turns=self.cfg.fastener.turns))
        self.assertGreater(report.thread.tilt_deg, self.cfg.scoring.wedge_angle_deg)
        self.assertFalse(report.thread.threaded)
        self.assertEqual(report.score, 25)

    def test_off_table_and_fixture_motion_override(self):
        seated = _scene(self.cfg, _gap_position(self.cfg, 0.0), turns=self.cfg.fastener.turns)
        off = SceneSnapshot(nut=seated.nut, fixture_position_xyz=seated.fixture_position_xyz, fixture_rotation_xyzw=seated.fixture_rotation_xyzw)
        off = SceneSnapshot(
            nut=NutSnapshot((2.0, 0.0, 0.9), (0.0, 0.0, 0.0, 1.0), False, True, False, thread_turns=self.cfg.fastener.turns),
            fixture_position_xyz=self.cfg.fixture_pose.position_xyz,
            fixture_rotation_xyzw=self.cfg.fixture_pose.rotation_xyzw,
        )
        off_report = self._score(off)
        self.assertEqual(off_report.score, 0)
        self.assertTrue(off_report.hard_failure)
        self.assertIn("off_table", off_report.hard_reasons)
        moved = _scene(
            self.cfg,
            _gap_position(self.cfg, 0.0),
            turns=self.cfg.fastener.turns,
            fixture_pos=(0.50, 0.0, self.cfg.fixture_pose.position_xyz[2]),
        )
        moved_report = self._score(moved)
        self.assertEqual(moved_report.score, 0)
        self.assertIn("fixture_moved", moved_report.hard_reasons)
        yaw = quat_from_axis_angle((0.0, 0.0, 1.0), math.radians(6.0))
        yaw_report = self._score(_scene(self.cfg, _gap_position(self.cfg, 0.0), turns=self.cfg.fastener.turns, fixture_quat=yaw))
        self.assertIn("fixture_moved", yaw_report.hard_reasons)

    def test_clockwise_turn_accumulates_and_leaving_clears_it(self):
        start = _scene(self.cfg, _gap_position(self.cfg, self.travel * 0.5), turns=0.0)
        updated, state = measure(self.cfg, start)
        self.assertEqual(state.thread_turns, 0.0)
        self.assertIsNotNone(updated.prev_yaw_rad)
        clockwise = quat_from_axis_angle((0.0, 0.0, 1.0), -math.pi / 2.0)
        turned = _scene(
            self.cfg,
            _gap_position(self.cfg, self.travel * 0.5),
            quat=clockwise,
            turns=updated.thread_turns,
            prev=updated.prev_yaw_rad,
        )
        _again, turned_state = measure(self.cfg, turned)
        self.assertAlmostEqual(turned_state.thread_turns, 0.25, places=5)
        away = _scene(
            self.cfg,
            (0.48, 0.20, 0.82),
            quat=quat_from_axis_angle((0.0, 0.0, 1.0), -math.pi),
            turns=turned_state.thread_turns,
            prev=updated.prev_yaw_rad,
        )
        left, left_state = measure(self.cfg, away)
        self.assertEqual(left_state.thread_turns, 0.0)
        self.assertIsNone(left.prev_yaw_rad)
        self.assertFalse(left.cross_threaded)

    def test_latches_and_clear(self):
        off = _scene(self.cfg, (2.0, 0.0, 0.9))
        updated, _report = step(self.cfg, off, 0.02)
        back = SceneSnapshot(
            nut=NutSnapshot(self.cfg.nut_pose.position_xyz, (0.0, 0.0, 0.0, 1.0), False, True, False),
            fixture_position_xyz=updated.fixture_position_xyz,
            fixture_rotation_xyzw=updated.fixture_rotation_xyzw,
            ever_off_table=updated.ever_off_table,
        )
        _again, report = step(self.cfg, back, 0.02)
        self.assertTrue(report.ever_off_table)
        self.assertEqual(report.score, 0)
        cleared = clear_episode_state(updated)
        self.assertFalse(cleared.ever_off_table)
        self.assertEqual(cleared.nut.thread_turns, 0.0)

    def _run(self, frames):
        """frames: (gap, clockwise_turns, holding). Steps the scorer with episode memory."""
        previous = None
        report = None
        for gap, turns_cw, holding in frames:
            quat = quat_from_axis_angle((0.0, 0.0, 1.0), -2.0 * math.pi * turns_cw)
            observed = _scene(self.cfg, _gap_position(self.cfg, gap), quat=quat, holding=holding)
            observed = clear_episode_state(observed) if previous is None else carry_episode_memory(previous, observed)
            previous, report = step(self.cfg, observed, 0.02)
        return report

    def _spin(self, gap, start, stop, holding=True, per_step=0.1):
        count = int(round((stop - start) / per_step))
        return [(gap, start + index * per_step, holding) for index in range(1, count + 1)]

    def test_screwing_down_with_coupled_descent_scores_100(self):
        turns = self.cfg.fastener.turns
        frames = [(0.05, 0.0, True), (self.travel, 0.0, True)]
        steps = int(round(turns / 0.1))
        frames += [(self.travel - (i * 0.1) * self.pitch, i * 0.1, True) for i in range(1, steps + 1)]
        frames += [(0.0, turns, True), (0.0, turns, False)]
        report = self._run(frames)
        self.assertEqual(report.score, 100)
        self.assertNotIn("cross_threaded", report.hard_reasons)

    def test_drop_to_seat_then_spin_is_cross_threaded(self):
        turns = self.cfg.fastener.turns
        frames = [(0.05, 0.0, True), (self.travel, 0.0, True), (0.0, 0.0, True)]
        frames += self._spin(0.0, 0.0, turns)
        frames += [(0.0, turns, False)]
        report = self._run(frames)
        self.assertEqual(report.score, 25)
        self.assertTrue(report.thread.cross_threaded)
        self.assertIn("cross_threaded", report.hard_reasons)

    def test_spin_at_top_then_drop_is_not_threaded(self):
        turns = self.cfg.fastener.turns
        frames = [(0.05, 0.0, True)] + self._spin(self.travel, 0.0, turns) + [(0.0, turns, True), (0.0, turns, False)]
        report = self._run(frames)
        self.assertEqual(report.score, 25)
        self.assertFalse(report.thread.threaded)

    def test_spin_just_below_first_thread_then_drop_is_cross_threaded(self):
        turns = self.cfg.fastener.turns
        gap = self.travel - 0.1 * self.pitch
        frames = [(0.05, 0.0, True), (self.travel, 0.0, True)] + self._spin(gap, 0.0, turns)
        frames += [(0.0, turns, True), (0.0, turns, False)]
        report = self._run(frames)
        self.assertEqual(report.score, 25)
        self.assertTrue(report.thread.cross_threaded)

    def test_backing_off_clears_cross_thread(self):
        turns = self.cfg.fastener.turns
        frames = [(0.05, 0.0, True), (self.travel, 0.0, True), (0.0, 0.0, True), (0.05, 0.0, True), (self.travel, 0.0, True)]
        steps = int(round(turns / 0.1))
        frames += [(self.travel - (i * 0.1) * self.pitch, i * 0.1, True) for i in range(1, steps + 1)]
        frames += [(0.0, turns, False)]
        report = self._run(frames)
        self.assertEqual(report.score, 100)

    def test_upside_down_nut_counts(self):
        flipped = quat_from_axis_angle((1.0, 0.0, 0.0), math.pi)
        report = self._score(_scene(self.cfg, _gap_position(self.cfg, 0.0), quat=flipped, turns=self.cfg.fastener.turns))
        self.assertAlmostEqual(report.thread.tilt_deg, 0.0, places=5)
        self.assertAlmostEqual(report.thread.gap_m, 0.0, places=9)
        self.assertEqual(report.score, 100)

    def test_jammed_nut_is_not_aligned(self):
        jam = jam_tilt_deg(self.cfg.fastener)
        self.assertLess(self.cfg.scoring.wedge_angle_deg, jam)
        quat = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(jam - 0.5))
        report = self._score(_scene(self.cfg, _gap_position(self.cfg, 0.002), quat=quat, turns=(self.travel - 0.002) / self.pitch))
        self.assertFalse(report.thread.aligned)
        self.assertEqual(report.score, 25)

    def test_fixture_tilt_counts_as_moved(self):
        tilt = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(6.0))
        report = self._score(_scene(self.cfg, _gap_position(self.cfg, 0.0), turns=self.cfg.fastener.turns, fixture_quat=tilt))
        self.assertIn("fixture_moved", report.hard_reasons)
        self.assertEqual(report.score, 0)

    def test_dt_must_be_positive(self):
        with self.assertRaises(ValueError):
            step(self.cfg, _scene(self.cfg, self.cfg.nut_pose.position_xyz), 0.0)

    def test_no_damage_reason(self):
        report = self._score(_scene(self.cfg, self.cfg.nut_pose.position_xyz))
        self.assertFalse(hasattr(report, "damage"))
        allowed = {"off_table", "fixture_moved", "never_lifted", "held_short_of_75", "cross_threaded"}
        self.assertTrue(set(report.hard_reasons) <= allowed)


class ActionAndGraspTests(unittest.TestCase):
    def test_zero_action_holds_home_and_open_fingers(self):
        cfg = load_config()
        mapped = map_bimanual_action(
            [0.0] * ACTION_DIM,
            home_arm_rad=cfg.robot.home_arm_rad,
            finger_open_m=cfg.robot.finger_open_m,
            finger_closed_m=cfg.robot.finger_closed_m,
            layout=cfg.robot.layout,
        )
        for side in ("left", "right"):
            for index, name in enumerate(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")):
                self.assertAlmostEqual(mapped[side][name], cfg.robot.home_arm_rad[index])
            self.assertAlmostEqual(mapped[side]["joint7"], cfg.robot.finger_open_m)
            self.assertAlmostEqual(mapped[side]["joint8"], cfg.robot.finger_open_m)
            self.assertEqual(mapped[side]["joint7"], mapped[side]["joint8"])

    def test_closed_gripper_and_combined_names(self):
        cfg = load_config()
        action = [0.0] * ACTION_DIM
        action[6] = 1.0
        action[13] = 1.0
        mapped = map_bimanual_action(
            action,
            home_arm_rad=cfg.robot.home_arm_rad,
            finger_open_m=cfg.robot.finger_open_m,
            finger_closed_m=cfg.robot.finger_closed_m,
            layout="combined",
        )
        closed = finger_command(1.0, cfg.robot.finger_open_m, cfg.robot.finger_closed_m)
        self.assertAlmostEqual(mapped["robot"]["left_joint7"], closed)
        self.assertAlmostEqual(mapped["robot"]["right_joint8"], closed)
        with self.assertRaises(ValueError):
            map_bimanual_action(action[:-1], home_arm_rad=cfg.robot.home_arm_rad, finger_open_m=-0.04, finger_closed_m=0.0, layout="combined")

    def test_joint_resolve_reports_the_live_list(self):
        live = ["joint1", "joint2"]
        with self.assertRaises(JointResolveError) as caught:
            resolve_joint_names(live, expected_joint_names("duplicated_single_arm"))
        self.assertIn("Live joints:", str(caught.exception))
        self.assertIn("joint1", str(caught.exception))
        home = initial_joint_positions((0.4, 1.2, 1.0, -0.5, 0.0, 0.0), -0.04695, "duplicated_single_arm", "left")
        self.assertAlmostEqual(home["joint8"], -0.04695)

    def test_grasp(self):
        cfg = load_config()
        self.assertAlmostEqual(finger_open_fraction(cfg.robot.finger_open_m, cfg.robot.finger_open_m, cfg.robot.finger_closed_m), 1.0)
        self.assertAlmostEqual(finger_open_fraction(0.0, cfg.robot.finger_open_m, cfg.robot.finger_closed_m), 0.0)
        self.assertTrue(
            is_holding(distance_m=0.01, open_fraction=0.2, radius_m=cfg.grasp.radius_m, hold_open_fraction_max=cfg.grasp.hold_open_fraction_max)
        )
        self.assertFalse(
            is_holding(distance_m=0.01, open_fraction=0.9, radius_m=cfg.grasp.radius_m, hold_open_fraction_max=cfg.grasp.hold_open_fraction_max)
        )
        self.assertFalse(is_holding(distance_m=None, open_fraction=0.0, radius_m=0.05, hold_open_fraction_max=0.35))
        self.assertFalse(any_arm_holding([None, None], radius_m=0.05, hold_open_fraction_max=0.35))
        # Fingers shut on nothing are not a hold.
        self.assertFalse(
            is_holding(
                distance_m=0.01,
                open_fraction=0.0,
                radius_m=cfg.grasp.radius_m,
                hold_open_fraction_max=cfg.grasp.hold_open_fraction_max,
                hold_open_fraction_min=cfg.grasp.hold_open_fraction_min,
            )
        )

    def test_camera_offset_uses_the_left_base(self):
        cfg = load_config()
        offset = camera_offset_position(cfg.cameras[0].eye_xyz, cfg.robot.left_base_xyz)
        self.assertAlmostEqual(offset[0], 0.30)
        self.assertAlmostEqual(offset[1], -0.22)
        self.assertAlmostEqual(offset[2], 0.70)


def _usda_prims(text: str) -> dict[str, dict]:
    """Minimal USDA reader for the generated files: prim name -> attributes and schemas."""
    prims = {}
    for match in re.finditer(r'def (\w+) "(\w+)"(.*?)\n    \}', text, re.S):
        kind, name, body = match.groups()
        prims[name] = {"kind": kind, "body": body}
    return prims


def _attr(body: str, name: str):
    match = re.search(re.escape(name) + r" = \(([^)]*)\)", body)
    return tuple(float(v) for v in match.group(1).split(",")) if match else None


def _usd_quat_to_xyzw(wxyz):
    w, x, y, z = wxyz
    return (x, y, z, w)


class GeometryTests(unittest.TestCase):
    def test_assets(self):
        cfg = load_config()
        walls = [prim for prim in nut_prims(cfg) if prim.name.startswith("wall_")]
        self.assertEqual(len(walls), 6)
        self.assertTrue(all(prim.collision for prim in walls))
        fixture = fixture_prims(cfg)
        stud = next(prim for prim in fixture if prim.name == "stud")
        self.assertTrue(stud.collision)
        self.assertAlmostEqual(stud.radius_m, cfg.fastener.core_radius_m)
        self.assertAlmostEqual(stud.height_m, required_travel_m(cfg) + cfg.fastener.lead_in_m)
        ridge = [prim for prim in fixture if prim.name.startswith("ridge_")]
        self.assertGreater(len(ridge), 6 * cfg.fastener.segments_per_turn)
        self.assertTrue(all(prim.collision for prim in ridge))
        self.assertFalse(any(prim.kind == "sphere" for prim in fixture))
        travel = required_travel_m(cfg)
        for name, completed in (("mark_100", 1.0), ("mark_75", 0.75), ("mark_50", 0.5)):
            mark = next(prim for prim in fixture if prim.name == name)
            self.assertFalse(mark.collision)
            self.assertAlmostEqual(mark.translate[2], mark_z_m(travel, completed))
        nut_text = nut_usda(cfg)
        fixture_text = fixture_usda(cfg)
        for text in (nut_text, fixture_text):
            self.assertEqual(text.count("PhysicsRigidBodyAPI"), 1)
            self.assertNotIn("PhysicsFixedJoint", text)
            self.assertNotIn("PhysicsArticulationRootAPI", text)
            self.assertIn("PhysicsMaterialAPI", text)
            self.assertIn("material:binding:physics", text)
            self.assertIn(f"staticFriction = {cfg.fastener.static_friction:.8f}", text)
            self.assertIn("solverPositionIterationCount = 32", text)
        self.assertIn("maxDepenetrationVelocity = 0.05", text)
            self.assertIn("solverVelocityIterationCount = 1", text)
            # Every collider applies PhysxCollisionAPI, so its contactOffset is honoured.
            self.assertEqual(text.count('"PhysicsCollisionAPI", "PhysxCollisionAPI"'), text.count("PhysicsCollisionAPI"))
        self.assertIn("physics:kinematicEnabled = true", fixture_text)
        self.assertIn("physics:kinematicEnabled = false", nut_text)
        self.assertIn(f"contactOffset = {cfg.fastener.thread_contact_offset_m:.8f}", fixture_text)
        backplate = next(prim for prim in fixture if prim.name == "backplate")
        face_x = backplate.translate[0] + backplate.scale[0]
        for name in ("mark_100", "mark_75", "mark_50"):
            mark = next(prim for prim in fixture if prim.name == name)
            self.assertLess(mark.translate[0] - mark.scale[0], face_x)
            self.assertGreater(mark.translate[0] + mark.scale[0], face_x)
        with tempfile.TemporaryDirectory() as directory:
            nut_path, fixture_path = ensure_generated_assets(cfg, directory)
            self.assertEqual(nut_path.read_text(encoding="utf-8"), nut_text)
            self.assertEqual(fixture_path.read_text(encoding="utf-8"), fixture_text)

    def test_generated_dir_is_outside_the_package_source(self):
        with unittest.mock.patch.dict(os.environ, {"ROBOCURVE_GENERATED_DIR": "/tmp/robocurve-gen"}):
            self.assertEqual(generated_dir(), Path("/tmp/robocurve-gen"))
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROBOCURVE_GENERATED_DIR", None)
            self.assertEqual(generated_dir(), ROOT / "outputs" / "generated")
        self.assertFalse((ROOT / "my_package" / "assets" / "generated").exists())

    def test_usda_ridge_matches_the_helix(self):
        """Parse the written fixture: every ridge box top must sit on the helix at the contact radius."""
        cfg = load_config()
        prims = _usda_prims(fixture_usda(cfg))
        ridge = {name: prim for name, prim in prims.items() if name.startswith("ridge_")}
        self.assertEqual(len(ridge), len(ridge_boxes(cfg)))
        step = 2.0 * math.pi / cfg.fastener.segments_per_turn
        for name, prim in ridge.items():
            index = int(name.split("_")[1])
            center = _attr(prim["body"], "float3 xformOp:translate")
            orient = _usd_quat_to_xyzw(_attr(prim["body"], "quatf xformOp:orient"))
            half = _attr(prim["body"], "float3 xformOp:scale")
            up = rotate(orient, (0.0, 0.0, 1.0))
            top_center = tuple(c + half[2] * u for c, u in zip(center, up))
            clockwise = (index + 0.5) * step
            self.assertAlmostEqual(top_center[2], ridge_top_z_m(cfg, clockwise), delta=2e-6)
            # Its tangential slope matches the lead angle: one pitch per turn.
            tangent = rotate(orient, (0.0, 1.0, 0.0))
            self.assertAlmostEqual(math.atan2(tangent[2], math.hypot(tangent[0], tangent[1])), lead_angle_rad(cfg), places=6)
        self.assertGreaterEqual(ridge_top_z_m(cfg, ridge_end_rad(cfg)), -1e-9)

    def test_pins_ride_the_ridge_at_every_turn(self):
        """For a nut at clockwise yaw psi with lower face at travel - pitch*psi/2pi, each pin bottom is on the ridge top."""
        cfg = load_config()
        f = cfg.fastener
        travel = required_travel_m(cfg)
        layout = pin_layout(cfg)
        self.assertEqual(len(layout), 3)
        for psi_turns in (0.1, 1.0, 2.37, 4.5, f.turns):
            psi = 2.0 * math.pi * psi_turns
            gap = travel - f.pitch_m * psi_turns
            for k, (angle, _z) in enumerate(layout):
                pin_clockwise = psi - angle
                pin_bottom = gap + pin_bottom_above_face_m(cfg, k)
                self.assertAlmostEqual(pin_bottom, ridge_top_z_m(cfg, pin_clockwise), delta=1e-9)
                self.assertLessEqual(pin_clockwise, ridge_end_rad(cfg))
                # The next turn up is above the pin's top by the groove play: no straight pull-off.
                above_bottom = ridge_top_z_m(cfg, pin_clockwise - 2.0 * math.pi) - f.ridge_thickness_m
                self.assertAlmostEqual(above_bottom - (pin_bottom + f.pin_thickness_m), f.pitch_m - f.ridge_thickness_m - f.pin_thickness_m, delta=1e-9)
        # Self-locking: the lead angle is far below the friction angle, so a released nut holds.
        self.assertLess(math.tan(lead_angle_rad(cfg)), 0.2 * f.static_friction)
        # Pins sit radially between the core and the ridge crest, and reach into the bore wall.
        for pin in nut_pin_boxes(cfg):
            radial = math.hypot(pin.translate[0], pin.translate[1])
            self.assertAlmostEqual(radial - pin.scale[0], f.pin_tip_radius_m, places=9)
            self.assertGreater(radial + pin.scale[0], f.hole_radius_m)
            self.assertGreater(pin.translate[2] - pin.scale[2], -f.nut_height_m / 2.0)


class RuntimeAndLaunchTests(unittest.TestCase):
    def test_same_step_does_not_double_count(self):
        cfg = load_config()
        runtime = EpisodeRuntime()
        travel = required_travel_m(cfg)
        first = _scene(cfg, _gap_position(cfg, travel * 0.5))
        _snapshot, report = runtime.observe(cfg, first, token=1, dt=0.02)
        clockwise = _scene(
            cfg,
            _gap_position(cfg, travel * 0.5),
            quat=quat_from_axis_angle((0.0, 0.0, 1.0), -math.pi / 2.0),
        )
        _again, cached = runtime.observe(cfg, clockwise, token=1, dt=0.02)
        self.assertEqual(cached.thread.thread_turns, report.thread.thread_turns)
        _later, turned = runtime.observe(cfg, clockwise, token=2, dt=0.02)
        self.assertAlmostEqual(turned.thread.thread_turns, report.thread.thread_turns + 0.25, places=5)
        runtime.on_reset()
        _reset, fresh = runtime.observe(cfg, first, token=3, dt=0.02)
        self.assertEqual(fresh.thread.thread_turns, 0.0)

    def test_pose_conversion(self):
        # The installed runtime is xyzw (measured); identity must stay identity, not become a 180 deg turn.
        self.assertEqual(SIM_QUAT_ORDER, "xyzw")
        self.assertEqual(quaternion_xyzw((0.0, 0.0, 0.0, 1.0)), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(quaternion_xyzw([0.0, 0.0, 0.0, 1.0]), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(quaternion_xyzw((1.0, 2.0, 3.0, 4.0), order="wxyz"), (2.0, 3.0, 4.0, 1.0))
        self.assertEqual(sim_quaternion((0.0, 0.0, 0.0, 1.0)), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(env_local_position((1.0, 2.0, 3.0), (0.25, 0.5, 0.75)), (0.75, 1.5, 2.25))
        cfg = load_config()
        # An untouched fixture read through the bridge conversion does not count as moved.
        _snap, report = step(cfg, _scene(cfg, cfg.nut_pose.position_xyz, fixture_quat=quaternion_xyzw((0.0, 0.0, 0.0, 1.0))), 0.02)
        self.assertFalse(report.ever_fixture_moved)

    def test_env_indices_accept_lists_tensors_and_none(self):
        class FakeTensor:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return list(self.values)

        self.assertEqual(env_indices(None, 3), [0, 1, 2])
        self.assertEqual(env_indices([2, 0], 3), [2, 0])
        self.assertEqual(env_indices((1,), 3), [1])
        self.assertEqual(env_indices(FakeTensor([1, 2]), 3), [1, 2])
        self.assertEqual(env_indices(slice(None), 2), [0, 1])

    def test_reset_and_recorder_accept_list_env_ids(self):
        from my_package.tasks import sim_bridge

        class Env:
            num_envs = 2

        env = Env()
        runs = sim_bridge.runtimes_for(env)
        for run in runs:
            run.pending_clear = False
        sim_bridge.reset_episode_memory(env, [1])
        self.assertEqual([run.pending_clear for run in runs], [False, True])
        sim_bridge.reset_episode_memory(env, None)
        self.assertEqual([run.pending_clear for run in runs], [True, True])
        text = (ROOT / "my_package" / "tasks" / "sim_bridge.py").read_text(encoding="utf-8")
        self.assertNotIn("env_ids.detach", text)
        self.assertNotIn("wxyz", text.split('"""', 2)[2])

    def test_unknown_gripper_is_never_released(self):
        cfg = load_config()
        travel = required_travel_m(cfg)
        self.assertFalse(hold_known([None, {"distance_m": 0.1, "open_fraction": 1.0}]))
        self.assertTrue(hold_known([{"distance_m": 0.1, "open_fraction": 1.0}] * 2))
        nut = NutSnapshot(
            position_xyz=_gap_position(cfg, 0.0),
            rotation_xyzw=quat_from_axis_angle((0.0, 0.0, 1.0), -2.0 * math.pi * cfg.fastener.turns),
            gripper_holding=False,
            released=False,
            off_table=False,
            hold_known=False,
            ever_lifted=True,
            thread_turns=cfg.fastener.turns,
            prev_yaw_rad=-2.0 * math.pi * cfg.fastener.turns,
        )
        scene = SceneSnapshot(nut=nut, fixture_position_xyz=cfg.fixture_pose.position_xyz, fixture_rotation_xyzw=cfg.fixture_pose.rotation_xyzw)
        _snap, report = step(cfg, scene, 0.02)
        self.assertTrue(report.thread.threaded)
        self.assertLess(report.score, 100)
        # Known and released, the same pose scores 100.
        import dataclasses

        known = dataclasses.replace(scene, nut=dataclasses.replace(nut, hold_known=True, released=True))
        self.assertEqual(step(cfg, known, 0.02)[1].score, 100)
        self.assertLess(travel, 0.01)

    def test_capture_png_and_steps(self):
        self.assertEqual(capture_steps(None), {1, 25, 49})
        self.assertEqual(capture_steps("3, 7"), {3, 7})
        data = png_bytes(bytes([255, 0, 0, 0, 255, 0]), 2, 1)
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height = struct.unpack(">II", data[16:24])
        self.assertEqual((width, height), (2, 1))
        start = data.index(b"IDAT") + 4
        length = struct.unpack(">I", data[start - 8 : start - 4])[0]
        self.assertEqual(zlib.decompress(data[start : start + length]), b"\x00" + bytes([255, 0, 0, 0, 255, 0]))

    def test_recorded_scores(self):
        self.assertEqual(summarize_recorded_scores([]), 0.0)
        self.assertEqual(summarize_recorded_scores([[100.0], [50.0]]), 75.0)

    def test_spawn_kwargs_stay_narrow(self):
        kwargs = usd_file_spawn_kwargs("nut.usda")
        self.assertEqual(tuple(kwargs), USD_FILE_FIELDS)
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("YAM_USD", None)
            self.assertEqual(resolve_robot_usd("assets/yam/arm.usda"), "assets/yam/arm.usda")
        with unittest.mock.patch.dict(os.environ, {"YAM_USD": "/tmp/yam.usda"}):
            self.assertEqual(resolve_robot_usd("assets/yam/arm.usda"), "/tmp/yam.usda")
        for path in (
            ROOT / "my_package" / "assets" / "spawn.py",
            ROOT / "my_package" / "embodiments" / "yam_bimanual.py",
        ):
            text = path.read_text(encoding="utf-8")
            for banned in ("rigid_props", "collision_props", "mass_props", "physics_material", "fix_root_link=True", "ARTICULATION"):
                self.assertNotIn(banned, text)

    def test_launch_argv_and_import_order(self):
        from launch_nut_threading import build_argv, make_parser

        arena = Path("/opt/arena")
        args = make_parser().parse_args(["--arena-root", str(arena)])
        argv = build_argv(args, ROOT)
        self.assertEqual(argv[0], str(arena / "isaaclab_arena/evaluation/policy_runner.py"))
        self.assertEqual(argv[argv.index("--visualizer") + 1], "none")
        self.assertEqual(argv[argv.index("--num_steps") + 1], "50")
        self.assertEqual(argv[argv.index("--output_base_dir") + 1], str(ROOT / "outputs"))
        self.assertIn("--no_solve_relations", argv)
        self.assertIn("zero_action", argv)
        self.assertIn("cuda:0", argv)
        self.assertEqual(argv[-1], "nut_threading")
        self.assertNotIn("--livestream", argv)
        stream = build_argv(
            make_parser().parse_args(["--stream", "--public-ip", "1.2.3.4", "--enable-cameras", "--nut-config", "x.yaml"]), ROOT
        )
        self.assertEqual(stream[stream.index("--visualizer") + 1], "kit")
        self.assertEqual(stream[stream.index("--num_steps") + 1], "1000000")
        self.assertEqual(stream[stream.index("--livestream") + 1], "1")
        # Runner flags precede the subcommand; environment flags follow it.
        sub = stream.index("nut_threading")
        self.assertLess(stream.index("--enable_cameras"), sub)
        self.assertLess(stream.index("--livestream"), sub)
        self.assertGreater(stream.index("--nut_config"), sub)
        self.assertTrue(Path(stream[stream.index("--nut_config") + 1]).is_absolute())
        text = (ROOT / "launch_nut_threading.py").read_text(encoding="utf-8")
        self.assertNotIn("headless", text)
        self.assertIn("ROBOCURVE_GENERATED_DIR", text)
        self.assertIn("ROBOCURVE_CAPTURE_DIR", text)
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", "") or ""
                names = [alias.name for alias in node.names]
                self.assertFalse(module.startswith("isaac"))
                self.assertNotIn("torch", names)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        imported = [alias.name for node in main.body if isinstance(node, ast.Import) for alias in node.names]
        self.assertLess(imported.index("torch"), imported.index("runpy"))

    def test_package_layout(self):
        env_dir = ROOT / "my_package" / "isaaclab_arena_environments"
        self.assertTrue((env_dir / "nut_environment.py").is_file())
        self.assertFalse((env_dir / "__init__.py").exists())
        embodiment = (ROOT / "my_package" / "embodiments" / "yam_bimanual.py").read_text(encoding="utf-8")
        self.assertLess(embodiment.index("if self.enable_cameras:"), embodiment.index("self.camera_config = _camera_cfg"))
        for relative in ("my_package/scoring", "my_package/assets/geometry.py", "my_package/embodiments/action_map.py"):
            path = ROOT / relative
            files = [path] if path.is_file() else list(path.glob("*.py"))
            for source in files:
                tree = ast.parse(source.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        self.assertFalse(node.module.startswith("isaac"))
                        self.assertFalse(node.module.startswith("omni"))


if __name__ == "__main__":
    unittest.main()


class TaskTermParamsTests(unittest.TestCase):
    def test_frozen_config_is_not_passed_through_term_params(self):
        # Isaac Lab's configclass post-init assigns into params and raised
        # FrozenInstanceError on the server when the NutConfig was passed there.
        text = (Path(__file__).resolve().parents[1] / "my_package" / "tasks" / "nut_task.py").read_text()
        self.assertNotIn('params={"cfg"', text)
        self.assertNotIn("partial(", text)
        self.assertIn("set_active_config(cfg)", text)


class ActionTermAbcTests(unittest.TestCase):
    def test_action_term_implements_every_abstract_property(self):
        # The server's ActionTerm ABC requires these; without them the env could not be built.
        text = (Path(__file__).resolve().parents[1] / "my_package" / "embodiments" / "yam_bimanual.py").read_text()
        for name in ("action_dim", "raw_actions", "processed_actions"):
            self.assertRegex(text, rf"@property\n    def {name}\(self\)")
        for name in ("process_actions", "apply_actions"):
            self.assertIn(f"def {name}(self", text)


class HoldPointTests(unittest.TestCase):
    def test_holding_is_measured_from_the_fingertips(self):
        text = (Path(__file__).resolve().parents[1] / "my_package" / "tasks" / "sim_bridge.py").read_text()
        self.assertIn('"tip_left"', text)
        self.assertIn('"tip_right"', text)
