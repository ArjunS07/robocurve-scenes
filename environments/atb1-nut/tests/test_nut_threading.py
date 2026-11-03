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

from my_package.assets.factory import resolve_factory_asset
from my_package.assets.geometry import (
    bolt_envelope_world,
    bolt_usda,
    ensure_generated_assets,
    fixture_boxes,
    fixture_boxes_world,
    fixture_usda,
    generated_dir,
    nut_usda,
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
from my_package.scoring.config import (
    DEFAULT_CONFIG_PATH,
    bolt_position_xyz,
    exposed_thread_m,
    load_config,
    nut_spawn_z,
    parse_config,
    start_mat_top_z,
    threaded_tilt_bound_deg,
    turns_to_seat,
)
from my_package.scoring.math3d import quat_from_axis_angle, quat_inv, rotate
from my_package.scoring.scorer import carry_episode_memory, measure, nut_centre_xyz, required_travel_m, step
from my_package.scoring.snapshot import NutSnapshot, SceneSnapshot, clear_episode_state
from my_package.scoring.summary import summarize_recorded_scores
from my_package.tasks.capture import capture_steps, png_bytes
from my_package.tasks.runtime_state import EpisodeRuntime
from my_package.tasks.sim_bridge import env_local_position
from my_package.tasks.state_values import SIM_QUAT_ORDER, env_indices, quaternion_xyzw, sim_quaternion

ROOT = Path(__file__).resolve().parents[1]


def _gap_position(cfg, gap: float, radial: float = 0.0):
    """The nut's centre (what the scorer reads) with its lower face ``gap`` above the seat."""
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
        f = cfg.fastener
        self.assertEqual(f.nominal, "M16")
        self.assertEqual(f.handedness, "right")
        self.assertAlmostEqual(f.pitch_m, 0.002)
        for url in (f.nut_usd, f.bolt_usd):
            self.assertTrue(url.startswith("https://omniverse-content-production.s3-us-west-2.amazonaws.com/"), url)
            self.assertIn("/IsaacLab/Factory/factory_", url)
        # About 16 mm of thread above the plate: 8 turns to seat.
        self.assertAlmostEqual(exposed_thread_m(cfg), 0.016)
        self.assertAlmostEqual(turns_to_seat(cfg), 8.0)
        self.assertAlmostEqual(required_travel_m(cfg), exposed_thread_m(cfg))
        self.assertTrue(Path(cfg.robot.usd_path).is_absolute())
        self.assertEqual(cfg.robot.gripper_body_name, "gripper")
        self.assertEqual(len(cfg.robot.finger_pad_offsets_m), 2)
        self.assertLess(threaded_tilt_bound_deg(f), cfg.scoring.wedge_angle_deg)
        self.assertGreaterEqual(cfg.scoring.radial_max_m, f.nut_bore_minor_radius_m - f.shank_minor_radius_m)
        self.assertEqual([camera.name for camera in cfg.cameras], ["overhead", "front", "side"])

    def test_seat_height_and_nut_spawn(self):
        cfg = load_config()
        f = cfg.fastener
        top = cfg.table.top_z_m
        # The bolt stands on the table; the seat face is the plate top above its head.
        self.assertEqual(bolt_position_xyz(cfg), (0.55, 0.0, top))
        self.assertAlmostEqual(cfg.fixture_pose.position_xyz[2], top + f.bolt_head_height_m + cfg.fixture_body.plate_thickness_m)
        self.assertAlmostEqual(cfg.fixture_pose.position_xyz[2], 0.022)
        # Bolt tip = seat + exposed thread.
        self.assertAlmostEqual(top + f.bolt_length_m, cfg.fixture_pose.position_xyz[2] + exposed_thread_m(cfg))
        # The nut's origin is 10 mm below its lower face, which rests on the mat top.
        self.assertAlmostEqual(start_mat_top_z(cfg), top + cfg.start_mat.thickness_m)
        self.assertAlmostEqual(nut_spawn_z(cfg), start_mat_top_z(cfg) - 0.010)
        self.assertAlmostEqual(cfg.nut_pose.position_xyz[2], -0.006)
        centre = nut_centre_xyz(cfg, cfg.nut_pose.position_xyz, cfg.nut_pose.rotation_xyzw)
        self.assertAlmostEqual(centre[2] - f.nut_height_m / 2.0, start_mat_top_z(cfg))
        self.assertAlmostEqual(math.dist(cfg.nut_pose.position_xyz[:2], cfg.fixture_pose.position_xyz[:2]), 0.15)

    def test_nut_centre_follows_the_nut_axis(self):
        cfg = load_config()
        up = cfg.fastener.nut_origin_to_lower_face_m + cfg.fastener.nut_height_m / 2.0
        flipped = quat_from_axis_angle((1.0, 0.0, 0.0), math.pi)
        centre = nut_centre_xyz(cfg, (0.5, 0.1, 0.2), flipped)
        self.assertAlmostEqual(centre[2], 0.2 - up)
        sideways = quat_from_axis_angle((0.0, 1.0, 0.0), math.pi / 2.0)
        centre = nut_centre_xyz(cfg, (0.5, 0.1, 0.2), sideways)
        self.assertAlmostEqual(centre[0], 0.5 + up)
        self.assertAlmostEqual(centre[2], 0.2)

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
            # A hole narrower than the thread crests would put the plate inside the bolt.
            (lambda d: d["fixture_body"].__setitem__("hole_half_m", 0.0075), "plate hole"),
            # A hole wider than the nut's flats leaves no seat face.
            (lambda d: d["fixture_body"].__setitem__("hole_half_m", 0.013), "seat face"),
            (lambda d: d["fixture_body"].__setitem__("pocket_half_m", 0.013), "pocket"),
            # A thin plate exposes more than 10 turns (and moves the seat).
            (lambda d: d["fixture_body"].__setitem__("plate_thickness_m", 0.003), "turns"),
            (lambda d: d["nut_pose"].__setitem__("position_xyz", [0.55, 0.15, 0.0095]), "nut z"),
            (lambda d: d["fixture_pose"].__setitem__("position_xyz", [0.55, 0.0, 0.019]), "fixture z"),
            (lambda d: d["fastener"].__setitem__("handedness", "left"), "handedness"),
            (lambda d: d["fastener"].__setitem__("nut_usd", "factory_nut_m16.usd"), "nut_usd"),
            (lambda d: d["start_mat"].__setitem__("half_xy_m", [0.005, 0.005]), "start mat"),
            (lambda d: d["start_mat"].__setitem__("half_xy_m", [0.02, 0.13]), "overlap the fixture"),
            (lambda d: d["scoring"].__setitem__("wedge_angle_deg", 3.0), "wedge"),
            (lambda d: d["scoring"].__setitem__("radial_max_m", 0.0003), "radial play"),
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
        report = self._score(_scene(self.cfg, _gap_position(self.cfg, 0.0), turns=turns_to_seat(self.cfg)))
        self.assertEqual(report.score, 100)
        self.assertTrue(report.thread.threaded)
        self.assertFalse(report.hard_failure)

    def test_held_at_full_seat_scores_75(self):
        report = self._score(
            _scene(self.cfg, _gap_position(self.cfg, 0.0), turns=turns_to_seat(self.cfg), holding=True, released=False)
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
        # Resting on the tip thread, above the first full turn.
        gap = self.travel + 0.0005
        report = self._score(_scene(self.cfg, _gap_position(self.cfg, gap), turns=20.0))
        self.assertFalse(report.thread.threaded)
        self.assertEqual(report.score, 25)

    def test_wedge_is_not_threaded(self):
        quat = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(30.0))
        report = self._score(_scene(self.cfg, (0.48, 0.0, 0.84), quat=quat, turns=turns_to_seat(self.cfg)))
        self.assertGreater(report.thread.tilt_deg, self.cfg.scoring.wedge_angle_deg)
        self.assertFalse(report.thread.threaded)
        self.assertEqual(report.score, 25)

    def test_off_table_and_fixture_motion_override(self):
        seated = _scene(self.cfg, _gap_position(self.cfg, 0.0), turns=turns_to_seat(self.cfg))
        off = SceneSnapshot(nut=seated.nut, fixture_position_xyz=seated.fixture_position_xyz, fixture_rotation_xyzw=seated.fixture_rotation_xyzw)
        off = SceneSnapshot(
            nut=NutSnapshot((2.0, 0.0, 0.9), (0.0, 0.0, 0.0, 1.0), False, True, False, thread_turns=turns_to_seat(self.cfg)),
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
            turns=turns_to_seat(self.cfg),
            fixture_pos=(0.50, 0.0, self.cfg.fixture_pose.position_xyz[2]),
        )
        moved_report = self._score(moved)
        self.assertEqual(moved_report.score, 0)
        self.assertIn("fixture_moved", moved_report.hard_reasons)
        yaw = quat_from_axis_angle((0.0, 0.0, 1.0), math.radians(6.0))
        yaw_report = self._score(_scene(self.cfg, _gap_position(self.cfg, 0.0), turns=turns_to_seat(self.cfg), fixture_quat=yaw))
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
        turns = turns_to_seat(self.cfg)
        frames = [(0.05, 0.0, True), (self.travel, 0.0, True)]
        steps = int(round(turns / 0.1))
        frames += [(self.travel - (i * 0.1) * self.pitch, i * 0.1, True) for i in range(1, steps + 1)]
        frames += [(0.0, turns, True), (0.0, turns, False)]
        report = self._run(frames)
        self.assertEqual(report.score, 100)
        self.assertNotIn("cross_threaded", report.hard_reasons)

    def test_drop_to_seat_then_spin_is_cross_threaded(self):
        turns = turns_to_seat(self.cfg)
        frames = [(0.05, 0.0, True), (self.travel, 0.0, True), (0.0, 0.0, True)]
        frames += self._spin(0.0, 0.0, turns)
        frames += [(0.0, turns, False)]
        report = self._run(frames)
        self.assertEqual(report.score, 25)
        self.assertTrue(report.thread.cross_threaded)
        self.assertIn("cross_threaded", report.hard_reasons)

    def test_spin_at_top_then_drop_is_not_threaded(self):
        turns = turns_to_seat(self.cfg)
        frames = [(0.05, 0.0, True)] + self._spin(self.travel, 0.0, turns) + [(0.0, turns, True), (0.0, turns, False)]
        report = self._run(frames)
        self.assertEqual(report.score, 25)
        self.assertFalse(report.thread.threaded)

    def test_spin_just_below_first_thread_then_drop_is_cross_threaded(self):
        turns = turns_to_seat(self.cfg)
        gap = self.travel - 0.1 * self.pitch
        frames = [(0.05, 0.0, True), (self.travel, 0.0, True)] + self._spin(gap, 0.0, turns)
        frames += [(0.0, turns, True), (0.0, turns, False)]
        report = self._run(frames)
        self.assertEqual(report.score, 25)
        self.assertTrue(report.thread.cross_threaded)

    def test_backing_off_clears_cross_thread(self):
        turns = turns_to_seat(self.cfg)
        frames = [(0.05, 0.0, True), (self.travel, 0.0, True), (0.0, 0.0, True), (0.05, 0.0, True), (self.travel, 0.0, True)]
        steps = int(round(turns / 0.1))
        frames += [(self.travel - (i * 0.1) * self.pitch, i * 0.1, True) for i in range(1, steps + 1)]
        frames += [(0.0, turns, False)]
        report = self._run(frames)
        self.assertEqual(report.score, 100)

    def test_upside_down_nut_counts(self):
        flipped = quat_from_axis_angle((1.0, 0.0, 0.0), math.pi)
        report = self._score(_scene(self.cfg, _gap_position(self.cfg, 0.0), quat=flipped, turns=turns_to_seat(self.cfg)))
        self.assertAlmostEqual(report.thread.tilt_deg, 0.0, places=5)
        self.assertAlmostEqual(report.thread.gap_m, 0.0, places=9)
        self.assertEqual(report.score, 100)

    def test_tilt_past_the_wedge_angle_is_not_aligned(self):
        tilt = self.cfg.scoring.wedge_angle_deg + 0.5
        quat = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(tilt))
        report = self._score(_scene(self.cfg, _gap_position(self.cfg, 0.002), quat=quat, turns=(self.travel - 0.002) / self.pitch))
        self.assertFalse(report.thread.aligned)
        self.assertEqual(report.score, 25)
        # Within the thread's tilt bound it still counts.
        quat = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(threaded_tilt_bound_deg(self.cfg.fastener)))
        report = self._score(_scene(self.cfg, _gap_position(self.cfg, 0.002), quat=quat, turns=(self.travel - 0.002) / self.pitch))
        self.assertTrue(report.thread.aligned)

    def test_fixture_tilt_counts_as_moved(self):
        tilt = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(6.0))
        report = self._score(_scene(self.cfg, _gap_position(self.cfg, 0.0), turns=turns_to_seat(self.cfg), fixture_quat=tilt))
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


def _overlap(lo_a, hi_a, lo_b, hi_b, tol: float = 1e-9) -> bool:
    """Open-interval overlap in every axis; touching faces do not count."""
    return all(a0 < b1 - tol and b0 < a1 - tol for a0, a1, b0, b1 in zip(lo_a, hi_a, lo_b, hi_b, strict=True))


def _box_cylinder_clearance(box, axis_xy, radius, z_low, z_high) -> float | None:
    """Horizontal clearance between a box and a vertical cylinder; None if their z ranges do not overlap."""
    if box.upper[2] <= z_low + 1e-9 or box.lower[2] >= z_high - 1e-9:
        return None
    dx = max(box.lower[0] - axis_xy[0], 0.0, axis_xy[0] - box.upper[0])
    dy = max(box.lower[1] - axis_xy[1], 0.0, axis_xy[1] - box.upper[1])
    return math.hypot(dx, dy) - radius


FAKE_NUT = "/cache/factory_nut_m16.usd"
FAKE_BOLT = "/cache/factory_bolt_m16.usd"


class GeometryTests(unittest.TestCase):
    def test_plate_hole_clears_the_bolt(self):
        cfg = load_config()
        bolt_xy = bolt_position_xyz(cfg)[:2]
        checked = 0
        for box in fixture_boxes_world(cfg):
            for name, radius, z_low, z_high in bolt_envelope_world(cfg):
                clearance = _box_cylinder_clearance(box, bolt_xy, radius, z_low, z_high)
                if clearance is None:
                    continue
                checked += 1
                # No fixture collider touches the bolt: at least 1 mm all round.
                self.assertGreater(clearance, 0.001, (box.name, name, clearance))
        self.assertGreaterEqual(checked, 8)
        plate = [box for box in fixture_boxes(cfg) if box.name.startswith("plate_")]
        pocket = [box for box in fixture_boxes(cfg) if box.name.startswith("pocket_")]
        self.assertEqual((len(plate), len(pocket)), (4, 4))
        # The plate top is the seat face (fixture origin) and its hole is narrower than the nut's flats.
        self.assertTrue(all(box.upper[2] == 0.0 for box in plate))
        self.assertLess(cfg.fixture_body.hole_half_m, cfg.fastener.nut_across_flats_m / 2.0)
        self.assertGreater(cfg.fixture_body.hole_half_m, cfg.fastener.shank_major_radius_m)

    def test_exposed_length_and_seat(self):
        cfg = load_config()
        fz = cfg.fixture_pose.position_xyz[2]
        shank = next(item for item in bolt_envelope_world(cfg) if item[0] == "shank")
        plate_top = max(box.upper[2] for box in fixture_boxes_world(cfg) if box.name.startswith("plate_"))
        self.assertAlmostEqual(plate_top, fz)
        self.assertAlmostEqual(shank[3] - plate_top, exposed_thread_m(cfg))
        self.assertAlmostEqual(exposed_thread_m(cfg) / cfg.fastener.pitch_m, 8.0)
        # The block sits on the table; the head is in the pocket layer, flush with its top.
        self.assertAlmostEqual(min(box.lower[2] for box in fixture_boxes_world(cfg) if box.name.startswith("pocket_")), cfg.table.top_z_m)
        head = next(item for item in bolt_envelope_world(cfg) if item[0] == "head")
        self.assertAlmostEqual(head[3], fz - cfg.fixture_body.plate_thickness_m)

    def test_start_mat_holds_the_nut_flat(self):
        cfg = load_config()
        mat = next(box for box in fixture_boxes_world(cfg) if box.name == "start_mat")
        self.assertAlmostEqual(mat.lower[2], cfg.table.top_z_m)
        self.assertAlmostEqual(mat.upper[2], start_mat_top_z(cfg))
        nx, ny, _ = cfg.nut_pose.position_xyz
        corner = cfg.fastener.nut_across_flats_m / math.sqrt(3.0)
        self.assertLessEqual(mat.lower[0], nx - corner)
        self.assertGreaterEqual(mat.upper[0], nx + corner)
        self.assertLessEqual(mat.lower[1], ny - corner)
        self.assertGreaterEqual(mat.upper[1], ny + corner)

    def test_no_collider_overlaps(self):
        cfg = load_config()
        boxes = fixture_boxes_world(cfg)
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                self.assertFalse(_overlap(a.lower, a.upper, b.lower, b.upper), (a.name, b.name))
        # The nut at its spawn pose (hex bounded by its corner circle) touches only the mat top.
        f = cfg.fastener
        centre = nut_centre_xyz(cfg, cfg.nut_pose.position_xyz, cfg.nut_pose.rotation_xyzw)
        corner = f.nut_across_flats_m / math.sqrt(3.0)
        nut_lo = (centre[0] - corner, centre[1] - corner, centre[2] - f.nut_height_m / 2.0)
        nut_hi = (centre[0] + corner, centre[1] + corner, centre[2] + f.nut_height_m / 2.0)
        for box in boxes:
            self.assertFalse(_overlap(nut_lo, nut_hi, box.lower, box.upper), box.name)
        self.assertAlmostEqual(nut_lo[2], start_mat_top_z(cfg))
        # And it is clear of the bolt.
        self.assertGreater(math.dist(centre[:2], bolt_position_xyz(cfg)[:2]), corner + f.bolt_head_corner_radius_m)

    def test_assets(self):
        cfg = load_config()
        fixture_text = fixture_usda(cfg)
        self.assertIn("physics:kinematicEnabled = true", fixture_text)
        self.assertEqual(fixture_text.count("PhysicsRigidBodyAPI"), 1)
        self.assertEqual(fixture_text.count('def Cube "'), len(fixture_boxes(cfg)))
        self.assertEqual(fixture_text.count("material:binding:physics"), len(fixture_boxes(cfg)))
        self.assertIn(f"staticFriction = {cfg.fixture_body.static_friction:.8f}", fixture_text)
        for text in (fixture_text,):
            self.assertNotIn("PhysicsFixedJoint", text)
            self.assertNotIn("ArticulationRootAPI", text)
        nut_text = nut_usda(cfg, FAKE_NUT)
        bolt_text = bolt_usda(cfg, FAKE_BOLT)
        self.assertIn(f"prepend references = @{FAKE_NUT}@", nut_text)
        self.assertIn(f"prepend references = @{FAKE_BOLT}@", bolt_text)
        self.assertIn('over "factory_nut_loose"', nut_text)
        self.assertIn('over "factory_bolt_loose"', bolt_text)
        self.assertIn("physics:kinematicEnabled = false", nut_text)
        self.assertIn("physics:kinematicEnabled = true", bolt_text)
        # The bolt's world fixed joint would make it an articulation: it is switched off.
        self.assertRegex(bolt_text, r'over "root_joint" \(\s*active = false')
        self.assertNotIn("root_joint", nut_text)
        p = cfg.physics
        for text in (nut_text, bolt_text):
            # A RigidObject must not contain an articulation root.
            self.assertIn('delete apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]', text)
            self.assertIn(f"solverPositionIterationCount = {p.fastener_solver_position_iterations}", text)
            self.assertIn(f"solverVelocityIterationCount = {p.fastener_solver_velocity_iterations}", text)
            self.assertIn(f"maxDepenetrationVelocity = {p.fastener_max_depenetration_velocity_m_s:.8f}", text)
            self.assertIn("enableGyroscopicForces = true", text)
            self.assertIn("maxContactImpulse = 1e+32", text)
            self.assertIn(f"contactOffset = {p.fastener_contact_offset_m:.8f}", text)
            self.assertIn(f"restOffset = {p.fastener_rest_offset_m:.8f}", text)
            self.assertIn('over "collisions"', text)
            self.assertIn(f"staticFriction = {p.fastener_friction:.8f}", text)
            # Wrappers only reference NVIDIA's files; nothing is vendored.
            self.assertNotIn("def Mesh", text)
        with tempfile.TemporaryDirectory() as directory:
            nut_path, bolt_path, fixture_path = ensure_generated_assets(cfg, FAKE_NUT, FAKE_BOLT, directory)
            self.assertEqual(nut_path.read_text(encoding="utf-8"), nut_text)
            self.assertEqual(bolt_path.read_text(encoding="utf-8"), bolt_text)
            self.assertEqual(fixture_path.read_text(encoding="utf-8"), fixture_text)
        self.assertFalse(list(ROOT.rglob("factory_*_m16.usd")))

    def test_missing_factory_asset_fails_clearly(self):
        with self.assertRaisesRegex(FileNotFoundError, "nut_usd.*not found.*Factory M16"):
            resolve_factory_asset("/nonexistent/factory_nut_m16.usd", "nut_usd")
        with tempfile.NamedTemporaryFile(suffix=".usd") as handle:
            self.assertEqual(resolve_factory_asset(handle.name, "nut_usd"), handle.name)
        import sys
        import types

        cfg = load_config()
        calls = []

        def fetch(path, force_download=True):
            calls.append((path, force_download))
            raise FileNotFoundError("404")

        module = types.ModuleType("isaaclab.utils.assets")
        module.retrieve_file_path = fetch
        with unittest.mock.patch.dict(sys.modules, {"isaaclab": types.ModuleType("isaaclab"), "isaaclab.utils": types.ModuleType("isaaclab.utils"), "isaaclab.utils.assets": module}):
            with self.assertRaisesRegex(FileNotFoundError, "bolt_usd could not be fetched from https://.*404"):
                resolve_factory_asset(cfg.fastener.bolt_usd, "bolt_usd")
        # Isaac Lab's cache is reused rather than downloading every launch.
        self.assertEqual(calls, [(cfg.fastener.bolt_usd, False)])

    def test_generated_dir_is_outside_the_package_source(self):
        with unittest.mock.patch.dict(os.environ, {"ROBOCURVE_GENERATED_DIR": "/tmp/robocurve-gen"}):
            self.assertEqual(generated_dir(), Path("/tmp/robocurve-gen"))
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROBOCURVE_GENERATED_DIR", None)
            self.assertEqual(generated_dir(), ROOT / "outputs" / "generated")
        self.assertFalse((ROOT / "my_package" / "assets" / "generated").exists())


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
            rotation_xyzw=quat_from_axis_angle((0.0, 0.0, 1.0), -2.0 * math.pi * turns_to_seat(cfg)),
            gripper_holding=False,
            released=False,
            off_table=False,
            hold_known=False,
            ever_lifted=True,
            thread_turns=turns_to_seat(cfg),
            prev_yaw_rad=-2.0 * math.pi * turns_to_seat(cfg),
        )
        scene = SceneSnapshot(nut=nut, fixture_position_xyz=cfg.fixture_pose.position_xyz, fixture_rotation_xyzw=cfg.fixture_pose.rotation_xyzw)
        _snap, report = step(cfg, scene, 0.02)
        self.assertTrue(report.thread.threaded)
        self.assertLess(report.score, 100)
        # Known and released, the same pose scores 100.
        import dataclasses

        known = dataclasses.replace(scene, nut=dataclasses.replace(nut, hold_known=True, released=True))
        self.assertEqual(step(cfg, known, 0.02)[1].score, 100)
        self.assertAlmostEqual(travel, 0.016)

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
        for relative in (
            "my_package/scoring",
            "my_package/assets/geometry.py",
            "my_package/embodiments/action_map.py",
        ):
            path = ROOT / relative
            files = [path] if path.is_file() else list(path.glob("*.py"))
            for source in files:
                tree = ast.parse(source.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        self.assertFalse(node.module.startswith("isaac"))
                        self.assertFalse(node.module.startswith("omni"))
        # The Factory resolver imports Isaac Lab only inside the function that fetches.
        factory = ast.parse((ROOT / "my_package" / "assets" / "factory.py").read_text(encoding="utf-8"))
        for node in factory.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertFalse(node.module.startswith(("isaac", "omni")))


if __name__ == "__main__":
    unittest.main()


class TaskTermParamsTests(unittest.TestCase):
    def test_environment_spawns_the_bolt_from_the_factory_wrappers(self):
        text = (ROOT / "my_package" / "isaaclab_arena_environments" / "nut_environment.py").read_text()
        self.assertIn("resolve_fastener_assets(cfg)", text)
        self.assertIn('make_rigid_usd_object("bolt", bolt_usd, bolt_position_xyz(cfg)', text)
        self.assertIn("nut, bolt, fixture", text)
        bridge = (ROOT / "my_package" / "tasks" / "sim_bridge.py").read_text()
        # Scored and held from the nut's centre, not the Factory body origin below its face.
        self.assertIn("nut_centre_xyz(cfg, env_local_position(nut_world, origin), nut_rotation)", bridge)

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
