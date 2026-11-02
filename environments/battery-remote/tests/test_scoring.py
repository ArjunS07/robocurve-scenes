"""Pure tests for the scoring state machine. They do not import Isaac Sim."""

from __future__ import annotations

import unittest

from my_package.assets.geometry import measure_battery
from my_package.scoring.config import default_config_path, load_config, with_single_slot
from my_package.scoring.grasp import gripper_is_holding
from my_package.scoring.math3d import rotate
from my_package.scoring.scorer import full_seat_depth_m, step
from my_package.scoring.snapshot import BatterySnapshot, SceneSnapshot, clear_episode_state, update_battery

PLUS_X = (1.0, 0.0, 0.0)
MINUS_X = (-1.0, 0.0, 0.0)
START_QUAT = (0.0, 0.7071067811865475, 0.0, 0.7071067811865476)


def _battery(name: str, **overrides) -> BatterySnapshot:
    axis = PLUS_X if name == "battery_a" else MINUS_X
    side = -0.09 if name == "battery_a" else 0.09
    values = dict(
        name=name,
        position_xyz=(0.50, side, 0.76),
        rotation_xyzw=START_QUAT,
        positive_axis_xyz=axis,
        seating_depth_m=0.039,
        radial_offset_m=0.001,
        polarity_aligned=True,
        gripper_holding=False,
        released=True,
        spring_contact=True,
        off_table=False,
        peak_force_n=0.0,
    )
    values.update(overrides)
    return BatterySnapshot(**values)


def _scene(*batteries: BatterySnapshot, **overrides) -> SceneSnapshot:
    values = dict(
        batteries=batteries,
        remote_position_xyz=(0.50, 0.0, 0.761),
        remote_rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        remote_displacement_m=0.0,
        remote_yaw_error_deg=0.0,
        damage_force_n=0.0,
    )
    values.update(overrides)
    return SceneSnapshot(**values)


def _both(**overrides) -> SceneSnapshot:
    return _scene(_battery("battery_a", **overrides), _battery("battery_b", **overrides))


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(default_config_path())

    def _to_score(self, snapshot: SceneSnapshot, dt: float = 0.5):
        current = snapshot
        report = None
        # Two half-second seated steps fill the 1.0 s window exactly.
        for _ in range(2):
            current, report = step(self.cfg, current, dt)
        return current, report

    def _inserted_after_lift(self, **battery_overrides):
        lifted = _both(
            position_xyz=(0.50, 0.0, 0.80),
            seating_depth_m=0.0,
            spring_contact=False,
            **battery_overrides,
        )
        # The shared overrides above put both cells on y=0. Split them again after the lift.
        lifted = update_battery(lifted, "battery_a", position_xyz=(0.50, -0.09, 0.80))
        lifted = update_battery(lifted, "battery_b", position_xyz=(0.50, 0.09, 0.80))
        current, _ = step(self.cfg, lifted, 0.5)
        current = update_battery(current, "battery_a", position_xyz=(0.50, -0.09, 0.76), seating_depth_m=0.039, spring_contact=True)
        current = update_battery(current, "battery_b", position_xyz=(0.50, 0.09, 0.76), seating_depth_m=0.039, spring_contact=True)
        return self._to_score(current)

    def test_both_inserted_scores_100(self):
        _, report = self._inserted_after_lift()
        self.assertEqual(report.score, 100)
        self.assertTrue(report.inserted["battery_a"])
        self.assertTrue(report.inserted["battery_b"])
        self.assertFalse(report.hard_failure)

    def test_one_inserted_scores_50(self):
        current, _ = step(self.cfg, _both(position_xyz=(0.50, 0.0, 0.80), seating_depth_m=0.0, spring_contact=False), 0.5)
        current = update_battery(current, "battery_a", position_xyz=(0.50, -0.09, 0.76), seating_depth_m=0.039, spring_contact=True)
        current = update_battery(current, "battery_b", position_xyz=(0.50, 0.09, 0.80), seating_depth_m=0.0, spring_contact=False)
        current, report = self._to_score(current)
        self.assertEqual(report.score, 50)
        self.assertTrue(report.inserted["battery_a"])
        self.assertFalse(report.inserted["battery_b"])

    def test_lift_without_insertion_scores_25(self):
        current, report = step(self.cfg, _both(position_xyz=(0.50, 0.0, 0.80), seating_depth_m=0.0, spring_contact=False), 0.5)
        self.assertEqual(report.score, 25)
        self.assertTrue(report.ever_lifted["battery_a"])
        self.assertFalse(any(report.inserted.values()))
        del current

    def test_resting_pose_is_not_lifted(self):
        _, report = step(self.cfg, _both(), 0.5)
        self.assertIn("never_lifted", report.hard_reasons)
        self.assertEqual(report.score, 0)

    def test_reversed_flag_does_not_insert(self):
        current, _ = step(self.cfg, _both(position_xyz=(0.50, 0.0, 0.80), seating_depth_m=0.0, spring_contact=False), 0.5)
        current = update_battery(current, "battery_a", polarity_aligned=False, seating_depth_m=0.039, spring_contact=True, position_xyz=(0.50, -0.09, 0.76))
        current = update_battery(current, "battery_b", polarity_aligned=False, seating_depth_m=0.039, spring_contact=True, position_xyz=(0.50, 0.09, 0.76))
        _, report = self._to_score(current)
        self.assertFalse(any(report.inserted.values()))
        self.assertEqual(report.score, 25)

    def test_true_flag_cannot_hide_a_reversed_axis(self):
        current, _ = step(self.cfg, _both(position_xyz=(0.50, 0.0, 0.80), seating_depth_m=0.0, spring_contact=False), 0.5)
        current = update_battery(
            current,
            "battery_a",
            positive_axis_xyz=MINUS_X,
            polarity_aligned=True,
            seating_depth_m=0.039,
            spring_contact=True,
            position_xyz=(0.50, -0.09, 0.76),
        )
        current = update_battery(current, "battery_b", seating_depth_m=0.0, spring_contact=False, position_xyz=(0.50, 0.09, 0.80))
        _, report = self._to_score(current)
        self.assertFalse(report.inserted["battery_a"])

    def test_partly_seated_does_not_insert(self):
        current, _ = step(self.cfg, _both(position_xyz=(0.50, 0.0, 0.80), seating_depth_m=0.0, spring_contact=False), 0.5)
        current = update_battery(current, "battery_a", seating_depth_m=0.01, position_xyz=(0.50, -0.09, 0.76))
        current = update_battery(current, "battery_b", seating_depth_m=0.01, position_xyz=(0.50, 0.09, 0.76))
        _, report = self._to_score(current)
        self.assertFalse(any(report.inserted.values()))
        self.assertEqual(report.score, 25)

    def test_missing_spring_contact_does_not_insert(self):
        current, _ = step(self.cfg, _both(position_xyz=(0.50, 0.0, 0.80), seating_depth_m=0.0, spring_contact=False), 0.5)
        current = update_battery(current, "battery_a", spring_contact=False, seating_depth_m=0.039, position_xyz=(0.50, -0.09, 0.76))
        current = update_battery(current, "battery_b", spring_contact=False, seating_depth_m=0.039, position_xyz=(0.50, 0.09, 0.76))
        _, report = self._to_score(current)
        self.assertFalse(any(report.inserted.values()))

    def test_large_radial_offset_does_not_insert(self):
        current, _ = step(self.cfg, _both(position_xyz=(0.50, 0.0, 0.80), seating_depth_m=0.0, spring_contact=False), 0.5)
        current = update_battery(current, "battery_a", radial_offset_m=0.05, seating_depth_m=0.039, spring_contact=True, position_xyz=(0.50, -0.09, 0.76))
        current = update_battery(current, "battery_b", radial_offset_m=0.05, seating_depth_m=0.039, spring_contact=True, position_xyz=(0.50, 0.09, 0.76))
        _, report = self._to_score(current)
        self.assertFalse(any(report.inserted.values()))

    def test_gripper_hold_blocks_the_retention_window(self):
        held = _both(gripper_holding=True, released=False, position_xyz=(0.50, 0.0, 0.80))
        current, report = step(self.cfg, held, 0.5)
        for _ in range(6):
            current, report = step(self.cfg, current, 0.5)
        self.assertEqual(report.retention_timer_s["battery_a"], 0.0)
        self.assertFalse(report.inserted["battery_a"])
        self.assertEqual(report.score, 25)

    def test_hold_overrides_a_released_flag(self):
        current, _ = step(self.cfg, _both(position_xyz=(0.50, 0.0, 0.80), seating_depth_m=0.0, spring_contact=False), 0.5)
        current = update_battery(current, "battery_a", gripper_holding=True, released=True, seating_depth_m=0.039, spring_contact=True, position_xyz=(0.50, -0.09, 0.76))
        current = update_battery(current, "battery_b", gripper_holding=True, released=True, seating_depth_m=0.039, spring_contact=True, position_xyz=(0.50, 0.09, 0.76))
        _, report = self._to_score(current)
        self.assertFalse(any(report.inserted.values()))

    def test_release_before_the_window_does_not_count(self):
        current, _ = step(self.cfg, _both(position_xyz=(0.50, 0.0, 0.80), seating_depth_m=0.0, spring_contact=False), 0.5)
        current = update_battery(current, "battery_a", seating_depth_m=0.039, spring_contact=True, position_xyz=(0.50, -0.09, 0.76))
        current = update_battery(current, "battery_b", seating_depth_m=0.039, spring_contact=True, position_xyz=(0.50, 0.09, 0.76))
        current, report = step(self.cfg, current, 0.5)
        self.assertAlmostEqual(report.retention_timer_s["battery_a"], 0.5)
        self.assertFalse(report.inserted["battery_a"])
        current = update_battery(current, "battery_a", seating_depth_m=0.0, spring_contact=False)
        current = update_battery(current, "battery_b", seating_depth_m=0.0, spring_contact=False)
        _, report = step(self.cfg, current, 0.5)
        self.assertEqual(report.retention_timer_s["battery_a"], 0.0)
        self.assertFalse(report.inserted["battery_a"])
        self.assertEqual(report.score, 25)

    def test_falling_out_after_the_window_drops_credit(self):
        current, report = self._inserted_after_lift()
        self.assertEqual(report.score, 100)
        current = update_battery(current, "battery_a", seating_depth_m=0.0, spring_contact=False, position_xyz=(0.50, -0.09, 0.80))
        current = update_battery(current, "battery_b", seating_depth_m=0.0, spring_contact=False, position_xyz=(0.50, 0.09, 0.80))
        _, report = step(self.cfg, current, 0.5)
        self.assertFalse(any(report.inserted.values()))
        self.assertEqual(report.score, 25)

    def test_never_lifted_overrides_an_inserted_signal(self):
        _, report = self._to_score(_both())
        self.assertTrue(report.inserted["battery_a"])
        self.assertTrue(report.inserted["battery_b"])
        self.assertEqual(report.score, 0)
        self.assertIn("never_lifted", report.hard_reasons)
        self.assertFalse(report.hard_failure)

    def test_off_table_latch_overrides_100_and_sticks(self):
        current, report = self._inserted_after_lift()
        self.assertEqual(report.score, 100)
        current = update_battery(current, "battery_a", off_table=True)
        current, report = step(self.cfg, current, 0.5)
        self.assertEqual(report.score, 0)
        self.assertTrue(report.inserted["battery_a"])
        self.assertIn("off_table", report.hard_reasons)
        current = update_battery(current, "battery_a", off_table=False)
        _, report = step(self.cfg, current, 0.5)
        self.assertEqual(report.score, 0)
        self.assertTrue(report.ever_off_table)

    def test_geometric_off_table_overrides_without_the_flag(self):
        current, _ = self._inserted_after_lift()
        current = update_battery(current, "battery_b", position_xyz=(0.50, 2.0, 0.80), off_table=False)
        _, report = step(self.cfg, current, 0.5)
        self.assertEqual(report.score, 0)
        self.assertIn("off_table", report.hard_reasons)

    def test_remote_motion_overrides_100(self):
        current, _ = self._inserted_after_lift()
        current, report = step(self.cfg, update_battery(current, "battery_a"), 0.5)
        moved = SceneSnapshot(
            batteries=current.batteries,
            remote_position_xyz=current.remote_position_xyz,
            remote_rotation_xyzw=current.remote_rotation_xyzw,
            remote_displacement_m=0.02,
            remote_yaw_error_deg=0.0,
            damage_force_n=0.0,
            ever_off_table=current.ever_off_table,
            ever_remote_moved=current.ever_remote_moved,
            ever_damage=current.ever_damage,
        )
        _, report = step(self.cfg, moved, 0.5)
        self.assertEqual(report.score, 0)
        self.assertIn("remote_moved", report.hard_reasons)
        self.assertTrue(report.inserted["battery_a"])

    def test_remote_yaw_overrides_100(self):
        current, _ = self._inserted_after_lift()
        yawed = SceneSnapshot(
            batteries=current.batteries,
            remote_position_xyz=current.remote_position_xyz,
            remote_rotation_xyzw=current.remote_rotation_xyzw,
            remote_displacement_m=0.0,
            remote_yaw_error_deg=10.0,
            damage_force_n=0.0,
            ever_off_table=current.ever_off_table,
            ever_remote_moved=False,
            ever_damage=False,
        )
        _, report = step(self.cfg, yawed, 0.5)
        self.assertEqual(report.score, 0)
        self.assertIn("remote_moved", report.hard_reasons)

    def test_damage_proxy_overrides_100(self):
        current, _ = self._inserted_after_lift()
        current = update_battery(current, "battery_a", peak_force_n=100.0)
        _, report = step(self.cfg, current, 0.5)
        self.assertEqual(report.score, 0)
        self.assertIn("damage_proxy", report.hard_reasons)
        self.assertTrue(report.inserted["battery_b"])

    def test_clear_episode_state_drops_memory(self):
        current, report = self._inserted_after_lift()
        self.assertEqual(report.score, 100)
        cleared = clear_episode_state(current)
        self.assertFalse(any(battery.ever_lifted for battery in cleared.batteries))
        self.assertTrue(all(battery.retention_timer_s == 0.0 for battery in cleared.batteries))
        self.assertFalse(cleared.ever_off_table)
        _, report = step(self.cfg, cleared, 0.5)
        self.assertEqual(report.score, 0)
        self.assertIn("never_lifted", report.hard_reasons)
        self.assertLess(report.retention_timer_s["battery_a"], 1.0)

    def test_single_slot_scores_100_for_one_battery(self):
        cfg = with_single_slot(self.cfg, True)
        lifted = _scene(_battery("battery_a", position_xyz=(0.50, -0.09, 0.80), seating_depth_m=0.0, spring_contact=False))
        current, _ = step(cfg, lifted, 0.5)
        current = update_battery(current, "battery_a", position_xyz=(0.50, -0.09, 0.76), seating_depth_m=0.039, spring_contact=True)
        current, report = step(cfg, current, 0.5)
        current, report = step(cfg, current, 0.5)
        self.assertEqual(report.score, 100)
        self.assertNotIn("battery_b", report.inserted)

    def test_inside_remote_bounds_is_not_insertion(self):
        slot = self.cfg.compartment.slots[0]
        measured = measure_battery(
            self.cfg,
            slot,
            self.cfg.remote_pose.position_xyz,
            (0.0, 0.0, 0.0, 1.0),
            self.cfg.remote_pose.position_xyz,
            self.cfg.remote_pose.rotation_xyzw,
        )
        self.assertFalse(measured.spring_contact)
        self.assertLess(measured.seating_depth_m, full_seat_depth_m(self.cfg) - self.cfg.scoring.axial_tolerance_m)
        current, _ = step(
            self.cfg,
            _both(position_xyz=(0.50, 0.0, 0.80), seating_depth_m=0.0, spring_contact=False),
            0.5,
        )
        current = update_battery(
            current,
            "battery_a",
            position_xyz=self.cfg.remote_pose.position_xyz,
            positive_axis_xyz=measured.positive_axis_xyz,
            seating_depth_m=measured.seating_depth_m,
            radial_offset_m=measured.radial_offset_m,
            spring_contact=measured.spring_contact,
            polarity_aligned=True,
        )
        _, report = self._to_score(current)
        self.assertNotEqual(report.score, 100)
        self.assertFalse(report.inserted["battery_a"])

    def test_positive_terminal_quat_points_along_world_x(self):
        pointed = rotate(START_QUAT, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(pointed[0], 1.0, places=6)
        self.assertAlmostEqual(pointed[1], 0.0, places=6)
        self.assertAlmostEqual(pointed[2], 0.0, places=6)

    def test_grasp_uses_finger_opening(self):
        closed = gripper_is_holding(
            self.cfg,
            (0.5, 0.0, 0.8),
            ((0.51, 0.0, 0.8), None),
            ((0.0, 0.0), None),
        )
        opened = gripper_is_holding(
            self.cfg,
            (0.5, 0.0, 0.8),
            ((0.51, 0.0, 0.8), None),
            ((self.cfg.robot.finger_open_m, self.cfg.robot.finger_open_m), None),
        )
        missing = gripper_is_holding(self.cfg, (0.5, 0.0, 0.8), (None, None), (None, None))
        self.assertTrue(closed)
        self.assertFalse(opened)
        self.assertFalse(missing)

    def test_dt_must_be_positive(self):
        with self.assertRaises(ValueError):
            step(self.cfg, _both(), 0.0)


if __name__ == "__main__":
    unittest.main()
