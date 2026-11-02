"""Pure tests for the scoring state machine. They do not import Isaac Sim."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

from my_package.assets.geometry import measure_battery, plunger_box, rest_center_z
from my_package.scoring.config import default_config_path, load_config, with_single_slot
from my_package.scoring.grasp import gripper_is_holding
from my_package.scoring.math3d import quat_from_axis_angle, rotate
from my_package.scoring.scorer import assign_cells, battery_is_lifted, battery_lowest_z, step
from my_package.scoring.snapshot import BatterySnapshot, SceneSnapshot, SlotFit, clear_episode_state, update_battery
from my_package.tasks.state_values import env_indices, quaternion_xyzw

PLUS_X = (1.0, 0.0, 0.0)
MINUS_X = (-1.0, 0.0, 0.0)
# +90 deg about Y: local +Z (positive terminal) onto world +X.
PLUS_X_QUAT = (0.0, 0.7071067811865475, 0.0, 0.7071067811865476)
# -90 deg about Y: positive terminal onto world -X.
MINUS_X_QUAT = (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)
REQUIRED = {"slot_a": PLUS_X, "slot_b": MINUS_X}
# World heights from the measured table top (0.003 m).
TABLE_Z = 0.003
SEATED_Z = 0.014 + 0.00475
LIFTED_Z = 0.05


def _fit(slot: str, seated: bool, **overrides) -> SlotFit:
    values = dict(
        slot=slot,
        axial_offset_m=0.0 if seated else 0.03,
        radial_offset_m=0.0005 if seated else 0.05,
        polarity_aligned=seated,
        spring_contact=seated,
    )
    values.update(overrides)
    return SlotFit(**values)


def _cell(name: str, *, seated_in: str | None = None, z: float = LIFTED_Z, axis=PLUS_X, fit_overrides=None, **overrides) -> BatterySnapshot:
    """A cell either seated in one slot (with polarity matching that slot) or away from both."""
    if seated_in is not None:
        axis = REQUIRED[seated_in]
        z = SEATED_Z
    fits = tuple(
        _fit(slot, slot == seated_in, **(fit_overrides or {}) if slot == seated_in else {})
        for slot in ("slot_a", "slot_b")
    )
    values = dict(
        name=name,
        position_xyz=(0.50, -0.09 if name == "battery_a" else 0.09, z),
        rotation_xyzw=PLUS_X_QUAT if axis == PLUS_X else MINUS_X_QUAT,
        positive_axis_xyz=axis,
        fits=fits,
        gripper_holding=False,
        released=True,
        off_table=False,
        peak_force_n=0.0,
    )
    values.update(overrides)
    return BatterySnapshot(**values)


def _scene(*batteries: BatterySnapshot, **overrides) -> SceneSnapshot:
    values = dict(
        batteries=batteries,
        remote_position_xyz=(0.50, 0.0, 0.014),
        remote_rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        remote_displacement_m=0.0,
        remote_yaw_error_deg=0.0,
        damage_force_n=0.0,
    )
    values.update(overrides)
    return SceneSnapshot(**values)


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(default_config_path())

    def _run(self, *scenes: SceneSnapshot, dt: float = 0.5, cfg=None):
        """Step each scene in turn, carrying memory from the previous result."""
        cfg = cfg or self.cfg
        current = None
        report = None
        for scene in scenes:
            if current is not None:
                scene = replace(
                    scene,
                    batteries=tuple(
                        replace(cell, ever_lifted=prior.ever_lifted)
                        for cell, prior in zip(scene.batteries, current.batteries)
                    ),
                    ever_off_table=current.ever_off_table,
                    ever_remote_moved=current.ever_remote_moved,
                    ever_damage=current.ever_damage,
                    retention=current.retention,
                )
            current, report = step(cfg, scene, dt)
        return current, report

    def _lifted(self):
        return _scene(_cell("battery_a"), _cell("battery_b"))

    def _seated(self, a_slot="slot_a", b_slot="slot_b", **overrides):
        return _scene(_cell("battery_a", seated_in=a_slot, **overrides), _cell("battery_b", seated_in=b_slot, **overrides))

    def test_both_inserted_scores_100(self):
        _, report = self._run(self._lifted(), self._seated(), self._seated())
        self.assertEqual(report.score, 100)
        self.assertEqual(report.inserted, {"slot_a": True, "slot_b": True})
        self.assertEqual(report.occupant, {"slot_a": "battery_a", "slot_b": "battery_b"})
        self.assertFalse(report.hard_failure)

    def test_cells_are_interchangeable(self):
        swapped = self._seated(a_slot="slot_b", b_slot="slot_a")
        _, report = self._run(self._lifted(), swapped, swapped)
        self.assertEqual(report.score, 100)
        self.assertEqual(report.occupant, {"slot_a": "battery_b", "slot_b": "battery_a"})

    def test_one_slot_cannot_be_counted_twice(self):
        # Both cells claim slot_a; only one may fill it.
        both_a = self._seated(a_slot="slot_a", b_slot="slot_a")
        _, report = self._run(self._lifted(), both_a, both_a)
        self.assertEqual(report.score, 50)
        self.assertTrue(report.inserted["slot_a"])
        self.assertFalse(report.inserted["slot_b"])

    def test_assignment_keeps_previous_occupant_on_tie(self):
        seated = {("battery_a", "slot_a"), ("battery_b", "slot_a")}
        chosen = assign_cells(seated, ("battery_a", "battery_b"), ("slot_a", "slot_b"), {"slot_a": "battery_b"})
        self.assertEqual(chosen, {"slot_a": "battery_b", "slot_b": None})

    def test_one_inserted_scores_50(self):
        one = _scene(_cell("battery_a", seated_in="slot_a"), _cell("battery_b"))
        _, report = self._run(self._lifted(), one, one)
        self.assertEqual(report.score, 50)
        self.assertTrue(report.inserted["slot_a"])
        self.assertFalse(report.inserted["slot_b"])

    def test_lift_without_insertion_scores_25(self):
        _, report = self._run(self._lifted())
        self.assertEqual(report.score, 25)
        self.assertTrue(report.ever_lifted["battery_a"])
        self.assertFalse(any(report.inserted.values()))

    def test_resting_pose_is_not_lifted(self):
        resting = _scene(_cell("battery_a", z=TABLE_Z + 0.00525), _cell("battery_b", z=TABLE_Z + 0.00525))
        _, report = self._run(resting)
        self.assertIn("never_lifted", report.hard_reasons)
        self.assertEqual(report.score, 0)

    def test_standing_cell_is_not_lifted(self):
        # A cell stood on its end has its center 22 mm up but its base on the table.
        standing = _cell("battery_a", z=TABLE_Z + 0.02225, rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        self.assertAlmostEqual(battery_lowest_z(self.cfg, standing), TABLE_Z, places=9)
        self.assertFalse(battery_is_lifted(self.cfg, standing))

    def test_tilted_cell_lowest_point_uses_orientation(self):
        tilt = quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(30.0))
        cell = _cell("battery_a", z=LIFTED_Z, rotation_xyzw=tilt)
        expected = LIFTED_Z - (0.00525 * math.sin(math.radians(30.0)) + 0.02225 * math.cos(math.radians(30.0)))
        self.assertAlmostEqual(battery_lowest_z(self.cfg, cell), expected, places=9)

    def test_wrong_polarity_for_the_slot_does_not_insert(self):
        # battery_a lies in slot_b but still points +X; slot_b needs -X.
        wrong = _scene(
            _cell("battery_a", seated_in="slot_b", positive_axis_xyz=PLUS_X, rotation_xyzw=PLUS_X_QUAT),
            _cell("battery_b"),
        )
        _, report = self._run(self._lifted(), wrong, wrong)
        self.assertFalse(any(report.inserted.values()))
        self.assertEqual(report.score, 25)

    def test_false_flag_does_not_insert(self):
        flagged = _scene(
            _cell("battery_a", seated_in="slot_a", fit_overrides={"polarity_aligned": False}),
            _cell("battery_b"),
        )
        _, report = self._run(self._lifted(), flagged, flagged)
        self.assertFalse(report.inserted["slot_a"])

    def test_partly_seated_does_not_insert(self):
        partly = self._seated(fit_overrides={"axial_offset_m": 0.01})
        _, report = self._run(self._lifted(), partly, partly)
        self.assertFalse(any(report.inserted.values()))
        self.assertEqual(report.score, 25)

    def test_axial_offset_has_an_upper_and_lower_bound(self):
        tol = self.cfg.scoring.axial_tolerance_m
        for offset in (tol * 2.0, -tol * 2.0):
            past = self._seated(fit_overrides={"axial_offset_m": offset})
            _, report = self._run(self._lifted(), past, past)
            self.assertFalse(any(report.inserted.values()), offset)
        inside = self._seated(fit_overrides={"axial_offset_m": tol * 0.5})
        _, report = self._run(self._lifted(), inside, inside)
        self.assertEqual(report.score, 100)

    def test_unknown_plunger_reading_never_inserts(self):
        """A missing plunger joint is unknown, not the authored rest pose: no credit, and reported."""
        unknown = self._seated(fit_overrides={"spring_contact": None})
        _, report = self._run(self._lifted(), unknown, unknown, unknown)
        self.assertEqual(report.inserted, {"slot_a": False, "slot_b": False})
        self.assertEqual(report.score, 25)
        self.assertEqual(report.spring_unknown, ("slot_a", "slot_b"))

    def test_measure_without_a_plunger_reading_reports_unknown_contact(self):
        slot = self.cfg.compartment.slots[0]
        remote = self.cfg.remote_pose
        seated = (remote.position_xyz[0] + self.cfg.compartment.seat_center_x, slot.center_y_m, remote.position_xyz[2] + rest_center_z(self.cfg))
        known = measure_battery(self.cfg, slot, seated, PLUS_X_QUAT, remote.position_xyz, remote.rotation_xyzw, plunger_joint_m=self.cfg.compartment.spring_preload_m)
        self.assertTrue(known.spring_contact)
        missing = measure_battery(self.cfg, slot, seated, PLUS_X_QUAT, remote.position_xyz, remote.rotation_xyzw, plunger_joint_m=None)
        self.assertIsNone(missing.spring_contact)

    def test_missing_spring_contact_does_not_insert(self):
        loose = self._seated(fit_overrides={"spring_contact": False})
        _, report = self._run(self._lifted(), loose, loose)
        self.assertFalse(any(report.inserted.values()))

    def test_large_radial_offset_does_not_insert(self):
        off = self._seated(fit_overrides={"radial_offset_m": 0.05})
        _, report = self._run(self._lifted(), off, off)
        self.assertFalse(any(report.inserted.values()))

    def test_gripper_hold_blocks_the_retention_window(self):
        held = self._seated(gripper_holding=True, released=False)
        _, report = self._run(self._lifted(), *([held] * 6))
        self.assertEqual(report.retention_timer_s["slot_a"], 0.0)
        self.assertFalse(report.inserted["slot_a"])
        self.assertEqual(report.score, 25)

    def test_hold_overrides_a_released_flag(self):
        held = self._seated(gripper_holding=True, released=True)
        _, report = self._run(self._lifted(), held, held)
        self.assertFalse(any(report.inserted.values()))

    def test_unknown_hold_is_not_a_release(self):
        unknown = self._seated(gripper_holding=None, released=True)
        _, report = self._run(self._lifted(), unknown, unknown, unknown)
        self.assertFalse(any(report.inserted.values()))
        self.assertEqual(report.score, 25)
        self.assertEqual(report.hold_unknown, ("battery_a", "battery_b"))

    def test_release_before_the_window_does_not_count(self):
        _, report = self._run(self._lifted(), self._seated())
        self.assertAlmostEqual(report.retention_timer_s["slot_a"], 0.5)
        self.assertFalse(report.inserted["slot_a"])
        _, report = self._run(self._lifted(), self._seated(), self._lifted())
        self.assertEqual(report.retention_timer_s["slot_a"], 0.0)
        self.assertEqual(report.score, 25)

    def test_swapping_occupant_restarts_the_window(self):
        swapped = self._seated(a_slot="slot_b", b_slot="slot_a")
        _, report = self._run(self._lifted(), self._seated(), swapped)
        self.assertAlmostEqual(report.retention_timer_s["slot_a"], 0.5)
        self.assertFalse(report.inserted["slot_a"])

    def test_falling_out_after_the_window_drops_credit(self):
        _, report = self._run(self._lifted(), self._seated(), self._seated(), self._lifted())
        self.assertFalse(any(report.inserted.values()))
        self.assertEqual(report.score, 25)

    def test_never_lifted_overrides_an_inserted_signal(self):
        _, report = self._run(self._seated(), self._seated())
        self.assertTrue(report.inserted["slot_a"])
        self.assertTrue(report.inserted["slot_b"])
        self.assertEqual(report.score, 0)
        self.assertIn("never_lifted", report.hard_reasons)
        self.assertFalse(report.hard_failure)

    def test_off_table_latch_overrides_100_and_sticks(self):
        off = _scene(_cell("battery_a", seated_in="slot_a", off_table=True), _cell("battery_b", seated_in="slot_b"))
        _, report = self._run(self._lifted(), self._seated(), self._seated(), off, self._seated())
        self.assertEqual(report.score, 0)
        self.assertIn("off_table", report.hard_reasons)
        self.assertTrue(report.ever_off_table)

    def test_geometric_off_table_overrides_without_the_flag(self):
        gone = _scene(_cell("battery_a", seated_in="slot_a"), _cell("battery_b", position_xyz=(0.50, 2.0, LIFTED_Z)))
        _, report = self._run(self._lifted(), gone)
        self.assertEqual(report.score, 0)
        self.assertIn("off_table", report.hard_reasons)

    def test_remote_motion_overrides_100(self):
        moved = replace(self._seated(), remote_displacement_m=0.02)
        _, report = self._run(self._lifted(), self._seated(), moved)
        self.assertEqual(report.score, 0)
        self.assertIn("remote_moved", report.hard_reasons)
        self.assertTrue(report.inserted["slot_a"])

    def test_remote_yaw_overrides_100(self):
        yawed = replace(self._seated(), remote_yaw_error_deg=10.0)
        _, report = self._run(self._lifted(), self._seated(), yawed)
        self.assertEqual(report.score, 0)
        self.assertIn("remote_moved", report.hard_reasons)

    def test_damage_proxy_overrides_100(self):
        hit = _scene(_cell("battery_a", seated_in="slot_a", peak_force_n=100.0), _cell("battery_b", seated_in="slot_b"))
        _, report = self._run(self._lifted(), self._seated(), hit)
        self.assertEqual(report.score, 0)
        self.assertIn("damage_proxy", report.hard_reasons)

    def test_clear_episode_state_drops_memory(self):
        current, report = self._run(self._lifted(), self._seated(), self._seated())
        self.assertEqual(report.score, 100)
        cleared = clear_episode_state(current)
        self.assertFalse(any(battery.ever_lifted for battery in cleared.batteries))
        self.assertEqual(cleared.retention, ())
        self.assertFalse(cleared.ever_off_table)
        _, report = step(self.cfg, cleared, 0.5)
        self.assertEqual(report.score, 0)
        self.assertIn("never_lifted", report.hard_reasons)
        self.assertLess(report.retention_timer_s["slot_a"], 1.0)

    def test_single_slot_scores_100_for_one_battery(self):
        cfg = with_single_slot(self.cfg, True)
        lifted = _scene(_cell("battery_a"))
        seated = _scene(_cell("battery_a", seated_in="slot_a"))
        _, report = self._run(lifted, seated, seated, cfg=cfg)
        self.assertEqual(report.score, 100)
        self.assertEqual(set(report.inserted), {"slot_a"})

    def test_dt_must_be_positive(self):
        with self.assertRaises(ValueError):
            step(self.cfg, self._lifted(), 0.0)


class MeasureTests(unittest.TestCase):
    """measure_battery against real slot geometry, not hand-written fits."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(default_config_path())
        cls.remote = cls.cfg.remote_pose

    def _seated_world(self, slot):
        x = self.remote.position_xyz[0] + self.cfg.compartment.seat_center_x
        y = self.remote.position_xyz[1] + slot.center_y_m
        z = self.remote.position_xyz[2] + rest_center_z(self.cfg)
        return (x, y, z)

    def _measure(self, slot, position, rotation, joint):
        return measure_battery(self.cfg, slot, position, rotation, self.remote.position_xyz, self.remote.rotation_xyzw, joint)

    def test_seated_cell_with_compressed_spring_measures_as_seated(self):
        preload = self.cfg.compartment.spring_preload_m
        for slot, quat in zip(self.cfg.compartment.slots, (PLUS_X_QUAT, MINUS_X_QUAT)):
            measured = self._measure(slot, self._seated_world(slot), quat, preload)
            self.assertAlmostEqual(measured.axial_offset_m, 0.0, places=9)
            self.assertLess(measured.radial_offset_m, 1e-9)
            self.assertLess(measured.polarity_angle_deg, 1e-6)
            self.assertTrue(measured.spring_contact, slot.name)

    def test_reversed_cell_fails_polarity(self):
        slot = self.cfg.compartment.slots[0]
        measured = self._measure(slot, self._seated_world(slot), MINUS_X_QUAT, self.cfg.compartment.spring_preload_m)
        self.assertGreater(measured.polarity_angle_deg, 170.0)

    def test_cell_in_the_other_slot_is_far_off_axis(self):
        slot_a, slot_b = self.cfg.compartment.slots
        measured = self._measure(slot_a, self._seated_world(slot_b), PLUS_X_QUAT, 0.0)
        self.assertGreater(measured.radial_offset_m, self.cfg.scoring.seat_radial_max_m)

    def test_cell_above_the_slot_is_not_spring_contact(self):
        slot = self.cfg.compartment.slots[0]
        x, y, z = self._seated_world(slot)
        measured = self._measure(slot, (x, y, z + 0.03), PLUS_X_QUAT, 0.0)
        self.assertFalse(measured.spring_contact)
        self.assertGreater(measured.radial_offset_m, self.cfg.scoring.seat_radial_max_m)

    def test_plunger_compresses_away_from_the_cell(self):
        for slot in self.cfg.compartment.slots:
            extended = plunger_box(self.cfg, slot, 0.0).center[0]
            compressed = plunger_box(self.cfg, slot, self.cfg.compartment.plunger_travel_m).center[0]
            away = math.copysign(1.0, extended - self.cfg.compartment.seat_center_x)
            self.assertGreater(away * (compressed - extended), 0.0, slot.name)

    def test_positive_terminal_quat_points_along_world_x(self):
        pointed = rotate(PLUS_X_QUAT, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(pointed[0], 1.0, places=6)
        self.assertAlmostEqual(pointed[2], 0.0, places=6)
        pointed = rotate(MINUS_X_QUAT, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(pointed[0], -1.0, places=6)


class GraspTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(default_config_path())

    def test_grasp_uses_finger_opening(self):
        near = ((0.51, 0.0, 0.8), (0.0, 0.5, 0.8))
        closed = gripper_is_holding(self.cfg, (0.5, 0.0, 0.8), near, ((0.0, 0.0), (0.0, 0.0)))
        open_m = self.cfg.robot.finger_open_m
        opened = gripper_is_holding(self.cfg, (0.5, 0.0, 0.8), near, ((open_m, open_m), (open_m, open_m)))
        self.assertIs(closed, True)
        self.assertIs(opened, False)

    def test_missing_gripper_names_make_hold_unknown(self):
        missing = gripper_is_holding(self.cfg, (0.5, 0.0, 0.8), (None, None), (None, None))
        self.assertIsNone(missing)
        open_m = self.cfg.robot.finger_open_m
        one_missing = gripper_is_holding(self.cfg, (0.5, 0.0, 0.8), ((0.51, 0.0, 0.8), None), ((open_m, open_m), None))
        self.assertIsNone(one_missing)

    def test_observed_hold_wins_over_a_missing_arm(self):
        holding = gripper_is_holding(self.cfg, (0.5, 0.0, 0.8), ((0.51, 0.0, 0.8), None), ((0.0, 0.0), None))
        self.assertIs(holding, True)


class _FakeTensor:
    """Stands in for a torch tensor: detach() -> cpu() -> tolist()."""

    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class StateValueTests(unittest.TestCase):
    def test_env_indices_accepts_lists_tensors_and_none(self):
        self.assertEqual(env_indices([0, 2], 4), [0, 2])
        self.assertEqual(env_indices(_FakeTensor([1, 3]), 4), [1, 3])
        self.assertEqual(env_indices(None, 3), [0, 1, 2])
        self.assertEqual(env_indices((1,), 3), [1])

    def test_quaternion_is_read_as_xyzw_without_reordering(self):
        # Server probe: a 90 deg X turn reads [0.707, 0, 0, 0.707] from root_quat_w.
        quat = quaternion_xyzw(_FakeTensor([0.7071, 0.0, 0.0, 0.7071]))
        self.assertEqual(quat, (0.7071, 0.0, 0.0, 0.7071))
        self.assertEqual(quaternion_xyzw([0.0, 0.0, 0.0, 1.0]), (0.0, 0.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            quaternion_xyzw([1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
