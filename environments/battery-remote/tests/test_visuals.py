"""Visual shell, materials, and capture-hook tests. No Isaac Sim imports."""

from __future__ import annotations

import os
import tempfile
import unittest

from my_package.assets.geometry import (
    _LOOKS,
    end_collider_outlines,
    shell_spec,
    slot_holes,
    static_boxes,
    visual_meshes,
    write_geometry,
)
from my_package.assets.remote_shell import plan_half_width, top_height
from my_package.scoring.config import default_config_path, load_config, with_single_slot
from my_package.tasks.capture import capture_steps, maybe_capture


def _centroid(mesh, triangle):
    points = [mesh.points[index] for index in triangle]
    return tuple(sum(p[k] for p in points) / 3.0 for k in range(3))


class ShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(default_config_path())
        cls.meshes = visual_meshes(cls.cfg)
        cls.spec = shell_spec(cls.cfg)

    def test_listed_outer_size_is_respected(self):
        """Listing: 23.5 x 4.8 x 2.4 cm. The shell must not exceed it."""
        points = self.meshes["shell"].points
        xs, ys, zs = zip(*points)
        self.assertAlmostEqual(max(xs) - min(xs), 0.235, places=6)
        self.assertLessEqual(max(ys) - min(ys), 0.048 + 1e-9)
        self.assertLessEqual(max(zs) - min(zs), 0.024 + 1e-9)

    def test_plan_tapers_from_the_nose_and_rounds_the_ends(self):
        spec = self.spec
        self.assertGreater(plan_half_width(spec, spec.nose_x + spec.end_radius), plan_half_width(spec, spec.taper_end_x + 0.01))
        self.assertLess(plan_half_width(spec, spec.nose_x), spec.nose_half_width - 0.5 * spec.end_radius)
        self.assertLess(plan_half_width(spec, spec.front_x), spec.bay_half_width - 0.5 * spec.end_radius)
        # Filleted top edge: lower at the side than on the centreline.
        x = spec.taper_end_x + 0.02
        self.assertLess(top_height(spec, x, plan_half_width(spec, x)), top_height(spec, x, 0.0))

    def test_slots_are_open_from_above(self):
        """No shell face covers a slot opening, and each slot has a floor at floor_z."""
        shell = self.meshes["shell"]
        for hole in slot_holes(self.cfg):
            covering = [
                t for t in shell.triangles
                if hole.x_min < _centroid(shell, t)[0] < hole.x_max
                and hole.y_min < _centroid(shell, t)[1] < hole.y_max
                and _centroid(shell, t)[2] > self.cfg.compartment.floor_z_m + 1e-6
                and abs(_centroid(shell, t)[2] - self.spec.half_height) < 1e-6
            ]
            self.assertEqual(covering, [], hole.name)

    def test_holes_match_the_collision_slots(self):
        boxes = {box.name: box for box in static_boxes(self.cfg)}
        for hole in slot_holes(self.cfg):
            contact = boxes[f"{hole.name}_fixed_contact"]
            face = contact.center[0] + (-1 if hole.contact_at_max_x else 1) * contact.size[0] / 2.0
            self.assertAlmostEqual(face, hole.x_max if hole.contact_at_max_x else hole.x_min, places=9)
            self.assertAlmostEqual(hole.y_max - hole.y_min, 2.0 * self.cfg.compartment.channel_y_half, places=9)

    def test_contact_plate_is_the_slot_end_wall(self):
        contacts = self.meshes["contacts"]
        holes = slot_holes(self.cfg)
        self.assertEqual(len(contacts.triangles), 2 * len(holes))
        xs = sorted({round(p[0], 9) for p in contacts.points})
        expected = sorted({round(h.x_max if h.contact_at_max_x else h.x_min, 9) for h in holes})
        self.assertEqual(xs, expected)

    def test_marks_stand_proud_and_start_below_their_surface(self):
        """No coplanar faces: every mark bar crosses its surface instead of lying on it."""
        floor = self.cfg.compartment.floor_z_m
        top = self.spec.half_height
        marks = self.meshes["marks"]
        for start in range(0, len(marks.points), 8):
            zs = [p[2] for p in marks.points[start:start + 8]]
            surface = floor if min(zs) < floor + 0.001 else top
            self.assertLess(min(zs), surface)
            self.assertGreater(max(zs), surface)

    def test_marks_include_plus_and_minus_for_each_slot(self):
        # Two bars for +, one for -; floor and spine copies for each slot.
        boxes = len(self.meshes["marks"].points) // 8
        self.assertEqual(boxes, len(slot_holes(self.cfg)) * 2 * (2 + 1))

    def test_end_colliders_fit_physx_hull_limits_and_the_shell(self):
        for name, outline in end_collider_outlines(self.cfg).items():
            self.assertLessEqual(2 * len(outline), 64, name)
            for x, y in outline:
                self.assertLessEqual(abs(y), plan_half_width(self.spec, x) + 1e-9)

    def test_colliders_are_guides_and_visuals_have_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            text = write_geometry(self.cfg, directory)["remote"].read_text()
        for name in ("floor", "rail_left", "spine", "rail_right", "slot_a_fixed_contact", "nose_collider", "handle_collider"):
            block = text.split(f'"{name}"', 1)[1].split("}", 1)[0]
            self.assertIn('uniform token purpose = "guide"', block, name)
        for look in _LOOKS:
            self.assertIn(f'def Material "{look}"', text)
        self.assertIn("rel material:binding = </remote/Looks/black_abs>", text)
        self.assertIn("rel material:binding = </remote/Looks/nickel>", text)
        self.assertEqual(text.count('def Mesh "coil"'), len(self.cfg.compartment.slots))

    def test_single_slot_cuts_one_hole(self):
        single = with_single_slot(self.cfg, True)
        self.assertEqual([hole.name for hole in slot_holes(single)], ["slot_a"])
        meshes = visual_meshes(single)
        self.assertEqual(len(meshes["contacts"].triangles), 2)


class CaptureTests(unittest.TestCase):
    def test_steps_parse(self):
        self.assertEqual(capture_steps(None), {1, 25, 49})
        self.assertEqual(capture_steps("3, 7"), {3, 7})

    def test_noop_without_capture_dir(self):
        previous = os.environ.pop("ROBOCURVE_CAPTURE_DIR", None)
        try:
            self.assertEqual(maybe_capture(object(), 1), [])
        finally:
            if previous is not None:
                os.environ["ROBOCURVE_CAPTURE_DIR"] = previous


if __name__ == "__main__":
    unittest.main()
