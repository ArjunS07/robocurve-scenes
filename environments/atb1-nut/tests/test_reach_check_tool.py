"""tools/reach_check.py compiles and its pure helpers behave. No Isaac Sim."""

from __future__ import annotations

import importlib.util
import math
import py_compile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "reach_check.py"


def _load():
    spec = importlib.util.spec_from_file_location("nut_reach_check", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # top level imports only the standard library
    return module


def _hexagon(vertex_phase_deg, across_flats=0.024, samples_per_edge=8):
    """Outline points of a hexagon whose vertices sit at vertex_phase + k * 60 degrees."""
    corner = across_flats / math.sqrt(3.0)
    vertices = [
        (corner * math.cos(math.radians(vertex_phase_deg + 60 * k)), corner * math.sin(math.radians(vertex_phase_deg + 60 * k)))
        for k in range(6)
    ]
    points = []
    for k in range(6):
        (x0, y0), (x1, y1) = vertices[k], vertices[(k + 1) % 6]
        points += [(x0 + (x1 - x0) * i / samples_per_edge, y0 + (y1 - y0) * i / samples_per_edge) for i in range(samples_per_edge)]
    return points


class ReachCheckToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _load()

    def test_compiles(self):
        py_compile.compile(str(TOOL), doraise=True)

    def test_wrap(self):
        self.assertAlmostEqual(self.tool.wrap_deg(190.0), -170.0)
        self.assertAlmostEqual(self.tool.wrap_deg(-35.0, 60.0), 25.0)
        self.assertAlmostEqual(self.tool.wrap_deg(29.0, 60.0), 29.0)

    def test_flat_phase_is_halfway_between_vertices(self):
        # Vertices at 0, 60, ... put the flat normals at 30, 90, ...
        self.assertAlmostEqual(self.tool.hex_flat_phase_deg(_hexagon(0.0)), 30.0, delta=0.5)
        self.assertAlmostEqual(self.tool.hex_flat_phase_deg(_hexagon(30.0)), 0.0, delta=0.5)
        self.assertAlmostEqual(self.tool.hex_flat_phase_deg(_hexagon(12.0)), 42.0, delta=0.5)

    def test_flat_phase_ignores_the_bore(self):
        bore = [(0.007 * math.cos(a / 10), 0.007 * math.sin(a / 10)) for a in range(63)]
        self.assertAlmostEqual(self.tool.hex_flat_phase_deg(_hexagon(0.0) + bore), 30.0, delta=0.5)

    def test_flats_toward_picks_the_nearest_flat_direction(self):
        self.assertAlmostEqual(self.tool.flats_toward(30.0, 90.0), 90.0)
        self.assertAlmostEqual(self.tool.flats_toward(10.0, 90.0), 70.0)
        self.assertAlmostEqual(self.tool.flats_toward(50.0, 90.0), 110.0)

    def test_wrist_target_lands_on_flats_nearest_the_goal_within_limits(self):
        choose = self.tool.choose_wrist_target
        # Fingers 20 deg short of the flats; joint6 turns them +1 rad of yaw per rad.
        j6 = choose(-1.5, 1.0, 20.0, 1.57, -2.04, 2.04)
        self.assertAlmostEqual(math.degrees(j6 + 1.5) % 60.0, 20.0, places=6)
        self.assertLess(abs(j6 - 1.57), math.radians(30.0) + 1e-9)
        # A reversed joint6 needs the opposite rotation for the same yaw.
        j6 = choose(1.5, -1.0, 20.0, -1.57, -2.04, 2.04)
        self.assertAlmostEqual(math.degrees(1.5 - j6) % 60.0, 20.0, places=6)
        self.assertIsNone(choose(0.0, 1.0, 20.0, 0.0, 0.1, 0.2))
        with self.assertRaises(ValueError):
            choose(0.0, 0.0, 20.0, 0.0, -2.0, 2.0)

    def test_torque_summary(self):
        summary = self.tool.summarize_torques([0.1, -0.3, None, float("nan"), 0.2])
        self.assertEqual(summary["samples"], 3)
        self.assertAlmostEqual(summary["max_abs_n_m"], 0.3)
        self.assertAlmostEqual(summary["mean_abs_n_m"], 0.2)
        self.assertEqual(self.tool.summarize_torques([]), {"samples": 0})


if __name__ == "__main__":
    unittest.main()
