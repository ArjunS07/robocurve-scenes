"""Check physical camera axes and projected task coverage without Isaac Sim."""
import math
import unittest

from my_package.scoring.config import default_config_path, load_config
from my_package.scoring.math3d import (
    dot, look_at_ros_xyzw, normalize, quat_inv, rotate, sub,
)


def project(camera, point):
    x, y, z = rotate(quat_inv(camera.rotation_xyzw), sub(point, camera.eye_xyz))
    if z <= 0:
        raise AssertionError('Task geometry is behind the camera')
    return (camera.width / 2 + camera.width * camera.focal_length / camera.horizontal_aperture * x / z,
            camera.height / 2 + camera.height * camera.focal_length / camera.vertical_aperture * y / z)


class CameraTests(unittest.TestCase):
    def test_oblique_camera_is_upright_and_looks_at_target(self):
        # Independent world-axis properties catch the old upside-down look-at
        # even if the YAML and helper are regenerated to agree with each other.
        for eye in ((1., 0., .4), (0., -1., .4), (-1., .3, .6)):
            with self.subTest(eye=eye):
                target = (0., 0., 0.)
                q = look_at_ros_xyzw(eye, target)
                forward = rotate(q, (0., 0., 1.))
                down = rotate(q, (0., 1., 0.))
                self.assertGreater(dot(forward, normalize(sub(target, eye))), 1 - 1e-12)
                self.assertLess(dot(down, (0., 0., 1.)), 0.)
                # An elevated point at the same ground position must appear above the target.
                local = rotate(quat_inv(q), sub((0., 0., .1), eye))
                self.assertLess(local[1], 0.)

    def test_near_vertical_views_remain_orthonormal_and_exactly_aimed(self):
        for eye in ((0., 0., 1.), (.08, .02, 1.), (0., 0., -1.)):
            q = look_at_ros_xyzw(eye, (0., 0., 0.))
            axes = [rotate(q, axis) for axis in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))]
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(dot(axes[i], axes[j]), float(i == j), places=12)
            self.assertGreater(dot(axes[2], normalize(sub((0., 0., 0.), eye))), 1 - 1e-12)

    def test_undefined_camera_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            look_at_ros_xyzw((0., 0., 0.), (0., 0., 0.))

    def test_board_and_piece_height_envelope_fit_all_views(self):
        cfg = load_config(default_config_path())
        cx, cy, _ = cfg.board_pose.position_xyz
        half = 4 * cfg.board.square_m + cfg.board.border_m
        top = cfg.table.top_z_m + cfg.board.thickness_m
        highest = top + max(piece.height_m for piece in cfg.pieces)
        points = [(x, y, z) for x in (cx - half, cx + half)
                  for y in (cy - half, cy + half) for z in (top, highest)]
        for camera in cfg.cameras:
            with self.subTest(camera=camera.name):
                uv = [project(camera, p) for p in points]
                for u, v in uv:
                    self.assertGreaterEqual(min(u, v), 12.)
                    self.assertLessEqual(u, camera.width - 12.)
                    self.assertLessEqual(v, camera.height - 12.)
                self.assertEqual(camera.width, camera.height)
                self.assertAlmostEqual(camera.horizontal_aperture, camera.vertical_aperture)
                self.assertLess(math.degrees(2 * math.atan(camera.horizontal_aperture / (2 * camera.focal_length))), 45.)
                board_uv = [project(camera, (x, y, top)) for x in (cx - half, cx + half)
                            for y in (cy - half, cy + half)]
                self.assertGreater(max(u for u, _ in board_uv) - min(u for u, _ in board_uv), 130.)

    def test_white_home_rank_is_at_bottom_of_overhead_view(self):
        cfg = load_config(default_config_path())
        camera = cfg.cameras[0]
        cx, cy, _ = cfg.board_pose.position_xyz
        z = cfg.table.top_z_m + cfg.board.thickness_m
        self.assertGreater(project(camera, (cx - .1, cy, z))[1],
                           project(camera, (cx + .1, cy, z))[1])
