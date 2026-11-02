"""Bootstrap regression checks without starting Isaac or loading scene assets."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robocurve_inspect.runtime import configure_episode_limit, configure_servos, rendering_args


class RenderingBootstrapTests(unittest.TestCase):
    def test_selects_camera_rendering_instead_of_plain_headless(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apps = root / 'submodules/IsaacLab/apps'
            apps.mkdir(parents=True)
            (apps / 'isaaclab.python.headless.kit').touch()
            rendering = apps / 'isaaclab.python.headless.rendering.kit'
            rendering.touch()
            (apps / 'alternative.headless.rendering.kit').touch()
            self.assertEqual(rendering_args(root, False), ['--experience', str(rendering.resolve())])

    def test_missing_or_ambiguous_experience_fails_before_kit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'headless camera-rendering'):
                rendering_args(root, False)
            apps = root / 'submodules/IsaacLab/apps'
            apps.mkdir(parents=True)
            for name in ('a.headless.rendering.kit', 'b.headless.rendering.kit'):
                (apps / name).touch()
            with self.assertRaisesRegex(RuntimeError, 'headless camera-rendering'):
                rendering_args(root, False)

    def test_stream_keeps_livestream_experience_selection(self):
        self.assertEqual(rendering_args(Path('/no-installation-required'), True),
                         ['--livestream', '1'])


class EpisodeBudgetTests(unittest.TestCase):
    def test_default_keeps_the_configured_timeout(self):
        cfg = NS(episode_length_s=90.)
        report = configure_episode_limit(cfg, None)
        self.assertEqual(cfg.episode_length_s, 90.)
        self.assertEqual(report['effective_episode_seconds'], 90.)
        self.assertFalse(report['task_timing_modified'])

    def test_120_second_run_changes_lab_timeout_and_records_original(self):
        cfg = NS(episode_length_s=90.)
        report = configure_episode_limit(cfg, 120.)
        self.assertEqual(cfg.episode_length_s, 120.)
        self.assertEqual(report, {'configured_episode_seconds': 90.,
                                 'effective_episode_seconds': 120., 'task_timing_modified': True})

    def test_invalid_timeout_is_rejected_without_changing_config(self):
        for invalid in (0., -1., float('inf'), float('nan')):
            cfg = NS(episode_length_s=90.)
            with self.assertRaises(ValueError):
                configure_episode_limit(cfg, invalid)
            self.assertEqual(cfg.episode_length_s, 90.)


class ServoProfileTests(unittest.TestCase):
    def test_tracking_profile_changes_gains_but_keeps_torque_limits(self):
        proximal=NS(stiffness=80.,damping=8.,effort_limit=20.)
        fingers=NS(stiffness=40.,damping=2.,effort_limit=10.)
        cfg=NS(scene=NS(robot=NS(actuators={'left_proximal':proximal,'left_fingers':fingers})))
        report=configure_servos(cfg,[NS(asset='robot')],'tracking')
        self.assertEqual((proximal.stiffness,proximal.damping),(400.,40.))
        self.assertEqual((fingers.stiffness,fingers.damping),(400.,8.))
        self.assertEqual(proximal.effort_limit,20.)
        self.assertEqual(fingers.effort_limit,10.)
        self.assertEqual(report['groups'][0]['original'],{'stiffness':80.,'damping':8.})

    def test_scene_profile_leaves_existing_settings_unchanged(self):
        cfg=NS(scene=None)
        self.assertEqual(configure_servos(cfg,[],'scene'),{'profile':'scene','groups':[]})
