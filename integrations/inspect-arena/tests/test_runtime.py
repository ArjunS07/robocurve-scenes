"""Bootstrap regression checks without starting Isaac or loading scene assets."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robocurve_inspect.runtime import rendering_args


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
