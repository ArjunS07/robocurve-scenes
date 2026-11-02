"""Use real Inspect frame files and FFmpeg; no simulator or model request."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from inspect_robots.frames import FrameStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robocurve_inspect.video import export_recording, require_ffmpeg


class RecordingPreflightTests(unittest.TestCase):
    def test_unusable_encoder_fails_before_a_rollout(self):
        with self.assertRaisesRegex(RuntimeError, 'Cannot start the MP4 encoder'):
            require_ffmpeg('/nonexistent/robocurve-test-ffmpeg')


@unittest.skipUnless(shutil.which('ffmpeg'), 'Real FFmpeg is needed for media integration tests')
class VideoTests(unittest.TestCase):
    def test_real_mp4s_decode_in_camera_order_from_partial_saved_run(self):
        ffmpeg = require_ffmpeg()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = FrameStore(str(root / 'inspect/frames/test-run'))
            colors = {'overhead': (255, 0, 0), 'front': (0, 255, 0), 'side': (0, 0, 255)}
            for step in range(3):
                for label, color in colors.items():
                    store.put('trial-0', step, label, np.full((8, 8, 3), color, dtype=np.uint8))
            # Simulate interruption during the next frame set, with no final eval JSON.
            store.put('trial-0', 3, 'overhead', np.zeros((8, 8, 3), dtype=np.uint8))
            report = export_recording(root, ffmpeg, 15.)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['camera_order'], ['overhead', 'front', 'side'])
            self.assertEqual(report['unpaired_frames_excluded'], {'overhead': 1, 'front': 0, 'side': 0})
            self.assertEqual(report['steps'], [0, 1, 2])
            for label in colors:
                self.assertGreater((root / 'videos' / f'{label}.mp4').stat().st_size, 0)
            decoded = subprocess.run([ffmpeg, '-v', 'error', '-i', str(root / 'rollout.mp4'),
                                      '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                                     capture_output=True, check=True, timeout=30).stdout
            video = np.frombuffer(decoded, dtype=np.uint8).reshape(3, 8, 24, 3)
            for index in range(3):
                mean = video[:, :, index * 8:(index + 1) * 8].mean(axis=(0, 1, 2))
                self.assertGreater(mean[index], 180.)
                self.assertLess(max(np.delete(mean, index)), 60.)
            self.assertEqual(len(list((root / 'inspect/frames').rglob('*.npy'))), 10)

    def test_missing_view_fails_loudly_and_preserves_raw_frames(self):
        ffmpeg = require_ffmpeg()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = FrameStore(str(root / 'inspect/frames/test-run'))
            store.put('trial-0', 0, 'overhead', np.zeros((8, 8, 3), dtype=np.uint8))
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                export_recording(root, ffmpeg, 15.)
            self.assertFalse((root / 'rollout.mp4').exists())
            self.assertEqual(json.loads((root / 'videos/recording.json').read_text())['status'], 'failed')
            self.assertEqual(len(list((root / 'inspect/frames').rglob('*.npy'))), 1)
