"""Check run-local camera changes and geometry-only policy calibration."""
import unittest
from types import SimpleNamespace as NS
import numpy as np
from robocurve_inspect.perception import configure_cameras, workspace_description


class PerceptionTests(unittest.TestCase):
    def test_resolution_changes_sensor_rendering_but_keeps_framing(self):
        cameras={name:NS(height=256,width=256,spawn=NS(focal_length=13.),offset=object())
                 for name in ('overhead_cam','front_cam','side_cam')}
        cfg=NS(scene=NS(**cameras))
        offsets=[c.offset for c in cameras.values()]
        report=configure_cameras(cfg,'chess-mate',512)
        self.assertEqual(len(report),3)
        for index,camera in enumerate(cameras.values()):
            self.assertEqual((camera.height,camera.width),(512,512))
            self.assertEqual(camera.spawn.focal_length,13.)
            self.assertIs(camera.offset,offsets[index])
            self.assertEqual(report[index]['original'],{'height':256,'width':256})

    def test_default_keeps_scene_resolution_and_supports_nut_names(self):
        cameras={name:NS(height=256,width=320) for name in ('overhead','front','side')}
        report=configure_cameras(NS(scene=NS(**cameras)),'atb1-nut')
        self.assertTrue(all(r['effective']==r['original'] for r in report))

    def test_workspace_docs_use_actual_square_size_without_reading_puzzle(self):
        class Config:
            table=NS(top_z_m=.003)
            board=NS(square_m=.030)
            board_pose=NS(position_xyz=(.55,0.,.007),rotation_xyzw=(0,0,0,1))
            @property
            def fen_sequence(self): raise AssertionError('Never reveal puzzle state')
        text=workspace_description(Config())
        self.assertIn('30 mm',text)
        self.assertIn('X=0.55+(r-4.5)*0.03',text)
        self.assertIn('Y=0+(3.5-f)*0.03',text)
        self.assertNotIn('d1',text)
        self.assertNotIn('h5',text)
        # Formula must agree with the configured board frame for all corners.
        points=[(.55+(r-4.5)*.03,(3.5-f)*.03) for f,r in ((0,1),(7,1),(0,8),(7,8))]
        np.testing.assert_allclose(points,[(.445,.105),(.445,-.105),(.655,.105),(.655,-.105)])

    def test_rotated_board_does_not_get_identity_world_axis_formula(self):
        cfg=NS(board=NS(square_m=.03),board_pose=NS(position_xyz=(0,0,0),rotation_xyzw=(0,0,.7071,.7071)))
        text=workspace_description(cfg)
        self.assertIn('30 mm',text)
        self.assertNotIn('square center world X=',text)
        self.assertIn('0.7071',text)
