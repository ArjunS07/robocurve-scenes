"""Run-local camera resolution and public workspace calibration; no puzzle answers."""
import numpy as np


def camera_names(scene):
    return ('overhead', 'front', 'side') if scene == 'atb1-nut' else ('overhead_cam', 'front_cam', 'side_cam')


def configure_cameras(env_cfg, scene, resolution=None):
    report = []
    for name in camera_names(scene):
        camera = getattr(env_cfg.scene, name)
        original = {'height': int(camera.height), 'width': int(camera.width)}
        if resolution is not None:
            if resolution not in (256, 512, 768, 1024):
                raise ValueError('Camera resolution must be 256, 512, 768 or 1024')
            camera.height = camera.width = resolution
        report.append({'sensor': name, 'original': original,
                       'effective': {'height': int(camera.height), 'width': int(camera.width)}})
    return report


def workspace_description(cfg):
    docs = ''
    table = getattr(cfg, 'table', None)
    if table is not None:
        docs += f'Table surface world Z={table.top_z_m} m. '
    pose = getattr(cfg, 'board_pose', None)
    board = getattr(cfg, 'board', None)
    if pose is None or board is None:
        return docs
    docs += (f'Board center world XYZ={tuple(pose.position_xyz)} m; orientation quaternion XYZW='
             f'{tuple(pose.rotation_xyzw)}. Chess square side={board.square_m:g} m '
             f'({board.square_m*1000:g} mm); use this configured scene calibration, not an estimated scale. ')
    q = np.asarray(pose.rotation_xyzw, dtype=float)
    if np.allclose(q, [0,0,0,1]) or np.allclose(q, [0,0,0,-1]):
        x,y,_ = pose.position_xyz
        s = board.square_m
        docs += ('Rank 1 is toward world -X and rank 8 toward +X. Files a to h run toward world -Y. '
                 f'For file index f=0 for a through 7 for h and rank r=1 through 8, square center world '
                 f'X={x:g}+(r-4.5)*{s:g}, Y={y:g}+(3.5-f)*{s:g}, in metres. '
                 'Convert WORLD targets to each arm base frame using its measured base pose. ')
    docs += ('This calibration does not identify pieces or give the solution. Read the actual pieces '
             'from all three images. Confirm piece identity and square before descending or closing; '
             'if views disagree, keep the gripper above the pieces and re-observe instead of acting on a guess. ')
    return docs
