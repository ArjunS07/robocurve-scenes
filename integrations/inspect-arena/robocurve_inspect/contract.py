"""Explicit two-arm action conversion; imports neither Isaac nor Inspect."""
from dataclasses import dataclass
import numpy as np

LABELS = tuple(f'{side}_joint{i}' if i < 7 else f'{side}_gripper_open'
               for side in ('left', 'right') for i in range(1, 8))


def array(value):
    if hasattr(value, 'torch'):
        value = value.torch
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class Arm:
    asset: str
    home: tuple[float, ...]
    finger_open: float
    finger_closed: float
    arm_scale: float = 1.0
    opening_offset: float = 1.0
    opening_scale: float = 1.0

    def __post_init__(self):
        values = (*self.home, self.finger_open, self.finger_closed, self.arm_scale,
                  self.opening_offset, self.opening_scale)
        if len(self.home) != 6 or not np.isfinite(values).all():
            raise ValueError('Arm configuration needs six finite home angles and finite scales')
        if self.arm_scale == 0 or self.opening_scale == 0 or self.finger_open == self.finger_closed:
            raise ValueError('Action scales and finger span must be nonzero')

    def native(self, canonical):
        value = np.asarray(canonical, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError('Each arm needs seven finite action values')
        return np.r_[(value[:6] - self.home) / self.arm_scale,
                     (value[6] - self.opening_offset) / self.opening_scale]

    def state(self, joint_positions, arm_ids, finger_ids):
        positions = np.asarray(joint_positions, dtype=np.float64)
        fingers = positions[list(finger_ids)]
        opening = (fingers.mean() - self.finger_closed) / (self.finger_open - self.finger_closed)
        return np.r_[positions[list(arm_ids)], np.clip(opening, 0, 1)]


def scene_arms(scene, cfg):
    robot = cfg.robot
    if scene == 'atb1-nut':
        if robot.layout != 'duplicated_single_arm':
            raise ValueError('Only the confirmed duplicated-single-arm layout is supported')
        return tuple(Arm(name, tuple(robot.home_arm_rad), robot.finger_open_m,
                         robot.finger_closed_m, opening_scale=-1.0)
                     for name in ('left_robot', 'right_robot'))
    if robot.arm_layout != 'duplicated_single_arm':
        raise ValueError('Only the confirmed duplicated-single-arm layout is supported')
    return tuple(Arm(name, tuple(robot.home_arm_rad), robot.finger_open_m,
                     robot.finger_closed_m, robot.arm_action_scale,
                     robot.home_open_fraction, robot.gripper_action_scale)
                 for name in ('robot', 'robot_right'))


def rgb_frame(value, height, width):
    frame = array(value)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.shape not in ((height, width, 3), (height, width, 4)) or frame.dtype != np.uint8:
        raise ValueError(f'Camera expected uint8 RGB/RGBA {height}x{width}, got {frame.shape} {frame.dtype}')
    return frame[:, :, :3].copy()


def validate_target(target, low, high, previous=None, max_step=None):
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (14,) or not np.isfinite(target).all():
        raise ValueError('Expected exactly 14 finite absolute joint/opening targets')
    if np.any(target < low - 1e-6) or np.any(target > high + 1e-6):
        raise ValueError('Target exceeds measured joint limits or gripper opening [0, 1]')
    if previous is not None and np.any(np.abs(target - previous) > max_step + 1e-6):
        raise ValueError('Target exceeds the declared per-step motion limit; interpolate the move')
    return np.minimum(np.maximum(target, low), high)
