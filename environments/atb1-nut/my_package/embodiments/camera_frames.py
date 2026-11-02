"""World-camera offsets relative to the left arm base. No Isaac Sim imports."""

from __future__ import annotations


def camera_offset_position(eye_xyz, base_xyz) -> tuple[float, float, float]:
    return tuple(float(eye) - float(base) for eye, base in zip(eye_xyz, base_xyz, strict=True))
