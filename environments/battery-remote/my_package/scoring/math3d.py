"""Small pose helpers for the battery task. No Isaac Sim dependency."""

from __future__ import annotations

import math


def sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: tuple[float, float, float]) -> tuple[float, float, float]:
    length = norm(a)
    if length < 1e-12:
        raise ValueError(f"cannot normalize a near-zero vector: {a}")
    return scale(a, 1.0 / length)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    length = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if length < 1e-12:
        raise ValueError(f"cannot normalize a near-zero quaternion: {q}")
    return (q[0] / length, q[1] / length, q[2] / length, q[3] / length)


def quat_inv(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Inverse of a unit quaternion stored as xyzw."""
    x, y, z, w = quat_normalize(q)
    return (-x, -y, -z, w)


def rotate(q_xyzw: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate a vector by a unit quaternion in xyzw order."""
    x, y, z, w = quat_normalize(q_xyzw)
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def quat_from_axis_angle(
    axis: tuple[float, float, float], angle_rad: float
) -> tuple[float, float, float, float]:
    """Unit quaternion xyzw for a right-handed rotation."""
    half = 0.5 * angle_rad
    s = math.sin(half)
    axis_n = normalize(axis)
    return quat_normalize((axis_n[0] * s, axis_n[1] * s, axis_n[2] * s, math.cos(half)))


def quat_from_axes(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Quaternion xyzw whose columns are the given world-frame axes."""
    # Columns of the rotation matrix that maps camera coordinates into world.
    m00, m10, m20 = x_axis
    m01, m11, m21 = y_axis
    m02, m12, m22 = z_axis
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return quat_normalize((x, y, z, w))


def look_at_ros_xyzw(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """ROS optical pose: camera +Z looks at the target, +X is right, +Y is down.

    Provisional aiming helper. The resulting quaternion is what the saved camera
    config is checked against; it is not a calibrated extrinsics solution.
    """
    forward = normalize(sub(target, eye))
    if abs(dot(forward, world_up)) > 0.99:
        right = (1.0, 0.0, 0.0)
    else:
        right = normalize(cross(world_up, forward))
    down = cross(forward, right)
    return quat_from_axes(right, down, forward)


def yaw_deg(q_xyzw: tuple[float, float, float, float]) -> float:
    x, y, z, w = quat_normalize(q_xyzw)
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_error_deg(current: tuple[float, float, float, float], reference: tuple[float, float, float, float]) -> float:
    delta = yaw_deg(current) - yaw_deg(reference)
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return abs(delta)


def angle_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    denom = norm(a) * norm(b)
    if denom < 1e-12:
        raise ValueError("angle_deg requires non-zero vectors")
    return math.degrees(math.acos(clamp(dot(a, b) / denom, -1.0, 1.0)))


def world_to_local(
    origin: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    return rotate(quat_inv(rotation_xyzw), sub(point, origin))


def local_to_world(
    origin: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    return add(origin, rotate(rotation_xyzw, point))
