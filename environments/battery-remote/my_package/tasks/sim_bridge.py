"""Read simulator state into the pure snapshot and step the scorer once per control step."""

from __future__ import annotations

import torch

from my_package.scoring.action_map import arm_bindings
from my_package.scoring.config import active_slots
from my_package.scoring.grasp import gripper_is_holding
from my_package.scoring.math3d import norm, sub, yaw_error_deg
from my_package.scoring.runtime_state import (
    already_updated,
    clear_env,
    get_active_config,
    get_report,
    get_snapshot,
    store,
)
from my_package.scoring.scorer import carry_episode_memory, step
from my_package.scoring.snapshot import BatterySnapshot, SceneSnapshot

_WARNED: set[str] = set()


def _warn(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(message)


def control_dt(env) -> float:
    step_dt = getattr(env, "step_dt", None)
    if step_dt is not None:
        return float(step_dt)
    cfg = getattr(env, "cfg", None)
    sim = getattr(env, "sim", None)
    if sim is not None and getattr(sim, "cfg", None) is not None:
        dt = float(sim.cfg.dt)
    elif cfg is not None and getattr(cfg, "sim", None) is not None:
        dt = float(cfg.sim.dt)
    else:
        raise RuntimeError("battery insertion could not read the control dt")
    decimation = int(getattr(cfg, "decimation", 1) or 1)
    return dt * decimation


def _step_id(env) -> int:
    if hasattr(env, "common_step_counter"):
        return int(env.common_step_counter)
    buf = getattr(env, "episode_length_buf", None)
    if buf is not None:
        return int(buf.max().item())
    return -1


def _as_tuple3(tensor) -> tuple[float, float, float]:
    values = tensor.detach().cpu().tolist()
    return (float(values[0]), float(values[1]), float(values[2]))


def _xyzw_from_wxyz(quat) -> tuple[float, float, float, float]:
    values = quat.detach().cpu().tolist()
    return (float(values[1]), float(values[2]), float(values[3]), float(values[0]))


def _find_one(asset, name: str, kind: str):
    finder = asset.find_joints if kind == "joint" else asset.find_bodies
    indices, names = finder(f"^{name}$", preserve_order=True)
    if len(names) != 1:
        return None
    return int(indices[0])


def _sensor(env, name: str):
    sensors = getattr(env.scene, "sensors", None)
    if isinstance(sensors, dict) and name in sensors:
        return sensors[name]
    try:
        return env.scene[name]
    except Exception:
        return None


def _peak_force(sensor, env_index: int) -> float:
    if sensor is None or not hasattr(sensor, "data"):
        return 0.0
    forces = getattr(sensor.data, "net_forces_w", None)
    if forces is None:
        return 0.0
    sample = forces[env_index].reshape(-1, 3)
    if sample.numel() == 0:
        return 0.0
    return float(torch.linalg.norm(sample, dim=-1).max())


def _gripper_observation(env, env_index: int, cfg):
    positions = []
    fingers = []
    for binding in arm_bindings(cfg):
        try:
            asset = env.scene[binding.asset_name]
        except Exception:
            positions.append(None)
            fingers.append(None)
            _warn(binding.asset_name, f"battery insertion could not find articulation {binding.asset_name}")
            continue
        body_index = _find_one(asset, binding.gripper_body, "body")
        finger_ids = [_find_one(asset, joint, "joint") for joint in binding.finger_joints]
        if body_index is None or any(index is None for index in finger_ids):
            positions.append(None)
            fingers.append(None)
            _warn(
                f"gripper-{binding.side}",
                "battery insertion could not find gripper body "
                f"{binding.gripper_body!r} or finger joints {binding.finger_joints} on {binding.asset_name}. "
                f"Live bodies: {list(getattr(asset, 'body_names', []))}. "
                "Holding is treated as false until the YAML names match the USDA.",
            )
            continue
        origin = env.scene.env_origins[env_index]
        body_pos = asset.data.body_pos_w[env_index, body_index] - origin
        positions.append(_as_tuple3(body_pos))
        joint_pos = asset.data.joint_pos[env_index]
        fingers.append((float(joint_pos[finger_ids[0]]), float(joint_pos[finger_ids[1]])))
    return tuple(positions), tuple(fingers)


def _plunger_position(env, env_index: int, joint_name: str) -> float:
    try:
        remote = env.scene["remote"]
    except Exception:
        return 0.0
    index = _find_one(remote, joint_name, "joint")
    if index is None:
        _warn(joint_name, f"plunger joint {joint_name} was not found; spring pose stays at the authored 0")
        return 0.0
    return float(remote.data.joint_pos[env_index, index])


def _read_env(env, env_index: int):
    from my_package.assets.geometry import measure_battery
    from my_package.scoring.math3d import rotate

    cfg = get_active_config()
    origin = env.scene.env_origins[env_index]
    remote = env.scene["remote"]
    remote_pos = _as_tuple3(remote.data.root_pos_w[env_index] - origin)
    remote_quat = _xyzw_from_wxyz(remote.data.root_quat_w[env_index])
    displacement = norm(sub(remote_pos, cfg.remote_pose.position_xyz))
    yaw_error = yaw_error_deg(remote_quat, cfg.remote_pose.rotation_xyzw)
    gripper_positions, finger_positions = _gripper_observation(env, env_index, cfg)
    batteries = []
    for slot in active_slots(cfg):
        asset = env.scene[slot.name]
        position = _as_tuple3(asset.data.root_pos_w[env_index] - origin)
        rotation = _xyzw_from_wxyz(asset.data.root_quat_w[env_index])
        joint_name = slot.name.replace("battery_", "slot_") + "_plunger"
        measured = measure_battery(
            cfg,
            slot,
            position,
            rotation,
            remote_pos,
            remote_quat,
            plunger_joint_m=_plunger_position(env, env_index, joint_name),
        )
        holding = gripper_is_holding(cfg, position, gripper_positions, finger_positions)
        batteries.append(
            BatterySnapshot(
                name=slot.name,
                position_xyz=position,
                rotation_xyzw=rotation,
                positive_axis_xyz=rotate(rotation, (0.0, 0.0, 1.0)),
                seating_depth_m=measured.seating_depth_m,
                radial_offset_m=measured.radial_offset_m,
                polarity_aligned=measured.polarity_angle_deg <= cfg.scoring.polarity_angle_deg,
                gripper_holding=holding,
                released=not holding,
                spring_contact=measured.spring_contact,
                off_table=False,
                peak_force_n=_peak_force(_sensor(env, f"contact_{slot.name}"), env_index),
            )
        )
    observed = SceneSnapshot(
        batteries=tuple(batteries),
        remote_position_xyz=remote_pos,
        remote_rotation_xyzw=remote_quat,
        remote_displacement_m=displacement,
        remote_yaw_error_deg=yaw_error,
        damage_force_n=_peak_force(_sensor(env, "contact_remote"), env_index),
    )
    return observed


def update_from_env(env) -> None:
    """Step the scorer once per environment per control step."""
    step_id = _step_id(env)
    if all(already_updated(index, step_id) for index in range(env.num_envs)):
        return
    cfg = get_active_config()
    dt = control_dt(env)
    for index in range(env.num_envs):
        if already_updated(index, step_id):
            continue
        observed = carry_episode_memory(get_snapshot(index), _read_env(env, index))
        updated, report = step(cfg, observed, dt)
        store(index, updated, report, step_id)


def reset_battery_insertion_state(env, env_ids) -> None:
    """Clear score memory. Physics poses are reset by the articulation and object events."""
    if env_ids is None:
        indices = range(env.num_envs)
    else:
        indices = env_ids.detach().cpu().tolist()
    for index in indices:
        clear_env(int(index))


def scores_tensor(env, env_ids):
    update_from_env(env)
    indices = env_ids.detach().cpu().tolist()
    values = []
    for index in indices:
        report = get_report(int(index))
        values.append(0.0 if report is None else float(report.score))
    return torch.tensor(values, device=env.device, dtype=torch.float32)


def batteries_retained(env):
    """Arena success predicate. True only when every active cell scores 100."""
    update_from_env(env)
    flags = []
    for index in range(env.num_envs):
        report = get_report(index)
        flags.append(report is not None and report.score == 100 and not report.hard_failure)
    return torch.tensor(flags, device=env.device, dtype=torch.bool)


def _latch(env, attr: str):
    update_from_env(env)
    flags = []
    for index in range(env.num_envs):
        report = get_report(index)
        flags.append(False if report is None else bool(getattr(report, attr)))
    return torch.tensor(flags, device=env.device, dtype=torch.bool)


def battery_off_table(env):
    return _latch(env, "ever_off_table")


def remote_moved(env):
    return _latch(env, "ever_remote_moved")


def damage_proxy(env):
    return _latch(env, "ever_damage")


def insertion_state_tracker(env):
    """Reward term whose only job is to advance the scorer before terminations."""
    update_from_env(env)
    return torch.zeros(env.num_envs, device=env.device)
