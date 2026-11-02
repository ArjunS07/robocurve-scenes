"""Read simulator state into the pure snapshot and step the scorer once per control step."""

from __future__ import annotations

import torch

from isaaclab_arena.utils.pose import Pose

from my_package.assets.geometry import piece_poses
from my_package.scoring.action_map import arm_bindings
from my_package.scoring.grasp import gripper_is_holding
from my_package.scoring.puzzles import Puzzle
from my_package.scoring.runtime_state import (
    active_puzzle_index,
    already_updated,
    get_active_config,
    get_report,
    get_snapshot,
    note_reset,
    store,
)
from my_package.scoring.scorer import carry_episode_memory, step
from my_package.scoring.snapshot import PieceSnapshot, SceneSnapshot

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
        raise RuntimeError("chess mate could not read the control dt")
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
    # Isaac Lab's root_quat_w is wxyz. The scorer stores xyzw, matching Arena's Pose.
    values = quat.detach().cpu().tolist()
    return (float(values[1]), float(values[2]), float(values[3]), float(values[0]))


def _find_one(asset, name: str, kind: str):
    finder = asset.find_joints if kind == "joint" else asset.find_bodies
    indices, names = finder(f"^{name}$", preserve_order=True)
    if len(names) != 1:
        return None
    return int(indices[0])


def _gripper_observation(env, env_index: int, cfg):
    positions = []
    fingers = []
    for binding in arm_bindings(cfg):
        try:
            asset = env.scene[binding.asset_name]
        except Exception:
            positions.append(None)
            fingers.append(None)
            _warn(binding.asset_name, f"chess mate could not find articulation {binding.asset_name}")
            continue
        body_index = _find_one(asset, binding.gripper_body, "body")
        finger_ids = [_find_one(asset, joint, "joint") for joint in binding.finger_joints]
        if body_index is None or any(index is None for index in finger_ids):
            positions.append(None)
            fingers.append(None)
            _warn(
                f"gripper-{binding.side}",
                "chess mate could not find gripper body "
                f"{binding.gripper_body!r} or finger joints {binding.finger_joints} on {binding.asset_name}. "
                "Holding is treated as false until the YAML names match the USDA.",
            )
            continue
        origin = env.scene.env_origins[env_index]
        body_pos = asset.data.body_pos_w[env_index, body_index] - origin
        positions.append(_as_tuple3(body_pos))
        joint_pos = asset.data.joint_pos[env_index]
        fingers.append((float(joint_pos[finger_ids[0]]), float(joint_pos[finger_ids[1]])))
    return tuple(positions), tuple(fingers)


def _active_puzzle(env_index: int) -> Puzzle:
    from my_package.scoring.config import puzzles_for

    cfg = get_active_config()
    return puzzles_for(cfg)[active_puzzle_index(env_index)]


def _read_env(env, env_index: int, puzzle: Puzzle) -> SceneSnapshot:
    cfg = get_active_config()
    origin = env.scene.env_origins[env_index]
    board = env.scene["board"]
    board_pos = _as_tuple3(board.data.root_pos_w[env_index] - origin)
    board_quat = _xyzw_from_wxyz(board.data.root_quat_w[env_index])
    gripper_positions, finger_positions = _gripper_observation(env, env_index, cfg)
    pieces = []
    for name in puzzle.homes:
        asset = env.scene[name]
        position = _as_tuple3(asset.data.root_pos_w[env_index] - origin)
        rotation = _xyzw_from_wxyz(asset.data.root_quat_w[env_index])
        holding = gripper_is_holding(cfg, position, gripper_positions, finger_positions)
        pieces.append(
            PieceSnapshot(
                name=name,
                position_xyz=position,
                rotation_xyzw=rotation,
                gripper_holding=holding,
                released=not holding,
                off_table=False,
            )
        )
    return SceneSnapshot(pieces=tuple(pieces), board_position_xyz=board_pos, board_rotation_xyzw=board_quat)


def update_from_env(env) -> None:
    step_id = _step_id(env)
    if all(already_updated(index, step_id) for index in range(env.num_envs)):
        return
    cfg = get_active_config()
    dt = control_dt(env)
    for index in range(env.num_envs):
        if already_updated(index, step_id):
            continue
        puzzle = _active_puzzle(index)
        observed = carry_episode_memory(get_snapshot(index), _read_env(env, index, puzzle))
        updated, report = step(cfg, observed, dt, puzzle)
        store(index, updated, report, step_id)


def _write_root(env, name: str, position, rotation, env_index: int) -> None:
    asset = env.scene[name]
    origin = env.scene.env_origins[env_index]
    pose = torch.tensor(
        [[
            position[0] + float(origin[0]),
            position[1] + float(origin[1]),
            position[2] + float(origin[2]),
            rotation[0],
            rotation[1],
            rotation[2],
            rotation[3],
        ]],
        device=env.device,
        dtype=torch.float32,
    )
    ids = torch.tensor([env_index], device=env.device)
    asset.write_root_pose_to_sim(pose, env_ids=ids)
    asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=ids)


def reset_chess_state(env, env_ids) -> None:
    """Pose the next puzzle, then drop score memory. The first reset stays on puzzle 0."""
    cfg = get_active_config()
    from my_package.scoring.config import puzzles_for

    puzzles = puzzles_for(cfg)
    if env_ids is None:
        indices = range(env.num_envs)
    else:
        indices = [int(value) for value in env_ids.detach().cpu().tolist()]
    for index in indices:
        puzzle_index = note_reset(index, len(puzzles))
        puzzle = puzzles[puzzle_index]
        for pose in piece_poses(cfg, puzzle):
            _write_root(env, pose.name, pose.position_xyz, pose.rotation_xyzw, index)
        _write_root(
            env,
            "board",
            cfg.board_pose.position_xyz,
            cfg.board_pose.rotation_xyzw,
            index,
        )
        # Pose is only used as a type check that the written rotation is xyzw.
        Pose(position_xyz=cfg.board_pose.position_xyz, rotation_xyzw=cfg.board_pose.rotation_xyzw)


def scores_tensor(env, env_ids):
    update_from_env(env)
    indices = env_ids.detach().cpu().tolist()
    values = []
    for index in indices:
        report = get_report(int(index))
        values.append(0.0 if report is None else float(report.score))
    return torch.tensor(values, device=env.device, dtype=torch.float32)


def mate_succeeded(env):
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


def piece_off_table(env):
    return _latch(env, "ever_off_table")


def board_moved(env):
    return _latch(env, "ever_board_moved")


def multiple_transferred(env):
    update_from_env(env)
    flags = []
    for index in range(env.num_envs):
        report = get_report(index)
        flags.append(False if report is None else len(report.transferred) >= 2)
    return torch.tensor(flags, device=env.device, dtype=torch.bool)


def mate_state_tracker(env):
    update_from_env(env)
    return torch.zeros(env.num_envs, device=env.device)
