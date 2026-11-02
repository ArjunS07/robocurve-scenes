"""Read the live scene into the pure scorer, once per control step.

Root orientations are read through ``state_values.quaternion_xyzw``; the
installed runtime was measured as xyzw, so they are not reordered. Env origins
are subtracted so the protocol poses stay in the table frame.
"""

from __future__ import annotations

from my_package.embodiments.action_map import FINGER_JOINTS
from my_package.embodiments.grasp import any_arm_holding, hold_known, mean_open_fraction
from my_package.scoring.math3d import norm, sub
from my_package.scoring.snapshot import NutSnapshot, SceneSnapshot
from my_package.tasks.runtime_state import EpisodeRuntime
from my_package.tasks.state_values import env_indices, quaternion_xyzw

_RUNTIMES: dict[int, list[EpisodeRuntime]] = {}
_WARNED: set[str] = set()


# The frozen NutConfig must not travel through Isaac Lab term params: configclass
# post-init deep-walks params and tries to assign into it (FrozenInstanceError on
# the server). Terms read it from here instead, as chess-mate and battery do.
_ACTIVE_CFG = None


def set_active_config(cfg) -> None:
    global _ACTIVE_CFG
    _ACTIVE_CFG = cfg


def active_config():
    if _ACTIVE_CFG is None:
        raise RuntimeError("nut threading config was not registered before the task stepped")
    return _ACTIVE_CFG


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(message, flush=True)


def env_local_position(world_xyz, origin_xyz) -> tuple[float, float, float]:
    return (
        float(world_xyz[0]) - float(origin_xyz[0]),
        float(world_xyz[1]) - float(origin_xyz[1]),
        float(world_xyz[2]) - float(origin_xyz[2]),
    )


def runtimes_for(env) -> list[EpisodeRuntime]:
    key = id(env)
    count = int(env.num_envs)
    runs = _RUNTIMES.get(key)
    if runs is None or len(runs) != count:
        runs = [EpisodeRuntime() for _ in range(count)]
        _RUNTIMES[key] = runs
    return runs


def _tensor(value):
    # Isaac Lab 3 data fields are ProxyArrays; implicit tensor use is deprecated.
    return value.torch if hasattr(value, "torch") else value


def _row(value, index):
    row = _tensor(value)[index]
    if hasattr(row, "detach"):
        row = row.detach().cpu().tolist()
    return row


def _root_pose_w(asset, index):
    data = asset.data
    if hasattr(data, "root_pos_w") and hasattr(data, "root_quat_w"):
        return _row(data.root_pos_w, index), _row(data.root_quat_w, index)
    state = _row(data.root_state_w, index)
    return state[0:3], state[3:7]


def _finger_names(layout: str, side: str) -> tuple[str, ...]:
    prefix = "" if layout == "duplicated_single_arm" else f"{side}_"
    return tuple(f"{prefix}{name}" for name in FINGER_JOINTS)


def _arm_reading(robot, env_index, origin, nut_position, body_name: str, finger_names, finger_open, finger_closed):
    body_names = [str(name) for name in robot.body_names]
    joint_names = [str(name) for name in robot.joint_names]
    if body_name not in body_names or any(name not in joint_names for name in finger_names):
        _warn_once(
            f"gripper-{body_name}",
            f"nut threading: gripper body {body_name!r} or fingers {list(finger_names)} not found. "
            f"Live bodies: {body_names}. Live joints: {joint_names}. "
            "Holding is UNKNOWN, so the nut is never treated as released.",
        )
        return None
    body_index = body_names.index(body_name)
    body_world = _row(_tensor(robot.data.body_pos_w)[env_index], body_index)
    body_local = env_local_position(body_world, origin)
    joint_pos = _row(robot.data.joint_pos, env_index)
    fingers = [float(joint_pos[joint_names.index(name)]) for name in finger_names]
    return {
        "distance_m": norm(sub(body_local, nut_position)),
        "open_fraction": mean_open_fraction(fingers, finger_open, finger_closed),
    }


def _robots(env, layout: str):
    if layout == "duplicated_single_arm":
        return (("left_robot", "left"), ("right_robot", "right"))
    if layout == "combined":
        return (("robot", "left"), ("robot", "right"))
    raise ValueError(f"unknown YAM layout: {layout}")


def observe_snapshot(env, cfg, env_index: int) -> SceneSnapshot:
    origin = _row(env.scene.env_origins, env_index)
    try:
        nut_asset = env.scene["nut"]
        fixture_asset = env.scene["fixture"]
    except KeyError as exc:
        raise RuntimeError("nut threading scene is missing the nut or the fixture") from exc
    nut_world, nut_quat = _root_pose_w(nut_asset, env_index)
    fixture_world, fixture_quat = _root_pose_w(fixture_asset, env_index)
    nut_position = env_local_position(nut_world, origin)
    robot = cfg.robot
    arms = []
    for scene_name, side in _robots(env, robot.layout):
        try:
            articulation = env.scene[scene_name]
        except KeyError as exc:
            raise RuntimeError(f"nut threading scene is missing {scene_name}") from exc
        arms.append(
            _arm_reading(
                articulation,
                env_index,
                origin,
                nut_position,
                robot.gripper_body_name,
                _finger_names(robot.layout, side),
                robot.finger_open_m,
                robot.finger_closed_m,
            )
        )
    known = hold_known(arms)
    holding = any_arm_holding(
        arms,
        radius_m=cfg.grasp.radius_m,
        hold_open_fraction_max=cfg.grasp.hold_open_fraction_max,
        hold_open_fraction_min=cfg.grasp.hold_open_fraction_min,
    )
    return SceneSnapshot(
        nut=NutSnapshot(
            position_xyz=nut_position,
            rotation_xyzw=quaternion_xyzw(nut_quat),
            gripper_holding=holding,
            released=known and not holding,
            off_table=False,
            hold_known=known,
        ),
        fixture_position_xyz=env_local_position(fixture_world, origin),
        fixture_rotation_xyzw=quaternion_xyzw(fixture_quat),
    )


def advance_all(env, cfg) -> None:
    counter = int(env.common_step_counter)
    lengths = env.episode_length_buf.detach().cpu().tolist()
    dt = float(env.step_dt)
    if dt <= 0.0:
        raise ValueError(f"env.step_dt must be positive, got {dt}")
    for index, runtime in enumerate(runtimes_for(env)):
        token = (counter, int(lengths[index]))
        if runtime.token == token and runtime.last_report is not None and not runtime.pending_clear:
            continue
        runtime.observe(cfg, observe_snapshot(env, cfg, index), token, dt)


def reset_episode_memory(env, env_ids) -> None:
    # No **kwargs: Isaac Lab's term resolver counts them as a required parameter.
    runs = runtimes_for(env)
    for index in env_indices(env_ids, len(runs)):
        runs[index].on_reset()


def advance_score_reward(env, cfg=None):
    cfg = cfg if cfg is not None else active_config()
    import torch

    from my_package.tasks.capture import maybe_capture

    advance_all(env, cfg)
    maybe_capture(env, cfg)
    return torch.zeros(env.num_envs, device=env.device)


def _flags(env, cfg, selector) -> "torch.Tensor":
    import torch

    advance_all(env, cfg)
    values = []
    for runtime in runtimes_for(env):
        report = runtime.last_report
        values.append(False if report is None else bool(selector(report)))
    return torch.tensor(values, dtype=torch.bool, device=env.device)


def success_fully_seated(env, cfg=None):
    cfg = cfg if cfg is not None else active_config()
    return _flags(env, cfg, lambda report: report.score >= 100)


def failure_off_table(env, cfg=None):
    cfg = cfg if cfg is not None else active_config()
    return _flags(env, cfg, lambda report: report.ever_off_table)


def failure_fixture_moved(env, cfg=None):
    cfg = cfg if cfg is not None else active_config()
    return _flags(env, cfg, lambda report: report.ever_fixture_moved)


def make_nut_score_metric():
    from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_arena.metrics.metric_base import MetricBase
    from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg

    from my_package.scoring.summary import summarize_recorded_scores

    class NutScoreRecorder(RecorderTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.name = cfg.name

        def record_pre_reset(self, env_ids):
            import torch

            runs = runtimes_for(self._env)
            values = []
            for index in env_indices(env_ids, len(runs)):
                report = runs[index].last_report
                values.append(0.0 if report is None else float(report.score))
            return self.name, torch.tensor(values, dtype=torch.float32, device=self._env.device)

    @configclass
    class NutScoreRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = NutScoreRecorder
        name: str = "nut_score"

    class NutScoreMetric(MetricBase):
        name = "nut_score"
        recorder_term_name = "nut_score"

        def get_recorder_term_cfg(self):
            return NutScoreRecorderCfg()

        def get_metric_term_cfg(self):
            return MetricTermCfg(
                compute_metric_func=summarize_recorded_scores,
                params={},
                recorder_term_name=self.recorder_term_name,
            )

    return NutScoreMetric()
