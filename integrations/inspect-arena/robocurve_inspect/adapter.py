"""Wrap an already-created Arena environment; never create another SimulationApp."""
from dataclasses import asdict, is_dataclass
import numpy as np
from inspect_robots import (ActionSemantics, Box, CameraSpec, EmbodimentInfo,
                            Observation, ObservationSpace, StateField, StateSpec, StepResult)
from .contract import LABELS, array, rgb_frame, validate_target


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class ArenaEmbodiment:
    """14 labeled absolute controls, real sensor images, and pre-reset task scores."""

    def __init__(self, env, arms, cameras, report_reader, *, owns_env=False,
                 arm_step=0.04, gripper_step=0.05, tensor_factory=None):
        self.env = env
        self.base = env.unwrapped
        if int(self.base.num_envs) != 1:
            raise ValueError('Inspect adapter requires num_envs=1')
        self.arms, self.cameras = tuple(arms), dict(cameras)
        if len(self.arms) != 2 or tuple(self.cameras) != ('overhead', 'front', 'side'):
            raise ValueError('Require two arms and overhead/front/side cameras')
        if not np.isfinite((arm_step, gripper_step)).all() or min(arm_step, gripper_step) <= 0:
            raise ValueError('Motion limits must be positive and finite')
        if not np.isfinite(float(self.base.step_dt)) or self.base.step_dt <= 0:
            raise ValueError('Arena must expose a positive step_dt')
        self.report_reader, self.owns_env = report_reader, owns_env
        self.tensor_factory = tensor_factory or self._torch_tensor
        self._resolved = []
        low, high = [], []
        for arm in self.arms:
            asset = self.base.scene[arm.asset]
            names = list(asset.joint_names)
            arm_ids = [names.index(f'joint{i}') for i in range(1, 7)]
            fingers = [names.index('joint7'), names.index('joint8')]
            limits = getattr(asset.data, 'soft_joint_pos_limits', None)
            if limits is None:
                limits = getattr(asset.data, 'joint_pos_limits', None)
            if limits is None:
                raise ValueError(f'No measured joint limits for {arm.asset}')
            limits = array(limits)
            if limits.ndim == 3:
                limits = limits[0]
            bounds = limits[arm_ids]
            if bounds.shape != (6, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
                raise ValueError(f'Invalid joint limits for {arm.asset}')
            low.extend((*bounds[:, 0], 0.0))
            high.extend((*bounds[:, 1], 1.0))
            self._resolved.append((arm, asset, arm_ids, fingers))
        self.low, self.high = np.asarray(low), np.asarray(high)
        self.max_step = np.array(([arm_step] * 6 + [gripper_step]) * 2)
        self.info = EmbodimentInfo(
            name='robocurve_arena_bimanual',
            action_space=Box((14,), low=self.low.copy(), high=self.high.copy(),
                             semantics=ActionSemantics('joint_pos', gripper='continuous', frame='base',
                                                       dim_labels=LABELS, max_step=tuple(self.max_step))),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(label, h, w) for label, (_, h, w) in self.cameras.items()),
                state=StateSpec(fields=(StateField('joint_command_state', (14,), 'rad+normalized'),
                                        StateField('joint_pos', (12,), 'rad'),
                                        StateField('gripper', (2,), 'normalized'),
                                        StateField('finger_joint_pos', (4,), 'm')))),
            control_hz=1.0 / float(self.base.step_dt), is_simulated=True,
            capabilities=frozenset({'resettable', 'seedable', 'renderable', 'privileged_success'}),
            docs=('Two YAM arms. Each has six absolute joint angles in radians plus its own gripper opening. '
                  'left_gripper_open and right_gripper_open: 0 fully closed, 1 fully open. '
                  'Each gripper couples two physical finger joints. Joint angles are in each arm joint frame. '
                  'Both arms share the table workspace. Images are labeled overhead, front, side. '
                  'joint_command_state follows the exact 14 action labels. '
                  'Task time is simulated time; physics is paused during model inference. '
                  'Command limits are not collision avoidance or an IK planner.'))
        self._in_step = False
        self._terminal = None
        self._last_target = None
        self._done = True
        self._closed = False
        self.instruction = ''
        # The launcher's pre-reset recorder invokes this before the scene clears task state.
        self.base._robocurve_inspect_capture = self.capture_terminal

    def _torch_tensor(self, value):
        import torch
        return torch.as_tensor(value[None, :], dtype=torch.float32, device=self.base.device)

    def observe(self):
        images = {}
        for label, (name, height, width) in self.cameras.items():
            sensor = self.base.scene[name]
            try:
                image = sensor.data.output['rgb']
            except (AttributeError, KeyError) as exc:
                raise RuntimeError(f'Camera {name} has no RGB output; enable cameras before Kit starts') from exc
            images[label] = rgb_frame(image, height, width)
        states, fingers = [], []
        for arm, asset, ids, finger_ids in self._resolved:
            row = array(asset.data.joint_pos)[0]
            states.append(arm.state(row, ids, finger_ids))
            fingers.extend(row[finger_ids])
        state = np.concatenate(states)
        if not np.isfinite(state).all():
            raise RuntimeError('Non-finite measured robot state')
        sim_time = float(getattr(self.base, 'common_step_counter', 0)) * float(self.base.step_dt)
        return Observation(images=images, state={
            'joint_command_state': state.copy(),
            'joint_pos': state[[*range(6), *range(7, 13)]].copy(),
            'gripper': state[[6, 13]].copy(), 'finger_joint_pos': np.asarray(fingers).copy()},
            instruction=self.instruction, image_times={name: sim_time for name in images}, state_time=sim_time)

    def _report(self):
        report = self.report_reader(self.base)
        if report is None:
            raise RuntimeError('Task scorer supplied no report; refusing to invent a success signal')
        result = plain(report)
        if not isinstance(result, dict) or 'score' not in result or 'hard_failure' not in result:
            raise RuntimeError('Task report must include score and hard_failure')
        score = result['score']
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score) or not 0 <= score <= 100:
            raise RuntimeError('Task score must be finite and in [0, 100]')
        return result

    def capture_terminal(self):
        if self._in_step:
            self._terminal = (self.observe(), self._report())

    def reset(self, scene, *, seed=None):
        if self._closed:
            raise RuntimeError('Adapter is closed')
        self.instruction = scene.instruction
        self._in_step = False
        self._terminal = None
        self.env.reset(seed=seed)
        observation = self.observe()
        self._last_target = observation.state['joint_command_state'].copy()
        self._done = False
        return observation

    def step(self, action):
        if self._done or self._closed:
            raise RuntimeError('Reset the adapter before stepping, including after termination')
        target = validate_target(action.data, self.low, self.high, self._last_target, self.max_step)
        native = np.concatenate([arm.native(target[i * 7:(i + 1) * 7]) for i, arm in enumerate(self.arms)])
        self._terminal = None
        self._in_step = True
        try:
            _, reward, terminated, truncated, _ = self.env.step(self.tensor_factory(native))
        finally:
            self._in_step = False
        self._last_target = target.copy()
        terminated, truncated = bool(array(terminated).reshape(-1)[0]), bool(array(truncated).reshape(-1)[0])
        self._done = terminated or truncated
        if self._done:
            if self._terminal is None:
                raise RuntimeError('Terminal pre-reset capture missing; refusing to grade a reset scene')
            observation, report = self._terminal
        else:
            observation, report = self.observe(), self._report()
        success = terminated and not report['hard_failure'] and report['score'] == 100
        reason = 'success' if success else ('task_failure' if terminated else ('time_limit' if truncated else None))
        return StepResult(observation, reward=float(array(reward).reshape(-1)[0]), terminated=terminated,
                          truncated=truncated, termination_reason=reason,
                          info={'success': success, 'task_score': report['score'], 'task_report': report})

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._done = True
        if getattr(self.base, '_robocurve_inspect_capture', None) == self.capture_terminal:
            del self.base._robocurve_inspect_capture
        if self.owns_env:
            self.env.close()


class ArenaTaskScore:
    name = 'arena_task_score'

    def __call__(self, record, target=None):
        from inspect_robots import Score
        if not record.steps:
            return Score(0, 'No simulated actions completed')
        info = record.steps[-1].result.info
        if 'task_score' not in info:
            raise RuntimeError('Recorded result is missing the task score')
        return Score(info['task_score'], 'Physics task scorer, not the model completion claim',
                     metadata={'task_report': info.get('task_report', {})})
