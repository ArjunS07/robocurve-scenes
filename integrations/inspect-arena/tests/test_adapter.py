"""Exercise the real Inspect SDK with fake simulator data; no GPU or paid requests."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspect_robots import Action, Scene
from robocurve_inspect.adapter import ArenaEmbodiment
from robocurve_inspect.artifacts import save_capture
from robocurve_inspect.contract import Arm, LABELS, rgb_frame

HOME = (0.4, 1.2, 1., -0.5, 0., 0.)
CAMERAS = {label: (label + '_cam', 4, 6) for label in ('overhead', 'front', 'side')}


class FakeEnv:
    def __init__(self, arms):
        self.unwrapped = self
        self.num_envs, self.step_dt, self.device = 1, 1 / 15, 'cpu'
        self.common_step_counter = 0
        self.scene = {}
        self.arms = arms
        self.native_calls = []
        self.report = {'score': 0, 'hard_failure': False}
        self.terminal = None
        self.call_hook = True
        for arm in arms:
            self.scene[arm.asset] = NS(joint_names=[f'joint{i}' for i in range(1, 9)], data=NS(
                joint_pos=np.array([[*HOME, arm.finger_open, arm.finger_open]]),
                soft_joint_pos_limits=np.array([[[-3., 3.]] * 8])))
        for index, (sensor, h, w) in enumerate(CAMERAS.values()):
            self.scene[sensor] = NS(data=NS(output={'rgb': np.full((1, h, w, 4), 10 + index, dtype=np.uint8)}))
        self.closed = 0

    def reset(self, seed=None):
        self.seed = seed
        return {}, {}

    def step(self, native):
        native = np.asarray(native).reshape(14)
        self.native_calls.append(native.copy())
        for index, arm in enumerate(self.arms):
            row = self.scene[arm.asset].data.joint_pos[0]
            block = native[index * 7:(index + 1) * 7]
            row[:6] = np.asarray(arm.home) + arm.arm_scale * block[:6]
            opening = arm.opening_offset + arm.opening_scale * block[6]
            row[6:] = arm.finger_closed + (arm.finger_open - arm.finger_closed) * opening
        self.common_step_counter += 1
        if self.terminal:
            self.report = dict(self.terminal['report'])
            for name, _, _ in CAMERAS.values():
                self.scene[name].data.output['rgb'][:] = 100
            if self.call_hook:
                self._robocurve_inspect_capture()
            # Reproduce the dangerous auto-reset: frames and score are now different.
            self.report = {'score': 0, 'hard_failure': False}
            for name, _, _ in CAMERAS.values():
                self.scene[name].data.output['rgb'][:] = 200
            return {}, np.array([0.]), np.array([self.terminal['terminated']]), np.array([self.terminal['truncated']]), {}
        return {}, np.array([0.]), np.array([False]), np.array([False]), {}

    def close(self):
        self.closed += 1


def setup(nut=False, owns_env=False):
    arms = tuple(Arm(name, HOME, -0.04695, 0., opening_scale=-1. if nut else 1.)
                 for name in ('left', 'right'))
    env = FakeEnv(arms)
    adapter = ArenaEmbodiment(env, arms, CAMERAS, lambda env: env.report,
                             tensor_factory=lambda value: value, owns_env=owns_env)
    obs = adapter.reset(Scene('test', 'Observe the task'), seed=42)
    return env, adapter, obs


class MappingTests(unittest.TestCase):
    def test_zero_native_holds_home_and_both_open_grippers(self):
        for nut in (False, True):
            env, adapter, obs = setup(nut)
            adapter.step(Action(obs.state['joint_command_state']))
            np.testing.assert_allclose(env.native_calls[-1], np.zeros(14))
            self.assertEqual(tuple(adapter.info.action_space.semantics.dim_labels), LABELS)
            self.assertAlmostEqual(adapter.info.control_hz, 15.)
            self.assertEqual(obs.state['joint_pos'].shape, (12,))
            self.assertEqual(obs.state['gripper'].shape, (2,))

    def test_grippers_are_independent_and_native_signs_differ(self):
        for nut, expected in ((False, -0.05), (True, 0.05)):
            env, adapter, obs = setup(nut)
            target = obs.state['joint_command_state'].copy()
            target[6] = 0.95
            adapter.step(Action(target))
            self.assertAlmostEqual(env.native_calls[-1][6], expected)
            self.assertEqual(env.native_calls[-1][13], 0.)
            np.testing.assert_allclose(env.scene['right'].data.joint_pos[0, 6:], [-0.04695] * 2)

    def test_absolute_target_is_converted_from_home_not_accumulated(self):
        env, adapter, obs = setup()
        target = obs.state['joint_command_state'].copy()
        target[0] += 0.02
        adapter.step(Action(target))
        adapter.step(Action(target))
        self.assertAlmostEqual(env.native_calls[-1][0], 0.02)
        self.assertAlmostEqual(env.scene['left'].data.joint_pos[0, 0], HOME[0] + 0.02)

    def test_nonunit_scales_are_inverted(self):
        arm = Arm('test', HOME, -0.04, 0., arm_scale=2., opening_offset=0.75, opening_scale=0.5)
        native = arm.native([*(np.array(HOME) + 0.2), 0.25])
        np.testing.assert_allclose(native[:6], 0.1)
        self.assertAlmostEqual(native[6], -1.)

    def test_invalid_targets_never_reach_simulator(self):
        for change in ('shape', 'nan', 'bounds', 'jump'):
            env, adapter, obs = setup()
            target = obs.state['joint_command_state'].copy()
            if change == 'shape': target = target[:-1]
            if change == 'nan': target[0] = np.nan
            if change == 'bounds': target[6] = 1.1
            if change == 'jump': target[0] += 0.5
            with self.assertRaises(ValueError): adapter.step(Action(target))
            self.assertEqual(len(env.native_calls), 0)


class ObservationTests(unittest.TestCase):
    def test_three_labeled_camera_arrays_are_copied_from_live_sensors(self):
        env, adapter, obs = setup()
        self.assertEqual(tuple(obs.images), ('overhead', 'front', 'side'))
        for index, (label, frame) in enumerate(obs.images.items()):
            self.assertEqual(frame.shape, (4, 6, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertTrue(np.all(frame == 10 + index))
        env.scene['overhead_cam'].data.output['rgb'][:] = 255
        self.assertTrue(np.all(obs.images['overhead'] == 10))
        self.assertNotIn('task_score', obs.extra)
        self.assertNotIn('task_report', obs.extra)

    def test_wrong_camera_format_fails(self):
        with self.assertRaises(ValueError): rgb_frame(np.zeros((4, 6, 3)), 4, 6)
        with self.assertRaises(ValueError): rgb_frame(np.zeros((2, 4, 6, 3), dtype=np.uint8), 4, 6)

    def test_capture_artifacts_have_all_labels_and_state(self):
        env, adapter, obs = setup()
        with tempfile.TemporaryDirectory() as directory:
            save_capture(directory, obs, adapter.info, env.report)
            root = Path(directory)
            for label in CAMERAS:
                self.assertTrue((root / (label + '.png')).read_bytes().startswith(b'\x89PNG'))
            data = json.loads((root / 'capture.json').read_text())
            self.assertEqual(len(data['action_labels']), 14)
            self.assertEqual(data['control_hz'], 15.)


class TerminalTests(unittest.TestCase):
    def test_terminal_score_and_images_are_captured_before_auto_reset(self):
        env, adapter, obs = setup()
        env.terminal = {'report': {'score': 100, 'hard_failure': False}, 'terminated': True, 'truncated': False}
        result = adapter.step(Action(obs.state['joint_command_state']))
        self.assertEqual(result.info['task_score'], 100)
        self.assertEqual(result.termination_reason, 'success')
        self.assertTrue(np.all(result.observation.images['front'] == 100))
        self.assertTrue(np.all(env.scene['front_cam'].data.output['rgb'] == 200))
        with self.assertRaises(RuntimeError): adapter.step(Action(obs.state['joint_command_state']))

    def test_failure_is_never_inferred_to_be_success(self):
        env, adapter, obs = setup()
        env.terminal = {'report': {'score': 100, 'hard_failure': True}, 'terminated': True, 'truncated': False}
        result = adapter.step(Action(obs.state['joint_command_state']))
        self.assertFalse(result.info['success'])
        self.assertEqual(result.termination_reason, 'task_failure')

    def test_timeout_keeps_partial_credit(self):
        env, adapter, obs = setup()
        env.terminal = {'report': {'score': 50, 'hard_failure': False}, 'terminated': False, 'truncated': True}
        result = adapter.step(Action(obs.state['joint_command_state']))
        self.assertEqual(result.info['task_score'], 50)
        self.assertFalse(result.info['success'])
        self.assertTrue(result.truncated)

    def test_missing_capture_fails_instead_of_grading_reset_scene(self):
        env, adapter, obs = setup()
        env.call_hook = False
        env.terminal = {'report': {'score': 50, 'hard_failure': False}, 'terminated': False, 'truncated': True}
        with self.assertRaisesRegex(RuntimeError, 'pre-reset capture missing'):
            adapter.step(Action(obs.state['joint_command_state']))

    def test_close_respects_env_ownership(self):
        env, adapter, _ = setup()
        adapter.close()
        self.assertEqual(env.closed, 0)
        env, adapter, _ = setup(owns_env=True)
        adapter.close()
        adapter.close()
        self.assertEqual(env.closed, 1)


class AgentContractTests(unittest.TestCase):
    def test_real_agent_binds_and_emits_only_left_gripper_motion(self):
        import httpx
        from run_inspect import create_agent_policy
        env, adapter, obs = setup()
        requests = []
        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            return httpx.Response(200, json={'id': 'offline-test', 'choices': [{'finish_reason': 'tool_calls', 'message': {
                'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'test_call', 'type': 'function', 'function': {
                    'name': 'move_joints', 'arguments': json.dumps({'targets': {'left_gripper_open': 0.9},
                    'note': 'Test the left gripper while the right holds position.'})}}]}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})
        # Use the launcher's actual options, not a separate approximation of them.
        policy = create_agent_policy('google/gemini-4-argon', 1,
            env={'OPENROUTER_API_KEY': 'offline-test'}, transport=httpx.MockTransport(respond))
        policy.bind(adapter.info)
        policy.reset(Scene('test', 'Test only the left gripper'))
        chunk = policy.act(obs)
        self.assertEqual(len(requests), 1)
        content = requests[0]['messages'][-1]['content']
        self.assertEqual(sum(part.get('type') == 'image_url' for part in content), 3)
        for action in chunk.actions:
            self.assertEqual(action.data.shape, (14,))
            self.assertEqual(action.data[13], 1.)
            adapter.step(action)
        self.assertAlmostEqual(chunk.actions[-1].data[6], 0.9)
        self.assertEqual(policy.config.base_url, 'https://openrouter.ai/api/v1')
        self.assertEqual(requests[0]['model'], 'google/gemini-4-argon')
        self.assertEqual(requests[0]['reasoning']['effort'], 'high')
        self.assertIsNone(policy.config.image_horizon)
        self.assertEqual(policy.config.wire, 'chat')
        self.assertIsNone(policy.config.max_output_tokens)
        self.assertEqual(policy.config.max_llm_calls, 1)
        self.assertEqual(policy.config.max_retries, 3)


    def test_real_inspect_eval_preserves_partial_task_score_and_failure_status(self):
        from inspect_robots import ActionChunk, PolicyConfig, PolicyInfo, Task, eval, success_at_end
        from robocurve_inspect.adapter import ArenaTaskScore
        env, adapter, _ = setup()
        env.terminal = {'report': {'score': 50, 'hard_failure': False}, 'terminated': False, 'truncated': True}
        class HoldPolicy:
            config = PolicyConfig()
            info = PolicyInfo('hold', adapter.info.action_space, adapter.info.observation_space)
            def reset(self, scene): pass
            def act(self, observation):
                return ActionChunk([Action(observation.state['joint_command_state'].copy())])
            def close(self): pass
        task = Task('fake_sim', [Scene('test', 'Hold position')], [ArenaTaskScore(), success_at_end()], max_steps=2)
        with tempfile.TemporaryDirectory() as directory:
            logs = eval(task, HoldPolicy(), adapter, log_dir=directory, store_frames=True, fail_on_error=True)
            self.assertEqual(logs[0].results.errored_trials, 0)
            self.assertEqual(logs[0].samples[0].reduced['arena_task_score'], 50.)
            self.assertEqual(logs[0].samples[0].reduced['success_at_end'], 0.)
            self.assertEqual(logs[0].samples[0].termination_reasons, ('time_limit',))
            from robocurve_inspect.results import save_eval_summary
            summary=save_eval_summary(logs,Path(directory),adapter.info.control_hz)
            self.assertEqual(summary['errors'],[])
            self.assertEqual(summary['runs'][0]['scenes'][0]['termination_reasons'],['time_limit'])
            self.assertTrue(list(Path(directory).rglob('*.json')))


class EvalSummaryTests(unittest.TestCase):
    def test_returned_embodiment_fault_is_reported_with_partial_step_count(self):
        from inspect_robots import ActionChunk, PolicyConfig, PolicyInfo, Task, eval
        from robocurve_inspect.adapter import ArenaTaskScore
        from robocurve_inspect.results import save_eval_summary
        env,adapter,_=setup()
        class FaultPolicy:
            config=PolicyConfig()
            info=PolicyInfo('fault',adapter.info.action_space,adapter.info.observation_space)
            def reset(self,scene): self.calls=0
            def act(self,observation):
                self.calls+=1
                target=observation.state['joint_command_state'].copy()
                if self.calls==2: target[0]+=.5
                return ActionChunk([Action(target)])
            def close(self): pass
        task=Task('fake_sim',[Scene('test','Hold')],[ArenaTaskScore()],max_steps=20)
        with tempfile.TemporaryDirectory() as directory:
            logs=eval(task,FaultPolicy(),adapter,log_dir=directory,store_frames=True,fail_on_error=True)
            # eval returns an error LOG rather than throwing this adapter fault.
            self.assertEqual(logs[0].status,'error')
            summary=save_eval_summary(logs,Path(directory),15.)
            self.assertEqual(summary['runs'][0]['total_steps'],1)
            self.assertAlmostEqual(summary['runs'][0]['simulation_seconds'],1/15)
            self.assertIn('per-step motion limit',';'.join(summary['errors']))
            self.assertIn('EmbodimentFault',';'.join(summary['errors']))
            self.assertEqual(json.loads((Path(directory)/'eval_summary.json').read_text()),summary)



if __name__ == '__main__':
    unittest.main()
