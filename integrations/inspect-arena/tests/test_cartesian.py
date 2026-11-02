"""Independent kinematic math and the real Inspect EEF tool against fake dynamics."""
import json
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import unittest
import numpy as np
from inspect_robots import Action, Scene
from test_adapter import HOME, CAMERAS, setup
from robocurve_inspect.cartesian import CartesianEmbodiment
from robocurve_inspect.cartesian_math import (relative_rotation, relative_angles, rotation_error,
                                             offset_jacobian, base_jacobian, dls_delta, jacobian_row)


class CartesianMathTests(unittest.TestCase):
    def test_yam_angle_convention_and_rotation_error(self):
        for angles in ((.3,.2,-.4),(-.8,-.7,.1),(0.,0.,0.)):
            np.testing.assert_allclose(relative_angles(relative_rotation(*angles)),angles,atol=1e-12)
        np.testing.assert_allclose(rotation_error(relative_rotation(np.pi/2,0,0),np.eye(3)),[0,0,np.pi/2])
        self.assertAlmostEqual(np.linalg.norm(rotation_error(relative_rotation(np.pi,0,0),np.eye(3))),np.pi)
        # Positive YAM pitch rotates a downward tool toward base +X.
        np.testing.assert_allclose(relative_rotation(0,np.pi/2,0) @ [0,0,-1],[1,0,0],atol=1e-12)

    def test_tcp_offset_adds_angular_velocity_cross_offset(self):
        j=np.zeros((6,6));j[5,0]=1.
        shifted=offset_jacobian(j,[.1,0,0])
        np.testing.assert_allclose(shifted[:,0],[0,.1,0,0,0,1.])
        base=relative_rotation(np.pi/2,0,0)
        np.testing.assert_allclose(base_jacobian(shifted,base)[:3,0],[.1,0,0],atol=1e-12)

    def test_fixed_and_anchored_floating_jacobian_indices(self):
        fixed=np.arange(1*8*6*8).reshape(1,8,6,8)
        np.testing.assert_equal(jacobian_row(fixed,6,list(range(6)),True,9,8),fixed[0,5,:,:6])
        floating=np.arange(1*9*6*14).reshape(1,9,6,14)
        np.testing.assert_equal(jacobian_row(floating,6,list(range(6)),False,9,8),floating[0,6,:,6:12])
        with self.assertRaises(ValueError): jacobian_row(floating,6,list(range(6)),True,9,8)

    def test_dls_joint_steps_are_finite_bounded_and_follow_error(self):
        dq=dls_delta(np.eye(6),[.1,0,0],[0,0,0])
        self.assertGreater(dq[0],0)
        self.assertLessEqual(abs(dq).max(),.025)
        np.testing.assert_equal(dls_delta(np.zeros((6,6)),[.1,0,0],[0,0,0]),np.zeros(6))
        with self.assertRaises(ValueError): dls_delta(np.eye(6)*np.nan,[0,0,0],[0,0,0])


class FakeLiveArm:
    def __init__(self,asset,joints):
        self.asset,self.joints=asset,joints
        self.calibration=None
    def calibrate(self): self.calibration={'source':'test kinematics'}
    def pose(self):
        q=self.asset.data.joint_pos[0,:6]-np.array(HOME)
        r=self.asset.test_base_r
        position=np.array([.3,0,.2])+.1*q[:3]
        return {'position':position,'rotation':relative_rotation(*q[3:]),
                'world_position':self.asset.test_base_pos+r @ position,
                'base_position':self.asset.test_base_pos,'base_rotation':r,
                'base_quaternion':self.asset.test_base_quat,'offset_world':np.zeros(3)}
    def jacobian(self):
        q=self.asset.data.joint_pos[0,:6]-np.array(HOME)
        j=np.zeros((6,6));j[:3,:3]=.1*np.eye(3)
        current=relative_rotation(*q[3:])
        for i in range(3):
            angles=q[3:].copy();angles[i]+=1e-6
            j[3:,i+3]=rotation_error(relative_rotation(*angles),current)/1e-6
        r=self.asset.test_base_r
        return np.vstack((r @ j[:3],r @ j[3:]))


def cartesian_setup():
    env,old,_=setup();old.close()
    for i,(arm,asset,ids,fingers) in enumerate(old._resolved):
        asset.test_base_r=relative_rotation(i*np.pi/2,0,0)
        asset.test_base_pos=np.array([.27,.22 if i==0 else -.22,.003])
        asset.test_base_quat=np.array([0,0,np.sin(i*np.pi/4),np.cos(i*np.pi/4)])
    adapter=CartesianEmbodiment(env,old.arms,CAMERAS,lambda env:env.report,
                                tensor_factory=lambda value:value,kinematics_factory=FakeLiveArm)
    obs=adapter.reset(Scene('test','Reach upward'),seed=42)
    return env,adapter,obs


class CartesianContractTests(unittest.TestCase):
    def test_measured_state_and_home_hold_for_independent_base_frames(self):
        env,adapter,obs=cartesian_setup()
        self.assertEqual(adapter.info.action_space.semantics.control_mode,'eef_abs_pose')
        self.assertEqual(obs.state['eef_state'].shape,(14,))
        np.testing.assert_allclose(obs.state['eef_state'][[0,1,2,7,8,9]],[.3,0,.2,.3,0,.2])
        np.testing.assert_allclose(obs.state['eef_world_pos'],[.57,.22,.203,.27,.08,.203],atol=1e-12)
        adapter.step(Action(obs.state['eef_state']))
        np.testing.assert_allclose(env.native_calls[-1],np.zeros(14),atol=1e-12)

    def test_only_requested_arm_rises_and_gripper_remains_independent(self):
        env,adapter,obs=cartesian_setup()
        target=obs.state['eef_state'].copy();target[2]+=.002;target[6]=.95
        result=adapter.step(Action(target))
        self.assertGreater(result.observation.state['eef_state'][2],.2)
        np.testing.assert_allclose(result.observation.state['eef_state'][7:10],[.3,0,.2])
        self.assertAlmostEqual(result.observation.state['gripper'][0],.95)
        self.assertAlmostEqual(result.observation.state['gripper'][1],1.)
        self.assertLessEqual(abs(env.native_calls[-1][:6]).max(),.025)

    def test_terminal_cartesian_frames_and_reason_survive_auto_reset(self):
        env,adapter,obs=cartesian_setup()
        env.terminal={'report':{'score':0,'hard_failure':True,'hard_reasons':['board moved']},
                      'terminated':True,'truncated':False}
        result=adapter.step(Action(obs.state['eef_state']))
        self.assertEqual(result.termination_reason,'task_failure')
        self.assertEqual(adapter.last_result['task_report']['hard_reasons'],['board moved'])
        self.assertTrue(np.all(result.observation.images['overhead']==100))
        self.assertEqual(result.observation.state['eef_state'].shape,(14,))

    def test_real_agent_advertises_move_to_and_uses_eef_reference(self):
        import httpx
        from run_inspect import create_agent_policy
        env,adapter,obs=cartesian_setup();requests=[]
        def respond(request):
            payload=json.loads(request.content);requests.append(payload)
            return httpx.Response(200,json={'choices':[{'finish_reason':'tool_calls','message':{
                'role':'assistant','content':None,'tool_calls':[{'id':'eef-1','type':'function','function':{
                'name':'move_to','arguments':json.dumps({'targets':{'left_z':.21},'note':'Lift left gripper 1 cm.'})}}]}}]})
        policy=create_agent_policy('google/gemini-4-argon',1,transport=httpx.MockTransport(respond),
                                   env={'OPENROUTER_API_KEY':'offline-test'})
        policy.bind(adapter.info);policy.reset(Scene('test','Lift left gripper'))
        chunk=policy.act(obs)
        names=[t['function']['name'] for t in requests[0]['tools']]
        self.assertIn('move_to',names);self.assertNotIn('move_joints',names)
        for action in chunk.actions: adapter.step(action)
        state=adapter.observe().state['eef_state']
        self.assertLess(abs(state[2]-.21),.001)
        np.testing.assert_allclose(state[7:10],[.3,0,.2])
        self.assertEqual(sum(p.get('type')=='image_url' for p in requests[0]['messages'][-1]['content']),3)

    def test_model_free_probe_tracks_both_arms_and_saves_all_camera_frames(self):
        from robocurve_inspect.probe import run_reach_check
        env,adapter,_=cartesian_setup()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output=Path(directory)
            report=run_reach_check(adapter,Scene('test','Tracking check'),output)
            self.assertTrue(report['passed'])
            self.assertEqual(report['model_calls'],0)
            self.assertEqual(len(report['phases']),8)
            self.assertTrue((output/'reach_check.json').exists())
            frames=list((output/'inspect/frames').rglob('*.npy'))
            self.assertGreater(len(frames),3)
            self.assertEqual(len(frames)%3,0)

    def test_cartesian_hold_builds_pd_offset_under_gravity_and_actuator_lag(self):
        env,adapter,obs=cartesian_setup()
        ideal_step=env.step
        def lagged_step(native):
            # q_next = q + alpha*(q_command-q) - gravity. A fixed target
            # needs a 0.075-rad tracking offset, larger than a per-step jog.
            previous=[asset.data.joint_pos.copy() for _,asset,_,_ in adapter._resolved]
            result=ideal_step(native)
            for before,(_,asset,_,_) in zip(previous,adapter._resolved):
                desired=asset.data.joint_pos.copy()
                asset.data.joint_pos[:,:6]=before[:,:6]+.2*(desired[:,:6]-before[:,:6])
                asset.data.joint_pos[:,2]-=.015
            return result
        env.step=lagged_step
        goal=obs.state['eef_state'].copy()
        for _ in range(150): result=adapter.step(Action(goal))
        measured=result.observation.state['eef_state']
        self.assertLess(abs(measured[2]-.2),.001)
        self.assertLess(abs(measured[9]-.2),.001)
        # The controller must retain load-supporting command error after
        # Cartesian convergence rather than replacing it with a zero jog.
        for index,(_,asset,_,_) in enumerate(adapter._resolved):
            command=adapter._joint_previous[index*7+2]
            self.assertGreater(command-asset.data.joint_pos[0,2],.06)
        commands=np.array(env.native_calls)
        self.assertLessEqual(np.abs(np.diff(commands[:,:6],axis=0)).max(),.0250001)
