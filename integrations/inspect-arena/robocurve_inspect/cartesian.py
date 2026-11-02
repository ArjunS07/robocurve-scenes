"""YAM-style 14-D EEF targets converted using the loaded simulator Jacobians."""
from dataclasses import replace
import json
import numpy as np
from inspect_robots import ActionSemantics, Box, StateField, StateSpec
from .adapter import ArenaEmbodiment
from .contract import array
from .cartesian_math import (EEF_LABELS, relative_rotation, relative_angles, rotation_error,
                             offset_jacobian, base_jacobian, dls_delta)
from .sim_kinematics import LiveArm


class CartesianEmbodiment(ArenaEmbodiment):
    def __init__(self,*args,kinematics_factory=None,rig_docs='',**kwargs):
        super().__init__(*args,**kwargs)
        factory=kinematics_factory or LiveArm
        self.kinematics=[factory(asset,ids) for _,asset,ids,_ in self._resolved]
        self.references=None
        self._joint_previous=None
        self.command_state_key='eef_state'
        self.low=np.array([.02,-.50,.008,-np.pi,-1.4,-np.pi,0.]*2)
        self.high=np.array([.60,.50,.60,np.pi,1.4,np.pi,1.]*2)
        self.max_step=np.array([.002,.002,.002,.04,.04,.04,.05]*2)
        semantics=ActionSemantics('eef_abs_pose',rotation_repr='none',gripper='continuous',frame='base',
                                  dim_labels=EEF_LABELS,max_step=tuple(self.max_step))
        fields=(*self.info.observation_space.state.fields, StateField('eef_state',(14,),'m,rad,normalized'),
                StateField('eef_world_pos',(6,),'m'),StateField('left_base_pose',(7,),'m,xyzw'),
                StateField('right_base_pose',(7,),'m,xyzw'),StateField('tool_z_base',(6,),'unit_vector'))
        docs=('Two YAM arms, each commanded by x,y,z,yaw,pitch,roll,gripper using move_to. '
              'Positions are metres in THAT arm base frame, +X forward, +Y left, +Z up. '
              'Each target is the grasp point between the DISTAL finger pads, not the wrist. '
              'yaw,pitch,roll are absolute offsets from the gripper orientation captured on reset: '
              'Rz(yaw) @ Ry(-pitch) @ Rx(roll) @ R_reset. They are not incremental rotations. '
              'Positive pitch tips a downward-facing tool toward base +X; rotations do not wrap. '
              'left_gripper/right_gripper: 0 closed, 1 open, independently. '
              'eef_state matches the 14 action labels. joint_command_state contains JOINT angles for logging; '
              'do not use it as Cartesian commands. eef_world_pos is left_xyz then right_xyz. '
              'left_base_pose/right_base_pose are measured world XYZ+quaternion XYZW. '
              'tool_z_base gives each measured tool +Z axis (the direction toward the fingertip ends). '
              'For top-down pinching, aim that axis downward while keeping the fingers clear. '
              'Use measured eef_state and camera feedback after every move. '
              'Approach above a piece with open fingers, descend, close, lift, transfer above the board, '
              'lower and release; do not sweep through neighbouring pieces. '
              'IK has joint/workspace rate limits but is not a collision-free motion planner. '
              'Simulation pauses during model inference. '+rig_docs)
        self.info=replace(self.info,name='robocurve_arena_yam_eef',
                          action_space=Box((14,),low=self.low.copy(),high=self.high.copy(),semantics=semantics),
                          observation_space=replace(self.info.observation_space,state=StateSpec(fields),state_keys=frozenset()),docs=docs)

    def _on_reset(self):
        self.references=[]
        joints=[]
        for kin,(arm,asset,ids,fingers) in zip(self.kinematics,self._resolved):
            kin.calibrate()
            self.references.append(kin.pose()['rotation'].copy())
            joints.append(arm.state(array(asset.data.joint_pos)[0],ids,fingers))
        self._joint_previous=np.concatenate(joints)
        initial=self.observe().state['eef_state']
        if np.any(initial<self.low-1e-6) or np.any(initial>self.high+1e-6):
            raise RuntimeError(f'Measured home grasp points are outside EEF bounds: {initial.tolist()}')
        # Verify the actual simulator exposes the expected Jacobian layout before any model call.
        for kin in self.kinematics: kin.jacobian()
        path=getattr(self,'control_report_path',None)
        if path is not None:
            path.write_text(json.dumps(self.control_report(),indent=2)+'\n')

    def observe(self):
        observation=super().observe()
        if self.references is None: raise RuntimeError('Reset before reading Cartesian observations')
        state=dict(observation.state)
        eef=[];world=[];axes=[]
        for index,(kin,reference) in enumerate(zip(self.kinematics,self.references)):
            pose=kin.pose()
            eef.extend((*pose['position'],*relative_angles(pose['rotation'] @ reference.T),state['gripper'][index]))
            world.extend(pose['world_position'])
            axes.extend(pose['rotation'][:,2])
            state[('left_base_pose','right_base_pose')[index]]=np.r_[pose['base_position'],pose['base_quaternion']].copy()
        state['eef_state']=np.asarray(eef)
        state['eef_world_pos']=np.asarray(world)
        state['tool_z_base']=np.asarray(axes)
        if not all(np.isfinite(v).all() for v in state.values()):
            raise RuntimeError('Non-finite Cartesian observation')
        return replace(observation,state=state)

    def _joint_target(self,target):
        output=[]
        for index,(kin,reference,resolved) in enumerate(zip(self.kinematics,self.references,self._resolved)):
            arm,asset,ids,fingers=resolved
            pose=kin.pose()
            goal=target[index*7:(index+1)*7]
            rotation=relative_rotation(*goal[3:6]) @ reference
            jac=base_jacobian(offset_jacobian(kin.jacobian(),pose['offset_world']),pose['base_rotation'])
            dq=dls_delta(jac,goal[:3]-pose['position'],rotation_error(rotation,pose['rotation']))
            limits=array(asset.data.soft_joint_pos_limits)
            if limits.ndim==3: limits=limits[0]
            previous=self._joint_previous[index*7:index*7+6]
            # Keep a persistent servo target. Re-seeding every update from the
            # sagging measurement caps the PD tracking error at 0.025 rad and
            # can remove the torque needed to resist gravity. The live pose
            # still supplies closed-loop error/Jacobian; integrate its bounded
            # correction into the held target, as the existing reach probe does.
            joint=np.clip(previous+dq,limits[ids,0],limits[ids,1])
            output.extend((*joint,goal[6]))
        self._joint_previous=np.asarray(output)
        return self._joint_previous.copy()

    def control_report(self):
        drives=[]
        for _,asset,_,_ in self._resolved:
            values={}
            for key in ('joint_stiffness','joint_damping','joint_effort_limits','joint_effort_limit','joint_velocity_limits'):
                try: value=getattr(asset.data,key,None)
                except (AttributeError,NotImplementedError): continue
                if value is not None: values[key]=array(value).tolist()
            drives.append(values)
        return {'interface':'eef_pos','labels':list(EEF_LABELS),'units':'m,rad,normalized',
                'servo_settings':getattr(self.base,'_robocurve_inspect_servo_report',None),
                'measured_drive_properties':drives,
                'arms':[{'calibration':kin.calibration,'measured_position_base_m':kin.pose()['position'].tolist(),
                         'measured_position_world_m':kin.pose()['world_position'].tolist(),
                         'reference_rotation_base':ref.tolist()}
                        for kin,ref in zip(self.kinematics,self.references)]}
