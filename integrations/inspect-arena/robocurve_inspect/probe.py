"""Model-free Cartesian tracking checks in the actual Arena environment."""
import json
import numpy as np
from inspect_robots import Action
from inspect_robots.frames import FrameStore


def run_reach_check(adapter,scene,output,max_steps=480):
    if adapter.command_state_key!='eef_state':
        raise ValueError('reach-check requires --control-interface eef_pos')
    observation=adapter.reset(scene,seed=42)
    store=FrameStore(str(output/'inspect'/'frames'/'reach-check'))
    def save(step,obs):
        for label,image in obs.images.items(): store.put('reach-check',step,label,image)
    save(0,observation)
    home=observation.state['eef_state'].copy()
    report={'kind':'model-free Cartesian tracking check','model_calls':0,'phases':[]}
    counter=0
    goal=home.copy()
    for arm in range(2):
        side=('left','right')[arm]
        for phase in ('raise_15mm','close_10_percent','open','return'):
            if phase=='raise_15mm': goal[arm*7+2]=home[arm*7+2]+.015
            elif phase=='close_10_percent': goal[arm*7+6]=max(0.,home[arm*7+6]-.10)
            elif phase=='open': goal[arm*7+6]=home[arm*7+6]
            else: goal=home.copy()
            row={'arm':side,'phase':phase,'target':goal.tolist()}
            for attempt in range(60):
                if counter>=max_steps:
                    row['step_budget_exhausted']=True
                    break
                previous=adapter._last_target
                command=previous+np.clip(goal-previous,-adapter.max_step,adapter.max_step)
                result=adapter.step(Action(command))
                observation=result.observation
                counter+=1
                save(counter,observation)
                state=observation.state['eef_state']
                position=np.array([np.linalg.norm(state[i*7:i*7+3]-goal[i*7:i*7+3]) for i in range(2)])
                opening=np.max(np.abs(state[[6,13]]-goal[[6,13]]))
                angles=np.max(np.abs(state[[3,4,5,10,11,12]]-goal[[3,4,5,10,11,12]]))
                trace={'step':counter,'commanded_eef':command.tolist(),'measured_eef':state.tolist(),
                       'commanded_joints':adapter._joint_previous.tolist(),
                       'measured_joints':observation.state['joint_command_state'].tolist(),
                       'position_error_mm':(position*1000).tolist()}
                with (output/'reach_trace.jsonl').open('a') as stream:
                    stream.write(json.dumps(trace)+'\n')
                if max(position)>.05 or angles>.3:
                    row['tracking_abort']='Over 50 mm position error or 0.3 rad orientation error'
                    break
                if result.terminated or result.truncated: break
                if attempt>=14 and max(position)<.003 and opening<.03 and angles<.04: break
            row.update({'position_error_mm':(position*1000).tolist(),'opening_error':float(opening),
                        'orientation_error_rad':float(angles),'steps':attempt+1,
                        'measured_eef':state.tolist(), 'commanded_joints':adapter._joint_previous.tolist(),
                        'measured_joints':observation.state['joint_command_state'].tolist(),
                        'passed':bool(max(position)<.003 and opening<.03 and angles<.04
                                      and not result.terminated and not result.truncated and not row.get('step_budget_exhausted') and not row.get('tracking_abort'))})
            if result.terminated or result.truncated: row['terminal']=result.info
            report['phases'].append(row)
            (output/'reach_check.json').write_text(json.dumps(report,indent=2)+'\n')
            print('Reach check: '+json.dumps(row),flush=True)
            if not row['passed']:
                report['passed']=False
                (output/'reach_check.json').write_text(json.dumps(report,indent=2)+'\n')
                return report
    report['passed']=True
    (output/'reach_check.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
