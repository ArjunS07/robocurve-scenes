"""NumPy-only Cartesian contract matching Inspect YAM's EEF interface."""
import numpy as np

EEF_LABELS = tuple(f'{side}_{axis}' for side in ('left', 'right')
                   for axis in ('x', 'y', 'z', 'yaw', 'pitch', 'roll', 'gripper'))


def rotation_xyzw(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError('Expected a finite nonzero xyzw quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def relative_rotation(yaw, pitch, roll):
    # YAM convention: Rz(yaw) @ Ry(-pitch) @ Rx(roll).
    a,b,c = yaw,pitch,roll
    ca,sa,cb,sb,cc,sc = np.cos(a),np.sin(a),np.cos(b),np.sin(b),np.cos(c),np.sin(c)
    rz=np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    ry=np.array([[cb,0,-sb],[0,1,0],[sb,0,cb]])
    rx=np.array([[1,0,0],[0,cc,-sc],[0,sc,cc]])
    return rz @ ry @ rx


def relative_angles(rotation):
    r = np.asarray(rotation)
    return np.array([np.arctan2(r[1,0], r[0,0]), np.arcsin(np.clip(r[2,0],-1,1)),
                     np.arctan2(r[2,1], r[2,2])])


def rotation_error(target, current):
    r = np.asarray(target) @ np.asarray(current).T
    angle = np.arccos(np.clip((np.trace(r)-1)/2, -1, 1))
    skew = np.array([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])
    if angle < 1e-7:
        return skew / 2
    if np.pi-angle < 1e-5:
        values,vectors=np.linalg.eigh((r+np.eye(3))/2)
        axis=vectors[:,np.argmax(values)]
        if np.dot(axis,skew)<0: axis=-axis
        return axis*angle
    return skew * angle/(2*np.sin(angle))


def offset_jacobian(jacobian, offset_world):
    j = np.asarray(jacobian, dtype=float).copy()
    # v_tcp = v_body + omega cross r. Applies the grasp-site offset.
    j[:3] += np.cross(j[3:].T, np.asarray(offset_world)).T
    return j


def base_jacobian(jacobian_world, rotation_base_world):
    r = np.asarray(rotation_base_world).T
    j = np.asarray(jacobian_world)
    return np.vstack((r @ j[:3], r @ j[3:]))


def dls_delta(jacobian, position_error, orientation_error, damping=.035, max_joint_step=.025):
    j=np.asarray(jacobian,dtype=float)
    error=np.r_[position_error, orientation_error]
    if j.shape != (6,6) or not np.isfinite(j).all() or not np.isfinite(error).all():
        raise ValueError('Invalid measured Jacobian or Cartesian error')
    weights=np.array([1.,1.,1.,.15,.15,.15])
    weighted=j*weights[:,None]
    dq=weighted.T @ np.linalg.solve(weighted @ weighted.T + damping*damping*np.eye(6), error*weights)
    # Scale the whole joint step rather than clipping axes independently.
    largest=np.max(np.abs(dq))
    if largest>max_joint_step: dq*=max_joint_step/largest
    return dq


def jacobian_row(jacobians, body_id, joint_ids, fixed, body_count, joint_count):
    j=np.asarray(jacobians)
    expected_rows=body_count-int(fixed)
    expected_cols=joint_count+(0 if fixed else 6)
    if j.ndim!=4 or j.shape[0]!=1 or j.shape[1:]!=(expected_rows,6,expected_cols):
        raise ValueError(f'Unexpected Jacobian shape {j.shape}; expected (1,{expected_rows},6,{expected_cols})')
    row=body_id-int(fixed)
    if row<0: raise ValueError('Root body has no fixed-base Jacobian row')
    return j[0,row][:,np.asarray(joint_ids)+(0 if fixed else 6)].copy()
