"""Measured pose/Jacobian access and CAD grasp point for already-loaded YAM arms."""
import numpy as np
from .contract import array
from .cartesian_math import rotation_xyzw, jacobian_row


def _body_data(asset, preferred, fallback):
    value=getattr(asset.data,preferred,None)
    if value is None: value=getattr(asset.data,fallback)
    return array(value)[0]


def cad_grasp_point(asset):
    # Read transforms and points consistently from USD, without mixing stale USD
    # world transforms with live Fabric poses. All points end in gripper-local axes.
    from pxr import Usd, UsdGeom, UsdPhysics
    import isaaclab.sim as sim_utils
    roots=sim_utils.find_matching_prims(asset.cfg.prim_path)
    if len(roots)!=1: raise RuntimeError('Need exactly one YAM USD root for TCP calibration')
    prims=list(Usd.PrimRange(roots[0], Usd.TraverseInstanceProxies()))
    named={name:[p for p in prims if p.GetName()==name and p.HasAPI(UsdPhysics.RigidBodyAPI)] for name in ('gripper','tip_left','tip_right')}
    if any(len(v)!=1 for v in named.values()):
        raise RuntimeError('Cannot identify unique gripper/tip_left/tip_right USD links')
    transforms=UsdGeom.XformCache()
    inverse=transforms.GetLocalToWorldTransform(named['gripper'][0]).GetInverse()
    tips=[]
    bounds=[]
    for name in ('tip_left','tip_right'):
        vertices=[]
        for prim in Usd.PrimRange(named[name][0], Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh): continue
            points=UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if points is None: continue
            matrix=np.asarray(transforms.GetLocalToWorldTransform(prim)*inverse)
            points=np.asarray(points,dtype=float)
            vertices.append((np.c_[points,np.ones(len(points))] @ matrix)[:,:3])
        if not vertices: raise RuntimeError(f'No loaded finger mesh points for {name}')
        points=np.concatenate(vertices)
        # YAM URDF finger direction is local +Z. Use the distal contact region,
        # not the prismatic-joint origin, which is several centimetres proximal.
        distal=points[points[:,2]>=points[:,2].max()-.002]
        tips.append((distal.min(axis=0)+distal.max(axis=0))/2)
        bounds.append({'link':name,'min':points.min(axis=0).tolist(),'max':points.max(axis=0).tolist()})
    tcp=np.mean(tips,axis=0)
    tcp[2]-=.002  # target is 2 mm inside the distal plane for finger-pad overlap
    if not np.isfinite(tcp).all() or not .08<tcp[2]<.22 or np.linalg.norm(tcp[:2])>.035:
        raise RuntimeError(f'Unexpected YAM CAD grasp point {tcp}; inspect the loaded finger geometry')
    return tcp, {'source':'loaded USD distal finger mesh bounds; 2 mm inset',
                 'offset_gripper_frame_m':tcp.tolist(),'finger_bounds_gripper_frame_m':bounds}


class LiveArm:
    def __init__(self, asset, joint_ids):
        self.asset=asset
        self.joint_ids=joint_ids
        names=list(asset.body_names)
        self.base_id=names.index('base')
        self.gripper_id=names.index('gripper')
        self.tcp=None
        self.calibration=None
        self.fixed=bool(asset.is_fixed_base)

    def calibrate(self):
        self.tcp,self.calibration=cad_grasp_point(self.asset)

    def pose(self):
        if self.tcp is None: raise RuntimeError('TCP not calibrated; reset the embodiment')
        pos=_body_data(self.asset,'body_link_pos_w','body_pos_w')
        quat=_body_data(self.asset,'body_link_quat_w','body_quat_w')
        base_r=rotation_xyzw(quat[self.base_id])
        grip_r=rotation_xyzw(quat[self.gripper_id])
        offset=grip_r @ self.tcp
        tcp_world=pos[self.gripper_id]+offset
        tcp_base=base_r.T @ (tcp_world-pos[self.base_id])
        return {'position':tcp_base,'rotation':base_r.T @ grip_r,'world_position':tcp_world,
                'base_position':pos[self.base_id], 'base_rotation':base_r,
                'base_quaternion':quat[self.base_id], 'offset_world':offset}

    def jacobian(self):
        value=getattr(self.asset.data,'body_link_jacobian_w',None)
        if value is None:
            view=getattr(self.asset,'root_physx_view',None)
            if view is None: view=getattr(self.asset,'root_view',None)
            if view is None: raise RuntimeError('Loaded YAM has no Jacobian view')
            value=view.get_jacobians()
        return jacobian_row(array(value),self.gripper_id,self.joint_ids,self.fixed,
                            len(self.asset.body_names),len(self.asset.joint_names))
