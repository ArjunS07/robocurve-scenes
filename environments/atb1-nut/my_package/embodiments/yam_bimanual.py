"""YAM bimanual embodiment for Arena 0.3.1.

The resolved USDA is referenced, not edited. duplicated_single_arm spawns that
file twice. A zero action holds both arms at home with the fingers open.
Cameras are created only when the runner asks for them.
"""

from __future__ import annotations

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

from my_package.assets.spawn import resolve_robot_usd, usd_file_spawn_kwargs
from my_package.embodiments.action_map import (
    ACTION_DIM,
    actuator_joint_expressions,
    expected_joint_names,
    initial_joint_positions,
    map_bimanual_action,
    resolve_joint_names,
)


def add_link_colliders(root_prim) -> dict:
    """Give every YAM link mesh a convex-hull collider. Returns counts for reporting.

    The converted YAM USDA has no collision geometry at all (measured on the
    server), so without this the fingers pass through the nut and cannot grasp
    or turn it. Same fix as chess-mate. This edits only the spawned prims on the
    stage, never the shared file. Instance proxies cannot be edited and are counted.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    added, skipped = 0, 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.IsInstanceProxy():
            skipped += 1
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
        added += 1
    return {"added": added, "instance_proxies_skipped": skipped}


def _spawn_yam_with_colliders(prim_path: str, cfg, *args, **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

    base_spawn = UsdFileCfg(usd_path=cfg.usd_path).func
    prim = base_spawn(prim_path, cfg, *args, **kwargs)
    stage = prim.GetStage() if hasattr(prim, "GetStage") else sim_utils.get_current_stage()
    # prim_path may be a regex such as /World/envs/env_.*/left_robot; handle every match.
    matches = sim_utils.find_matching_prims(prim_path, stage=stage) if hasattr(sim_utils, "find_matching_prims") else [prim]
    for match in matches:
        counts = add_link_colliders(match)
        print(f"[yam] {match.GetPath()}: added {counts['added']} link colliders, skipped {counts['instance_proxies_skipped']} instance proxies")
    return prim


class YamBimanualAction(ActionTerm):
    cfg: "YamBimanualActionCfg"

    def __init__(self, cfg: "YamBimanualActionCfg", env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._resolved = {}
        if cfg.layout == "duplicated_single_arm":
            bindings = (("left", cfg.asset_name), ("right", cfg.right_asset_name))
            required = expected_joint_names(cfg.layout)
        elif cfg.layout == "combined":
            bindings = (("robot", cfg.asset_name),)
            required = expected_joint_names(cfg.layout)
        else:
            raise ValueError(f"unknown YAM layout: {cfg.layout}")
        for key, asset_name in bindings:
            asset = env.scene[asset_name]
            names = resolve_joint_names(asset.joint_names, required if cfg.layout == "combined" else expected_joint_names("duplicated_single_arm"))
            live = [str(name) for name in asset.joint_names]
            self._resolved[key] = (asset, names, [live.index(name) for name in names])
        self._targets = None

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions
        self._targets = [
            map_bimanual_action(
                actions[index].detach().cpu().tolist(),
                home_arm_rad=self.cfg.home_arm_rad,
                finger_open_m=float(self.cfg.finger_open_m),
                finger_closed_m=float(self.cfg.finger_closed_m),
                layout=self.cfg.layout,
            )
            for index in range(actions.shape[0])
        ]

    def apply_actions(self):
        for key, (asset, names, joint_ids) in self._resolved.items():
            rows = [[env_targets[key][name] for name in names] for env_targets in self._targets]
            asset.set_joint_position_target(torch.tensor(rows, device=self.device, dtype=torch.float32), joint_ids=joint_ids)


@configclass
class YamBimanualActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = YamBimanualAction
    asset_name: str = "left_robot"
    right_asset_name: str = "right_robot"
    layout: str = "duplicated_single_arm"
    home_arm_rad: tuple[float, float, float, float, float, float] = (0.4, 1.2, 1.0, -0.5, 0.0, 0.0)
    finger_open_m: float = -0.04695
    finger_closed_m: float = 0.0


class YamBimanualEmbodiment(EmbodimentBase):
    name = "yam_bimanual"
    tags = ["embodiment"]
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(self, cfg, enable_cameras: bool = False):
        super().__init__(enable_cameras=enable_cameras, initial_pose=None, arm_mode=ArmMode.DUAL_ARM)
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
        from isaaclab.managers import EventTermCfg
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
        from isaaclab_arena.terms.events import reset_all_articulation_joints
        from isaaclab_arena.utils.configclass import make_configclass

        robot = cfg.robot
        identity = (0.0, 0.0, 0.0, 1.0)

        def actuators(side: str) -> dict:
            expressions = actuator_joint_expressions(robot.layout, side)
            specs = {"proximal": robot.proximal, "distal": robot.distal, "finger": robot.finger}
            return {
                name: ImplicitActuatorCfg(
                    joint_names_expr=expressions[name],
                    stiffness=spec.stiffness,
                    damping=spec.damping,
                    effort_limit=spec.effort,
                    velocity_limit=spec.velocity,
                )
                for name, spec in specs.items()
            }

        def articulation(prim_name: str, position, joint_pos, side: str) -> ArticulationCfg:
            return ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/" + prim_name,
                spawn=UsdFileCfg(
                    func=_spawn_yam_with_colliders,
                    **usd_file_spawn_kwargs(resolve_robot_usd(robot.usd_path)),
                    # Neighbouring link hulls overlap at the joints.
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
                ),
                init_state=ArticulationCfg.InitialStateCfg(pos=position, rot=identity, joint_pos=joint_pos),
                actuators=actuators(side),
            )

        if robot.layout == "duplicated_single_arm":
            joint_pos = initial_joint_positions(robot.home_arm_rad, robot.finger_open_m, robot.layout, "left")
            fields = [
                ("left_robot", ArticulationCfg, articulation("left_robot", robot.left_base_xyz, joint_pos, "left")),
                ("right_robot", ArticulationCfg, articulation("right_robot", robot.right_base_xyz, joint_pos, "right")),
            ]
            asset_name = "left_robot"
        elif robot.layout == "combined":
            joint_pos = {}
            joint_pos.update(initial_joint_positions(robot.home_arm_rad, robot.finger_open_m, robot.layout, "left"))
            joint_pos.update(initial_joint_positions(robot.home_arm_rad, robot.finger_open_m, robot.layout, "right"))
            fields = [
                (
                    "robot",
                    ArticulationCfg,
                    articulation("robot", robot.left_base_xyz, joint_pos, "left"),
                )
            ]
            asset_name = "robot"
        else:
            raise ValueError(f"unknown YAM layout: {robot.layout}")
        self.scene_config = make_configclass("YamSceneCfg", fields)()
        action = YamBimanualActionCfg(
            asset_name=asset_name,
            right_asset_name="right_robot",
            layout=robot.layout,
            home_arm_rad=tuple(robot.home_arm_rad),
            finger_open_m=robot.finger_open_m,
            finger_closed_m=robot.finger_closed_m,
        )
        self.action_config = make_configclass("YamActionsCfg", [("yam_action", ActionTermCfg, action)])()
        self.event_config = make_configclass(
            "YamEventsCfg",
            [("reset_all", EventTermCfg, EventTermCfg(func=reset_all_articulation_joints, mode="reset"))],
        )()
        if self.enable_cameras:
            self.camera_config = _camera_cfg(cfg)
        else:
            self.camera_config = None

    def get_ee_frame_transformer_names(self) -> list[str]:
        return []


def _camera_cfg(cfg):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab_arena.utils.cameras import ArenaCameraCfg
    from isaaclab_arena.utils.configclass import make_configclass

    from my_package.embodiments.camera_frames import camera_offset_position

    items = []
    for camera in cfg.cameras:
        items.append(
            (
                camera.name,
                CameraCfg,
                CameraCfg(
                    prim_path="{ENV_REGEX_NS}/left_robot/" + camera.name + "_cam",
                    update_period=0.0,
                    height=84,
                    width=84,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=2.8,
                        focus_distance=28.0,
                        horizontal_aperture=5.376,
                        vertical_aperture=3.024,
                    ),
                    offset=CameraCfg.OffsetCfg(
                        pos=camera_offset_position(camera.eye_xyz, cfg.robot.left_base_xyz),
                        rot=camera.rotation_xyzw,
                        convention="ros",
                    ),
                ),
            )
        )
    return make_configclass("YamCameraCfg", items, bases=(ArenaCameraCfg,))()
