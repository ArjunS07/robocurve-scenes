"""YAM bimanual embodiment for Arena 0.3.1.

The policy action is 14 numbers: each arm is six joint offsets from home plus
one gripper opening. Those two gripper numbers are written to the two finger
joints. Joint names are confirmed on the source URDF, not on the USDA. If the
loaded asset uses different names, initialization raises and prints the live
joint list.

ActionTerm's method set is the Isaac Lab ABC. The Arena 0.3.1 tag subclasses
Isaac Lab's binary gripper action instead of ActionTerm directly, so this
class was not executed against the simulator.
"""

from __future__ import annotations

from pathlib import Path

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import Pose

from my_package.scoring.action_map import ACTION_DIM, map_bimanual_action
from my_package.scoring.config import BatteryInsertionConfig


def _resolve_ids(asset, names: tuple[str, ...], kind: str) -> list[int]:
    finder = asset.find_joints if kind == "joint" else asset.find_bodies
    resolved = []
    missing = []
    for name in names:
        indices, found = finder(f"^{name}$", preserve_order=True)
        if len(found) != 1 or found[0] != name:
            missing.append(name)
            continue
        resolved.append(int(indices[0]))
    if missing:
        live = list(asset.joint_names) if kind == "joint" else list(getattr(asset, "body_names", []))
        raise RuntimeError(
            f"YAM asset is missing {kind}s {missing}. Live names: {live}. "
            "Edit robot.arm_layout in my_package/config/battery_insertion.yaml. "
            "duplicated_single_arm expects joint1..joint8 on each copy of the USDA. "
            "combined expects left_joint1..8 and right_joint1..8 on one articulation. "
            "If the finger sign is flipped, also change finger_open_m and finger_closed_m."
        )
    return resolved


class YamBimanualAction(ActionTerm):
    """14-dim action. ``processed_actions`` stays 14-wide; 16 joint targets are applied separately."""

    cfg: "YamBimanualActionCfg"

    def __init__(self, cfg: "YamBimanualActionCfg", env):
        super().__init__(cfg, env)
        count = int(getattr(self, "num_envs", env.num_envs))
        device = getattr(self, "device", env.device)
        self._raw_actions = torch.zeros(count, ACTION_DIM, device=device)
        self._processed = torch.zeros(count, ACTION_DIM, device=device)
        self._groups = []
        bindings = (
            (
                cfg.left_asset,
                cfg.left_arm_joints,
                cfg.left_finger_joints,
            ),
            (
                cfg.right_asset,
                cfg.right_arm_joints,
                cfg.right_finger_joints,
            ),
        )
        # Combined layout names both arms on the same articulation, so group by asset.
        ordered: dict[str, list[str]] = {}
        for asset_name, arm_joints, finger_joints in bindings:
            ordered.setdefault(asset_name, [])
            ordered[asset_name].extend(list(arm_joints) + list(finger_joints))
        for asset_name, joint_names in ordered.items():
            asset = env.scene[asset_name]
            ids = _resolve_ids(asset, tuple(joint_names), "joint")
            self._groups.append(
                {
                    "asset_name": asset_name,
                    "joint_names": tuple(joint_names),
                    "ids": ids,
                    "values": torch.zeros(count, len(ids), device=device),
                }
            )

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed[:] = actions
        home = tuple(self.cfg.home_arm)
        for env_index in range(self._raw_actions.shape[0]):
            action = tuple(float(value) for value in actions[env_index].tolist())
            targets = map_bimanual_action(
                action,
                self.cfg.bindings(),
                home,
                self.cfg.finger_closed_m,
                self.cfg.finger_open_m,
                self.cfg.home_open_fraction,
                self.cfg.arm_action_scale,
                self.cfg.gripper_action_scale,
            )
            for group in self._groups:
                for column, joint_name in enumerate(group["joint_names"]):
                    group["values"][env_index, column] = targets[(group["asset_name"], joint_name)]

    def apply_actions(self):
        for group in self._groups:
            asset = self._env.scene[group["asset_name"]]
            asset.set_joint_position_target(group["values"], joint_ids=group["ids"])

    def reset(self, env_ids=None):
        if env_ids is None:
            self._raw_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
        self.process_actions(self._raw_actions)


@configclass
class YamBimanualActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = YamBimanualAction
    asset_name: str = "robot"
    layout: str = "duplicated_single_arm"
    home_arm: tuple[float, float, float, float, float, float] = (0.4, 1.2, 1.0, -0.5, 0.0, 0.0)
    finger_closed_m: float = 0.0
    finger_open_m: float = -0.04695
    home_open_fraction: float = 1.0
    arm_action_scale: float = 1.0
    gripper_action_scale: float = 1.0
    left_asset: str = "robot"
    right_asset: str = "robot_right"
    left_arm_joints: tuple[str, str, str, str, str, str] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    left_finger_joints: tuple[str, str] = ("joint7", "joint8")
    left_gripper_body: str = "gripper"
    right_arm_joints: tuple[str, str, str, str, str, str] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    right_finger_joints: tuple[str, str] = ("joint7", "joint8")
    right_gripper_body: str = "gripper"

    def bindings(self):
        from my_package.scoring.action_map import ArmBinding

        return (
            ArmBinding("left", self.left_asset, self.left_arm_joints, self.left_finger_joints, self.left_gripper_body),
            ArmBinding(
                "right", self.right_asset, self.right_arm_joints, self.right_finger_joints, self.right_gripper_body
            ),
        )


def _actuator(names: tuple[str, ...] | list[str], stiffness: float, damping: float, effort: float, velocity: float):
    return ImplicitActuatorCfg(
        joint_names_expr=list(names),
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort,
        velocity_limit=velocity,
    )


def add_link_colliders(root_prim) -> dict:
    """Give every YAM link mesh a convex-hull collider. Returns counts for reporting.

    The converted YAM USDA has no collision geometry at all (the source URDF has
    none), so without this the arms pass through the board and cannot pinch a
    piece. This edits only the spawned prims on the stage, never the shared file.
    Instance proxies cannot be edited and are counted instead.
    """
    from collections import Counter

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = root_prim.GetStage()
    # Geometry may sit in payloads; load everything under the robot first.
    stage.Load(root_prim.GetPath())
    everything = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    # The link visuals are instanced, and a plain PrimRange does not enter instances
    # (the first server run added 0 colliders). Instance proxies cannot be edited, so
    # de-instance them on this stage only, repeating for nested instances.
    deinstanced = 0
    for _ in range(8):
        instances = [prim for prim in Usd.PrimRange(root_prim, everything) if prim.IsInstance() and not prim.IsInstanceProxy()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
            deinstanced += 1
    added, skipped = 0, 0
    types: Counter = Counter()
    for prim in Usd.PrimRange(root_prim, everything):
        types[prim.GetTypeName() or "<untyped>"] += 1
        if not prim.IsA(UsdGeom.Gprim) or prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsActive():
            continue
        if prim.IsInstanceProxy():
            skipped += 1
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
        added += 1
    # Prim types under the robot, so a zero count can be diagnosed from the log alone.
    return {
        "added": added,
        "instance_proxies_skipped": skipped,
        "deinstanced": deinstanced,
        "prim_types": dict(types.most_common(10)),
    }


def _spawn_yam_with_colliders(prim_path: str, cfg, *args, **kwargs):
    base_spawn = UsdFileCfg(usd_path=cfg.usd_path).func
    prim = base_spawn(prim_path, cfg, *args, **kwargs)
    import isaaclab.sim as sim_utils

    stage = prim.GetStage() if hasattr(prim, "GetStage") else sim_utils.get_current_stage()
    # prim_path may be a regex such as /World/envs/env_.*/RobotLeft; colour every match.
    for match in sim_utils.find_matching_prims(prim_path, stage=stage) if hasattr(sim_utils, "find_matching_prims") else [prim]:
        counts = add_link_colliders(match)
        print(f"[yam] {match.GetPath()}: added {counts['added']} link colliders, de-instanced {counts['deinstanced']}, skipped {counts['instance_proxies_skipped']} instance proxies; prim types {counts['prim_types']}")
    return prim


def _articulation(cfg: BatteryInsertionConfig, prim_path: str, pose, binding) -> ArticulationCfg:
    robot = cfg.robot
    joint_pos = {name: float(value) for name, value in zip(binding.arm_joints, robot.home_arm_rad)}
    for finger in binding.finger_joints:
        joint_pos[finger] = float(robot.finger_open_m)
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=UsdFileCfg(
            func=_spawn_yam_with_colliders,
            usd_path=robot.usda_path,
            activate_contact_sensors=False,
            # Link hulls of neighbouring links overlap at the joints.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(pose.position_xyz),
            rot=tuple(pose.rotation_xyzw),
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        actuators={
            f"{binding.side}_proximal": _actuator(
                binding.arm_joints[:3],
                robot.proximal_stiffness,
                robot.proximal_damping,
                robot.arm_effort_limit,
                robot.arm_velocity_limit,
            ),
            f"{binding.side}_distal": _actuator(
                binding.arm_joints[3:],
                robot.distal_stiffness,
                robot.distal_damping,
                robot.arm_effort_limit,
                robot.arm_velocity_limit,
            ),
            f"{binding.side}_fingers": _actuator(
                binding.finger_joints,
                robot.finger_stiffness,
                robot.finger_damping,
                robot.finger_effort_limit,
                robot.finger_velocity_limit,
            ),
        },
    )


def _observation_cfg(include_right: bool, concatenate: bool):
    if include_right:

        @configclass
        class PolicyCfg(ObsGroup):
            actions: ObsTerm = ObsTerm(func=mdp.last_action)
            joint_pos: ObsTerm = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
            joint_pos_right: ObsTerm = ObsTerm(
                func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_right")}
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = concatenate

    else:

        @configclass
        class PolicyCfg(ObsGroup):
            actions: ObsTerm = ObsTerm(func=mdp.last_action)
            joint_pos: ObsTerm = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = concatenate

    @configclass
    class ObservationsCfg:
        policy: PolicyCfg = PolicyCfg()

    return ObservationsCfg()


def _camera_cfg(cfg: BatteryInsertionConfig):
    fields = []
    for camera in cfg.cameras:
        fields.append(
            (
                camera.name,
                CameraCfg,
                CameraCfg(
                    prim_path="{ENV_REGEX_NS}/" + camera.name,
                    update_period=0.0,
                    height=camera.height,
                    width=camera.width,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=camera.focal_length,
                        focus_distance=camera.focus_distance,
                        horizontal_aperture=camera.horizontal_aperture,
                        vertical_aperture=camera.vertical_aperture,
                    ),
                    # Offset is relative to the env origin. convention="ros" matches Arena's Franka cameras,
                    # which store xyzw. These poses are provisional.
                    offset=CameraCfg.OffsetCfg(
                        pos=camera.eye_xyz,
                        rot=camera.rotation_xyzw,
                        convention="ros",
                    ),
                ),
            )
        )
    return make_configclass("YamCameraCfg", fields, bases=(ArenaCameraCfg,))()


@register_asset
class YamBimanualEmbodiment(EmbodimentBase):
    """Two YAM arms loaded from the resolved USDA. The file itself is not modified."""

    name = "yam_bimanual"
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        battery_cfg: BatteryInsertionConfig,
        enable_cameras: bool = False,
        concatenate_observation_terms: bool = False,
    ):
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=Pose(
                position_xyz=tuple(battery_cfg.robot.left_pose.position_xyz),
                rotation_xyzw=tuple(battery_cfg.robot.left_pose.rotation_xyzw),
            ),
            concatenate_observation_terms=concatenate_observation_terms,
        )
        usda = Path(battery_cfg.robot.usda_path)
        if not usda.is_file():
            raise FileNotFoundError(
                f"YAM USDA not found at {usda}. This package only references that file and does not modify it."
            )
        from my_package.scoring.action_map import arm_bindings

        left, right = arm_bindings(battery_cfg)
        duplicated = battery_cfg.robot.arm_layout == "duplicated_single_arm"
        left_cfg = _articulation(
            battery_cfg,
            "{ENV_REGEX_NS}/RobotLeft",
            battery_cfg.robot.left_pose,
            left,
        )
        fields = [("robot", ArticulationCfg, left_cfg)]
        if duplicated:
            right_cfg = _articulation(
                battery_cfg,
                "{ENV_REGEX_NS}/RobotRight",
                battery_cfg.robot.right_pose,
                right,
            )
            fields.append(("robot_right", ArticulationCfg, right_cfg))
        else:
            # One articulation carries both arms. The base class only rewrites `.robot`.
            merged = _articulation(
                battery_cfg,
                "{ENV_REGEX_NS}/Robot",
                battery_cfg.robot.left_pose,
                left,
            )
            extra = _articulation(
                battery_cfg,
                "{ENV_REGEX_NS}/Robot",
                battery_cfg.robot.right_pose,
                right,
            )
            merged.actuators.update(extra.actuators)
            merged.init_state.joint_pos.update(extra.init_state.joint_pos)
            fields = [("robot", ArticulationCfg, merged)]
        self.scene_config = make_configclass("YamSceneCfg", fields)()
        action = YamBimanualActionCfg(
            layout=battery_cfg.robot.arm_layout,
            home_arm=battery_cfg.robot.home_arm_rad,
            finger_closed_m=battery_cfg.robot.finger_closed_m,
            finger_open_m=battery_cfg.robot.finger_open_m,
            home_open_fraction=battery_cfg.robot.home_open_fraction,
            arm_action_scale=battery_cfg.robot.arm_action_scale,
            gripper_action_scale=battery_cfg.robot.gripper_action_scale,
            left_asset=left.asset_name,
            right_asset=right.asset_name,
            left_arm_joints=left.arm_joints,
            left_finger_joints=left.finger_joints,
            left_gripper_body=left.gripper_body,
            right_arm_joints=right.arm_joints,
            right_finger_joints=right.finger_joints,
            right_gripper_body=right.gripper_body,
        )
        self.action_config = make_configclass("YamActionsCfg", [("arm_action", ActionTermCfg, action)])()
        self.observation_config = _observation_cfg(duplicated, concatenate_observation_terms)
        self.camera_config = _camera_cfg(battery_cfg) if enable_cameras else None
        self.event_config = None
        self.gripper = None

    def get_ee_frame_transformer_names(self) -> list[str]:
        # No confirmed end-effector prim. Trajectory recording is not part of the smoke launch.
        return []
