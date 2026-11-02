"""USD objects for the board and the pieces.

UsdFileCfg's rigid-body fields are the same ones Arena's procedural cuboids use
on CuboidCfg. The 0.3.1 tag itself only passes usd_path, scale, and
activate_contact_sensors to UsdFileCfg. If this Isaac Lab build rejects the
extra fields, startup fails immediately and the fields should be dropped.
"""

from __future__ import annotations

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.pose import Pose

from my_package.assets.geometry import piece_poses
from my_package.scoring.config import ChessMateConfig, profile_for
from my_package.scoring.puzzles import Puzzle


class ConfiguredUsdObject(Object):
    def __init__(self, physics: dict | None = None, **kwargs):
        self._physics = physics
        super().__init__(**kwargs)

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        if self._physics is None:
            return UsdFileCfg(usd_path=self.usd_path, scale=self.scale, activate_contact_sensors=False)
        physics = self._physics
        return UsdFileCfg(
            usd_path=self.usd_path,
            scale=self.scale,
            activate_contact_sensors=activate_contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=physics["solver_position_iterations"],
                solver_velocity_iteration_count=physics["solver_velocity_iterations"],
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=physics["contact_offset_m"]),
            mass_props=sim_utils.MassPropertiesCfg(mass=physics["mass_kg"]),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=physics["static_friction"],
                dynamic_friction=physics["dynamic_friction"],
                restitution=physics["restitution"],
            ),
        )

    def _generate_rigid_cfg(self):
        from isaaclab.assets import RigidObjectCfg

        cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=self._get_spawn_cfg(activate_contact_sensors=False),
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(cfg)


def _physics(cfg: ChessMateConfig, mass_kg: float) -> dict:
    board = cfg.board
    return {
        "solver_position_iterations": board.solver_position_iterations,
        "solver_velocity_iterations": board.solver_velocity_iterations,
        "contact_offset_m": board.contact_offset_m,
        "mass_kg": mass_kg,
        "static_friction": board.static_friction,
        "dynamic_friction": board.dynamic_friction,
        "restitution": board.restitution,
    }


def make_board(cfg: ChessMateConfig, usd_path: str) -> ConfiguredUsdObject:
    pose = cfg.board_pose
    return ConfiguredUsdObject(
        name="board",
        prim_path="{ENV_REGEX_NS}/board",
        object_type=ObjectType.RIGID,
        usd_path=usd_path,
        initial_pose=Pose(position_xyz=tuple(pose.position_xyz), rotation_xyzw=tuple(pose.rotation_xyzw)),
        physics=_physics(cfg, cfg.board.mass_kg),
    )


def make_piece(cfg: ChessMateConfig, puzzle: Puzzle, name: str, usd_path: str) -> ConfiguredUsdObject:
    pose = next(item for item in piece_poses(cfg, puzzle) if item.name == name)
    return ConfiguredUsdObject(
        name=name,
        prim_path="{ENV_REGEX_NS}/" + name,
        object_type=ObjectType.RIGID,
        usd_path=usd_path,
        initial_pose=Pose(position_xyz=tuple(pose.position_xyz), rotation_xyzw=tuple(pose.rotation_xyzw)),
        physics=_physics(cfg, profile_for(cfg, pose.role).mass_kg),
    )


def make_marks(usd_path: str) -> ConfiguredUsdObject:
    return ConfiguredUsdObject(
        name="table_marks",
        prim_path="{ENV_REGEX_NS}/table_marks",
        object_type=ObjectType.BASE,
        usd_path=usd_path,
        initial_pose=Pose.identity(),
        physics=None,
    )
