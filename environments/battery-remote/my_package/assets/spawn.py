"""USD objects for the compartment and the cells.

Friction and restitution are authored as USD physics materials in the
generated files, so the spawner does not pass ``physics_material`` (the chess
scene shows the server's ``UsdFileCfg`` accepts it; it is just not needed).

The remote is a fixed-base articulation in its own USDA (a ``root_joint``
anchors the chassis, as in the YAM asset). The spawner does not request
``fix_root_link``: this Isaac Lab build cannot apply it when the articulation
root is not the first rigid link.
"""

from __future__ import annotations

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.pose import Pose

from my_package.assets.physics_props import battery_physics, remote_physics
from my_package.scoring.config import BatteryInsertionConfig, active_slots, battery_pose, plunger_joint_name


class ConfiguredUsdObject(Object):
    """Object whose spawn config carries the provisional contact and solver settings."""

    def __init__(self, physics: dict | None = None, actuators: dict | None = None, initial_joint_pos: dict | None = None, **kwargs):
        self._physics = physics
        self._actuators = actuators or {}
        self._initial_joint_pos = initial_joint_pos or {}
        super().__init__(**kwargs)

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        if self._physics is None:
            return UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=False,
            )
        physics = self._physics
        mass = physics.get("mass_kg")
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
            # None keeps the USD-authored per-body masses.
            mass_props=None if mass is None else sim_utils.MassPropertiesCfg(mass=mass),
            # The remote (the only object with actuators) must not sleep, or its spring drives
            # stop responding to targets; this overrides the USD value at spawn time too.
            articulation_props=(
                sim_utils.ArticulationRootPropertiesCfg(sleep_threshold=0.0, stabilization_threshold=0.0)
                if self._actuators
                else None
            ),
        )

    def _generate_rigid_cfg(self):
        from isaaclab.assets import RigidObjectCfg

        cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=self._get_spawn_cfg(activate_contact_sensors=bool(self._physics and self._physics.get("contacts"))),
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(cfg)

    def _generate_articulation_cfg(self):
        from isaaclab.assets import ArticulationCfg

        cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=self._get_spawn_cfg(
                activate_contact_sensors=bool(self._physics and self._physics.get("contacts"))
            ),
            actuators=self._actuators,
            **self.asset_cfg_addon,
        )
        cfg = self._add_initial_pose_to_cfg(cfg)
        if self._initial_joint_pos:
            cfg.init_state.joint_pos = dict(self._initial_joint_pos)
        return cfg


def _plunger_actuators(cfg: BatteryInsertionConfig):
    from isaaclab.actuators import ImplicitActuatorCfg

    names = [plunger_joint_name(slot) for slot in active_slots(cfg)]
    compartment = cfg.compartment
    return {
        "plungers": ImplicitActuatorCfg(
            joint_names_expr=names,
            stiffness=compartment.plunger_stiffness,
            damping=compartment.plunger_damping,
            effort_limit=compartment.plunger_effort_limit_n,
            velocity_limit=compartment.plunger_velocity_limit_m_s,
        )
    }


def make_remote(cfg: BatteryInsertionConfig, usd_path: str) -> ConfiguredUsdObject:
    pose = cfg.remote_pose
    return ConfiguredUsdObject(
        name="remote",
        prim_path="{ENV_REGEX_NS}/remote",
        object_type=ObjectType.ARTICULATION,
        usd_path=usd_path,
        initial_pose=Pose(position_xyz=tuple(pose.position_xyz), rotation_xyzw=tuple(pose.rotation_xyzw)),
        physics=remote_physics(cfg),
        actuators=_plunger_actuators(cfg),
        # Drive target 0 is the extended plunger; a seated cell compresses it to +preload.
        initial_joint_pos={plunger_joint_name(slot): 0.0 for slot in active_slots(cfg)},
    )


def make_battery(cfg: BatteryInsertionConfig, cell_name: str, usd_path: str) -> ConfiguredUsdObject:
    pose = battery_pose(cfg, cell_name)
    return ConfiguredUsdObject(
        name=cell_name,
        prim_path="{ENV_REGEX_NS}/" + cell_name,
        object_type=ObjectType.RIGID,
        usd_path=usd_path,
        initial_pose=Pose(position_xyz=tuple(pose.position_xyz), rotation_xyzw=tuple(pose.rotation_xyzw)),
        physics=battery_physics(cfg),
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
