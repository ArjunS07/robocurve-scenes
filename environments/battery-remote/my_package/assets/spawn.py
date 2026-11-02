"""USD objects for the compartment and the cells.

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

from my_package.scoring.config import BatteryInsertionConfig, active_slots


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
            mass_props=(None if self._actuators else sim_utils.MassPropertiesCfg(mass=physics["mass_kg"])),
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


def _remote_physics(cfg: BatteryInsertionConfig) -> dict:
    compartment = cfg.compartment
    return {
        "solver_position_iterations": compartment.solver_position_iterations,
        "solver_velocity_iterations": compartment.solver_velocity_iterations,
        "contact_offset_m": compartment.contact_offset_m,
        "mass_kg": compartment.mass_kg,
        "static_friction": compartment.static_friction,
        "dynamic_friction": compartment.dynamic_friction,
        "restitution": compartment.restitution,
        "contacts": cfg.scoring.enable_contact_sensors,
    }


def _battery_physics(cfg: BatteryInsertionConfig) -> dict:
    battery = cfg.battery
    compartment = cfg.compartment
    return {
        "solver_position_iterations": compartment.solver_position_iterations,
        "solver_velocity_iterations": compartment.solver_velocity_iterations,
        "contact_offset_m": compartment.contact_offset_m,
        "mass_kg": battery.mass_kg,
        "static_friction": battery.static_friction,
        "dynamic_friction": battery.dynamic_friction,
        "restitution": battery.restitution,
        "contacts": cfg.scoring.enable_contact_sensors,
    }


def _plunger_actuators(cfg: BatteryInsertionConfig):
    from isaaclab.actuators import ImplicitActuatorCfg

    names = [slot.name.replace("battery_", "slot_") + "_plunger" for slot in active_slots(cfg)]
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
        physics=_remote_physics(cfg),
        actuators=_plunger_actuators(cfg),
        initial_joint_pos={slot.name.replace("battery_", "slot_") + "_plunger": 0.0 for slot in active_slots(cfg)},
    )


def make_battery(cfg: BatteryInsertionConfig, slot_name: str, usd_path: str) -> ConfiguredUsdObject:
    pose = cfg.battery_a_pose if slot_name == "battery_a" else cfg.battery_b_pose
    return ConfiguredUsdObject(
        name=slot_name,
        prim_path="{ENV_REGEX_NS}/" + slot_name,
        object_type=ObjectType.RIGID,
        usd_path=usd_path,
        initial_pose=Pose(position_xyz=tuple(pose.position_xyz), rotation_xyzw=tuple(pose.rotation_xyzw)),
        physics=_battery_physics(cfg),
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
