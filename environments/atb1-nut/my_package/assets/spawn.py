"""USD spawn configs for the nut and the fixture.

Both are plain rigid objects. Mass, collision, contact offsets, friction, solver
counts, and the fixture's kinematic flag are authored in the generated USDA, so
UsdFileCfg only gets the path, scale, and contact-sensor flag. The fixture is
not an articulation: a kinematic body needs no fix_root_link, which this
Isaac Lab build cannot apply to an articulation whose root prim is not itself
the rigid body (the battery-remote startup failure).
"""

from __future__ import annotations

USD_FILE_FIELDS = ("usd_path", "scale", "activate_contact_sensors")


def resolve_robot_usd(usd_path: str) -> str:
    """Use YAM_USD when it is set. Otherwise keep the (absolute) YAML path."""
    import os

    override = os.environ.get("YAM_USD")
    return override if override else str(usd_path)


def usd_file_spawn_kwargs(usd_path: str) -> dict:
    return {
        "usd_path": str(usd_path),
        "scale": (1.0, 1.0, 1.0),
        "activate_contact_sensors": False,
    }


def make_rigid_usd_object(name: str, usd_path: str, position_xyz, rotation_xyzw):
    """Arena Object whose spawn config is a narrow UsdFileCfg (same pattern as chess-mate)."""
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    class UsdRigidObject(Object):
        def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
            return UsdFileCfg(**usd_file_spawn_kwargs(self.usd_path))

        def _generate_rigid_cfg(self):
            from isaaclab.assets import RigidObjectCfg

            cfg = RigidObjectCfg(prim_path=self.prim_path, spawn=self._get_spawn_cfg(), **self.asset_cfg_addon)
            return self._add_initial_pose_to_cfg(cfg)

    return UsdRigidObject(
        name=name,
        prim_path="{ENV_REGEX_NS}/" + name,
        object_type=ObjectType.RIGID,
        usd_path=str(usd_path),
        initial_pose=Pose(
            position_xyz=tuple(float(value) for value in position_xyz),
            rotation_xyzw=tuple(float(value) for value in rotation_xyzw),
        ),
    )
