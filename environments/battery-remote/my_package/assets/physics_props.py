"""Spawner physics overrides for the remote and the cells. No Isaac Sim imports.

``mass_kg`` of None means the spawner must not pass ``mass_props``: the USDA
already authors per-body masses, and an Isaac Lab mass override is applied to
every rigid body under the prim, which would give each spring pad the chassis
mass. ``fix_root_link`` models the secured fixture.
"""

from __future__ import annotations

from my_package.scoring.config import BatteryInsertionConfig


def remote_physics(cfg: BatteryInsertionConfig) -> dict:
    compartment = cfg.compartment
    return {
        "solver_position_iterations": compartment.solver_position_iterations,
        "solver_velocity_iterations": compartment.solver_velocity_iterations,
        "contact_offset_m": compartment.contact_offset_m,
        # Chassis and spring-pad masses are authored separately in the USDA.
        "mass_kg": None,
        # Secured face down at its mark: the articulation root is fixed to the world.
        "fix_root_link": True,
        "contacts": cfg.scoring.enable_contact_sensors,
    }


def battery_physics(cfg: BatteryInsertionConfig) -> dict:
    compartment = cfg.compartment
    return {
        "solver_position_iterations": compartment.solver_position_iterations,
        "solver_velocity_iterations": compartment.solver_velocity_iterations,
        "contact_offset_m": compartment.contact_offset_m,
        "mass_kg": cfg.battery.mass_kg,
        "fix_root_link": False,
        "contacts": cfg.scoring.enable_contact_sensors,
    }
