"""Spawner physics overrides for the remote and the cells. No Isaac Sim imports.

``mass_kg`` of None means the spawner must not pass ``mass_props``: the USDA
already authors per-body masses, and an Isaac Lab mass override is applied to
every rigid body under the prim, which would give each spring pad the chassis
mass. The secured fixture is authored in the remote USDA (``root_joint``), not
requested from the spawner, so there is no ``fix_root_link`` here.
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
        "contacts": cfg.scoring.enable_contact_sensors,
    }


def battery_physics(cfg: BatteryInsertionConfig) -> dict:
    compartment = cfg.compartment
    return {
        "solver_position_iterations": compartment.solver_position_iterations,
        "solver_velocity_iterations": compartment.solver_velocity_iterations,
        "contact_offset_m": compartment.contact_offset_m,
        "mass_kg": cfg.battery.mass_kg,
        "contacts": cfg.scoring.enable_contact_sensors,
    }
