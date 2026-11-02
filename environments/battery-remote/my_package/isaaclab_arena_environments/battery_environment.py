"""External Arena environment. Do not replace the Franka soup-can example with this module."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

from my_package.assets.table import maple_table
from my_package.assets.geometry import write_geometry
from my_package.assets.spawn import make_battery, make_marks, make_remote
from my_package.embodiments.yam_bimanual import YamBimanualEmbodiment
from my_package.scoring.config import cell_names, default_config_path, load_config, with_single_slot
from my_package.scoring.runtime_state import set_active_config
from my_package.tasks.battery_insertion_task import BatteryInsertionTask


class BatteryInsertionEnvironment(ExampleEnvironmentBase):
    """TV-remote battery insertion. Single-slot mode is a flag on this same class."""

    name = "battery_insertion"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        config_arg = getattr(args_cli, "battery_config", None)
        cfg = load_config(config_arg or default_config_path())
        if getattr(args_cli, "single_slot", False):
            cfg = with_single_slot(cfg, True)
        generated = Path(os.environ.get("ROBOCURVE_GENERATED_DIR", str(Path(__file__).resolve().parents[2] / "outputs" / "generated")))
        paths = write_geometry(cfg, generated)
        set_active_config(cfg)

        background = maple_table(self.asset_registry)
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()
        remote = make_remote(cfg, str(paths["remote"]))
        batteries = [make_battery(cfg, name, str(paths["battery"])) for name in cell_names(cfg)]
        marks = make_marks(str(paths["marks"]))
        # No On() relations. Marked poses are authoritative, so the launch also passes --no_solve_relations.
        embodiment = YamBimanualEmbodiment(
            cfg,
            enable_cameras=bool(getattr(args_cli, "enable_cameras", False)),
        )
        scene = Scene(assets=[background, light, directional_light, remote, *batteries, marks])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=BatteryInsertionTask(cfg),
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--single_slot",
            action="store_true",
            help="Cut and score only slot A. The two-slot scene stays in the same package.",
        )
        parser.add_argument(
            "--battery_config",
            type=str,
            default=None,
            help="YAML file of provisional parameters. Defaults to my_package/config/battery_insertion.yaml.",
        )
