"""External Arena environment: one approximate M8 nut-and-stud station inspired by NIST ATB1."""

from __future__ import annotations

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class NutThreadingEnvironment(ExampleEnvironmentBase):
    name: str = "nut_threading"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        from my_package.assets.geometry import ensure_generated_assets
        from my_package.assets.spawn import make_rigid_usd_object
        from my_package.embodiments.yam_bimanual import YamBimanualEmbodiment
        from my_package.scoring.config import load_config
        from my_package.tasks.nut_task import NutThreadingTask

        config_path = getattr(args_cli, "nut_config", None)
        cfg = load_config(config_path)
        # Written to ROBOCURVE_GENERATED_DIR (set per run by the launcher), never the source tree.
        nut_usd, fixture_usd = ensure_generated_assets(cfg)
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        light = self.asset_registry.get_asset_by_name("light")()
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()
        nut = make_rigid_usd_object("nut", nut_usd, cfg.nut_pose.position_xyz, cfg.nut_pose.rotation_xyzw)
        fixture = make_rigid_usd_object(
            "fixture", fixture_usd, cfg.fixture_pose.position_xyz, cfg.fixture_pose.rotation_xyzw
        )
        embodiment = YamBimanualEmbodiment(cfg, enable_cameras=bool(getattr(args_cli, "enable_cameras", False)))
        scene = Scene(assets=[background, light, directional_light, nut, fixture])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=NutThreadingTask(cfg),
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--nut_config", type=str, default=None)
