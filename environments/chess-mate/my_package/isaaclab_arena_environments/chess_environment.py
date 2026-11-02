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
from my_package.assets.spawn import make_board, make_marks, make_piece
from my_package.embodiments.yam_bimanual import YamBimanualEmbodiment
from my_package.scoring.config import default_config_path, load_config, puzzles_for
from my_package.scoring.runtime_state import set_active_config
from my_package.tasks.chess_mate_task import ChessMateTask


class ChessMateEnvironment(ExampleEnvironmentBase):
    name = "chess_mate"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        config_arg = getattr(args_cli, "chess_config", None)
        cfg = load_config(config_arg or default_config_path())
        generated = Path(os.environ.get("ROBOCURVE_GENERATED_DIR", str(Path(__file__).resolve().parents[2] / "outputs" / "generated")))
        paths = write_geometry(cfg, generated)
        set_active_config(cfg)
        puzzle = puzzles_for(cfg)[0]

        background = maple_table(self.asset_registry)
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()
        board = make_board(cfg, str(paths["board"]))
        pieces = [
            make_piece(cfg, puzzle, name, str(paths[f"{name.split('_')[0]}_{name.split('_')[1]}"]))
            for name in sorted(puzzle.homes)
        ]
        marks = make_marks(str(paths["marks"]))
        embodiment = YamBimanualEmbodiment(cfg, enable_cameras=bool(getattr(args_cli, "enable_cameras", False)))
        scene = Scene(assets=[background, light, directional_light, board, *pieces, marks])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=ChessMateTask(cfg),
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--chess_config",
            type=str,
            default=None,
            help="YAML file of provisional parameters. Defaults to my_package/config/chess_mate.yaml.",
        )
