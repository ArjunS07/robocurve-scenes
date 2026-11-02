"""Launch the chess scene with the Arena checkout's own interpreter.

Import torch before Arena starts. This install's physics path is
``--visualizer none``. The script does not pass a headless flag and does not
record a viewport video.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Physics smoke for the YAM chess mate-in-one scene.")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Zero-action physics steps. This is not the 90 second episode and it does not move a piece.",
    )
    parser.add_argument(
        "--enable-cameras",
        action="store_true",
        help="Forward --enable_cameras. The default smoke run does not spawn cameras.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    arena = Path("/home/ubuntu/workspace/IsaacLab-Arena")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(arena))

    import torch  # before Arena / runpy; CUDA import-order workaround for this machine
    import runpy

    argv = [
        str(arena / "isaaclab_arena/evaluation/policy_runner.py"),
        "--visualizer",
        "none",
        "--device",
        "cuda:0",
        "--num_envs",
        "1",
        "--policy_type",
        "zero_action",
        "--num_steps",
        str(args.num_steps),
        "--no_solve_relations",
        "--external_environment_class_path",
        "my_package.isaaclab_arena_environments.chess_environment:ChessMateEnvironment",
        "chess_mate",
    ]
    if args.enable_cameras:
        argv.append("--enable_cameras")
    sys.argv = argv
    runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
