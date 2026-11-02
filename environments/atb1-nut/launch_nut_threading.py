"""Run the nut-threading scene using the existing Arena install, optionally with WebRTC streaming."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def make_parser():
    parser = argparse.ArgumentParser(description="Zero-action scene preview or physics smoke test.")
    parser.add_argument("--arena-root", type=Path, default=Path("/home/ubuntu/workspace/IsaacLab-Arena"))
    parser.add_argument("--stream", action="store_true", help="Keep a Kit viewport open for the WebRTC client.")
    parser.add_argument("--public-ip", default=os.environ.get("PUBLIC_IP"), help="Instance public IP for WebRTC.")
    parser.add_argument("--num-steps", type=int, default=None, help="Default: 50 smoke steps, or 1000000 when streaming.")
    parser.add_argument("--enable-cameras", action="store_true", help="Also create the three task cameras.")
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=None,
        help="Save task-camera PNGs here at ROBOCURVE_CAPTURE_STEPS (default 1,25,49). Needs --enable-cameras.",
    )
    parser.add_argument("--nut-config", type=Path, default=None, help="Alternative YAML; default config/nut_threading.yaml.")
    return parser


def build_argv(args, root):
    argv = [
        str(args.arena_root / "isaaclab_arena/evaluation/policy_runner.py"),
        "--visualizer", "kit" if args.stream else "none",
        "--device", "cuda:0",
        "--num_envs", "1",
        "--policy_type", "zero_action",
        "--num_steps", str(args.num_steps if args.num_steps is not None else (1000000 if args.stream else 50)),
        "--output_base_dir", str(root / "outputs"),
        "--no_solve_relations",
        "--external_environment_class_path",
        "my_package.isaaclab_arena_environments.nut_environment:NutThreadingEnvironment",
    ]
    if args.stream:
        argv.extend(["--livestream", "1"])
    if args.enable_cameras:
        argv.append("--enable_cameras")
    # All runner options precede the environment subcommand.
    argv.append("nut_threading")
    if args.nut_config is not None:
        argv.extend(["--nut_config", str(args.nut_config.resolve())])
    return argv


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.num_steps is not None and args.num_steps <= 0:
        parser.error("--num-steps must be positive")
    if args.stream and not args.public_ip:
        parser.error("Streaming needs --public-ip INSTANCE_IP (or PUBLIC_IP).")
    if args.capture_dir is not None and not args.enable_cameras:
        parser.error("--capture-dir needs --enable-cameras")
    root = Path(__file__).resolve().parent
    runner = args.arena_root / "isaaclab_arena/evaluation/policy_runner.py"
    if not runner.is_file():
        parser.error(f"Arena runner not found: {runner}. Use --arena-root to select the existing install.")
    if args.stream:
        os.environ["PUBLIC_IP"] = args.public_ip
        print("WebRTC client: global.prd.ga.run.brev.nvidia.com | Signal 24317 | Stream 28337", flush=True)
        print("Preview runs until Ctrl+C or the step limit; it does not perform the task.", flush=True)
    if args.capture_dir is not None:
        os.environ["ROBOCURVE_CAPTURE_DIR"] = str(args.capture_dir.resolve())
    # Keep generated assets separate from tracked files and other users' runs.
    (root / "outputs").mkdir(exist_ok=True)
    generated = Path(tempfile.mkdtemp(prefix="assets-", dir=root / "outputs"))
    os.environ["ROBOCURVE_GENERATED_DIR"] = str(generated)
    sys.path.insert(0, str(args.arena_root))
    sys.path.insert(0, str(root))
    import torch  # before Kit: prevents its older CUDA library from shadowing PyTorch's
    import runpy

    sys.argv = build_argv(args, root)
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
