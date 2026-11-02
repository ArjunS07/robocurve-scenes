# Copy this package onto the remote machine

The files in this directory are the drop-in. The scene has been built on the server once; it has not completed a rollout.

## Location

On the server this package lives in the repository checkout at `/home/ubuntu/workspace/robocurve-scenes/environments/chess-mate`. Pull the branch there; do not copy it into Isaac Lab-Arena.

## Do not touch

- `/home/ubuntu/workspace/IsaacLab-Arena/uv.lock`, and the rest of `/home/ubuntu/workspace/IsaacLab-Arena`
- `/home/ubuntu/workspace/robocurve/assets/yam/yam_source_resolved/yam_source_resolved.usda`
- `/home/ubuntu/workspace/robocurve/battery-remote/my_package/isaaclab_arena_environments/my_environment.py`

Do not run `uv sync`, do not install Isaac Sim, and do not use Docker.

This package does not ship `my_package/isaaclab_arena_environments/__init__.py`, so a copy will not overwrite one that imports another example. Launch imports the chess class by module path.

## Run

Physics smoke, zero action, 50 steps. This is not a 90 second episode and the arm does not move a piece.

```bash
cd /home/ubuntu/workspace/robocurve-scenes/environments/chess-mate
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python launch_chess_mate.py
```

Live preview for the WebRTC client (one streaming scene at a time on these ports):

```bash
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python launch_chess_mate.py --stream --public-ip 160.211.46.91
```

Client settings that worked for the Franka example: server `global.prd.ga.run.brev.nvidia.com`, signal port `24317`, stream port `28337`. Streaming defaults to 1,000,000 zero-action steps so the preview stays up; it does not perform the task.

Options: `--enable-cameras`, `--num-steps N`, `--arena-root PATH`.

Visual and resting check (headless Kit with cameras, no streaming, no Arena runner). It builds the same table, board, pieces, and arms, settles 3 s, and writes PNGs and `report.json` to `outputs/render_check/<git revision>/`:

```bash
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python tools/render_check.py
```

Rebuilding the committed chess-set assets (only when changing board dimensions or the mesh source; needs network, numpy, Pillow):

```bash
python3 tools/build_chess_assets.py
```

The launcher puts this directory and the Arena checkout on `sys.path`, imports `torch` before Kit (CUDA library order on this machine), then runs `isaaclab_arena/evaluation/policy_runner.py`. Runner flags come before the `chess_mate` subcommand:

```text
--visualizer none            (kit with --stream)
--device cuda:0
--num_envs 1
--policy_type zero_action
--num_steps 50               (1000000 with --stream)
--output_base_dir <this dir>/outputs
--no_solve_relations
--external_environment_class_path my_package.isaaclab_arena_environments.chess_environment:ChessMateEnvironment
[--livestream 1] [--enable_cameras]
chess_mate
```

Generated USD and the evaluator sheet are written to a fresh `outputs/assets-*` directory per launch (`ROBOCURVE_GENERATED_DIR`), never into the source tree (`my_package/assets/generated/` keeps only the evaluator sheets). `outputs/` is git-ignored. The task string does not name the mating piece or square; do not put `puzzle_record.txt` into the policy prompt.

## If startup fails

- `UsdFileCfg` rejects `rigid_props`, `collision_props`, `mass_props`, or `physics_material`: those fields are copied from Arena's procedural cuboid spawner. Drop the rejected fields in `my_package/assets/spawn.py` if this Isaac Lab build does not accept them.
- The board trips `board_moved` at step 0, or pieces spawn rotated after the first reset: the installed quaternion order differs. Change `SIM_QUAT_ORDER` in `my_package/tasks/state_values.py`.
- A warning says the gripper body or finger joints were not found: release is then treated as unknown and the score cannot reach 100. Fix the names in `my_package/scoring/action_map.py` from the printed live body list.
- The action term prints the live joint list: set `robot.arm_layout` to `duplicated_single_arm` or `combined`.
- Finger direction is backwards: change `finger_open_m` and `finger_closed_m` together. The USDA file is not edited.
