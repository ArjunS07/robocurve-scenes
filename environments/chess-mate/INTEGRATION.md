# Copy this package onto the remote machine

The files in this directory are the drop-in. Nothing here was run in Isaac Sim.

## Copy

Give this directory its own folder. Do not merge it into Isaac Lab-Arena and do not use `--delete` against a folder that already holds another scene.

```bash
rsync -a --exclude '__pycache__' ./ /home/ubuntu/workspace/robocurve/chess-mate/
```

That adds `launch_chess_mate.py`, `my_package/`, and `tests/`.

## Do not touch

- `/home/ubuntu/workspace/IsaacLab-Arena/uv.lock`, and the rest of `/home/ubuntu/workspace/IsaacLab-Arena`
- `/home/ubuntu/workspace/robocurve/assets/yam/yam_source_resolved/yam_source_resolved.usda`
- `/home/ubuntu/workspace/robocurve/battery-remote/my_package/isaaclab_arena_environments/my_environment.py`

Do not run `uv sync`, do not install Isaac Sim, and do not use Docker.

This package does not ship `my_package/isaaclab_arena_environments/__init__.py`, so a copy will not overwrite one that imports another example. Launch imports the chess class by module path.

## Run

Physics smoke, zero action, 50 steps. This is not a 90 second episode and the arm does not move a piece.

```bash
cd /home/ubuntu/workspace/robocurve/chess-mate
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python launch_chess_mate.py
```

Cameras, still without a viewport recording:

```bash
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python launch_chess_mate.py --enable-cameras
```

The script puts this directory and `/home/ubuntu/workspace/IsaacLab-Arena` on `sys.path`, imports `torch`, then runs `isaaclab_arena/evaluation/policy_runner.py` with:

```text
--visualizer none
--device cuda:0
--num_envs 1
--policy_type zero_action
--num_steps 50
--no_solve_relations
--external_environment_class_path my_package.isaaclab_arena_environments.chess_environment:ChessMateEnvironment
chess_mate
```

`--enable_cameras` is appended for `--enable-cameras`. The script does not pass a headless flag and does not pass `--record_viewport_video`.

Launch rewrites `my_package/assets/generated/` from the YAML before the scene is built. `--no_solve_relations` keeps the marked board pose from being moved by Arena's `On()` solver. This scene does not add `On()` relations.

The task string does not name the mating piece or the destination square. The evaluator sheet is `my_package/assets/generated/puzzle_record.txt`, rewritten on launch. Do not put that file into the policy prompt.

## If startup fails

- `UsdFileCfg` rejects `rigid_props`, `collision_props`, `mass_props`, or `physics_material`: those fields are copied from Arena's procedural cuboid spawner. Drop the rejected fields in `my_package/assets/spawn.py` if this Isaac Lab build does not accept them.
- The action term prints the live joint list: set `robot.arm_layout` to `duplicated_single_arm` or `combined`.
- Finger direction is backwards: change `finger_open_m` and `finger_closed_m` together. The USDA file is not edited.
