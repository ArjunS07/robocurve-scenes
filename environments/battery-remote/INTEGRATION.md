# Copy this package onto the remote machine

The files in this directory are the drop-in. Nothing here was run on robocurve-sim.

## Copy

From a checkout of this directory, copy onto the remote project root. Do not use `--delete`, or the copy will remove the Franka example that already lives there.

```bash
rsync -a --exclude '__pycache__' ./ /home/ubuntu/workspace/robocurve/battery-remote/
```

That adds `launch_battery_insertion.py`, `my_package/` (battery scene, task, embodiment, config), and `tests/`. It does not replace a file this package does not contain.

## Do not touch

- `/home/ubuntu/workspace/IsaacLab-Arena/uv.lock`, and the rest of `/home/ubuntu/workspace/IsaacLab-Arena`
- `/home/ubuntu/workspace/robocurve/assets/yam/yam_source_resolved/yam_source_resolved.usda`
- `/home/ubuntu/workspace/robocurve_rotate`
- `/home/ubuntu/workspace/robocurve/battery-remote/my_package/isaaclab_arena_environments/my_environment.py`

Do not run `uv sync`, do not install Isaac Sim, and do not use Docker. There is no GR00T policy and no `inspect-robots` hookup.

This package does not ship `my_package/isaaclab_arena_environments/__init__.py`, so a copy will not overwrite one that imports the Franka example. Launch imports the battery class by module path, which works without that file. If you want the class imported for a side effect, add this line to the existing `__init__.py`. Do not replace the file:

```python
from my_package.isaaclab_arena_environments.battery_environment import BatteryInsertionEnvironment  # noqa: F401
```

## Run

Physics smoke, zero action, 50 steps. This is not a 90 second episode and the arm does not insert a battery.

```bash
cd /home/ubuntu/workspace/robocurve/battery-remote
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python launch_battery_insertion.py
```

Slot A only, same package:

```bash
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python launch_battery_insertion.py --single-slot
```

Cameras, still without a viewport recording:

```bash
/home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python launch_battery_insertion.py --enable-cameras
```

The script puts this directory and `/home/ubuntu/workspace/IsaacLab-Arena` on `sys.path`, imports `torch`, then runs `isaaclab_arena/evaluation/policy_runner.py` with:

```text
--visualizer none
--device cuda:0
--num_envs 1
--policy_type zero_action
--num_steps 50
--no_solve_relations
--external_environment_class_path my_package.isaaclab_arena_environments.battery_environment:BatteryInsertionEnvironment
battery_insertion
```

`--single_slot` is appended for `--single-slot`. `--enable_cameras` is appended for `--enable-cameras`. The script does not pass a headless flag and does not pass `--record_viewport_video`.

Launch rewrites `my_package/assets/generated/` from the YAML before the scene is built. `--no_solve_relations` keeps the marked poses from being moved by Arena's `On()` solver. This scene does not add `On()` relations.

## If startup fails

- Contact-sensor prims abort the app: set `scoring.enable_contact_sensors: false` in `my_package/config/battery_insertion.yaml`. The damage proxy then stays at 0.
- The action term prints the live joint list: set `robot.arm_layout` to `duplicated_single_arm` (USDA spawned twice, joints `joint1`..`joint8` on each) or `combined` (one articulation, `left_joint*` / `right_joint*`). Those prefixed names are the station URDF, not something confirmed on the USDA.
- Finger direction is backwards: the source URDF opens toward negative joint values (`finger_open_m: -0.04695`, `finger_closed_m: 0`). The station URDF uses the positive range. Change both numbers together. The USDA file is not edited.
- `UsdFileCfg` rejects `rigid_props`, `collision_props`, `mass_props`, or `physics_material`: those fields are copied from Arena's procedural cuboid spawner. The 0.3.1 tag only passes `usd_path`, `scale`, and `activate_contact_sensors` into `UsdFileCfg`. Drop the rejected fields in `my_package/assets/spawn.py` if this Isaac Lab build does not accept them.
