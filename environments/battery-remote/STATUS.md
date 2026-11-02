---
cursor:
  subagentId: "bc-cb898d1a-d858-5870-b600-a0eb229b5ec4"
---

# Battery insertion status

Nothing in this package was executed on robocurve-sim, Isaac Sim, or a GPU. The remote machine was not contacted.

## What was checked against Arena `release/0.3.1`

Local read-only clone: commit `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`. Where the public docs and this tag differ, the tag was followed.

Confirmed from that tree and used here:

- External environments still use `ExampleEnvironmentBase.get_env` and `add_cli_args`. `build(cfg)` raises for that base class. `isaaclab_arena_environments/cli.py` registers `--external_environment_class_path` and then the positional name `battery_insertion`.
- `policy_runner.py` is the launch entry. `--policy_type zero_action`, `--num_steps`, and `--no_solve_relations` are real flags. `--no_solve_relations` stores `solve_relations=False`.
- `TaskBase` requires `get_scene_cfg`, `get_termination_cfg`, `get_events_cfg`, `get_mimic_env_cfg`, and `get_metrics`. Default episode length is 70; this task sets 90.
- `TaskTerminationCfg` takes `timeout_s`, a list of `CompletionCriteria`, and a dict of `TerminationTermCfg`. Failure names cannot be `success` or `time_out`. A success predicate may be a plain callable.
- `MetricBase` needs a recorder term and a `MetricTermCfg`. `SuccessRateMetric` is included unchanged.
- `EmbodimentBase` expects `scene_config`, `action_config`, and `observation_config`. Cameras are merged only when `enable_cameras` is true. `get_scene_cfg` rewrites the pose of `.robot` only, so the second arm's pose is stored on `robot_right.init_state`.
- `Object` takes `usd_path` and `object_type`. `reset_all_articulation_joints` restores every articulation's root state and joint defaults. It does not create a weld.
- Procedural rigid props in `object_library.py` use `RigidBodyPropertiesCfg` (`solver_position_iteration_count`, `solver_velocity_iteration_count`), `CollisionPropertiesCfg(contact_offset=...)`, `MassPropertiesCfg`, and `RigidBodyMaterialCfg`. `ImplicitActuatorCfg` in the tag uses `effort_limit` and `velocity_limit`.
- Franka cameras use `CameraCfg.OffsetCfg(..., convention="ros")` with an xyzw quaternion. The same call shape is used here.
- The maple table, `light.set_intensity`, and `directional_light` are the assets the Franka example already builds. The tag does not record the maple table's top height. `table.top_z_m: 0.75` is a guess. No `On()` relation is added.

Not confirmed, because they are outside the Arena tag or were not executed:

- `--visualizer` is not declared in the Arena Python that was read. It comes from Isaac Lab's `AppLauncher`. This install is described as accepting `--visualizer none` and rejecting a headless flag. The launch script follows that description and was not run.
- `ActionTerm` (`action_dim`, `raw_actions`, `processed_actions`, `process_actions`, `apply_actions`) is Isaac Lab, not a class the tag subclasses directly. The tag's custom actions subclass `BinaryJointPositionAction`.
- `UsdFileCfg` in the tag is constructed with `usd_path`, `scale`, and `activate_contact_sensors` only. The extra physics fields match `CuboidCfg`, and startup will fail immediately if this build rejects them.
- USD prismatic-joint frames, cylinder collision, and PhysX contact-offset attributes were not opened in Isaac Sim.
- Joint names inside `yam_source_resolved.usda` were not read. The only confirmed joint list is the source URDF.

## Asset record

Source URDF `yam.urdf` at commit `fe78bae67334b9af0b79833dd8ba141a61767f2a`: `joint1`..`joint6` revolute, `joint7` and `joint8` prismatic, every `effort` and `velocity` entry equal to 1, no collision geometry. Finger limits on that file run from about `-0.04695` to `0`, so the same joint value opens both fingers. The station URDF uses `left_joint*` / `right_joint*` and the opposite finger sign. That is the `combined` layout, and it is not confirmed on the USDA.

Drive gains in the embodiment config override those placeholders inside the simulator only. They are not a calibration and they do not edit the USDA. The 14-dim action is two arms times (six arm offsets + one gripper). Each gripper value is written to both finger joints. A zero action holds the home pose.

## What the pure tests covered

Command, on this VM only:

```bash
cd /cursor/stores/bc-f58ccd4f-c5ff-41b8-99d4-8b82a5dc3f2b/docs/battery-remote
python3 tests/run_pure_tests.py
```

Result: `Ran 34 tests in 0.040s` — `OK`.

The tests drive `step(cfg, snapshot, dt)` and the geometry report. They do not import Isaac Sim. Cases that passed:

- reversed polarity flag, and a true flag with a reversed axis
- partly seated depth, missing plunger contact, and a large radial offset
- gripper still holding, including a hold that contradicts `released=True`
- release before the 1.0 s window, and falling out after the window
- score 100 for both cells, 50 for one, 25 for a lift with no insertion
- hard overrides that force 0 even when an inserted signal is present: never lifted, off table (flag and geometry, latched), remote translation, remote yaw, damage-force proxy
- `clear_episode_state` drops ever-lifted, timers, and latches
- single-slot scores 100 for the one active cell
- a pose inside the remote's outer bounds is not plunger contact and is not full depth
- zero action maps to 16 joint targets: home arms and both fingers open; a gripper command of -1 closes only that arm's two fingers
- the channel is wider than the cell, the top opening is narrower, the seated body clears the walls, a 3 mm lift hits a lip, and the USDA has prismatic plungers and no `PhysicsFixedJoint`

`never_lifted` forces score 0 but is not an early termination. Off table, remote motion, and the damage proxy are the failure terms.

## Remote shape

The scoring object is still a procedural body, not a mesh of the linked product. That listing is an Angrox replacement for an LG-style IR remote, ASIN B07T4N7NTX, and its page gives a black ABS wand of 23.5 × 4.8 × 2.4 cm that takes two AAA cells ([Amazon listing](https://www.amazon.com/dp/B07T4N7NTX)). The silhouette length is now 235 mm, black, with the battery bay on the back and the mouth at the handle end. The bay cross-section stays 50 × 22 mm rather than the listed 48 × 24 mm, because those two millimetres are inside the clearance budget already checked. Side keys and the IR window are visual only. The button face is against the table.

## What is still approximate

Until someone runs this on the L40S, treat the following as unproven:

- whether the C-channel and the plunger actually retain a cell, and whether the clearances survive PhysX contact offsets
- friction, restitution, and the provisional drive gains
- the maple table's real top height and footprint, and therefore every world Z
- camera frames (`overhead_cam`, `front_cam`, `side_cam`)
- USDA joint names, gripper body name, and finger sign
- the force threshold of 30 N, which is a proxy and is not evidence of visible damage
- contact-sensor prim paths (`{ENV_REGEX_NS}/battery_a`, the remote chassis, and the plunger links)

Insertion is not an axis-aligned box around the remote. A cell counts only when its center has passed the seat depth, it is near that slot's centerline, its positive axis matches that slot, the plunger contact test is true, the gripper has released it, and those stay true for `retention_window_s`. The plunger is a prismatic joint. The cell is not teleported and not welded.
