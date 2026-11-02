---
cursor:
  subagentId: "bc-cb898d1a-d858-5870-b600-a0eb229b5ec4"
---

# Battery insertion status

## Code-checked vs. demonstrated

Code-checked on a Mac with no Isaac Sim (`python3 tests/run_pure_tests.py`: `Ran 60 tests` — `OK`), plus a stubbed run of `sim_bridge.update_from_env` / `scores_tensor` / the reset callback with list env ids.

Measured on the server (Isaac Lab 17.0.2 / Kit 110.3) with `environments/tools/probe_runtime.py`:

- `root_quat_w` is **xyzw** (a 90° X turn reads `[0.707, 0, 0, 0.707]`). `state_values.quaternion_xyzw` keeps that order; the old wxyz reordering is gone.
- `maple_table_robolab` spawns at the origin. Its collision top is z = 0.003 and its footprint is x [0.1986, 0.8986], y [-0.4779, 0.5221] (`probe_runtime.py --sim`). Every world z in the YAML was shifted by -0.747 m from the old 0.75 guess, and `validate` now requires both robot bases to sit at the table top.
- YAM USDA: fixed-base single arm, `joint1`..`joint6` revolute, `joint7`/`joint8` prismatic in [-0.04695, 0], bodies `base`, `link1`..`link5`, `gripper`, `tip_left`, `tip_right`. `duplicated_single_arm` and `gripper` are correct. The home pose is inside the limits.

**Not demonstrated in physics yet.** No physics run since these changes has shown that a cell can be inserted, stays seated after release, or that the fixed remote and the plungers behave as designed. The pure tests prove the scorer's logic and the generated geometry's clearances, not PhysX behavior.

## Task model

- **Cover removed.** Each slot is an open-top trough with no ceiling and no lips: floor, side walls, a fixed contact at the positive end, and a spring plunger at the negative end (slot A: + toward +X; slot B: + toward -X). Red marks on the rim sit over the fixed (+) contacts.
- **Insertion.** The cell is pressed down from above, negative end first into the plunger. With the plunger fully compressed there is 2.5 mm of slack, so the cell can be lowered straight down. `design_report` sweeps that vertical drop against every chassis collider **and the compressed plunger**. It also records that the extended plunger blocks the same drop, so the spring has to be pressed.
- **Spring.** A prismatic joint per slot. Its frame is rotated so that +joint always compresses the spring, with limits [0, travel]. The implicit drive targets 0. A seated cell compresses it by `spring_preload_m` (1.5 mm), which pushes the cell against the fixed contact.
- **Retention after release** comes from gravity, the side walls (the seated cell top sits 1 mm below the rim), and the spring preload. No lip, weld, or teleport holds the cell.
- **Secured remote.** The spawner passes `ArticulationRootPropertiesCfg(fix_root_link=True)`, so Isaac Lab fixes the articulation root to the world at the marked pose. The `remote_moved` failure stays, but it should fire only if the fixture breaks.
- **Masses.** The USDA sets the chassis to 0.080 kg and each spring pad to 0.002 kg. The remote's spawner passes `mass_props=None` (`physics_props.remote_physics` sets `mass_kg: None`), so those masses are kept. Friction and restitution are now USD physics materials bound with `material:binding:physics`. `physics_material=` was dropped from `UsdFileCfg`: it is a procedural-shape config field, not a file-spawner field.
- **Joint relationships.** `physics:body0/body1` are sibling-relative (`<../chassis>`, `<../slot_x_plunger_link>`). A test parses the generated USDA and resolves both relationships to existing prims. It also checks with `pxr` when that module is installed; it was not installed here.

## Scoring

- **Interchangeable cells.** Each cell is measured against every slot. Each step, the scorer picks a one-to-one match with the most seated pairs, and on a tie it keeps the previous occupant. Each slot has its own retention timer, which restarts when its occupant changes. Polarity is checked against the slot the cell is in.
- **Seated** means the axial offset from the seat is within ±`axial_tolerance_m` (bounds on both sides), the cell is within `seat_radial_max_m` of the slot's seated axis, its polarity is right on both the axis and the flag, it has geometric spring contact against the live plunger joint pose, and it is released. It must hold that state for `retention_window_s`.
- **Lift** uses the cylinder's lowest point for its orientation: `z - (r·sqrt(1-a_z²) + h·|a_z|)`. A cell standing on its end does not count as lifted.
- **Holding is a geometric proxy:** the `gripper` body is within `grasp_radius_m` and the fingers are shut. It is not a contact test (`tip_left`/`tip_right` would be the bodies for one). If either arm's names are missing, holding is **unknown**, and unknown counts as not released, so no slot can earn credit.
- **Damage is a contact-force proxy** (`damage_force_proxy_n`, 30 N). It is not a model of visible damage.
- Off-table, remote motion, and the damage proxy latch and force 0. `never_lifted` also forces 0, but it is not an early termination.

## Still unverified (needs the GPU run)

- Whether `fix_root_link=True` is accepted on this build's `UsdFileCfg.articulation_props`, and whether the fixed joint lands on `chassis`.
- PhysX insertion and retention with these clearances and contact offsets. Also whether the spring drive reaches 1.5 mm compression without jitter, and the drive gains in general.
- Contact-sensor prim paths (`{ENV_REGEX_NS}/battery_a`, `.../remote/chassis`, `.../remote/slot_*_plunger_link`).
- Launcher flags `--output_base_dir`, `--visualizer kit`, and `--livestream 1` against the installed `policy_runner.py` / AppLauncher. If `--output_base_dir` is rejected, drop it from `build_argv`.
- Reachability. The robot bases at x = -0.05 sit 0.25 m behind the table's measured near edge (x = 0.1986). The remote is at x = 0.50.
- Camera frames.
