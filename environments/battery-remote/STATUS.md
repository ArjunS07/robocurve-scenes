# Battery insertion status

Three kinds of evidence, kept apart below: **verified on the server**, **code-checked** (pure tests, USD composition with `usd-core`, no PhysX), and **unverified**.

## Verified on the server

Isaac Lab 17.0.2 / Kit 110.3 on the L40S, via `environments/tools/probe_runtime.py` and earlier launches:

- `root_quat_w` is **xyzw** (a 90° X turn reads `[0.707, 0, 0, 0.707]`); `state_values.quaternion_xyzw` keeps that order.
- `maple_table_robolab` spawns at the origin; collision top z = 0.003; footprint x [0.1986, 0.8986], y [-0.4779, 0.5221].
- YAM USDA: fixed-base single arm (`root_joint` is a `PhysicsFixedJoint` from the `/yam` Xform to the `base` rigid body, which carries `ArticulationRootAPI`), joints `joint1`..`joint8`, bodies `base`, `link1`..`link5`, `gripper`, `tip_left`, `tip_right`. Correction: the YAM USDA does carry collision geometry: 9 colliders per arm, inside instanced link prims (server run c4f4e56). An earlier scan that did not enter instances reported 0, which was wrong. All 9 are enabled (chess-mate ed6aa50 run). The spawn hook only adds colliders to meshes that lack one, so it currently adds none.
- `UsdFileCfg` accepts `physics_material` (the chess scene passes it and ran). The earlier note here that it is "not a file-spawner field" was wrong. This package still authors friction as USD physics materials, which also works; nothing depends on the spawner field.
- Startup at `30f984e` failed: `fix_root_link=True` cannot be applied when `ArticulationRootAPI` sits on a non-rigid Xform. Fixed by the YAM-style `root_joint` below.
- `tools/physics_check.py` at `bc99711` (scripted cell poses, no robot):
  - **Fixture holds**: the remote moves 0.004 mm through settle, insertion, and disturbance. Isaac Lab reports `is_fixed_base: false` (the fixed joint is not counted as part of the articulation), but the chassis does not move.
  - **Springs work**: with no cell, each plunger follows a 3.92 mm target and returns to 0 within ~0.2 s after release. Before `5332b00` the hand-written prismatic joints had no `PhysicsDriveAPI`, so PhysX applied no drive even though Isaac Lab reported the gains; a sleeping articulation was ruled out first (sleep is now disabled anyway).
  - **Insertion and retention in both slots**: a cell lowered into either slot and released ends seated (axial offset 0.00 mm, spring contact, 1.46 mm compression against the 1.5 mm design preload) and stays seated through a 0.05 m/s lateral kick.
  - **Polarity**: a reversed cell sits in the slot touching the spring but is not counted as seated.
  - **Open top**: a 0.6 m/s upward pull lifts the cell out, as expected with the cover removed.
  - **Start pose**: on the bare table battery_b rolled ~20 mm in 2 s and was still accelerating (that patch of the table collider is not level). With the flat 1 mm start mats and angular damping 0.5 (`226053a`), both cells drift 1.4 mm while settling onto the mat and come to 0.7 mm/s, inside the 3 mm mark margin.
- `tools/reach_check.py` at `be1f60b` (real YAM arm, scripted damped-least-squares IK on the measured fingertip contact points, orientation-aware): the left arm reached battery_b within 2.1 mm with the gripper 3° from vertical and the fingers 2° from across the cell, closed on it (pad gap 29.1 mm vs 24.0 mm closed on nothing) and **lifted it 64 mm**. The right arm's attempt on battery_a ended 39.5 mm off and did not lift. Getting there needed stiffer arm drives (proximal 400/40, distal 150/15, 30 N m) and finger drives (400/20).
- `launch_battery_insertion.py` at `278daba`: the Arena runner builds the scene and completes 50 zero-action steps with no false failure (`num_episodes: 0`, insertion score 0 as expected with no motion).

## Code-checked (not physics)

`python3 tests/run_pure_tests.py`: 85 tests, OK (1 skipped here because `pxr` is not installed; it passes with `usd-core`).

- **Fixture.** Authored like the YAM asset: `chassis` is the root rigid body with `PhysicsArticulationRootAPI`; `root_joint` (`PhysicsFixedJoint`) goes from the enclosing `remote` Xform (not a rigid body, so the world) to `chassis`. The spawner no longer requests `fix_root_link`. Plunger links are rigid bodies nested under `chassis` (YAM nests its links the same way). Composed under `/World/envs/env_0/remote`: one articulation root, on the chassis; every joint target resolves; three rigid bodies.
- **Appearance** (likeness of ASIN B07T4N7NTX). MEASURED from the listing: 23.5 × 4.8 × 2.4 cm, black ABS, two AAA. ESTIMATED: everything about the bay (the front photo shows no compartment), the 46 mm bay-block width and 22 mm height, edge fillet 4 mm, end radius 12 mm.
  - Visual shell: one mesh, tapered from 48 mm at the IR nose to 46 mm at the bay, rounded ends, filleted top edges, the two slots cut out of the top with their own walls and floors. Black ABS material.
  - Positive contacts: nickel plates that *are* each slot's end wall at the fixed contact face (no overlay). Plunger pads nickel; a conical steel coil rides on each plunger link.
  - Moulded `+`/`-` marks on each slot floor and on the spine beside each slot's ends; every bar crosses its surface (no coplanar faces).
  - Colliders are `purpose = "guide"` (not rendered): boxes around the bay, and convex-hull prisms that follow the shell's tapered, rounded plan outline at the nose and handle ends (≤ 64 hull vertices each).
- **Spring.** Prismatic joint per slot, frame rotated so +joint compresses; limits [0, 4 mm]; drive target 0; seated preload 1.5 mm → 0.33 N at the configured 220 N/m. These are design numbers, not measured.
- **Masses.** Chassis 0.080 kg, each pad 0.002 kg, authored in the USDA; the spawner passes `mass_props=None`.
- **Unknown readings fail closed.** Missing gripper names → hold unknown → not released. Missing plunger joint → `spring_contact = None` (reported as `spring_unknown`) → no insertion credit; the authored rest pose is never substituted.
- **Mounting.** Bases at (0.27, ±0.20, 0.003): on the tabletop, 7 cm inside the near edge. `validate` checks base z, base XY inside the footprint with a 5 cm margin, and that the bay and both cells are 0.20–0.45 m from the nearer base (they are 0.32–0.34 m). Objects moved to x = 0.57 accordingly.
- **Scoring** (unchanged): interchangeable cells with one-to-one slot assignment; seated = axial ±1.5 mm, radial ≤ 2 mm, polarity, spring contact, released, held for 1 s; orientation-aware lift. **Holding is a geometric proxy** (gripper body distance + finger closure), not contact. **Damage is a 30 N contact-force proxy**, not visible damage.

## Unverified (needs the server)

Run `tools/physics_check.py` (see INTEGRATION.md). Until then, all of these are open:

- Whether the fixed-base authoring starts, and the remote stays put (`remote_displacement_mm` ≈ 0, `is_fixed_base` true).
- Physical insertion into both slots and retention after release; actual spring compression and force at rest; jitter; behavior under a small lateral kick; that an upward pull removes the cell (nothing welds it); that a reversed cell is not counted.
- Contact-sensor prim paths (`.../remote/chassis`, `.../remote/chassis/slot_*_plunger_link`).
- YAM base-link footprint and the arm's facing direction. The probe and `physics_check.py` now report the base bounds counting only geometry owned by the base body. At the old chess placement the arm at home spanned roughly 0.27 m behind the base, so the arm may overhang the near edge; a render will show which way it faces.
- Rendered appearance against the reference photo, and the collision overlay.
- Camera frames; `--capture-dir` output.
