# Status

## What this scene is

An M16 nut-and-bolt station inspired by NIST ATB1, using Isaac Lab's Factory M16 assets (NVIDIA, loaded at runtime). It is not the full ATB1 board, and it is not built from the NIST CAD. The language instruction is only "Tighten the nut onto the bolt." The 120 s limit and the 100 / 75 / 50 / 25 / 0 tiers live in the scorer.

### Why it moved from the M8 box thread to Factory M16

The earlier procedural M8 × 1.25 thread (a helical ridge of tilted box colliders, three pins in the nut) worked only below about 5 mN·m of turning torque. Driven at 2 mN·m it threaded 3 turns at 1.248 mm/turn and the seated, released nut scored 100 (`be4c8c8`). At 5–50 mN·m it skipped turns or threw the nut off the stud, both free and with the nut held upright. A robot gripper turning a nut applies far more than 5 mN·m, so that thread could not be used with a real policy. Isaac Lab's Factory nut and bolt carry real thread meshes with SDF collision and are what Isaac Lab's own nut-threading task uses.

### Layout (all measured from the Factory meshes with usd-core; `config/nut_threading.yaml`)

- Bolt: right-hand, **2.0 mm pitch** (helix fit of the crest points), 35 mm long, hex head 10 mm tall (corner radius 13.86 mm), shank crest radius 7.84 mm. Its origin is the bottom of the head; it stands on the table at (0.55, 0, 0.003).
- Fixture: a kinematic block. Its lower layer has a pocket around the head (half width 15 mm) and its 9 mm upper plate has a clearance hole around the shank (half width 9 mm, 1.16 mm clear of the crests). The plate top, z = 0.022, is the **seat face** and the fixture origin. No fixture collider touches the bolt.
- Exposed thread: 35 − 10 − 9 = **16 mm, 8 turns** from the tip to the seat. This is the scorer's travel.
- Nut: 13 mm tall, 24 mm across flats, 50 g. Its body origin is **10 mm below its lower face**, so it spawns at z = −0.006 to rest flat on a 1 mm start mat (part of the fixture body) 15 cm from the bolt. The scorer and the hold check use the nut's centre, computed from the origin and the nut's axis.
- Loading: the YAML names the production-bucket URLs (the same bucket as the maple table). `my_package/assets/factory.py` fetches them through Isaac Lab's `retrieve_file_path` into its `/tmp` cache; a fetch failure or a missing local path stops the launch with an error naming the asset and how to override it. Nothing from NVIDIA is stored in this repo.
- Wrappers: `assets/geometry.py` writes `nut.usda` and `bolt.usda` that reference the cached USDs and author overs. They delete the articulation root (a RigidObject cannot contain one), deactivate the bolt's world fixed joint, and make the bolt kinematic, the same pattern as the fixture, so nothing needs `fix_root_link`. They also set Factory's per-body settings: 192 position iterations, max depenetration velocity 5 m/s, gyroscopic forces, uncapped contact impulse, contact offset 5 mm and rest offset 0 on the SDF meshes, and friction 0.75.
- **Not set here:** Factory also tunes scene-wide PhysX settings (solver iteration limits, GPU contact buffers, bounce threshold). Those belong to the Arena environment, not this scene, and are left at Arena's defaults.

## Evidence levels

**Verified on the server** (other packages, same runtime; not this package):

- Root quaternions are xyzw: `environments/tools/probe_runtime.py --sim` read a 90° X turn back as `[0.707, 0, 0, 0.707]`.
- The YAM USDA's bodies are `base`, `link1`…`link5`, `gripper`, `tip_left`, `tip_right`; fingers are `joint7`/`joint8`, range −0.04695…0 m.
- maple_table_robolab spawns at the origin; collision top z = 0.003; footprint x [0.1986, 0.8986], y [−0.4779, 0.5221].
- An articulation whose root prim is not itself the rigid body cannot take `fix_root_link` (battery-remote startup failure). This package no longer uses that pattern.

**Verified on the server, this package:**

- `launch_nut_threading.py` completed 50 zero-action steps with no false failure with the earlier M8 scene. The Factory M16 scene **has not yet been run on the server**: the bolt as a kinematic SDF body, the wrappers' overs, and threading under a 10–500 mN·m drive are all unverified. Run `tools/physics_check.py` (INTEGRATION.md).
- Earlier M8 box-thread results, kept for comparison: 2 mN·m threaded at 1.248 mm/turn, was retained when released, held on a pull, scored 100 seated, and backed off free; 5–50 mN·m skipped or threw the nut, free or held upright.

**Code-checked only** (`python3 tests/run_pure_tests.py`):

- quaternions read through `tasks/state_values.py` (xyzw, identity stays identity);
- recorder and reset callbacks accept lists, tensors, slices, and `None`;
- gripper body `gripper`; if it or a finger joint is missing, holding is UNKNOWN, a warning prints the live names once, and the nut is never treated as released (no 100 or 50 credit); holding is measured from the finger pad centres;
- the plate hole and head pocket clear the bolt by more than 1 mm; the fixture boxes do not overlap each other; the nut at its spawn pose touches only the mat top; 16 mm exposed, 8 turns; seat at z = 0.022; nut spawn z from the 10 mm origin offset;
- the nut and bolt wrappers reference the Factory USDs, remove the articulation root, switch off the bolt's root joint, and carry the Factory settings; a missing asset raises a clear error; Isaac Lab's cache is reused;
- the scorer's tiers, cross-thread detection at the 2.0 mm pitch, the 1 mm seat tolerance, and the 75% travel tier;
- wedge angle (5°) above the most a nut can tilt on the thread (4.57°, minor-diameter clearance over the nut height); radial tolerance (1 mm) above the radial play (0.52 mm);
- robot bases on the measured tabletop, 4 cm inside its edge, 0.31–0.37 m from the station; robot USDA path absolute;
- generated wrappers go to `ROBOCURVE_GENERATED_DIR` (the launcher makes a fresh `outputs/assets-*`), never the source tree; tests use temp dirs;
- launcher flags, ordering, and import order.

## The thread is physical

Turning the nut screws it down the bolt only through PhysX contacts between the two SDF meshes. Nothing is welded, toggled, or snapped at runtime, and the scorer does not enforce threading; it checks every step that rotation and descent agree (cross-thread detection) and that the nut is coaxial. Seating is the nut's face meeting the plate's seat face; tightening torque and preload are not modelled.

**Not yet demonstrated:** that the kinematic SDF bolt threads stably with the nut at dt = 1/120 s under Arena's scene settings, at the torques a gripper applies. `tools/physics_check.py` sweeps 10, 50, 200 and 500 mN·m, free and held upright, then measures descent per turn, helix deviation, tilt, release drift, pull-off rise, seat, and back-off.

## Other pieces

- Fixture and bolt: kinematic rigid bodies (no articulation, no fixed joint), so contact cannot move them. The fixture-moved failure (> 5 mm or > 5°) only fires if a reset or another writer moves it.
- Holding is a GEOMETRIC PROXY: the `gripper` body within 55 mm of the nut with fingers between 5 % and 35 % open. It is not contact-verified.
- Visible damage is not modelled and there is no force threshold for it.
- Correction: the YAM USDA does carry collision geometry: 9 colliders per arm, inside instanced link prims (server run c4f4e56). An earlier scan that did not enter instances reported 0, which was wrong. All 9 are enabled (chess-mate ed6aa50 run). The spawn hook only adds colliders to meshes that lack one, so it currently adds none. Not yet run in this scene.
- Base z assumes the YAM root frame is at the bottom of its base; the base footprint was not measured, hence the 4 cm margin. In chess-mate's check the arm at home spanned about 0.27 m behind its base origin, so at x = 0.25 links may hang past the table's near edge (x = 0.1986); that is clear of the table but unverified for collisions. Reach is unverified.
- Robot gains (400/150/400), fixture friction (0.6/0.45), and camera intrinsics are provisional. Nut and bolt friction (0.75) follow Factory.
- Reset writes the authored nut, bolt and fixture poses and clears episode memory; it does not animate an unscrew.
