# Status

## What this scene is

One approximate M8 × 1.25 nut-and-stud station inspired by the 8 mm nut on NIST Assembly Task Board 1. It is not the full ATB1 board, and it is not built from the NIST CAD; every dimension here is an approximation chosen for simulation. The language instruction is only "Tighten the nut onto the bolt." The 120 s limit and the 100 / 75 / 50 / 25 / 0 tiers live in the scorer.

## Evidence levels

**Verified on the server** (other packages, same runtime; not this package):

- Root quaternions are xyzw: `environments/tools/probe_runtime.py --sim` read a 90° X turn back as `[0.707, 0, 0, 0.707]`.
- The YAM USDA's bodies are `base`, `link1`…`link5`, `gripper`, `tip_left`, `tip_right`; fingers are `joint7`/`joint8`, range −0.04695…0 m.
- maple_table_robolab spawns at the origin; collision top z = 0.003; footprint x [0.1986, 0.8986], y [−0.4779, 0.5221].
- An articulation whose root prim is not itself the rigid body cannot take `fix_root_link` (battery-remote startup failure). This package no longer uses that pattern.

**Nothing in this package has run on the server yet.** No screenshot exists. Run `tools/physics_check.py` and the launcher (see INTEGRATION.md) and record the revision they print.

**Code-checked only** (44 pure tests, `python3 tests/run_pure_tests.py`):

- quaternions read through `tasks/state_values.py` (xyzw, identity stays identity);
- recorder and reset callbacks accept lists, tensors, slices, and `None`;
- gripper body `gripper`; if it or a finger joint is missing, holding is UNKNOWN, a warning prints the live names once, and the nut is never treated as released (no 100 or 50 credit);
- the generated ridge boxes lie on the helix (parsed back from the USDA), the pins sit on the ridge at every turn, and the turn above blocks a straight pull-off;
- thread clearances exceed twice the contact offset; wedge angle sits between the threaded tilt (2.45°) and the jam tilt (5.36°);
- robot bases on the measured tabletop, 4 cm inside its edge, 0.31–0.37 m from the station; robot USDA path absolute;
- generated assets go to `ROBOCURVE_GENERATED_DIR` (the launcher makes a fresh `outputs/assets-*`), never the source tree; tests use temp dirs;
- launcher flags, ordering, and import order.

## The thread is physical

The stud is a core cylinder (r 3.3 mm) plus a right-hand helical ridge (crest r 4.0 mm, 0.5 mm thick, 173 tilted box colliders, 24 per turn, about 7.2 turns). The nut's bore (r 4.3 mm) carries three pins, 120° apart and stepped by a third of the pitch, reaching to r 3.5 mm. Box tops follow the helix at the pin contact radius, so:

- clockwise rotation (seen from above) slides the pins down the ridge, one pitch per turn; counter-clockwise backs the nut off;
- the ridge turn above each pin blocks pulling the nut straight off (0.25 mm axial play);
- the lead angle (3.0°) is far below the friction angle (atan 0.6), so a released, partly threaded nut stays put. The 50 tier is now physically reachable;
- a nut set on the stud tip rests on the ridge start; it is threaded only after it turns.

These are ordinary PhysX contacts between box colliders. Nothing is welded, toggled, or snapped at runtime, and the scorer does not enforce threading; it still checks every step that rotation and descent agree (cross-thread detection) and that the nut is coaxial.

**Not yet demonstrated:** that PhysX resolves these 0.1–0.3 mm clearances stably at dt = 1/120 s with 16 position iterations (thread contact offset 0.04 mm), that the pins neither jam nor tunnel through the ridge, and that the seams between ridge boxes do not snag. `tools/physics_check.py` measures each of these (descent per turn, helix deviation, tilt, release drift, pull-off rise, seat, back-off). If it shows snagging, raise `segments_per_turn`; if pins tunnel, raise solver iterations or lower the drive torque.

Thread form is simplified: rectangular ridge and pins, not a 60° profile, so there is no radial wedging and no preload model. Seating is the nut's face meeting the collar; tightening torque is not modelled.

## Other pieces

- Fixture: a kinematic rigid body (no articulation, no fixed joint), so contact cannot move it. The fixture-moved failure (> 5 mm or > 5°) only fires if a reset or another writer moves it.
- Holding is a GEOMETRIC PROXY: the `gripper` body within 55 mm of the nut with fingers between 5 % and 35 % open. It is not contact-verified.
- Visible damage is not modelled and there is no force threshold for it.
- The converted YAM USDA has no colliders (found by chess-mate's render check). The embodiment adds convex-hull colliders to every link mesh on the spawned prims only (self-collision off), as chess-mate does; without this the fingers could not grasp or turn the nut. Not yet run in this scene.
- Base z assumes the YAM root frame is at the bottom of its base; the base footprint was not measured, hence the 4 cm margin. In chess-mate's check the arm at home spanned about 0.27 m behind its base origin, so at x = 0.25 links may hang past the table's near edge (x = 0.1986); that is clear of the table but unverified for collisions. Reach is unverified.
- Robot gains (400/150/400), friction (0.6/0.45), and camera intrinsics are provisional.
- Reset writes the authored nut and fixture poses and clears episode memory; it does not animate an unscrew.
