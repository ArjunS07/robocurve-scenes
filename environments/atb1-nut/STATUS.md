# Status

## What this scene is

One approximate M8 × 1.25 nut-and-stud station inspired by the 8 mm nut on NIST Assembly Task Board 1. It is not the full ATB1 board, and it is not built from the NIST CAD; every dimension here is an approximation chosen for simulation. The language instruction is only "Tighten the nut onto the bolt." The 120 s limit and the 100 / 75 / 50 / 25 / 0 tiers live in the scorer.

## Evidence levels

**Verified on the server** (other packages, same runtime; not this package):

- Root quaternions are xyzw: `environments/tools/probe_runtime.py --sim` read a 90° X turn back as `[0.707, 0, 0, 0.707]`.
- The YAM USDA's bodies are `base`, `link1`…`link5`, `gripper`, `tip_left`, `tip_right`; fingers are `joint7`/`joint8`, range −0.04695…0 m.
- maple_table_robolab spawns at the origin; collision top z = 0.003; footprint x [0.1986, 0.8986], y [−0.4779, 0.5221].
- An articulation whose root prim is not itself the rigid body cannot take `fix_root_link` (battery-remote startup failure). This package no longer uses that pattern.

**Verified on the server, this package** (`tools/physics_check.py --no-render` and the launcher; scripted torque on the nut, no robot):

- `launch_nut_threading.py` completes 50 zero-action steps with no false failure (after the ActionTerm properties, observation group, reset-term and frozen-config fixes).
- Engage: dropped 0.1 mm above the first turn, the nut lands on the ridge (8.038 mm gap vs 8.056 expected). Before the thread contact offset was raised from 0.04 to 0.8 mm the pins tunnelled through every turn and the nut accelerated at ~2 g to the seat.
- At `be4c8c8`, driven at 2 mN m (free): 3 turns at **1.248 mm/turn** (pitch 1.25), within 0.066 mm of the helix, 1.5° max tilt; released mid-thread it is **retained**; a pull of 2× its weight does not lift it (**held on stud**); seated and released it **scores 100**; backed off at 6 mN m held upright it unscrews 6.5 turns and rises 9.2 mm (**freed**).
- With 24 segments per turn the three pins crossed seams in lockstep and stalled at 0.39 turns; 32 segments (not a multiple of 3) fixed that.
- **Robustness limit:** held upright, 2 mN m tracks to 0.032 mm but is slow; 5 mN m reaches the goal at 1.62 mm/turn (partly skipping); 10–20 mN m throw the nut off the stud even with tilt held at 0, as does 20 mN m free. The likely cause is the check's own rate controller: torque builds to the cap at a seam and the 5 g nut spins up when it breaks free. A robot turning the nut by wrist rotation is a different load and **has not been tested**.

**Code-checked only** (50 pure tests, `python3 tests/run_pure_tests.py`):

- quaternions read through `tasks/state_values.py` (xyzw, identity stays identity);
- recorder and reset callbacks accept lists, tensors, slices, and `None`;
- gripper body `gripper`; if it or a finger joint is missing, holding is UNKNOWN, a warning prints the live names once, and the nut is never treated as released (no 100 or 50 credit);
- the generated ridge boxes lie on the helix (parsed back from the USDA), the pins sit on the ridge at every turn, and the turn above blocks a straight pull-off;
- thread clearances are positive and the thread contact offset (0.8 mm, a detection margin with restOffset 0) is at least the ridge thickness; wedge angle sits between the threaded tilt (2.45°) and the jam tilt (5.36°);
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

**Not yet demonstrated:** that PhysX resolves these 0.1–0.3 mm clearances stably at dt = 1/120 s with 16 position iterations (thread contact offset 0.8 mm; at 0.04 mm the 942249a server run showed the pins tunnelling through every ridge turn, the nut accelerating at about 2 g to the seat), that the pins neither jam nor tunnel through the ridge, and that the seams between ridge boxes do not snag. `tools/physics_check.py` measures each of these (descent per turn, helix deviation, tilt, release drift, pull-off rise, seat, back-off). If it shows snagging, raise `segments_per_turn`; if pins tunnel, raise solver iterations or lower the drive torque.

Thread form is simplified: rectangular ridge and pins, not a 60° profile, so there is no radial wedging and no preload model. Seating is the nut's face meeting the collar; tightening torque is not modelled.

## Other pieces

- Fixture: a kinematic rigid body (no articulation, no fixed joint), so contact cannot move it. The fixture-moved failure (> 5 mm or > 5°) only fires if a reset or another writer moves it.
- Holding is a GEOMETRIC PROXY: the `gripper` body within 55 mm of the nut with fingers between 5 % and 35 % open. It is not contact-verified.
- Visible damage is not modelled and there is no force threshold for it.
- Correction: the YAM USDA does carry collision geometry: 9 colliders per arm, inside instanced link prims (server run c4f4e56). An earlier scan that did not enter instances reported 0, which was wrong. All 9 are enabled (chess-mate ed6aa50 run). The spawn hook only adds colliders to meshes that lack one, so it currently adds none. Not yet run in this scene.
- Base z assumes the YAM root frame is at the bottom of its base; the base footprint was not measured, hence the 4 cm margin. In chess-mate's check the arm at home spanned about 0.27 m behind its base origin, so at x = 0.25 links may hang past the table's near edge (x = 0.1986); that is clear of the table but unverified for collisions. Reach is unverified.
- Robot gains (400/150/400), friction (0.6/0.45), and camera intrinsics are provisional.
- Reset writes the authored nut and fixture poses and clears episode memory; it does not animate an unscrew.
