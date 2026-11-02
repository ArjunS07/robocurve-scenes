---
cursor:
  subagentId: "bc-cb898d1a-d858-5870-b600-a0eb229b5ec4"
---

# Chess mate status

Nothing in this package has been shown to work physically. Pure tests pass on a Mac without Isaac Sim. An earlier server run built the scene and then crashed on the first rollout step (`env_ids` arrived as a Python list); that path is fixed in code but has not been re-run on the GPU.

## The recorded move (spec resolved: White mates)

Task 7 says White delivers mate and the white pieces are nearest the robot. The earlier FEN had Black to move (`d8h4`), which contradicted that. It has been replaced by its color mirror:

`rnbqkbnr/ppppp3/5p1p/6p1/4PN1P/3NB1P1/PPPP1P2/R1BQRK2 w kq - 0 2`

White is to move. The only legal checkmate is the white queen from d1 to h5 (`d1h5`), a quiet move onto an empty square. With Black to move instead, there is no mate.

```text
8 r n b q k b n r
7 p p p p p . . .
6 . . . . . p . p
5 . . . . . . p .
4 . . . . P N . P
3 . . . N B . P .
2 P P P P . P . .
1 R . B Q R K . .
```

Queen h5 checks the black king on e8 along h5-g6-f7. g6 and f7 are empty, nothing can interpose or capture, and the king's flight squares are covered or occupied.

`build_puzzles` now rejects any puzzle that is not White to move, or whose mating move is a capture, castling, promotion, or en passant, because each of those moves or removes a second piece and contradicts "move only one piece, all other pieces remain". The robot task string says "Play White" but does not name the piece or square. `my_package/assets/generated/puzzle_record.txt` is the evaluator sheet.

## Scoring rules as implemented

- 100: the queen was lifted clear at some point, is now upright, resting, released, with its base entirely inside h5, and every other piece is upright and resting on its own square.
- 75: lifted at some point, upright on h5, not released.
- 50: the queen was lifted clear. 25: another piece was lifted and the queen never was.
- 0: nothing lifted; or any of these latches: a piece off the table, the board off its mark, two or more pieces fully on new squares.
- Sliding the queen onto h5 without lifting it earns no 100 or 75. A different piece that was lifted and put back on its own square does not block 100 (the spec only says the others must remain on their squares).
- A toppled piece still inside its own square counts as disturbed and blocks 100.
- Lift uses the lowest point of a tilted cylinder (base radius and height, origin at the base center), so an inverted piece is not counted as lifted.
- Holding is a proxy: a gripper body within `grasp_radius_m` of the piece origin with fingers mostly closed. It is not a contact check. If any arm's gripper body or finger joints cannot be found, release is treated as unknown and 100 is never awarded (the run tops out at 75) and a warning lists the live body names.
- There is no retention timer; the spec does not ask for one.

## Code-level checks

```bash
cd environments/chess-mate
python3 tests/run_pure_tests.py
```

Result on the Mac: `Ran 28 tests` — `OK`. The tests drive `step`, the puzzle builder, the geometry writer, and the pure `state_values` helpers. They do not import Isaac Sim. New since the first draft:

- the White-to-move FEN has exactly one mate, `d1h5`, onto an empty square; Black to move has none
- the puzzle builder rejects the old Black-to-move FEN and the scholar's-mate capture
- lifting a pawn and sliding the queen to h5 stays at 25 (held or released)
- a lifted-and-replaced pawn does not block 100
- unknown gripper state never scores 100, even with a stale `released=True`
- a toppled neighbour inside its own square blocks 100
- an inverted queen resting on its head is not "lifted"
- `env_indices` accepts lists, tuples, tensor-like objects, `None`, and slices
- quaternion reading and writing round-trip in either component order

## Quaternion order (one switch)

`my_package/tasks/state_values.py` has `SIM_QUAT_ORDER = "xyzw"`. Both reading `root_quat_w` and writing `write_root_pose_to_sim` go through it. Isaac Lab 3 documents xyzw; Isaac Lab 2.x used wxyz. This has not been checked on the installed runtime. If an untouched board immediately trips `board_moved`, or pieces spawn rotated after the first reset, check the installed order and change only that constant.

## Still unproven until a GPU run

- the scene steps past the first rollout step and the recorder/reset callbacks accept the env-id types they receive
- the installed quaternion order (above)
- whether the pieces stand, can be pinched, and stay put when a neighbour is moved
- friction, mass, and the provisional drive gains
- the maple table top is measured (z = 0.003, footprint x 0.199–0.899, y −0.478–0.522, via `environments/tools/probe_runtime.py --sim`); whether pieces actually settle on it is not yet seen in a run
- robot base poses, home pose, reach to d1 and h5, and collisions with the board
- USDA joint names, the `gripper` body name, and finger sign
- camera frames
- the launcher's `--output_base_dir`, `--visualizer kit`, and `--livestream 1` flags against the installed runner

## The set

The board is a procedural slab, not a mesh of the photographed 3-in-1 set. Square size 32 mm, border 28 mm, overall about 312 mm, king height 62 mm, and pawn height 32 mm are provisional. a1 is dark. Rank 1 faces -X, toward the robot bases. Pieces are separate rigid bodies. The board is a heavy rigid body on a visual mark, not a weld.
