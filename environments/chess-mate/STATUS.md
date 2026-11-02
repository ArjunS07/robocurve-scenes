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

`my_package/tasks/state_values.py` has `SIM_QUAT_ORDER = "xyzw"`. Both reading `root_quat_w` and writing `write_root_pose_to_sim` go through it. **Verified on the server** (Isaac Lab 17.0.2, Kit 110.3): a cube turned 90° about X reads back `[0.707, 0, 0, 0.707]`, and the init-state default is `(0, 0, 0, 1)`.

## Verified on the server

- revision 33733f1 (before the visual rework): a 50-step zero-action run completes with no false failure (`num_episodes: 0`); recorder and reset callbacks accept what they receive; the YAM joints `joint1`..`joint8` load with the configured gains and limits; WebRTC streaming works
- quaternion order xyzw; maple table spawned at the origin, collision top z = 0.003, footprint x 0.1986–0.8986, y −0.4779–0.5221 (`environments/tools/probe_runtime.py --sim`)

- revision 159ae2b, `tools/render_check.py`: the new board and pieces render; the grazing view shows no z-fighting; all six piece types are distinct; after a 3 s settle the worst piece drift is 0.6 mm, base-to-board penetration 0.023 mm, tilt 0.016°, board drift 0.11 mm; 321 colliders (10 per piece + board) render in the overlay as stacked prisms that follow the base, stem, and head; both YAM bases stand on the tabletop with the arms facing +X toward the board
- correction: the YAM USDA does carry collision geometry: 9 colliders per arm, inside instanced link prims (server run c4f4e56). An earlier scan that did not enter instances reported 0, which was wrong. Whether they are enabled is reported by tools/render_check.py from ed6aa50 on. The same run showed over-glossy materials and over-bright render lighting (retuned afterwards, not yet re-rendered)

- revision feddfae: the Arena launcher completes 50 zero-action steps on the new board and pieces with no false failure (`num_episodes: 0`); the maple table now loads from the production S3 bucket because the staging copy returns 404
- still open: YAM link colliders. The spawn hook de-instanced 18 prims per arm but found 0 shapes to collide; the next run prints the prim types under each arm

## Not yet verified (needs the server)

- the reworked visuals: textured board, Poly Haven pieces, materials (run `tools/render_check.py`)
- pieces resting without penetration or creep on the new colliders (same script, `settle` section)
- the YAM base footprint and origin against the tabletop at the new x = 0.27 placement (same script, `robots` section), reach to d1 and h5, and whether the YAM asset has colliders at all
- whether pieces can be pinched; friction, mass, and drive gains
- camera frames for policy observations

## The set

Reference: Regal Games 10.5 × 10.5 in reversible wooden board (ASIN B08C1QBGQJ) and the user's product photo.

| Quantity | Value | Source |
| --- | --- | --- |
| Board outer size | 0.2667 m (10.5 in) | listing |
| Square / border | 30 mm / 13.35 mm | estimated from the photo (border a little under half a square) |
| Thickness | 8 mm | estimated; the listing's "0.04 in" is not plausible |
| Inlay line, corner radius | 0.8 mm line 2.5 mm outside the field; 8 mm corners | estimated |
| King / pawn height | 60.8 mm / 34.4 mm | estimated (king about two squares tall) |
| Board mass, piece masses | unchanged | estimates |

**Board.** One rigid body. Its visual is a rounded slab: a single textured top face (squares, inlay line, and border all in one texture), a wood edge, and a bottom. Nothing overlaps the top face, which removes the z-fighting the old 64 coplanar square cubes caused. The collider is a separate flat-topped convex hull (purpose `guide`, 56 vertices, under PhysX's 64 limit). The texture is generated with a fixed seed by `tools/build_chess_assets.py`: honey oak veneer, near-black squares, a1 dark.

**Pieces.** Meshes are Poly Haven's "Chess Set" by Riley Queen (CC0), <https://polyhaven.com/a/chess_set>, converted to meters, Z-up, base-centred, the knight's muzzle toward +X (black pieces are yawed 180°). They are Staunton pieces, not the Regal set's simpler turned profiles, and no CC0 model of that set was found; the itch.io set by Deplorable Mountaineer (CC0) was inspected too and is also Staunton, multi-part, and textured as aged wood. To approach the Regal look the meshes are scaled 0.52 in XY and 0.64 in Z (slimmer, shorter) and get cream-lacquer and glossy-black `UsdPreviewSurface` materials instead of marble. Collision is 10 stacked convex prisms per piece, each the convex hull of the mesh section in its height band, so the base, stem, collar, and head are matched. The scorer's `height_m` and `base_radius_m` are the scaled mesh values; config loading fails if they drift from `assets/chess_set/manifest.json`.

**Reproducing the assets.** `python3 tools/build_chess_assets.py` downloads the glTF with pinned SHA-256s, converts it, writes the collision prisms, draws the textures, and writes `manifest.json` and `LICENSE-CC0.txt`. Launching never downloads anything; if `assets/chess_set` is incomplete, geometry generation raises `FileNotFoundError` naming the missing files instead of falling back to placeholders.

**Placement.** a1 is dark. Rank 1 faces −X, toward the robots. The board is centred on the table's centre line at x = 0.55. Both YAM bases now sit on the tabletop at x = 0.27 (7 cm inside the near edge), y = ±0.22; before, they were at x = −0.05, 0.25 m in front of the table, which is why they looked suspended. Config loading rejects a base outside the table footprint.
