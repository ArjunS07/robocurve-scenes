---
cursor:
  subagentId: "bc-cb898d1a-d858-5870-b600-a0eb229b5ec4"
---

# Chess mate status

Nothing in this package was executed in Isaac Sim or on a GPU. There is no screenshot and no viewport recording.

## The recorded move

The supplied FEN is `r1bqrk2/pppp1p2/3nb1p1/4pn1p/6P1/5P1P/PPPPP3/RNBQKBNR b KQ - 0 2`. Black is to move. The only legal checkmate is the black queen from d8 to h4 (`d8h4`). The same piece placement with White to move has no checking move that mates. White's castling rights in the FEN are unused: both wings are blocked by White's own pieces.

The task sentence that says White delivers mate does not match this FEN. The scene keeps the FEN, puts the white pieces on the side nearer the robot, and scores `d8h4`. The robot task string does not name that piece or that square. `my_package/assets/generated/puzzle_record.txt` is the evaluator sheet.

```text
8 r . b q r k . .
7 p p p p . p . .
6 . . . n b . p .
5 . . . . p n . p
4 . . . . . . P .
3 . . . . . P . P
2 P P P P P . . .
1 R N B Q K B N R
```

Queen h4 attacks the white king on e1 through g3 and f2. No white capture, block, or king step answers it.

## What the pure tests covered

```bash
cd /cursor/stores/bc-f58ccd4f-c5ff-41b8-99d4-8b82a5dc3f2b/docs/chess-mate
python3 tests/run_pure_tests.py
```

Result on this VM: `Ran 19 tests in 6.604s` — `OK`.

The tests drive `step` and the geometry writer. They do not import Isaac Sim. Cases:

- the FEN has one mating move, `d8h4`, and White to move has none
- a separate scholar's-mate position has one mating move, `h5f7`
- score 100 after the queen is lifted and then released upright, resting, with its base inside h4 and every other piece at home
- score 75 while a gripper still holds the queen on h4
- score 50 when the queen is lifted clear and is not on h4
- score 25 when a different piece is lifted and the queen never is
- score 0 at the start, when two pieces each sit fully on a new square, when a piece leaves the table, and when the board yaws off its mark
- a 40 degree tilt, and a base that straddles the edge of h4, do not score 100 or 75
- the first reset poses puzzle 0; a later reset advances and wraps
- zero action maps to 16 joint targets
- the board USDA has 64 visual squares, no fixed joint, and does not contain `d8h4`

`never_lifted` forces score 0 but is not an early termination. Off table, board motion, and two transferred pieces are the failure terms. There is no retention timer: the spec does not ask for one.

## The set

The board is a procedural slab, not a mesh of the photographed 3-in-1 set. The photo shows a light wood frame, nearly black squares, and turned light and dark pieces, with checkers and tic-tac-toe on the other views of the same board. This scene builds the chess side only. Square size 32 mm, border 28 mm, overall about 312 mm, king height 62 mm, and pawn height 32 mm are provisional. a1 is dark. Rank 1 faces -X, toward the robot bases. Pieces are separate rigid bodies. The board is a heavy rigid body on a visual mark, not a weld.

## What is still approximate

Until someone runs this on the GPU machine:

- whether the pieces stand and can be pinched
- friction, mass, and the provisional drive gains
- the maple table's real top height, and therefore every world Z
- camera frames
- USDA joint names and finger sign
- the wxyz-to-xyzw read of `root_quat_w`, which matches the battery adapter and was not executed here
