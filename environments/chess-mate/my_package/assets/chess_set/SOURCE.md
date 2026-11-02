# Chess set assets

- Piece meshes: "Chess Set" by Riley Queen, Poly Haven, <https://polyhaven.com/a/chess_set>, CC0 1.0 (`LICENSE-CC0.txt`). Source files and SHA-256 are in `manifest.json`.
- Board textures: generated procedurally by `tools/build_chess_assets.py` (seed in `manifest.json`); no third-party image.
- `pieces/<role>.usda`: one white source mesh per role, meters, Z-up, origin at the base centre, knight muzzle toward +X, source scale (the scene applies its own scale).
- `pieces/collision.json`: per role, 10 stacked convex prisms in source scale.

Rebuild everything with `python3 tools/build_chess_assets.py` from `environments/chess-mate`. Do not hand-edit these files.
