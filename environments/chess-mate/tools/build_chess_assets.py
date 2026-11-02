"""Rebuild the committed chess-set assets from their CC0 source. Needs numpy and Pillow.

Run from environments/chess-mate:

    python3 tools/build_chess_assets.py

What it does, reproducibly:

1. Downloads the Poly Haven "Chess Set" glTF (CC0, https://polyhaven.com/a/chess_set)
   into ``--cache`` and checks the pinned SHA-256 of every file.
2. Takes one white mesh per role, converts glTF Y-up to Z-up, keeps meters, puts the
   origin at the center of the base, and turns the knight so its muzzle faces +X.
3. Writes ``my_package/assets/chess_set/pieces/<role>.usda`` (visual mesh at source
   scale) and ``collision.json`` (stacked convex prisms per role, source scale).
4. Draws the board-top and board-edge wood textures from the board dimensions in
   ``my_package/config/chess_mate.yaml`` with a fixed seed.
5. Writes ``manifest.json`` with source hashes, measurements, and texture parameters.

Launching the scene never downloads anything. It reads only the committed outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "my_package" / "assets" / "chess_set"

SOURCE_URL = "https://polyhaven.com/a/chess_set"
SOURCE_FILES = {
    "chess_set_1k.gltf": (
        "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/chess_set/chess_set_1k.gltf",
        "3e21ecd0c2ffb703dfdb3f3ea7a379c1dbd4ec513ba2cb8e6cb1e5f276d7eb91",
    ),
    "chess_set.bin": (
        "https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/chess_set/chess_set.bin",
        "c73476fd032a2cb4a9c256f70573eed260708132fdc42fec161c6216711560b3",
    ),
}
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt"

# One white mesh per role. Black pieces reuse the same mesh with another material.
SOURCE_NODES = {
    "pawn": "piece_pawn_white_01",
    "rook": "piece_rook_white_01",
    "knight": "piece_knight_white_01",
    "bishop": "piece_bishop_white_01",
    "queen": "piece_queen_white",
    "king": "piece_king_white",
}

# Collision: this many stacked prisms per piece, each the convex hull of the mesh
# section inside its height band, extruded over the band. PhysX convex hulls are
# limited to 64 vertices, so each prism ring keeps at most HULL_POINTS points.
COLLISION_BANDS = 10
HULL_POINTS = 24
# Base radius is measured over the bottom slice of this fraction of the height.
BASE_SLICE_FRACTION = 0.03

_COMPONENT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _download(url: str) -> bytes:
    # Some hosts reject urllib's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "robocurve-asset-build/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_sources(cache: Path) -> dict[str, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (url, digest) in SOURCE_FILES.items():
        path = cache / name
        if not path.is_file() or _sha256(path) != digest:
            print(f"downloading {url}")
            path.write_bytes(_download(url))
        actual = _sha256(path)
        if actual != digest:
            raise RuntimeError(f"{name} SHA-256 is {actual}, expected {digest}; the upstream file changed")
        paths[name] = path
    return paths


def _accessor(gltf: dict, buffers: list[bytes], index: int) -> np.ndarray:
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(_COMPONENT[accessor["componentType"]])
    width = _WIDTH[accessor["type"]]
    count = accessor["count"]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride") or width * dtype.itemsize
    data = buffers[view["buffer"]]
    rows = [
        np.frombuffer(data, dtype=dtype, count=width, offset=start + row * stride) for row in range(count)
    ] if stride != width * dtype.itemsize else [np.frombuffer(data, dtype=dtype, count=count * width, offset=start)]
    return np.concatenate(rows).reshape(count, width)


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_node_mesh(gltf_path: Path, node_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points, normals, and triangle indices of one node, in glTF world space (Y-up)."""
    gltf = json.loads(gltf_path.read_text())
    buffers = [(gltf_path.parent / buffer["uri"]).read_bytes() for buffer in gltf["buffers"]]
    node = next(item for item in gltf["nodes"] if item.get("name") == node_name)
    if "matrix" in node:
        raise ValueError(f"{node_name} uses a matrix transform; this converter expects TRS")
    rotation = _quat_matrix(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]))
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]))
    points, normals, faces = [], [], []
    offset = 0
    for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
        if primitive.get("mode", 4) != 4:
            raise ValueError(f"{node_name} has a non-triangle primitive")
        p = _accessor(gltf, buffers, primitive["attributes"]["POSITION"]).astype(np.float64)
        n = _accessor(gltf, buffers, primitive["attributes"]["NORMAL"]).astype(np.float64)
        f = _accessor(gltf, buffers, primitive["indices"]).reshape(-1, 3).astype(np.int64)
        points.append((p * scale) @ rotation.T + translation)
        # Normals transform with the inverse transpose of rotation*scale.
        n = (n / scale) @ rotation.T
        normals.append(n / np.linalg.norm(n, axis=1, keepdims=True))
        faces.append(f + offset)
        offset += len(p)
    return np.vstack(points), np.vstack(normals), np.vstack(faces)


def to_piece_frame(role: str, points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Y-up to Z-up, origin at the base center, knight muzzle toward +X."""
    to_z_up = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    points = points @ to_z_up.T
    normals = normals @ to_z_up.T
    height = points[:, 2].max() - points[:, 2].min()
    base = points[points[:, 2] <= points[:, 2].min() + BASE_SLICE_FRACTION * height]
    center = np.array([(base[:, 0].max() + base[:, 0].min()) / 2.0, (base[:, 1].max() + base[:, 1].min()) / 2.0, points[:, 2].min()])
    points = points - center
    if role == "knight":
        upper = points[points[:, 2] > 0.55 * height]
        tip = upper[np.argmax(np.hypot(upper[:, 0], upper[:, 1]))]
        angle = -math.atan2(tip[1], tip[0])
        c, s = math.cos(angle), math.sin(angle)
        yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        points = points @ yaw.T
        normals = normals @ yaw.T
    return points, normals


def _convex_hull_2d(xy: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. Counter-clockwise, no repeated end point."""
    pts = np.unique(np.round(xy, 7), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _limit_hull(hull: np.ndarray, count: int) -> np.ndarray:
    """At most ``count`` points. Keeps the farthest hull point in each angular sector."""
    if len(hull) <= count:
        return hull
    center = hull.mean(axis=0)
    angles = np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0])
    sectors = ((angles + math.pi) / (2 * math.pi) * count).astype(int) % count
    keep = []
    for sector in range(count):
        members = np.where(sectors == sector)[0]
        if len(members):
            radii = np.hypot(*(hull[members] - center).T)
            keep.append(hull[members[np.argmax(radii)]])
    return _convex_hull_2d(np.array(keep))


def _section_points(points: np.ndarray, faces: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """Mesh vertices inside [z0, z1] plus every edge crossing z0 or z1."""
    inside = points[(points[:, 2] >= z0) & (points[:, 2] <= z1)][:, :2]
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a, b = points[edges[:, 0]], points[edges[:, 1]]
    cuts = [inside]
    for plane in (z0, z1):
        crossing = (a[:, 2] - plane) * (b[:, 2] - plane) < 0
        t = (plane - a[crossing, 2]) / (b[crossing, 2] - a[crossing, 2])
        cuts.append(a[crossing, :2] + (b[crossing, :2] - a[crossing, :2]) * t[:, None])
    return np.vstack(cuts)


def collision_prisms(points: np.ndarray, faces: np.ndarray) -> list[dict]:
    height = float(points[:, 2].max())
    edges = np.linspace(0.0, height, COLLISION_BANDS + 1)
    prisms = []
    for z0, z1 in zip(edges[:-1], edges[1:]):
        section = _section_points(points, faces, z0, z1)
        hull = _limit_hull(_convex_hull_2d(section), HULL_POINTS)
        prisms.append({"z0": float(z0), "z1": float(z1), "xy": [[float(x), float(y)] for x, y in hull]})
    return prisms


def measure(points: np.ndarray) -> dict:
    height = float(points[:, 2].max())
    radial = np.hypot(points[:, 0], points[:, 1])
    base = radial[points[:, 2] <= BASE_SLICE_FRACTION * height]
    return {
        "height_m": height,
        "base_radius_m": float(base.max()),
        "max_radius_m": float(radial.max()),
        "half_extent_x_m": float(np.abs(points[:, 0]).max()),
        "half_extent_y_m": float(np.abs(points[:, 1]).max()),
    }


def _fmt(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def piece_usda(role: str, points: np.ndarray, normals: np.ndarray, faces: np.ndarray) -> str:
    lo, hi = points.min(axis=0), points.max(axis=0)
    pts = ", ".join(f"({_fmt(x)}, {_fmt(y)}, {_fmt(z)})" for x, y, z in points)
    nrm = ", ".join(f"({_fmt(x)}, {_fmt(y)}, {_fmt(z)})" for x, y, z in normals)
    idx = ", ".join(str(int(i)) for i in faces.reshape(-1))
    counts = ", ".join("3" for _ in range(len(faces)))
    return f"""#usda 1.0
(
    defaultPrim = "mesh"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "Visual {role} mesh from Poly Haven Chess Set (CC0, {SOURCE_URL}). Source scale, meters, Z-up, origin at the base center. Generated by tools/build_chess_assets.py; do not edit."
)

def Mesh "mesh"
{{
    uniform token subdivisionScheme = "none"
    float3[] extent = [({_fmt(lo[0])}, {_fmt(lo[1])}, {_fmt(lo[2])}), ({_fmt(hi[0])}, {_fmt(hi[1])}, {_fmt(hi[2])})]
    int[] faceVertexCounts = [{counts}]
    int[] faceVertexIndices = [{idx}]
    point3f[] points = [{pts}]
    normal3f[] normals = [{nrm}] (
        interpolation = "vertex"
    )
}}
"""


# ---------------------------------------------------------------------------
# Board textures


def _fbm(shape: tuple[int, int], rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    """Smooth value noise in [0, 1] built from bilinearly upsampled random grids."""
    h, w = shape
    total = np.zeros(shape)
    amplitude = 1.0
    for octave in range(octaves):
        cells = 4 * 2**octave
        grid = rng.random((cells + 1, cells + 1))
        img = Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        total += amplitude * np.asarray(img, dtype=np.float64) / 255.0
        amplitude *= 0.5
    total -= total.min()
    return total / total.max()


def _streaks(shape: tuple[int, int], rng: np.random.Generator, along_x: bool, stretch: int, octaves: int) -> np.ndarray:
    """Noise stretched along the grain so it reads as long fibres, in [0, 1]."""
    h, w = shape
    total = np.zeros(shape)
    amplitude = 1.0
    for octave in range(octaves):
        across = 16 * 2**octave
        along = max(2, across // stretch)
        cells = (across, along) if along_x else (along, across)
        grid = rng.random((cells[0] + 1, cells[1] + 1))
        img = Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        total += amplitude * np.asarray(img, dtype=np.float64) / 255.0
        amplitude *= 0.55
    total -= total.min()
    return total / total.max()


def wood(shape: tuple[int, int], rng: np.random.Generator, base, dark, along_x: bool, lines: float, line_strength: float = 1.0) -> np.ndarray:
    """Straight-grain veneer in linear RGB: broad colour streaks plus thin, uneven grain lines."""
    h, w = shape
    broad = _streaks(shape, rng, along_x, stretch=24, octaves=4)
    fibre = _streaks(shape, rng, along_x, stretch=40, octaves=6)
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    across = yy if along_x else xx
    phase = across * lines + 0.35 * _fbm(shape, rng, octaves=3)
    # Thin lines: narrow peaks of a sine with jittered width.
    line = np.clip((np.sin(2 * math.pi * phase) - 0.86) / 0.14, 0.0, 1.0) * (0.4 + 0.6 * fibre)
    mix = np.clip(0.55 * broad + 0.25 * fibre + line_strength * 0.55 * line - 0.25, 0.0, 1.0)[..., None]
    return np.asarray(base)[None, None, :] * (1 - mix) + np.asarray(dark)[None, None, :] * mix


def board_textures(board: dict, size: int, seed: int) -> tuple[Image.Image, Image.Image]:
    """Top texture covers the whole top face; UV (0,0) is the board's (-X, -Y) corner.

    Board frame: +X toward rank 8, +Y toward the a-file, so a1 is at (-X, +Y).
    """
    rng = np.random.default_rng(seed)
    square = board["square_m"]
    border = board["border_m"]
    half = 4 * square + border
    # Pixel (row, col) -> board (x, y). Row 0 is the top of the image, which is UV t = 1 (+Y).
    coords = (np.arange(size) + 0.5) / size * 2 * half - half
    x = coords[None, :].repeat(size, axis=0)
    y = coords[::-1][:, None].repeat(size, axis=1)
    # Linear RGB. Honey oak veneer and near-black stained squares, estimated from the photo.
    light_wood = wood((size, size), rng, (0.62, 0.40, 0.19), (0.38, 0.21, 0.08), along_x=True, lines=70)
    border_wood = wood((size, size), rng, (0.60, 0.38, 0.18), (0.36, 0.20, 0.08), along_x=False, lines=60)
    dark = wood((size, size), rng, (0.010, 0.0095, 0.009), (0.022, 0.019, 0.016), along_x=True, lines=70, line_strength=0.4)
    img = border_wood.copy()
    field = (np.abs(x) < 4 * square) & (np.abs(y) < 4 * square)
    rank = np.floor((x + 4 * square) / square).astype(int)
    file = np.floor((4 * square - y) / square).astype(int)
    is_dark = field & ((rank + file) % 2 == 0)
    is_light = field & ~is_dark
    img[is_light] = light_wood[is_light]
    img[is_dark] = dark[is_dark]
    # Thin dark inlay line just outside the playing field, as on the reference board.
    gap = board["inlay_gap_m"]
    width = board["inlay_width_m"]
    edge = np.maximum(np.abs(x), np.abs(y))
    inlay = (edge >= 4 * square + gap) & (edge <= 4 * square + gap + width)
    img[inlay] = (0.012, 0.011, 0.010)
    top = Image.fromarray((np.clip(img, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
    # The routed edge in the photo is lighter, almost blond.
    side = wood((256, 1024), rng, (0.74, 0.52, 0.28), (0.55, 0.35, 0.16), along_x=True, lines=10)
    edge_img = Image.fromarray((np.clip(side, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
    return top, edge_img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs" / "asset_cache")
    parser.add_argument("--texture-size", type=int, default=2048)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    import yaml

    raw = yaml.safe_load((ROOT / "my_package" / "config" / "chess_mate.yaml").read_text())
    board = raw["board"]

    sources = fetch_sources(args.cache)
    (ASSET_DIR / "pieces").mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / "textures").mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "source": {
            "name": "Chess Set",
            "author": "Riley Queen",
            "page": SOURCE_URL,
            "license": "CC0 1.0",
            "files": {name: {"url": url, "sha256": digest} for name, (url, digest) in SOURCE_FILES.items()},
            "nodes": SOURCE_NODES,
        },
        "pieces": {},
        "collision": {"bands": COLLISION_BANDS, "hull_points": HULL_POINTS},
    }
    collision = {}
    for role, node in SOURCE_NODES.items():
        points, normals, faces = load_node_mesh(sources["chess_set_1k.gltf"], node)
        points, normals = to_piece_frame(role, points, normals)
        (ASSET_DIR / "pieces" / f"{role}.usda").write_text(piece_usda(role, points, normals, faces))
        collision[role] = collision_prisms(points, faces)
        manifest["pieces"][role] = {"node": node, "triangles": int(len(faces)), **measure(points)}
        print(role, {k: round(v, 4) if isinstance(v, float) else v for k, v in manifest["pieces"][role].items()})
    (ASSET_DIR / "pieces" / "collision.json").write_text(json.dumps(collision, indent=1) + "\n")

    texture_params = {
        "square_m": board["square_m"],
        "border_m": board["border_m"],
        "inlay_gap_m": board["inlay_gap_m"],
        "inlay_width_m": board["inlay_width_m"],
        "size_px": args.texture_size,
        "seed": 7,
    }
    top, edge = board_textures(texture_params, args.texture_size, texture_params["seed"])
    top.save(ASSET_DIR / "textures" / "board_top.png", optimize=True)
    edge.save(ASSET_DIR / "textures" / "board_edge.png", optimize=True)
    manifest["board_texture"] = texture_params

    license_path = ASSET_DIR / "LICENSE-CC0.txt"
    if not license_path.is_file():
        license_path.write_bytes(_download(LICENSE_URL))
    (ASSET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"wrote {ASSET_DIR}")


if __name__ == "__main__":
    main()
