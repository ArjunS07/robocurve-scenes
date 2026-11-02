"""Board and piece USD for the chess scene. No Isaac Sim imports.

The board is one rigid body: a rounded slab whose top face carries a single
texture (squares, inlay line, and border), so no visible faces overlap. Its
collider is a separate flat-topped convex hull. Pieces are separate rigid
bodies that reference the committed Poly Haven meshes (CC0) in
``assets/chess_set`` and collide through stacked convex prisms fitted to each
piece's base, stem, and head. Nothing welds a piece to the board.

Committed assets are required. If they are missing this module raises instead
of falling back to placeholder shapes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from my_package.scoring.config import ChessMateConfig, profile_for, puzzles_for
from my_package.scoring.math3d import local_to_world, quat_mul
from my_package.scoring.puzzles import Puzzle
from my_package.scoring.scorer import square_center_local

_BLACK_YAW = (0.0, 0.0, 1.0, 0.0)

ASSET_DIR = Path(__file__).resolve().parent / "chess_set"
ROLES = ("pawn", "rook", "knight", "bishop", "queen", "king")
# A scaled mesh must match its config profile to this tolerance.
_PROFILE_TOLERANCE_M = 0.0002
# Rounded-corner segments: smooth for the visual slab; few enough for PhysX's
# 64-vertex convex hull limit on the collider.
_VISUAL_CORNER_SEGMENTS = 12
_COLLIDER_CORNER_SEGMENTS = 6


def required_asset_paths() -> list[Path]:
    paths = [ASSET_DIR / "manifest.json", ASSET_DIR / "LICENSE-CC0.txt", ASSET_DIR / "pieces" / "collision.json"]
    paths += [ASSET_DIR / "pieces" / f"{role}.usda" for role in ROLES]
    paths += [ASSET_DIR / "textures" / "board_top.png", ASSET_DIR / "textures" / "board_edge.png"]
    return paths


def require_assets() -> None:
    missing = [str(path) for path in required_asset_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "chess set assets are missing; run `python3 tools/build_chess_assets.py` from "
            "environments/chess-mate (needs network, numpy, Pillow). Missing:\n- " + "\n- ".join(missing)
        )


def asset_manifest() -> dict:
    require_assets()
    return json.loads((ASSET_DIR / "manifest.json").read_text())


def collision_prisms() -> dict[str, list[dict]]:
    require_assets()
    return json.loads((ASSET_DIR / "pieces" / "collision.json").read_text())


def scaled_piece_size(cfg: ChessMateConfig, role: str) -> dict[str, float]:
    """Height and radii of the committed mesh after the config's piece scale."""
    measured = asset_manifest()["pieces"][role]
    sxy, sz = cfg.piece_mesh.scale_xy, cfg.piece_mesh.scale_z
    return {
        "height_m": measured["height_m"] * sz,
        "base_radius_m": measured["base_radius_m"] * sxy,
        "max_radius_m": measured["max_radius_m"] * sxy,
    }


@dataclass(frozen=True)
class PiecePose:
    name: str
    color: str
    role: str
    square: str
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


def board_half_extent(cfg: ChessMateConfig) -> float:
    return 4.0 * cfg.board.square_m + cfg.board.border_m


def board_top_local_z(cfg: ChessMateConfig) -> float:
    return cfg.board.thickness_m / 2.0


def piece_pose(cfg: ChessMateConfig, puzzle: Puzzle, name: str) -> PiecePose:
    square = puzzle.homes[name]
    color, role, _square = name.split("_", 2)
    center = square_center_local(cfg, square)
    local = (center[0], center[1], board_top_local_z(cfg))
    position = local_to_world(cfg.board_pose.position_xyz, cfg.board_pose.rotation_xyzw, local)
    rotation = cfg.board_pose.rotation_xyzw
    if color == "black":
        rotation = quat_mul(rotation, _BLACK_YAW)
    return PiecePose(name, color, role, square, position, rotation)


def piece_poses(cfg: ChessMateConfig, puzzle: Puzzle) -> tuple[PiecePose, ...]:
    return tuple(piece_pose(cfg, puzzle, name) for name in sorted(puzzle.homes))


def design_report(cfg: ChessMateConfig) -> dict:
    problems: list[str] = []
    puzzles = puzzles_for(cfg)
    half = board_half_extent(cfg)
    center_x, center_y = cfg.board_pose.position_xyz[0], cfg.board_pose.position_xyz[1]
    table_x, table_y = cfg.table.center_xy
    table_hx, table_hy = cfg.table.half_extents_xy
    if center_x - half < table_x - table_hx or center_x + half > table_x + table_hx:
        problems.append("the board hangs off the table in x")
    if center_y - half < table_y - table_hy or center_y + half > table_y + table_hy:
        problems.append("the board hangs off the table in y")
    poses = piece_poses(cfg, puzzles[0])
    if len(poses) != 32:
        problems.append(f"expected 32 pieces, got {len(poses)}")
    by_square = {}
    for pose in poses:
        if pose.square in by_square:
            problems.append(f"two pieces claim {pose.square}")
        by_square[pose.square] = pose
        if abs(pose.position_xyz[2] - (cfg.table.top_z_m + cfg.board.thickness_m)) > 1e-9:
            problems.append(f"{pose.name} does not sit on the board top")
        profile = profile_for(cfg, pose.role)
        if profile.base_radius_m >= cfg.board.square_m / 2.0:
            problems.append(f"{pose.role} base does not fit in a square")
    for index, first in enumerate(poses):
        for second in poses[index + 1 :]:
            gap = math.hypot(first.position_xyz[0] - second.position_xyz[0], first.position_xyz[1] - second.position_xyz[1])
            need = profile_for(cfg, first.role).base_radius_m + profile_for(cfg, second.role).base_radius_m + 0.001
            if gap < need:
                problems.append(f"{first.name} overlaps {second.name}")
    white_king = next(pose for pose in poses if pose.role == "king" and pose.color == "white")
    black_king = next(pose for pose in poses if pose.role == "king" and pose.color == "black")
    if white_king.position_xyz[0] >= black_king.position_xyz[0]:
        problems.append("white pieces are not nearer the robot than black")
    dest = puzzles[0].move.destination_square
    if dest in by_square:
        problems.append(f"mating square {dest} is occupied at the start")
    problems.extend(_asset_problems(cfg))
    return {"ok": not problems, "problems": problems, "piece_count": len(poses)}


def _asset_problems(cfg: ChessMateConfig) -> list[str]:
    """The committed meshes and textures must agree with the config the scorer reads."""
    problems: list[str] = []
    manifest = asset_manifest()
    for profile in cfg.pieces:
        size = scaled_piece_size(cfg, profile.role)
        for key in ("height_m", "base_radius_m"):
            if abs(getattr(profile, key) - size[key]) > _PROFILE_TOLERANCE_M:
                problems.append(
                    f"{profile.role} {key} is {getattr(profile, key)} in the config but {size[key]:.5f} for the scaled mesh"
                )
        # Widest part (e.g. the knight's head) must clear a neighbour's widest part.
        if 2.0 * size["max_radius_m"] + 0.001 > cfg.board.square_m:
            problems.append(f"{profile.role} is too wide for neighbouring squares")
    texture = manifest["board_texture"]
    for key in ("square_m", "border_m", "inlay_gap_m", "inlay_width_m"):
        if abs(texture[key] - getattr(cfg.board, key)) > 1e-9:
            problems.append(f"board texture was drawn for {key}={texture[key]}; rebuild assets for {getattr(cfg.board, key)}")
    if not 0.0 < cfg.board.corner_radius_m < cfg.board.border_m:
        problems.append("corner_radius_m must be positive and smaller than the border")
    return problems


def write_geometry(cfg: ChessMateConfig, directory: str | Path) -> dict[str, Path]:
    require_assets()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = design_report(cfg)
    if not report["ok"]:
        raise ValueError("refusing to write geometry:\n- " + "\n- ".join(report["problems"]))
    puzzles = puzzles_for(cfg)
    board_path = directory / "board.usda"
    marks_path = directory / "table_marks.usda"
    board_path.write_text(_board_usda(cfg))
    marks_path.write_text(_marks_usda(cfg))
    paths = {"board": board_path, "marks": marks_path}
    for color in ("white", "black"):
        for profile in cfg.pieces:
            path = directory / f"{color}_{profile.role}.usda"
            path.write_text(_piece_usda(cfg, profile.role, color))
            paths[f"{color}_{profile.role}"] = path
    diagram_path = directory / "setup_diagram.txt"
    record_path = directory / "puzzle_record.txt"
    diagram_path.write_text("\n\n".join(f"puzzle {puzzle.index}\n{puzzle.diagram()}" for puzzle in puzzles) + "\n")
    record_path.write_text(_record_text(puzzles))
    paths["diagram"] = diagram_path
    paths["record"] = record_path
    return paths


def _record_text(puzzles: tuple[Puzzle, ...]) -> str:
    blocks = []
    for puzzle in puzzles:
        move = puzzle.move
        blocks.append(
            "\n".join(
                [
                    f"puzzle {puzzle.index}",
                    f"fen {puzzle.fen}",
                    f"piece {move.piece_name}",
                    f"start {move.start_square}",
                    f"destination {move.destination_square}",
                    f"uci {move.uci}",
                    puzzle.diagram(),
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def _fmt(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _rgb(color: tuple[float, float, float]) -> str:
    return ", ".join(_fmt(channel) for channel in color)


def _vec3_list(points) -> str:
    return ", ".join(f"({_fmt(x)}, {_fmt(y)}, {_fmt(z)})" for x, y, z in points)


def _vec2_list(points) -> str:
    return ", ".join(f"({_fmt(x)}, {_fmt(y)})" for x, y in points)


def _ints(values) -> str:
    return ", ".join(str(int(value)) for value in values)


def rounded_rect(half: float, radius: float, segments: int) -> list[tuple[float, float]]:
    """Counter-clockwise outline of a square with rounded corners."""
    outline = []
    for cx, cy, start in ((half - radius, half - radius, 0.0), (radius - half, half - radius, 90.0), (radius - half, radius - half, 180.0), (half - radius, radius - half, 270.0)):
        for step in range(segments + 1):
            angle = math.radians(start + 90.0 * step / segments)
            outline.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return outline


def _cap(outline, z: float, up: bool):
    """Fan-triangulated cap: center vertex first, then the outline."""
    points = [(0.0, 0.0, z)] + [(x, y, z) for x, y in outline]
    n = len(outline)
    faces = []
    for k in range(n):
        a, b = 1 + k, 1 + (k + 1) % n
        faces.extend((0, a, b) if up else (0, b, a))
    return points, faces


def _walls(outline, z0: float, z1: float):
    n = len(outline)
    points = [(x, y, z0) for x, y in outline] + [(x, y, z1) for x, y in outline]
    faces = []
    for k in range(n):
        a, b = k, (k + 1) % n
        faces.extend((a, b, n + b, a, n + b, n + a))
    return points, faces


def _mesh(name: str, points, faces, extra: str = "", schemas: tuple[str, ...] = (), indent: str = "    ") -> str:
    header = f'{indent}def Mesh "{name}"'
    if schemas:
        header += f' (\n{indent}    prepend apiSchemas = [{", ".join(chr(34) + s + chr(34) for s in schemas)}]\n{indent})'
    body = f"""
{indent}{{
{indent}    uniform token subdivisionScheme = "none"
{indent}    int[] faceVertexCounts = [{_ints([3] * (len(faces) // 3))}]
{indent}    int[] faceVertexIndices = [{_ints(faces)}]
{indent}    point3f[] points = [{_vec3_list(points)}]{extra}
{indent}}}"""
    return header + body


def _preview_material(path: str, name: str, *, color=None, texture: Path | None = None, roughness: float, clearcoat: float = 0.0, indent: str = "        ") -> str:
    """UsdPreviewSurface; a texture, if given, drives diffuseColor through the ``st`` primvar."""
    base = f"{path}/{name}"
    if texture is not None:
        diffuse = f"color3f inputs:diffuseColor.connect = <{base}/texture.outputs:rgb>"
        readers = f"""
{indent}    def Shader "texture"
{indent}    {{
{indent}        uniform token info:id = "UsdUVTexture"
{indent}        asset inputs:file = @{texture.as_posix()}@
{indent}        token inputs:sourceColorSpace = "sRGB"
{indent}        token inputs:wrapS = "repeat"
{indent}        token inputs:wrapT = "repeat"
{indent}        float2 inputs:st.connect = <{base}/uv.outputs:result>
{indent}        float3 outputs:rgb
{indent}    }}
{indent}    def Shader "uv"
{indent}    {{
{indent}        uniform token info:id = "UsdPrimvarReader_float2"
{indent}        string inputs:varname = "st"
{indent}        float2 outputs:result
{indent}    }}"""
    else:
        diffuse = f"color3f inputs:diffuseColor = ({_rgb(color)})"
        readers = ""
    return f"""{indent}def Material "{name}"
{indent}{{
{indent}    token outputs:surface.connect = <{base}/surface.outputs:surface>
{indent}    def Shader "surface"
{indent}    {{
{indent}        uniform token info:id = "UsdPreviewSurface"
{indent}        {diffuse}
{indent}        float inputs:roughness = {_fmt(roughness)}
{indent}        float inputs:metallic = 0
{indent}        float inputs:clearcoat = {_fmt(clearcoat)}
{indent}        float inputs:clearcoatRoughness = 0.05
{indent}        token outputs:surface
{indent}    }}{readers}
{indent}}}"""


def _bind(material_path: str) -> str:
    return f"\n        rel material:binding = <{material_path}>"


def _board_usda(cfg: ChessMateConfig) -> str:
    half = board_half_extent(cfg)
    thickness = cfg.board.thickness_m
    top_z, bottom_z = thickness / 2.0, -thickness / 2.0
    radius = cfg.board.corner_radius_m
    outline = rounded_rect(half, radius, _VISUAL_CORNER_SEGMENTS)

    top_points, top_faces = _cap(outline, top_z, up=True)
    # UV (0,0) is the board's (-X, -Y) corner, matching tools/build_chess_assets.py.
    top_uv = [((x + half) / (2.0 * half), (y + half) / (2.0 * half)) for x, y, _z in top_points]
    wall_points, wall_faces = _walls(outline, bottom_z, top_z)
    perimeter = [0.0]
    for (x0, y0), (x1, y1) in zip(outline, outline[1:]):
        perimeter.append(perimeter[-1] + math.hypot(x1 - x0, y1 - y0))
    total = perimeter[-1] + math.hypot(outline[0][0] - outline[-1][0], outline[0][1] - outline[-1][1])
    wall_uv = [(4.0 * d / total, 0.0) for d in perimeter] + [(4.0 * d / total, 1.0) for d in perimeter]
    bottom_points, bottom_faces = _cap(outline, bottom_z, up=False)
    collider_outline = rounded_rect(half, radius, _COLLIDER_CORNER_SEGMENTS)
    hull_points = [(x, y, bottom_z) for x, y in collider_outline] + [(x, y, top_z) for x, y in collider_outline]
    hull_faces = _walls(collider_outline, bottom_z, top_z)[1]

    looks = "/board/Looks"
    uv_attr = lambda uv: f"""
        texCoord2f[] primvars:st = [{_vec2_list(uv)}] (
            interpolation = "vertex"
        )"""
    meshes = [
        _mesh("top", top_points, top_faces, uv_attr(top_uv) + _bind(f"{looks}/top"), ("MaterialBindingAPI",)),
        _mesh("edge", wall_points, wall_faces, uv_attr(wall_uv) + _bind(f"{looks}/edge"), ("MaterialBindingAPI",)),
        _mesh("bottom", bottom_points, bottom_faces, _bind(f"{looks}/edge"), ("MaterialBindingAPI",)),
        _mesh(
            "collider",
            hull_points,
            hull_faces,
            '\n        uniform token physics:approximation = "convexHull"\n        uniform token purpose = "guide"',
            ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"),
        ),
    ]
    textures = ASSET_DIR / "textures"
    looks_body = "\n".join(
        [
            _preview_material(looks, "top", texture=textures / "board_top.png", roughness=0.65),
            _preview_material(looks, "edge", texture=textures / "board_edge.png", roughness=0.5),
        ]
    )
    body = "\n".join(meshes)
    return f"""#usda 1.0
(
    defaultPrim = "board"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
    doc = "Regal-style 10.5 in wooden chessboard. One textured top face (no coplanar overlays). The collider is a separate convex hull with a flat top."
)

def Xform "board" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)
{{
    float physics:mass = {_fmt(cfg.board.mass_kg)}
    float physics:staticFriction = {_fmt(cfg.board.static_friction)}
    float physics:dynamicFriction = {_fmt(cfg.board.dynamic_friction)}
    float physics:restitution = {_fmt(cfg.board.restitution)}
    int physxRigidBody:solverPositionIterationCount = {cfg.board.solver_position_iterations}
    int physxRigidBody:solverVelocityIterationCount = {cfg.board.solver_velocity_iterations}
{body}
    def Scope "Looks"
    {{
{looks_body}
    }}
}}
"""


def _prism(name: str, prism: dict, sxy: float, sz: float) -> str:
    ring = [(x * sxy, y * sxy) for x, y in prism["xy"]]
    points, faces = _walls(ring, prism["z0"] * sz, prism["z1"] * sz)
    n = len(ring)
    # Caps as fans over the ring's own vertices (the ring is convex).
    for k in range(1, n - 1):
        faces.extend((0, k + 1, k))
        faces.extend((n, n + k, n + k + 1))
    return _mesh(
        name,
        points,
        faces,
        '\n        uniform token physics:approximation = "convexHull"\n        uniform token purpose = "guide"',
        ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"),
        indent="        ",
    )


def _piece_usda(cfg: ChessMateConfig, role: str, color: str) -> str:
    profile = profile_for(cfg, role)
    mesh = cfg.piece_mesh
    sxy, sz = mesh.scale_xy, mesh.scale_z
    rgb = mesh.white_rgb if color == "white" else mesh.black_rgb
    roughness = mesh.white_roughness if color == "white" else mesh.black_roughness
    prisms = "\n".join(_prism(f"band_{index:02d}", prism, sxy, sz) for index, prism in enumerate(collision_prisms()[role]))
    source = (ASSET_DIR / "pieces" / f"{role}.usda").as_posix()
    material = _preview_material("/piece/Looks", "finish", color=rgb, roughness=roughness, clearcoat=0.35 if color == "black" else 0.1)
    return f"""#usda 1.0
(
    defaultPrim = "piece"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
    doc = "{color} {role}: Poly Haven Chess Set mesh (CC0), scaled {_fmt(sxy)} in XY and {_fmt(sz)} in Z. Origin is the base center. Collision is stacked convex prisms."
)

def Xform "piece" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)
{{
    float physics:mass = {_fmt(profile.mass_kg)}
    float physics:staticFriction = {_fmt(cfg.board.static_friction)}
    float physics:dynamicFriction = {_fmt(cfg.board.dynamic_friction)}
    float physics:restitution = {_fmt(cfg.board.restitution)}
    int physxRigidBody:solverPositionIterationCount = {cfg.board.solver_position_iterations}
    int physxRigidBody:solverVelocityIterationCount = {cfg.board.solver_velocity_iterations}
    def Xform "visual"
    {{
        float3 xformOp:scale = ({_fmt(sxy)}, {_fmt(sxy)}, {_fmt(sz)})
        uniform token[] xformOpOrder = ["xformOp:scale"]
        def "mesh" (
            prepend apiSchemas = ["MaterialBindingAPI"]
            prepend references = @{source}@
        )
        {{
            rel material:binding = </piece/Looks/finish>
        }}
    }}
    def Xform "collision"
    {{
{prisms}
    }}
    def Scope "Looks"
    {{
{material}
    }}
}}
"""


def _marks_usda(cfg: ChessMateConfig) -> str:
    """Four thin strips just outside the board footprint. Nothing sits under the board."""
    half = board_half_extent(cfg)
    width = 0.004
    gap = 0.002
    z = cfg.table.top_z_m + 0.0003
    cx, cy = cfg.board_pose.position_xyz[0], cfg.board_pose.position_xyz[1]
    outer = half + gap + width
    strips = []
    for name, (x0, x1, y0, y1) in (
        ("near", (-outer, -half - gap, -outer, outer)),
        ("far", (half + gap, outer, -outer, outer)),
        ("right", (-half - gap, half + gap, -outer, -half - gap)),
        ("left", (-half - gap, half + gap, half + gap, outer)),
    ):
        points = [(cx + x, cy + y, z) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        strips.append(
            _mesh(f"mark_{name}", points, [0, 1, 2, 0, 2, 3], "\n        color3f[] primvars:displayColor = [(0.15, 0.45, 0.22)]")
        )
    body = "\n".join(strips)
    return f"""#usda 1.0
(
    defaultPrim = "marks"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "Visual outline of the board mark, outside the board footprint. No physics and no mating-square marker."
)

def Xform "marks"
{{
{body}
}}
"""
