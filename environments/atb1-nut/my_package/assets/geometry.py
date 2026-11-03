"""Author the fixture and the Factory nut/bolt wrappers as USDA. No Isaac Sim imports.

An M16 nut-and-bolt station inspired by NIST Assembly Task Board 1, using Isaac
Lab's Factory M16 nut and bolt (NVIDIA assets, loaded at runtime, not vendored).
Their collision meshes carry real right-hand 2.0 mm threads with SDF collision,
so turning the nut screws it down the bolt through PhysX contacts alone.

The bolt stands on the table with its hex head in a pocket of a kinematic
fixture block. The block's upper plate has a clearance hole around the shank and
its top face is the seat; no fixture collider touches the bolt. The nut starts
lying flat on a thin start mat, also part of the kinematic fixture.

The Factory assets are wrapped, not edited: each wrapper references the cached
USD and authors overs. The nut and bolt become plain rigid bodies (their
articulation roots are removed and the bolt's world fixed joint is deactivated);
the bolt is kinematic, the same pattern as the fixture, so nothing needs
fix_root_link. Solver, contact and friction settings follow Isaac Lab's Factory task.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from my_package.scoring.config import PACKAGE_ROOT, NutThreadingConfig, bolt_position_xyz, start_mat_top_z

# Prim paths inside the Factory USDs (measured with usd-core): the rigid body and its SDF mesh.
NUT_BODY = "factory_nut_loose"
BOLT_BODY = "factory_bolt_loose"
BOLT_ROOT_JOINT = "root_joint"
COLLISION_MESH = "collisions"


def generated_dir() -> Path:
    """Runtime output, never the source tree. The launcher sets ROBOCURVE_GENERATED_DIR."""
    override = os.environ.get("ROBOCURVE_GENERATED_DIR")
    return Path(override) if override else PACKAGE_ROOT / "outputs" / "generated"


@dataclass(frozen=True)
class BoxSpec:
    """An axis-aligned collision box in the fixture frame (origin at the seat face)."""

    name: str
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    color: tuple[float, float, float] = (0.25, 0.27, 0.30)

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple(0.5 * (a + b) for a, b in zip(self.lower, self.upper, strict=True))

    @property
    def half(self) -> tuple[float, float, float]:
        return tuple(0.5 * (b - a) for a, b in zip(self.lower, self.upper, strict=True))


def _frame_boxes(prefix: str, inner_half: float, outer_half_xy, z_low: float, z_high: float, color) -> list[BoxSpec]:
    """Four boxes around a square opening of half width ``inner_half``."""
    ox, oy = outer_half_xy
    return [
        BoxSpec(f"{prefix}_pos_x", (inner_half, -oy, z_low), (ox, oy, z_high), color),
        BoxSpec(f"{prefix}_neg_x", (-ox, -oy, z_low), (-inner_half, oy, z_high), color),
        BoxSpec(f"{prefix}_pos_y", (-inner_half, inner_half, z_low), (inner_half, oy, z_high), color),
        BoxSpec(f"{prefix}_neg_y", (-inner_half, -oy, z_low), (inner_half, -inner_half, z_high), color),
    ]


def fixture_boxes(cfg: NutThreadingConfig) -> list[BoxSpec]:
    """Plate with a shank hole, head pocket layer below it, and the start mat. Fixture frame."""
    body = cfg.fixture_body
    head = cfg.fastener.bolt_head_height_m
    plate = body.plate_thickness_m
    boxes = _frame_boxes("plate", body.hole_half_m, body.block_half_xy_m, -plate, 0.0, (0.25, 0.27, 0.30))
    boxes += _frame_boxes("pocket", body.pocket_half_m, body.block_half_xy_m, -plate - head, -plate, (0.20, 0.22, 0.24))
    fx, fy, fz = cfg.fixture_pose.position_xyz
    mx = cfg.nut_pose.position_xyz[0] - fx
    my = cfg.nut_pose.position_xyz[1] - fy
    hx, hy = cfg.start_mat.half_xy_m
    mat_top = start_mat_top_z(cfg) - fz
    boxes.append(
        BoxSpec("start_mat", (mx - hx, my - hy, mat_top - cfg.start_mat.thickness_m), (mx + hx, my + hy, mat_top), (0.15, 0.35, 0.55))
    )
    return boxes


def fixture_boxes_world(cfg: NutThreadingConfig) -> list[BoxSpec]:
    """The fixture boxes at the fixture pose (identity rotation, checked by the config)."""
    offset = cfg.fixture_pose.position_xyz
    return [
        BoxSpec(
            box.name,
            tuple(a + o for a, o in zip(box.lower, offset, strict=True)),
            tuple(b + o for b, o in zip(box.upper, offset, strict=True)),
            box.color,
        )
        for box in fixture_boxes(cfg)
    ]


def bolt_envelope_world(cfg: NutThreadingConfig) -> list[tuple[str, float, float, float]]:
    """(name, radius, z_low, z_high) cylinders that bound the bolt's mesh, in world z."""
    fastener = cfg.fastener
    _, _, z0 = bolt_position_xyz(cfg)
    head_top = z0 + fastener.bolt_head_height_m
    return [
        ("head", fastener.bolt_head_corner_radius_m, z0, head_top),
        ("shank", fastener.shank_major_radius_m, head_top, z0 + fastener.bolt_length_m),
    ]


def _fmt(value: float) -> str:
    return f"{value:.8f}"


def _header(root_name: str) -> list[str]:
    return [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{root_name}"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        "    kilogramsPerUnit = 1",
        ")",
        "",
    ]


def _material_block(static_friction: float, dynamic_friction: float, indent: str = "    ") -> list[str]:
    return [
        f'{indent}def Material "physics_material" (',
        f'{indent}    prepend apiSchemas = ["PhysicsMaterialAPI"]',
        f"{indent}) {{",
        f"{indent}    float physics:staticFriction = {_fmt(static_friction)}",
        f"{indent}    float physics:dynamicFriction = {_fmt(dynamic_friction)}",
        f"{indent}    float physics:restitution = 0",
        f"{indent}}}",
    ]


def _box_block(box: BoxSpec, contact_offset_m: float, material_path: str) -> list[str]:
    cx, cy, cz = box.center
    sx, sy, sz = box.half
    r, g, b = box.color
    return [
        f'    def Cube "{box.name}" (',
        '        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]',
        "    ) {",
        f"        float3 xformOp:translate = ({_fmt(cx)}, {_fmt(cy)}, {_fmt(cz)})",
        f"        float3 xformOp:scale = ({_fmt(sx)}, {_fmt(sy)}, {_fmt(sz)})",
        '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
        f"        float physxCollision:contactOffset = {_fmt(contact_offset_m)}",
        "        float physxCollision:restOffset = 0",
        f"        rel material:binding:physics = <{material_path}>",
        f"        color3f[] primvars:displayColor = [({_fmt(r)}, {_fmt(g)}, {_fmt(b)})]",
        "    }",
    ]


def fixture_usda(cfg: NutThreadingConfig) -> str:
    """Kinematic: the secured fixture cannot be pushed, and there is no articulation to fix."""
    body = cfg.fixture_body
    physics = cfg.physics
    material = "/Fixture/physics_material"
    lines = _header("Fixture") + [
        'def Xform "Fixture" (',
        '    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]',
        ") {",
        "    bool physics:rigidBodyEnabled = true",
        "    bool physics:kinematicEnabled = true",
        f"    float physics:mass = {_fmt(body.mass_kg)}",
        f"    int physxRigidBody:solverPositionIterationCount = {physics.fixture_solver_position_iterations}",
        f"    int physxRigidBody:solverVelocityIterationCount = {physics.fixture_solver_velocity_iterations}",
    ]
    lines += _material_block(body.static_friction, body.dynamic_friction)
    for box in fixture_boxes(cfg):
        lines += _box_block(box, physics.contact_offset_m, material)
    lines += ["}", ""]
    return "\n".join(lines)


def _fastener_wrapper(cfg: NutThreadingConfig, root_name: str, body: str, asset_path: str, kinematic: bool) -> list[str]:
    physics = cfg.physics
    material = f"/{root_name}/physics_material"
    lines = _header(root_name) + [
        f'def Xform "{root_name}" (',
        f"    prepend references = @{asset_path}@",
        ") {",
        f'    over "{body}" (',
        # A RigidObject must not contain an articulation root.
        '        delete apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]',
        '        prepend apiSchemas = ["PhysxRigidBodyAPI"]',
        "    ) {",
        "        bool physics:rigidBodyEnabled = true",
        f"        bool physics:kinematicEnabled = {'true' if kinematic else 'false'}",
        f"        int physxRigidBody:solverPositionIterationCount = {physics.fastener_solver_position_iterations}",
        f"        int physxRigidBody:solverVelocityIterationCount = {physics.fastener_solver_velocity_iterations}",
        f"        float physxRigidBody:maxDepenetrationVelocity = {_fmt(physics.fastener_max_depenetration_velocity_m_s)}",
        f"        bool physxRigidBody:enableGyroscopicForces = {'true' if physics.fastener_enable_gyroscopic_forces else 'false'}",
        f"        float physxRigidBody:maxContactImpulse = {physics.fastener_max_contact_impulse:.6g}",
        f'        over "{COLLISION_MESH}" (',
        '            prepend apiSchemas = ["PhysxCollisionAPI", "MaterialBindingAPI"]',
        "        ) {",
        f"            float physxCollision:contactOffset = {_fmt(physics.fastener_contact_offset_m)}",
        f"            float physxCollision:restOffset = {_fmt(physics.fastener_rest_offset_m)}",
        f"            rel material:binding:physics = <{material}>",
        "        }",
        "    }",
    ]
    lines += _material_block(physics.fastener_friction, physics.fastener_friction)
    return lines


def nut_usda(cfg: NutThreadingConfig, asset_path: str) -> str:
    """The Factory nut as a free rigid body."""
    lines = _fastener_wrapper(cfg, "Nut", NUT_BODY, asset_path, kinematic=False)
    return "\n".join(lines + ["}", ""])


def bolt_usda(cfg: NutThreadingConfig, asset_path: str) -> str:
    """The Factory bolt as a kinematic body. Its world fixed joint would make it an articulation."""
    lines = _fastener_wrapper(cfg, "Bolt", BOLT_BODY, asset_path, kinematic=True)
    lines += [f'    over "{BOLT_ROOT_JOINT}" (', "        active = false", "    ) {", "    }"]
    return "\n".join(lines + ["}", ""])


def ensure_generated_assets(
    cfg: NutThreadingConfig, nut_asset: str, bolt_asset: str, directory: Path | None = None
) -> tuple[Path, Path, Path]:
    """Write the nut and bolt wrappers (around local copies of the Factory USDs) and the fixture."""
    target = generated_dir() if directory is None else Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    paths = (target / "nut.usda", target / "bolt.usda", target / "fixture.usda")
    paths[0].write_text(nut_usda(cfg, nut_asset), encoding="utf-8")
    paths[1].write_text(bolt_usda(cfg, bolt_asset), encoding="utf-8")
    paths[2].write_text(fixture_usda(cfg), encoding="utf-8")
    return paths
