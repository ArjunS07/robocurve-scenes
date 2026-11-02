"""Author the nut and the bolt fixture as USDA. No Isaac Sim imports.

One approximate M8 x 1.25 station inspired by NIST Assembly Task Board 1, not
the full board and not the NIST CAD.

The thread is physical. The stud is a core cylinder plus a right-hand helical
ridge built from tilted convex boxes. The nut bore carries three small pins,
120 degrees apart and stepped by a third of the pitch, that ride in the groove
between ridge turns. So:

- turning the nut clockwise (seen from above) slides the pins down the ridge
  and the nut descends one pitch per turn; counter-clockwise backs it off;
- the ridge turn above each pin blocks a straight pull-off;
- the lead angle (about 3 deg) is far below the friction angle (atan 0.6), so a
  released nut stays where it is instead of sliding to the seat.

All of this comes from PhysX contacts between ordinary box colliders. Nothing
is welded, toggled, or snapped at runtime, and the scorer does not enforce it.

The fixture is a kinematic rigid body (immovable by contact, no articulation,
no fixed joint). Both parts carry a USD physics material.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from my_package.scoring.config import PACKAGE_ROOT, NutThreadingConfig
from my_package.scoring.math3d import quat_from_axis_angle, quat_mul
from my_package.scoring.scorer import required_travel_m

PLATE_XY_M = (0.08, 0.04)
COLLAR_RADIUS_M = 0.008
BACKPLATE_CENTER_X_M = -0.024
BACKPLATE_HALF_X_M = 0.002
MARK_HALF_X_M = 0.001
# Marks sit half embedded in the backplate face that looks at the stud.
MARK_X_M = BACKPLATE_CENTER_X_M + BACKPLATE_HALF_X_M + MARK_HALF_X_M / 2.0
MARK_COMPLETED = (("mark_100", 1.0), ("mark_75", 0.75), ("mark_50", 0.5))
# Lowest pin's bottom sits this far above the nut's lower face.
PIN_BOTTOM_MARGIN_M = 0.0001
# Ridge boxes reach this far into the core so no seam opens between them.
RIDGE_EMBED_M = 0.0001
# Pins reach this far into the bore wall.
PIN_EMBED_M = 0.0002
# Neighbouring ridge boxes overlap by this fraction of their chord.
RIDGE_OVERLAP = 0.06


def generated_dir() -> Path:
    """Runtime output, never the source tree. The launcher sets ROBOCURVE_GENERATED_DIR."""
    override = os.environ.get("ROBOCURVE_GENERATED_DIR")
    return Path(override) if override else PACKAGE_ROOT / "outputs" / "generated"


@dataclass(frozen=True)
class PrimSpec:
    name: str
    kind: str
    translate: tuple[float, float, float]
    orient_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] | None = None
    radius_m: float | None = None
    height_m: float | None = None
    collision: bool = False
    color: tuple[float, float, float] = (0.7, 0.7, 0.72)
    # Per-collider PhysX contact offset; None uses physics.contact_offset_m.
    contact_offset_m: float | None = None


def mark_z_m(travel_m: float, completed_fraction: float) -> float:
    """Height of the nut's lower face after this fraction of the required travel."""
    return (1.0 - completed_fraction) * travel_m


def _z_quat(angle_rad: float) -> tuple[float, float, float, float]:
    half = 0.5 * angle_rad
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _helix_tilt_quat(math_angle: float, lead_rad: float) -> tuple[float, float, float, float]:
    """Local +X radial at ``math_angle``; local +Y (counter-clockwise tangent) rises by ``lead_rad``."""
    return quat_mul(_z_quat(math_angle), quat_from_axis_angle((1.0, 0.0, 0.0), lead_rad))


def contact_radius_m(cfg: NutThreadingConfig) -> float:
    """Mid radius of the pin/ridge overlap, where the lead angle is matched."""
    return 0.5 * (cfg.fastener.pin_tip_radius_m + cfg.fastener.major_radius_m)


def lead_angle_rad(cfg: NutThreadingConfig) -> float:
    return math.atan2(cfg.fastener.pitch_m, 2.0 * math.pi * contact_radius_m(cfg))


def pin_layout(cfg: NutThreadingConfig) -> list[tuple[float, float]]:
    """(math angle, z of the pin center) in the nut frame, origin at the nut center.

    Pin k sits 120*k degrees clockwise of pin 0 and k/3 of a pitch lower, so all
    three rest on the same helix.
    """
    f = cfg.fastener
    n = f.pin_count
    lowest = -f.nut_height_m / 2.0 + PIN_BOTTOM_MARGIN_M + f.pin_thickness_m / 2.0
    return [(-2.0 * math.pi * k / n, lowest + f.pitch_m * (n - 1 - k) / n) for k in range(n)]


def pin_bottom_above_face_m(cfg: NutThreadingConfig, k: int) -> float:
    """Height of pin k's bottom above the nut's lower face."""
    _angle, z = pin_layout(cfg)[k]
    return z - cfg.fastener.pin_thickness_m / 2.0 + cfg.fastener.nut_height_m / 2.0


def ridge_start_top_z_m(cfg: NutThreadingConfig) -> float:
    """Ridge top at its start (clockwise angle 0). Pin 0 meets it when the lower face is at the travel."""
    return required_travel_m(cfg) + pin_bottom_above_face_m(cfg, 0)


def ridge_top_z_m(cfg: NutThreadingConfig, clockwise_rad: float) -> float:
    """Ridge top surface along the helix, at the contact radius. Descends one pitch per clockwise turn."""
    return ridge_start_top_z_m(cfg) - cfg.fastener.pitch_m * clockwise_rad / (2.0 * math.pi)


def ridge_end_rad(cfg: NutThreadingConfig) -> float:
    """The ridge runs down until its top reaches the collar face (fixture z = 0)."""
    return 2.0 * math.pi * ridge_start_top_z_m(cfg) / cfg.fastener.pitch_m


def thread_turns_available(cfg: NutThreadingConfig) -> float:
    return ridge_end_rad(cfg) / (2.0 * math.pi)


def nut_wall_boxes(cfg: NutThreadingConfig) -> list[PrimSpec]:
    fastener = cfg.fastener
    apothem = fastener.across_flats_m / 2.0
    side = fastener.across_flats_m / math.sqrt(3.0)
    thickness = apothem - fastener.hole_radius_m
    if thickness <= 0.0:
        raise ValueError("nut across-flats is too small for the hole")
    radial = fastener.hole_radius_m + thickness / 2.0
    boxes = []
    for index in range(6):
        angle = index * math.pi / 3.0
        boxes.append(
            PrimSpec(
                name=f"wall_{index}",
                kind="cube",
                translate=(radial * math.cos(angle), radial * math.sin(angle), 0.0),
                orient_xyzw=_z_quat(angle),
                scale=(thickness / 2.0, side / 2.0, fastener.nut_height_m / 2.0),
                collision=True,
                color=(0.75, 0.75, 0.78),
                contact_offset_m=fastener.thread_contact_offset_m,
            )
        )
    return boxes


def nut_pin_boxes(cfg: NutThreadingConfig) -> list[PrimSpec]:
    f = cfg.fastener
    inner = f.pin_tip_radius_m
    outer = f.hole_radius_m + PIN_EMBED_M
    lead = lead_angle_rad(cfg)
    pins = []
    for k, (angle, z) in enumerate(pin_layout(cfg)):
        mid = 0.5 * (inner + outer)
        pins.append(
            PrimSpec(
                name=f"pin_{k}",
                kind="cube",
                translate=(mid * math.cos(angle), mid * math.sin(angle), z),
                orient_xyzw=_helix_tilt_quat(angle, lead),
                scale=((outer - inner) / 2.0, f.pin_width_m / 2.0, f.pin_thickness_m / 2.0),
                collision=True,
                color=(0.55, 0.56, 0.60),
                contact_offset_m=f.thread_contact_offset_m,
            )
        )
    return pins


def nut_prims(cfg: NutThreadingConfig) -> list[PrimSpec]:
    return nut_wall_boxes(cfg) + nut_pin_boxes(cfg)


def ridge_boxes(cfg: NutThreadingConfig) -> list[PrimSpec]:
    f = cfg.fastener
    step = 2.0 * math.pi / f.segments_per_turn
    count = int(math.floor(ridge_end_rad(cfg) / step))
    inner = f.core_radius_m - RIDGE_EMBED_M
    mid = 0.5 * (inner + f.major_radius_m)
    half_chord = f.major_radius_m * math.tan(step / 2.0) * (1.0 + RIDGE_OVERLAP)
    lead = lead_angle_rad(cfg)
    boxes = []
    for j in range(count):
        clockwise = (j + 0.5) * step
        angle = -clockwise
        boxes.append(
            PrimSpec(
                name=f"ridge_{j:03d}",
                kind="cube",
                translate=(
                    mid * math.cos(angle),
                    mid * math.sin(angle),
                    ridge_top_z_m(cfg, clockwise) - f.ridge_thickness_m / 2.0,
                ),
                orient_xyzw=_helix_tilt_quat(angle, lead),
                scale=((f.major_radius_m - inner) / 2.0, half_chord, f.ridge_thickness_m / 2.0),
                collision=True,
                color=(0.72, 0.55, 0.12),
                contact_offset_m=f.thread_contact_offset_m,
            )
        )
    return boxes


def fixture_prims(cfg: NutThreadingConfig) -> list[PrimSpec]:
    fastener = cfg.fastener
    body = cfg.fixture_body
    travel = required_travel_m(cfg)
    stud_height = travel + fastener.lead_in_m
    prims = [
        PrimSpec(
            name="plate",
            kind="cube",
            translate=(0.0, 0.0, -(body.collar_height_m + body.plate_thickness_m / 2.0)),
            scale=(PLATE_XY_M[0] / 2.0, PLATE_XY_M[1] / 2.0, body.plate_thickness_m / 2.0),
            collision=True,
            color=(0.25, 0.27, 0.30),
        ),
        PrimSpec(
            name="collar",
            kind="cylinder",
            translate=(0.0, 0.0, -body.collar_height_m / 2.0),
            radius_m=COLLAR_RADIUS_M,
            height_m=body.collar_height_m,
            collision=True,
            color=(0.35, 0.37, 0.40),
            contact_offset_m=fastener.thread_contact_offset_m,
        ),
        PrimSpec(
            name="stud",
            kind="cylinder",
            translate=(0.0, 0.0, stud_height / 2.0),
            radius_m=fastener.core_radius_m,
            height_m=stud_height,
            collision=True,
            color=(0.62, 0.64, 0.66),
            contact_offset_m=fastener.thread_contact_offset_m,
        ),
        PrimSpec(
            name="backplate",
            kind="cube",
            translate=(BACKPLATE_CENTER_X_M, 0.0, 0.002),
            scale=(BACKPLATE_HALF_X_M, 0.015, 0.015),
            collision=True,
            color=(0.20, 0.22, 0.24),
        ),
    ]
    prims.extend(ridge_boxes(cfg))
    for name, completed in MARK_COMPLETED:
        prims.append(
            PrimSpec(
                name=name,
                kind="cube",
                translate=(MARK_X_M, 0.0, mark_z_m(travel, completed)),
                scale=(MARK_HALF_X_M, 0.008, 0.0004),
                collision=False,
                color=(0.15, 0.75, 0.25) if completed == 1.0 else (0.85, 0.75, 0.20),
            )
        )
    return prims


def _fmt(value: float) -> str:
    return f"{value:.8f}"


def _usd_quat(xyzw) -> str:
    x, y, z, w = xyzw
    return f"({_fmt(w)}, {_fmt(x)}, {_fmt(y)}, {_fmt(z)})"


def _prim_block(prim: PrimSpec, contact_offset_m: float) -> list[str]:
    header = f'    def {prim.kind.capitalize()} "{prim.name}"'
    if prim.collision:
        lines = [header + " (", '        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]', "    ) {"]
    else:
        lines = [header + " {"]
    lines += _prim_body(prim, contact_offset_m)
    return lines


def _prim_body(prim: PrimSpec, contact_offset_m: float) -> list[str]:
    lines = []
    tx, ty, tz = prim.translate
    lines.append(f"        float3 xformOp:translate = ({_fmt(tx)}, {_fmt(ty)}, {_fmt(tz)})")
    order = ["xformOp:translate"]
    if prim.kind == "cube":
        lines.append(f"        quatf xformOp:orient = {_usd_quat(prim.orient_xyzw)}")
        sx, sy, sz = prim.scale
        lines.append(f"        float3 xformOp:scale = ({_fmt(sx)}, {_fmt(sy)}, {_fmt(sz)})")
        order.extend(["xformOp:orient", "xformOp:scale"])
    elif prim.kind == "cylinder":
        lines.append(f"        double radius = {_fmt(prim.radius_m)}")
        lines.append(f"        double height = {_fmt(prim.height_m)}")
        lines.append('        token axis = "Z"')
    elif prim.kind == "sphere":
        lines.append(f"        double radius = {_fmt(prim.radius_m)}")
    else:
        raise ValueError(f"unsupported prim kind {prim.kind}")
    quoted = ", ".join(f'"{item}"' for item in order)
    lines.append(f"        uniform token[] xformOpOrder = [{quoted}]")
    if prim.collision:
        offset = contact_offset_m if prim.contact_offset_m is None else prim.contact_offset_m
        lines.append(f"        float physxCollision:contactOffset = {_fmt(offset)}")
        lines.append("        float physxCollision:restOffset = 0")
    r, g, b = prim.color
    lines.append(f"        color3f[] primvars:displayColor = [({_fmt(r)}, {_fmt(g)}, {_fmt(b)})]")
    lines.append("    }")
    return lines


def _indent(lines: list[str], spaces: int) -> list[str]:
    pad = " " * spaces
    return [pad + line if line else line for line in lines]


def _material_block(cfg: NutThreadingConfig) -> list[str]:
    fastener = cfg.fastener
    return [
        '    def Material "physics_material" (',
        '        prepend apiSchemas = ["PhysicsMaterialAPI"]',
        "    ) {",
        f"        float physics:staticFriction = {_fmt(fastener.static_friction)}",
        f"        float physics:dynamicFriction = {_fmt(fastener.dynamic_friction)}",
        "        float physics:restitution = 0",
        "    }",
    ]


def _rigid_body_lines(mass_kg: float, material_path: str, cfg: NutThreadingConfig, kinematic: bool = False) -> list[str]:
    physics = cfg.physics
    return [
        "    bool physics:rigidBodyEnabled = true",
        f"    bool physics:kinematicEnabled = {'true' if kinematic else 'false'}",
        f"    float physics:mass = {_fmt(mass_kg)}",
        f"    int physxRigidBody:solverPositionIterationCount = {physics.solver_position_iterations}",
        f"    int physxRigidBody:solverVelocityIterationCount = {physics.solver_velocity_iterations}",
        f"    rel material:binding:physics = <{material_path}>",
    ]


_RIGID_SCHEMAS = '["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI", "MaterialBindingAPI"]'


def _header(root_name: str) -> list[str]:
    return [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{root_name}"',
        "    metersPerUnit = 1",
        "    upAxis = \"Z\"",
        "    kilogramsPerUnit = 1",
        ")",
        "",
    ]


def build_usda(root_name: str, prims: list[PrimSpec], mass_kg: float, cfg: NutThreadingConfig, kinematic: bool = False) -> str:
    """A single rigid body with its collision prims and material. ``kinematic`` holds it in place."""
    lines = _header(root_name) + [
        f'def Xform "{root_name}" (',
        f"    prepend apiSchemas = {_RIGID_SCHEMAS}",
        ") {",
    ]
    lines += _rigid_body_lines(mass_kg, f"/{root_name}/physics_material", cfg, kinematic=kinematic)
    lines += _material_block(cfg)
    for prim in prims:
        lines.extend(_prim_block(prim, cfg.physics.contact_offset_m))
    lines += ["}", ""]
    return "\n".join(lines)


def nut_usda(cfg: NutThreadingConfig) -> str:
    return build_usda("Nut", nut_prims(cfg), cfg.fastener.nut_mass_kg, cfg)


def fixture_usda(cfg: NutThreadingConfig) -> str:
    """Kinematic: the secured fixture cannot be pushed, and there is no articulation to fix."""
    return build_usda("Fixture", fixture_prims(cfg), cfg.fixture_body.mass_kg, cfg, kinematic=True)


def ensure_generated_assets(cfg: NutThreadingConfig, directory: Path | None = None) -> tuple[Path, Path]:
    target = generated_dir() if directory is None else Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    nut_path = target / "nut.usda"
    fixture_path = target / "fixture.usda"
    nut_path.write_text(nut_usda(cfg), encoding="utf-8")
    fixture_path.write_text(fixture_usda(cfg), encoding="utf-8")
    return nut_path, fixture_path
