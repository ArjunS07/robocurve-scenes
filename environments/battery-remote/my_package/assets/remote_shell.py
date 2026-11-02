"""Visual meshes for the face-down remote. No Isaac Sim imports.

The colliders stay simple boxes (see ``geometry.static_boxes``) and are authored
with ``purpose = "guide"`` so they do not render. Everything here is visual:

- ``shell``: a rounded, tapered wand. Plan view narrows from the IR nose to the
  bay, with rounded ends; the top edges carry a fillet. The battery slots are
  cut out of the top, and their walls and floors are part of the same mesh.
- ``contacts``: the flat positive contact plate of each slot (the slot's end
  wall at the fixed contact face).
- ``marks``: moulded + and - symbols on each slot floor and on the spine.
- ``coil``: a conical coil spring that rides on each plunger link.

Faces never coincide: the slot walls are the hole boundaries of the top
surface, the contact plate replaces one slot wall instead of overlaying it, and
the raised marks start below the surface they sit on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Point = tuple[float, float, float]


@dataclass
class MeshData:
    points: list[Point] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)

    def add_point(self, point: Point) -> int:
        self.points.append(point)
        return len(self.points) - 1

    def quad(self, a: int, b: int, c: int, d: int) -> None:
        """Counter-clockwise a, b, c, d as seen from the side the face points to."""
        self.triangles.append((a, b, c))
        self.triangles.append((a, c, d))

    def wall(self, start: tuple[float, float], end: tuple[float, float], z_top: float, z_bottom: float) -> None:
        """Vertical quad from start to end. It faces left of the start->end direction."""
        a = self.add_point((start[0], start[1], z_top))
        b = self.add_point((end[0], end[1], z_top))
        c = self.add_point((end[0], end[1], z_bottom))
        d = self.add_point((start[0], start[1], z_bottom))
        # (b - a) x (c - a) points left of travel when +Z is up.
        self.quad(a, b, c, d)

    def box(self, center: Point, size: Point) -> None:
        cx, cy, cz = center
        hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
        base = len(self.points)
        for dz in (-hz, hz):
            for dy in (-hy, hy):
                for dx in (-hx, hx):
                    self.points.append((cx + dx, cy + dy, cz + dz))
        # Corner index = x + 2y + 4z.
        for quad in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
            self.quad(*(base + index for index in quad))


@dataclass(frozen=True)
class ShellSpec:
    """Shell dimensions in the remote frame (+X toward the handle end, +Z up, origin mid-height)."""

    nose_x: float
    front_x: float
    nose_half_width: float
    bay_half_width: float
    half_height: float
    edge_fillet: float
    end_radius: float
    floor_z: float
    taper_end_x: float


@dataclass(frozen=True)
class SlotHole:
    """Open part of one slot: from the fixed contact face to the bay end at the spring."""

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    contact_at_max_x: bool


def plan_half_width(spec: ShellSpec, x: float) -> float:
    """Half width in plan: linear taper from the nose, then constant, with rounded ends."""
    if x <= spec.taper_end_x:
        t = (x - spec.nose_x) / (spec.taper_end_x - spec.nose_x)
        width = spec.nose_half_width + (spec.bay_half_width - spec.nose_half_width) * t
    else:
        width = spec.bay_half_width
    r = spec.end_radius
    for distance in (x - spec.nose_x, spec.front_x - x):
        if distance < r:
            inset = r - distance
            width = min(width, width - r + math.sqrt(max(0.0, r * r - inset * inset)))
    return width


def _fillet(distance_to_edge: float, radius: float, top: float) -> float:
    """Height of a quarter-round fillet at ``distance_to_edge`` inward from the outer edge."""
    if distance_to_edge >= radius:
        return top
    inset = radius - max(0.0, distance_to_edge)
    return top - radius + math.sqrt(max(0.0, radius * radius - inset * inset))


def top_height(spec: ShellSpec, x: float, y: float) -> float:
    width = plan_half_width(spec, x)
    across = _fillet(width - abs(y), spec.edge_fillet, spec.half_height)
    along = _fillet(min(x - spec.nose_x, spec.front_x - x), spec.edge_fillet, spec.half_height)
    return min(across, along)


def _stations(start: float, stop: float, step: float) -> list[float]:
    count = max(1, int(math.ceil((stop - start) / step)))
    return [start + (stop - start) * index / count for index in range(count + 1)]


def _unique(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in sorted(values):
        if not out or value - out[-1] > 1e-7:
            out.append(value)
    return out


def x_stations(spec: ShellSpec, holes: tuple[SlotHole, ...]) -> list[float]:
    ends = []
    for k in range(9):
        # Cosine spacing packs stations into the rounded ends.
        f = 1.0 - math.cos(0.5 * math.pi * k / 8)
        ends.append(spec.nose_x + spec.end_radius * f)
        ends.append(spec.front_x - spec.end_radius * f)
    body = _stations(spec.nose_x + spec.end_radius, spec.front_x - spec.end_radius, 0.008)
    special = [spec.taper_end_x] + [hole.x_min for hole in holes] + [hole.x_max for hole in holes]
    return _unique(ends + body + special)


def u_stations(spec: ShellSpec, holes: tuple[SlotHole, ...]) -> list[float]:
    """Fractions of the local half width. Hole edges land on grid lines where the width is constant."""
    w = spec.bay_half_width
    fillet = 1.0 - spec.edge_fillet / w
    values = [0.0, 1.0, -1.0, fillet, -fillet]
    for k in range(1, 6):
        f = fillet + (1.0 - fillet) * (1.0 - math.cos(0.5 * math.pi * k / 6))
        values += [f, -f]
    for hole in holes:
        values += [hole.y_min / w, hole.y_max / w]
    values += [k / 10.0 for k in range(-9, 10)]
    return _unique([max(-1.0, min(1.0, value)) for value in values])


def _in_hole(holes: tuple[SlotHole, ...], x: float, y: float) -> SlotHole | None:
    for hole in holes:
        if hole.x_min < x < hole.x_max and hole.y_min < y < hole.y_max:
            return hole
    return None


def build_shell(spec: ShellSpec, holes: tuple[SlotHole, ...]) -> tuple[MeshData, MeshData]:
    """Shell (black) and the positive contact plates (metal)."""
    for hole in holes:
        if hole.x_min <= spec.taper_end_x:
            raise ValueError("slot holes must lie where the plan width is constant")
        if max(abs(hole.y_min), abs(hole.y_max)) >= spec.bay_half_width - spec.edge_fillet:
            raise ValueError("slot holes must stay inside the flat part of the top")
    xs = x_stations(spec, holes)
    us = u_stations(spec, holes)
    shell = MeshData()
    contacts = MeshData()

    # Top surface on an (x, u) grid; y = u * half_width(x).
    grid: dict[tuple[int, int], int] = {}
    for i, x in enumerate(xs):
        width = plan_half_width(spec, x)
        for j, u in enumerate(us):
            y = u * width
            grid[(i, j)] = shell.add_point((x, y, top_height(spec, x, y)))
    for i in range(len(xs) - 1):
        for j in range(len(us) - 1):
            cx = 0.5 * (xs[i] + xs[i + 1])
            cy = 0.5 * (us[j] + us[j + 1]) * plan_half_width(spec, cx)
            if _in_hole(holes, cx, cy):
                continue
            shell.quad(grid[(i, j)], grid[(i + 1, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)])

    # Outer side walls, outward facing: walk the outline clockwise seen from above.
    bottom = -spec.half_height
    outline: list[tuple[float, float, float]] = []
    for i, x in enumerate(xs):
        outline.append((x, -plan_half_width(spec, x), top_height(spec, x, -plan_half_width(spec, x))))
    for j, u in enumerate(us[1:], start=1):
        x = xs[-1]
        y = u * plan_half_width(spec, x)
        outline.append((x, y, top_height(spec, x, y)))
    for x in reversed(xs[:-1]):
        y = plan_half_width(spec, x)
        outline.append((x, y, top_height(spec, x, y)))
    for u in reversed(us[1:-1]):
        x = xs[0]
        y = u * plan_half_width(spec, x)
        outline.append((x, y, top_height(spec, x, y)))
    count = len(outline)
    for k in range(count):
        a = outline[k]
        b = outline[(k + 1) % count]
        # Counter-clockwise outline; the wall must face right of travel (outward), so reverse.
        top_a = shell.add_point(a)
        top_b = shell.add_point(b)
        bot_b = shell.add_point((b[0], b[1], bottom))
        bot_a = shell.add_point((a[0], a[1], bottom))
        shell.quad(top_b, top_a, bot_a, bot_b)

    # Flat bottom (the button face, on the table), facing down.
    bottom_ids = {}
    for i, x in enumerate(xs):
        width = plan_half_width(spec, x)
        for j, u in enumerate(us):
            bottom_ids[(i, j)] = shell.add_point((x, u * width, bottom))
    for i in range(len(xs) - 1):
        for j in range(len(us) - 1):
            shell.quad(bottom_ids[(i, j)], bottom_ids[(i, j + 1)], bottom_ids[(i + 1, j + 1)], bottom_ids[(i + 1, j)])

    # Slot walls face into the hole: walk each hole counter-clockwise. Floors face up.
    top = spec.half_height
    for hole in holes:
        corners = [(hole.x_min, hole.y_min), (hole.x_max, hole.y_min), (hole.x_max, hole.y_max), (hole.x_min, hole.y_max)]
        for k in range(4):
            start, end = corners[k], corners[(k + 1) % 4]
            contact_wall = start[0] == end[0] and (start[0] == hole.x_max) == hole.contact_at_max_x
            (contacts if contact_wall else shell).wall(start, end, top, spec.floor_z)
        floor = [shell.add_point((x, y, spec.floor_z)) for x, y in corners]
        shell.quad(*floor)
    return shell, contacts


def polarity_marks(
    holes: tuple[SlotHole, ...],
    floor_z: float,
    top_z: float,
    plus_x: dict[str, float],
    minus_x: dict[str, float],
    spine_marks: list[tuple[float, float, bool]],
) -> MeshData:
    """Raised + and - symbols. Each bar starts 0.1 mm below its surface and stands 0.3 mm proud."""
    marks = MeshData()
    bar_long, bar_short, relief = 0.004, 0.0008, 0.0003

    def symbol(x: float, y: float, z_surface: float, plus: bool, size: float = 1.0) -> None:
        z = z_surface + (relief - 0.0001) / 2.0
        height = relief + 0.0001
        marks.box((x, y, z), (bar_long * size, bar_short * size, height))
        if plus:
            marks.box((x, y, z), (bar_short * size, bar_long * size, height))

    for hole in holes:
        y = 0.5 * (hole.y_min + hole.y_max)
        symbol(plus_x[hole.name], y, floor_z, True)
        symbol(minus_x[hole.name], y, floor_z, False)
    for x, y, plus in spine_marks:
        symbol(x, y, top_z, plus, size=0.85)
    return marks


def coil_spring(
    x_start: float,
    x_end: float,
    radius_start: float,
    radius_end: float,
    wire_radius: float,
    turns: float,
    segments_per_turn: int = 20,
    sides: int = 6,
) -> MeshData:
    """Conical helix tube along X, centred on y = z = 0 of its parent frame."""
    mesh = MeshData()
    total = max(2, int(turns * segments_per_turn))
    rings = []
    for k in range(total + 1):
        t = k / total
        angle = 2.0 * math.pi * turns * t
        x = x_start + (x_end - x_start) * t
        r = radius_start + (radius_end - radius_start) * t
        center = (x, r * math.cos(angle), r * math.sin(angle))
        # Tube cross-section in the plane spanned by the radial direction and X.
        radial = (0.0, math.cos(angle), math.sin(angle))
        ring = []
        for s in range(sides):
            phi = 2.0 * math.pi * s / sides
            ring.append(
                mesh.add_point(
                    (
                        center[0] + wire_radius * math.sin(phi),
                        center[1] + wire_radius * math.cos(phi) * radial[1],
                        center[2] + wire_radius * math.cos(phi) * radial[2],
                    )
                )
            )
        rings.append(ring)
    for k in range(total):
        for s in range(sides):
            a, b = rings[k][s], rings[k][(s + 1) % sides]
            c, d = rings[k + 1][(s + 1) % sides], rings[k + 1][s]
            mesh.quad(a, b, c, d)
    return mesh
