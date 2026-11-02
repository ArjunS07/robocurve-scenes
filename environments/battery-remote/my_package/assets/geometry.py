"""Procedural remote, battery, and table-mark geometry. No Isaac Sim imports.

The battery cover is removed, so the bay is open from above. Each slot is an
open-top trough: floor, side walls, a fixed contact at the cell's positive end,
and a spring plunger at its negative end. A cell is inserted from above by
pressing its negative end into the plunger, then lowering the positive end
onto the floor. The plunger is a prismatic joint whose drive pushes the seated
cell against the fixed contact. Nothing welds the cell to the remote and nothing
teleports it into the slot. Retention after release comes from gravity, the
walls, and the spring preload.

The red nub is visual only. Collision is the cylinder body, so the nub does
not fight the plunger or the fixed contact. Polarity is the cylinder axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from my_package.assets.remote_shell import MeshData, ShellSpec, SlotHole, build_shell, coil_spring, plan_half_width
from my_package.assets.remote_shell import polarity_marks as shell_polarity_marks
from my_package.scoring.config import (
    BatteryInsertionConfig,
    SlotSpec,
    active_slots,
    battery_pose,
    cell_names,
    plunger_joint_name,
    plunger_link_name,
)
from my_package.scoring.math3d import add, clamp, norm, normalize, scale, sub


@dataclass(frozen=True)
class Box:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    collision: bool
    color: tuple[float, float, float]


@dataclass(frozen=True)
class BatteryMeasure:
    """One cell measured against one slot."""

    axial_offset_m: float
    radial_offset_m: float
    positive_axis_xyz: tuple[float, float, float]
    polarity_angle_deg: float
    spring_contact: bool | None
    local_position_xyz: tuple[float, float, float]


def _box(name: str, bounds: tuple[float, float, float, float, float, float], color: tuple[float, float, float], collision: bool = True) -> Box:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    if xmax <= xmin or ymax <= ymin or zmax <= zmin:
        raise ValueError(f"{name} has non-positive size {bounds}")
    center = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)
    size = (xmax - xmin, ymax - ymin, zmax - zmin)
    return Box(name=name, center=center, size=size, collision=collision, color=color)


def _aabb(box: Box) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    half = scale(box.size, 0.5)
    return sub(box.center, half), add(box.center, half)


def slot_y_bounds(slot: SlotSpec, channel_y_half: float) -> tuple[float, float]:
    return slot.center_y_m - channel_y_half, slot.center_y_m + channel_y_half


def spring_side(slot: SlotSpec) -> float:
    """+1 when the spring is at the slot's +X end. The spring sits at the negative terminal."""
    return -math.copysign(1.0, slot.required_axis_xyz[0])


def rest_center_z(cfg: BatteryInsertionConfig) -> float:
    """Center height of a cell sitting on the slot floor."""
    return cfg.compartment.floor_z_m + cfg.battery.diameter_m / 2.0


def bay_half_length(cfg: BatteryInsertionConfig) -> float:
    """Half the bay length: seated cell, preload, plunger, travel, and end gap."""
    compartment = cfg.compartment
    return (
        cfg.battery.length_m / 2.0
        - compartment.spring_preload_m
        + compartment.plunger_size_xyz[0]
        + compartment.plunger_travel_m
        + compartment.plunger_end_gap_m
    )


def bay_x_bounds(cfg: BatteryInsertionConfig) -> tuple[float, float]:
    half = bay_half_length(cfg)
    return cfg.compartment.seat_center_x - half, cfg.compartment.seat_center_x + half


def fixed_contact_x(cfg: BatteryInsertionConfig, slot: SlotSpec) -> float:
    """Face of the fixed contact. A seated cell's positive end touches it."""
    return cfg.compartment.seat_center_x - spring_side(slot) * cfg.battery.length_m / 2.0


def static_boxes(cfg: BatteryInsertionConfig) -> tuple[Box, ...]:
    """Chassis colliders. There is no ceiling: the cover is off."""
    compartment = cfg.compartment
    outer_y = compartment.size_xyz[1] / 2.0
    outer_z = compartment.size_xyz[2] / 2.0
    x_rear = compartment.rear_face_x
    x_front = compartment.front_face_x
    bay_min, bay_max = bay_x_bounds(cfg)
    floor_z = compartment.floor_z_m
    slots = active_slots(cfg)
    body = (0.06, 0.06, 0.07)
    contact_color = (0.62, 0.62, 0.58)
    nose = compartment.front_face_x - compartment.silhouette_length_m
    boxes: list[Box] = [
        _box("nose", (nose, x_rear, -outer_y, outer_y, -outer_z, outer_z), body),
        _box("back_cap", (x_rear, bay_min, -outer_y, outer_y, -outer_z, outer_z), body),
        _box("front_cap", (bay_max, x_front, -outer_y, outer_y, -outer_z, outer_z), body),
        _box("floor", (bay_min, bay_max, -outer_y, outer_y, -outer_z, floor_z), body),
    ]
    first_min, first_max = slot_y_bounds(slots[0], compartment.channel_y_half)
    boxes.append(_box("rail_left", (bay_min, bay_max, -outer_y, first_min, floor_z, outer_z), body))
    if len(slots) == 1:
        boxes.append(_box("rail_right", (bay_min, bay_max, first_max, outer_y, floor_z, outer_z), body))
    else:
        second_min, second_max = slot_y_bounds(slots[1], compartment.channel_y_half)
        boxes.append(_box("spine", (bay_min, bay_max, first_max, second_min, floor_z, outer_z), body))
        boxes.append(_box("rail_right", (bay_min, bay_max, second_max, outer_y, floor_z, outer_z), body))
    for slot in slots:
        ymin, ymax = slot_y_bounds(slot, compartment.channel_y_half)
        face = fixed_contact_x(cfg, slot)
        if spring_side(slot) > 0.0:
            bounds = (bay_min, face, ymin, ymax, floor_z, outer_z)
        else:
            bounds = (face, bay_max, ymin, ymax, floor_z, outer_z)
        boxes.append(_box(f"{slot.name}_fixed_contact", bounds, contact_color))
    return tuple(boxes)


def shell_spec(cfg: BatteryInsertionConfig) -> ShellSpec:
    compartment = cfg.compartment
    return ShellSpec(
        nose_x=compartment.front_face_x - compartment.silhouette_length_m,
        front_x=compartment.front_face_x,
        nose_half_width=compartment.shell_nose_half_width_m,
        bay_half_width=compartment.size_xyz[1] / 2.0,
        half_height=compartment.size_xyz[2] / 2.0,
        edge_fillet=compartment.shell_edge_fillet_m,
        end_radius=compartment.shell_end_radius_m,
        floor_z=compartment.floor_z_m,
        taper_end_x=compartment.rear_face_x,
    )


def slot_holes(cfg: BatteryInsertionConfig) -> tuple[SlotHole, ...]:
    """Open length of each slot: from the fixed + contact face to the bay end at the spring."""
    bay_min, bay_max = bay_x_bounds(cfg)
    holes = []
    for slot in active_slots(cfg):
        face = fixed_contact_x(cfg, slot)
        ymin, ymax = slot_y_bounds(slot, cfg.compartment.channel_y_half)
        if spring_side(slot) > 0.0:
            holes.append(SlotHole(slot.name, face, bay_max, ymin, ymax, contact_at_max_x=False))
        else:
            holes.append(SlotHole(slot.name, bay_min, face, ymin, ymax, contact_at_max_x=True))
    return tuple(holes)


def visual_meshes(cfg: BatteryInsertionConfig) -> dict[str, MeshData]:
    """Shell, + contact plates, and moulded polarity marks, in the remote frame."""
    spec = shell_spec(cfg)
    holes = slot_holes(cfg)
    shell, contacts = build_shell(spec, holes)
    plus_x, minus_x, spine = {}, {}, []
    for slot in active_slots(cfg):
        side = spring_side(slot)
        face = fixed_contact_x(cfg, slot)
        plus_x[slot.name] = face + side * 0.007
        minus_x[slot.name] = cfg.compartment.seat_center_x + side * 0.012
        # On the spine between the slots, beside each slot's ends, so the marks stay visible
        # with a cell seated. Offset toward the slot they describe.
        y = 0.003 if slot.center_y_m > 0.0 else -0.003
        spine.append((face + side * 0.004, y, True))
        spine.append((cfg.compartment.seat_center_x + side * 0.018, y, False))
    marks = shell_polarity_marks(holes, cfg.compartment.floor_z_m, spec.half_height, plus_x, minus_x, spine)
    return {"shell": shell, "contacts": contacts, "marks": marks}


def end_collider_outlines(cfg: BatteryInsertionConfig) -> dict[str, list[tuple[float, float]]]:
    """Plan outlines of the convex nose and handle-end colliders (they follow the shell)."""
    spec = shell_spec(cfg)
    bay_min, bay_max = bay_x_bounds(cfg)

    def outline(x0: float, x1: float, rounded_at_start: bool) -> list[tuple[float, float]]:
        r = spec.end_radius
        if rounded_at_start:
            xs = [x0 + r * (1.0 - math.cos(0.5 * math.pi * k / 8)) for k in range(9)] + [spec.taper_end_x, x1]
        else:
            xs = [x0] + [x1 - r * (1.0 - math.cos(0.5 * math.pi * k / 8)) for k in range(8, -1, -1)]
        xs = sorted(set(round(x, 9) for x in xs if x0 - 1e-12 <= x <= x1 + 1e-12))
        lower = [(x, -plan_half_width(spec, x)) for x in xs]
        upper = [(x, plan_half_width(spec, x)) for x in reversed(xs)]
        return lower + upper

    return {
        "nose_collider": outline(spec.nose_x, bay_min, True),
        "handle_collider": outline(bay_max, spec.front_x, False),
    }


def plunger_box(cfg: BatteryInsertionConfig, slot: SlotSpec, joint_pos_m: float = 0.0) -> Box:
    """Plunger pad at a prismatic joint position. Zero is extended; positive compresses it."""
    compartment = cfg.compartment
    size = compartment.plunger_size_xyz
    side = spring_side(slot)
    face = compartment.seat_center_x + side * (cfg.battery.length_m / 2.0 - compartment.spring_preload_m)
    center_x = face + side * (size[0] / 2.0 + joint_pos_m)
    return Box(
        name=f"{slot.name}_plunger",
        center=(center_x, slot.center_y_m, rest_center_z(cfg)),
        size=size,
        collision=True,
        color=(0.55, 0.57, 0.6),
    )


def aabb_gap(
    a_min: tuple[float, float, float],
    a_max: tuple[float, float, float],
    b_min: tuple[float, float, float],
    b_max: tuple[float, float, float],
) -> float:
    """Positive separation, or negative penetration, between two AABBs."""
    gaps = []
    separated = False
    for index in range(3):
        if a_max[index] < b_min[index]:
            gaps.append(b_min[index] - a_max[index])
            separated = True
        elif b_max[index] < a_min[index]:
            gaps.append(a_min[index] - b_max[index])
            separated = True
        else:
            gaps.append(0.0)
    if not separated:
        overlap = [
            min(a_max[index], b_max[index]) - max(a_min[index], b_min[index]) for index in range(3)
        ]
        return -min(overlap)
    positive = [gap for gap in gaps if gap > 0.0]
    return math.sqrt(sum(gap * gap for gap in positive))


def _circle_rect_signed(cx: float, cy: float, radius: float, xmin: float, xmax: float, ymin: float, ymax: float) -> float:
    qx = clamp(cx, xmin, xmax)
    qy = clamp(cy, ymin, ymax)
    dx = cx - qx
    dy = cy - qy
    outside = math.hypot(dx, dy)
    if outside > 1e-12:
        return outside - radius
    inside = min(cx - xmin, xmax - cx, cy - ymin, ymax - cy)
    return -(inside + radius)


def cylinder_aabb_gap(
    center: tuple[float, float, float],
    radius: float,
    half_length: float,
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
) -> float:
    """Signed gap from an X-axis flat-capped cylinder to an AABB."""
    x0 = center[0] - half_length
    x1 = center[0] + half_length
    radial = _circle_rect_signed(center[1], center[2], radius, box_min[1], box_max[1], box_min[2], box_max[2])
    if x1 < box_min[0]:
        axial = box_min[0] - x1
    elif x0 > box_max[0]:
        axial = x0 - box_max[0]
    else:
        axial = 0.0
    if axial == 0.0:
        return radial
    if radial >= 0.0:
        return math.hypot(axial, radial)
    return axial


def point_aabb_distance(
    point: tuple[float, float, float],
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
) -> float:
    delta = tuple(max(box_min[index] - point[index], 0.0, point[index] - box_max[index]) for index in range(3))
    return norm(delta)


def spring_contacts_plunger(
    center_local: tuple[float, float, float],
    axis_local: tuple[float, float, float],
    radius: float,
    half_length: float,
    plunger: Box,
    contact_distance_m: float,
) -> bool:
    """Contact with the plunger itself, not with the remote's outer box.

    Interior centerline samples count when they come within a radius of the
    plunger. The flat caps count only when the endpoint itself is within
    ``contact_distance_m``. Using the radius on a cap would call a cell
    'touching' while a real axial gap remained.
    """
    axis = normalize(axis_local)
    start = add(center_local, scale(axis, -half_length))
    end = add(center_local, scale(axis, half_length))
    box_min, box_max = _aabb(plunger)
    if point_aabb_distance(start, box_min, box_max) <= contact_distance_m:
        return True
    if point_aabb_distance(end, box_min, box_max) <= contact_distance_m:
        return True
    samples = 8
    for index in range(1, samples):
        point = add(start, scale(sub(end, start), index / samples))
        if point_aabb_distance(point, box_min, box_max) <= radius + contact_distance_m:
            return True
    return False


def _world_axis_to_remote(remote_rotation_xyzw, axis_world):
    from my_package.scoring.math3d import quat_inv, rotate

    return rotate(quat_inv(remote_rotation_xyzw), axis_world)


def _world_point_to_remote(remote_position, remote_rotation, point):
    from my_package.scoring.math3d import quat_inv, rotate

    return rotate(quat_inv(remote_rotation), sub(point, remote_position))


def measure_battery(
    cfg: BatteryInsertionConfig,
    slot: SlotSpec,
    battery_position_xyz: tuple[float, float, float],
    battery_rotation_xyzw: tuple[float, float, float, float],
    remote_position_xyz: tuple[float, float, float],
    remote_rotation_xyzw: tuple[float, float, float, float],
    plunger_joint_m: float | None = 0.0,
) -> BatteryMeasure:
    """Measure one cell against one slot along the slot axis, not with an AABB test.

    ``plunger_joint_m`` of None means the joint could not be read; spring contact is then
    None (unknown), never a guess from the authored rest pose.
    """
    from my_package.scoring.math3d import angle_deg, rotate

    local = _world_point_to_remote(remote_position_xyz, remote_rotation_xyzw, battery_position_xyz)
    positive_world = rotate(battery_rotation_xyzw, (0.0, 0.0, 1.0))
    positive_local = _world_axis_to_remote(remote_rotation_xyzw, positive_world)
    required_world = rotate(remote_rotation_xyzw, slot.required_axis_xyz)
    axial = local[0] - cfg.compartment.seat_center_x
    radial = math.hypot(local[1] - slot.center_y_m, local[2] - rest_center_z(cfg))
    if plunger_joint_m is None:
        contact = None
    else:
        contact = spring_contacts_plunger(
            local,
            positive_local,
            cfg.battery.diameter_m / 2.0,
            cfg.battery.length_m / 2.0,
            plunger_box(cfg, slot, plunger_joint_m),
            cfg.compartment.spring_contact_distance_m,
        )
    return BatteryMeasure(
        axial_offset_m=axial,
        radial_offset_m=radial,
        positive_axis_xyz=positive_world,
        polarity_angle_deg=angle_deg(positive_world, required_world),
        spring_contact=contact,
        local_position_xyz=local,
    )


def _seated_center(cfg: BatteryInsertionConfig, slot: SlotSpec) -> tuple[float, float, float]:
    return (cfg.compartment.seat_center_x, slot.center_y_m, rest_center_z(cfg))


def _drop_center_x(cfg: BatteryInsertionConfig, slot: SlotSpec) -> float:
    """Middle of the free length when the plunger is fully compressed."""
    compartment = cfg.compartment
    slack = compartment.plunger_travel_m - compartment.spring_preload_m
    return compartment.seat_center_x + spring_side(slot) * slack / 2.0


def design_report(cfg: BatteryInsertionConfig) -> dict:
    """Check the provisional clearances. ``ok`` is false when a number contradicts another."""
    problems: list[str] = []
    compartment = cfg.compartment
    diameter = cfg.battery.diameter_m
    radius = diameter / 2.0
    half_length = cfg.battery.length_m / 2.0
    outer_z = compartment.size_xyz[2] / 2.0
    channel_width = 2.0 * compartment.channel_y_half
    bay_min, bay_max = bay_x_bounds(cfg)
    if channel_width <= diameter:
        problems.append(f"channel width {channel_width} is not larger than the cell diameter {diameter}")
    if not (compartment.rear_face_x < bay_min < compartment.seat_center_x < bay_max < compartment.front_face_x):
        problems.append("remote x stations are not in rear < bay start < seat < bay end < front order")
    if not -outer_z < compartment.floor_z_m < outer_z:
        problems.append("floor_z_m must sit inside the shell height")
    seated_top = rest_center_z(cfg) + radius
    if seated_top > outer_z:
        problems.append(f"a seated cell top ({seated_top:.4f}) sticks out above the rim ({outer_z:.4f})")
    wall_height = outer_z - compartment.floor_z_m
    if wall_height < 0.5 * diameter:
        problems.append("slot walls are lower than the cell axis, so a seated cell is not retained laterally")
    outer_x = compartment.front_face_x - compartment.rear_face_x
    nose = compartment.front_face_x - compartment.silhouette_length_m
    if abs(outer_x - compartment.size_xyz[0]) > 1e-9:
        problems.append("front_face_x - rear_face_x must equal the battery-bay length")
    if compartment.silhouette_length_m <= outer_x:
        problems.append("silhouette_length_m must be longer than the battery bay")
    if not nose < compartment.rear_face_x:
        problems.append("the nose must sit entirely past the rear face")
    outer_y = compartment.size_xyz[1] / 2.0
    slot_reach = max(abs(slot.center_y_m) + compartment.channel_y_half for slot in active_slots(cfg))
    if outer_y - slot_reach < 0.003:
        problems.append(f"outer rail is {outer_y - slot_reach:.4f} m wide; need at least 0.003 m")
    if slot_reach >= outer_y - compartment.shell_edge_fillet_m:
        problems.append("the top-edge fillet reaches into a slot opening")
    if compartment.shell_nose_half_width_m < outer_y:
        problems.append("the shell must not be narrower at the nose than at the bay")
    if not 0.0 < compartment.shell_end_radius_m < outer_y:
        problems.append("shell_end_radius_m must be positive and smaller than the half width")
    if 2.0 * compartment.size_xyz[1] / 2.0 > 0.048 + 1e-9:
        problems.append("the bay block is wider than the listed 48 mm remote")
    if compartment.size_xyz[2] > 0.024 + 1e-9:
        problems.append("the bay block is taller than the listed 24 mm remote")
    if not 0.0 < compartment.spring_preload_m < compartment.plunger_travel_m:
        problems.append("spring_preload_m must be positive and smaller than plunger_travel_m")
    plunger_z = compartment.plunger_size_xyz[2] / 2.0
    if rest_center_z(cfg) - plunger_z <= compartment.floor_z_m:
        problems.append("the plunger pad reaches the slot floor")
    if compartment.plunger_size_xyz[1] >= channel_width:
        problems.append("the plunger pad is not narrower than the slot")
    drop_slack = compartment.plunger_travel_m - compartment.spring_preload_m
    numbers = {
        "channel_width_m": channel_width,
        "lateral_clearance_m": (channel_width - diameter) / 2.0,
        "seated_rim_margin_m": outer_z - seated_top,
        "bay_length_m": bay_max - bay_min,
        "drop_slack_m": drop_slack,
        "spring_preload_m": compartment.spring_preload_m,
        "rest_center_z_m": rest_center_z(cfg),
        "contact_offset_m": compartment.contact_offset_m,
        "silhouette_length_m": compartment.silhouette_length_m,
        "nose_x_m": nose,
    }
    if compartment.contact_offset_m >= numbers["lateral_clearance_m"]:
        problems.append("contact_offset_m is larger than the lateral clearance and would glue the cell")
    if drop_slack <= 2.0 * compartment.contact_offset_m:
        problems.append("a fully compressed plunger leaves no room to lower the cell straight down")

    boxes = [box for box in static_boxes(cfg) if box.collision]
    travel = compartment.plunger_travel_m
    for slot in active_slots(cfg):
        touch_ok = {"floor", f"{slot.name}_fixed_contact"}
        center = _seated_center(cfg, slot)
        for box in boxes:
            # A flat cap flush against the fixed contact reads as full overlap, so shave 10 um.
            gap = cylinder_aabb_gap(center, radius, half_length - 1e-5, *_aabb(box))
            if box.name in touch_ok:
                if gap < -1e-4:
                    problems.append(f"{slot.name} seated cell penetrates {box.name} ({gap:.6f} m)")
            elif gap < 0.0002:
                problems.append(f"{slot.name} seated cell clearance to {box.name} is {gap:.6f} m")
        extended = plunger_box(cfg, slot, 0.0)
        compressed = plunger_box(cfg, slot, travel)
        for label, plunger in (("extended", extended), ("compressed", compressed)):
            for box in boxes:
                gap = aabb_gap(*_aabb(plunger), *_aabb(box))
                if gap < 0.0005:
                    problems.append(f"{slot.name} {label} plunger clearance to {box.name} is {gap:.6f} m")
        # Seated, the cell compresses the plunger by the preload and no more than the travel.
        side = spring_side(slot)
        seated_end = center[0] + side * half_length
        extended_face = extended.center[0] - side * extended.size[0] / 2.0
        overlap = side * (seated_end - extended_face)
        if not 0.0 < overlap < travel:
            problems.append(f"{slot.name} seated preload {overlap:.6f} m is not inside the {travel} m travel")
        # Vertical drop from above the rim with the plunger fully compressed.
        drop_x = _drop_center_x(cfg, slot)
        top_z = outer_z + radius + 0.004
        samples = 16
        for index in range(samples + 1):
            z = top_z + (rest_center_z(cfg) - top_z) * index / samples
            sample = (drop_x, slot.center_y_m, z)
            hit = None
            for box in (*boxes, compressed):
                gap = cylinder_aabb_gap(sample, radius, half_length, *_aabb(box))
                if box.name == "floor":
                    if gap < -1e-4:
                        hit = (box.name, gap)
                elif gap < 0.0002:
                    hit = (box.name, gap)
                if hit:
                    problems.append(f"{slot.name} insertion drop hits {hit[0]} at z={z:.4f} (gap {hit[1]:.6f})")
                    break
            if hit:
                break
        # With the plunger extended the same drop must hit it: the spring has to be pressed.
        seated_drop = (drop_x, slot.center_y_m, rest_center_z(cfg))
        numbers[f"{slot.name}_extended_plunger_blocks_drop"] = (
            cylinder_aabb_gap(seated_drop, radius, half_length, *_aabb(extended)) < 0.0
        )
    # A cell between the slots is inside the remote's outer box but not on a plunger.
    if len(active_slots(cfg)) == 2:
        between = measure_battery(
            cfg,
            cfg.compartment.slots[0],
            cfg.remote_pose.position_xyz,
            (0.0, 0.0, 0.0, 1.0),
            cfg.remote_pose.position_xyz,
            cfg.remote_pose.rotation_xyzw,
        )
        if between.spring_contact:
            problems.append("a cell at the remote origin touches a plunger; the slot test is too broad")
        if between.radial_offset_m <= cfg.scoring.seat_radial_max_m:
            problems.append("a cell at the remote origin already counts as on a slot axis")
    return {"ok": not problems, "problems": problems, "numbers": numbers}


def _fmt(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _cube(box: Box, collision: bool, contact_offset_m: float = 0.0003, guide: bool = False) -> str:
    schemas = 'prepend apiSchemas = ["PhysicsCollisionAPI"]' if collision else ""
    header = f"    def Cube \"{box.name}\" (\n        {schemas}\n    )" if collision else f"    def Cube \"{box.name}\""
    color = ", ".join(_fmt(channel) for channel in box.color)
    contact = ""
    if collision:
        contact = f"\n        float physxCollision:contactOffset = {_fmt(contact_offset_m)}"
        if guide:
            contact += '\n        uniform token purpose = "guide"'
    return f"""{header}
    {{
        double size = 1
        double3 xformOp:translate = ({_fmt(box.center[0])}, {_fmt(box.center[1])}, {_fmt(box.center[2])})
        float3 xformOp:scale = ({_fmt(box.size[0])}, {_fmt(box.size[1])}, {_fmt(box.size[2])})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [({color})]{contact}
    }}"""


def _physics_material(name: str, static: float, dynamic: float, restitution: float) -> str:
    return f"""    def Material "{name}" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = {_fmt(static)}
        float physics:dynamicFriction = {_fmt(dynamic)}
        float physics:restitution = {_fmt(restitution)}
    }}"""


def write_geometry(cfg: BatteryInsertionConfig, directory: str | Path) -> dict[str, Path]:
    """Write the remote articulation, the AAA cell, and the visual table marks."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = design_report(cfg)
    if not report["ok"]:
        raise ValueError("refusing to write geometry:\n- " + "\n- ".join(report["problems"]))
    remote_path = directory / ("remote_compartment_single.usda" if cfg.single_slot else "remote_compartment.usda")
    battery_path = directory / "aaa_battery.usda"
    marks_path = directory / ("table_marks_single.usda" if cfg.single_slot else "table_marks.usda")
    remote_path.write_text(_remote_usda(cfg, report["numbers"]))
    battery_path.write_text(_battery_usda(cfg))
    marks_path.write_text(_marks_usda(cfg))
    return {"remote": remote_path, "battery": battery_path, "marks": marks_path}


def _mesh_usda(name: str, mesh: MeshData, indent: str, extra: str = "", schemas: tuple[str, ...] = ()) -> str:
    points = ", ".join(f"({_fmt(x)}, {_fmt(y)}, {_fmt(z)})" for x, y, z in mesh.points)
    indices = ", ".join(f"{a}, {b}, {c}" for a, b, c in mesh.triangles)
    counts = ", ".join("3" for _ in mesh.triangles)
    header = f'{indent}def Mesh "{name}"'
    if schemas:
        header += f' (\n{indent}    prepend apiSchemas = [{", ".join(chr(34) + item + chr(34) for item in schemas)}]\n{indent})'
    return f"""{header}
{indent}{{
{indent}    uniform token subdivisionScheme = "none"
{indent}    int[] faceVertexCounts = [{counts}]
{indent}    int[] faceVertexIndices = [{indices}]
{indent}    point3f[] points = [{points}]{extra}
{indent}}}"""


def _prism_mesh(outline: list[tuple[float, float]], z0: float, z1: float) -> MeshData:
    """Closed convex prism over a counter-clockwise outline."""
    mesh = MeshData()
    n = len(outline)
    bottom = [mesh.add_point((x, y, z0)) for x, y in outline]
    top = [mesh.add_point((x, y, z1)) for x, y in outline]
    for k in range(n):
        mesh.quad(bottom[k], bottom[(k + 1) % n], top[(k + 1) % n], top[k])
    for k in range(1, n - 1):
        mesh.triangles.append((bottom[0], bottom[k + 1], bottom[k]))
        mesh.triangles.append((top[0], top[k], top[k + 1]))
    return mesh


# Visual materials (UsdPreviewSurface). Black ABS, nickel contacts, slightly lighter moulded marks.
_LOOKS = {
    "black_abs": {"color": (0.018, 0.018, 0.02), "roughness": 0.38, "metallic": 0.0},
    "nickel": {"color": (0.72, 0.71, 0.68), "roughness": 0.22, "metallic": 1.0},
    "moulded_mark": {"color": (0.30, 0.30, 0.31), "roughness": 0.55, "metallic": 0.0},
    "spring_steel": {"color": (0.62, 0.62, 0.64), "roughness": 0.3, "metallic": 1.0},
}
# Collider colours, only seen when a collision overlay makes guide prims visible.
_COLLIDER_COLOR = (0.95, 0.4, 0.1)


def _looks_usda(root: str, indent: str = "    ") -> str:
    blocks = []
    for name, look in _LOOKS.items():
        color = ", ".join(_fmt(channel) for channel in look["color"])
        blocks.append(
            f"""{indent}    def Material "{name}"
{indent}    {{
{indent}        token outputs:surface.connect = </{root}/Looks/{name}/surface.outputs:surface>
{indent}        def Shader "surface"
{indent}        {{
{indent}            uniform token info:id = "UsdPreviewSurface"
{indent}            color3f inputs:diffuseColor = ({color})
{indent}            float inputs:roughness = {_fmt(look["roughness"])}
{indent}            float inputs:metallic = {_fmt(look["metallic"])}
{indent}            token outputs:surface
{indent}        }}
{indent}    }}"""
        )
    return f'{indent}def Scope "Looks"\n{indent}{{\n' + "\n".join(blocks) + f"\n{indent}}}"


def _bound(look: str, color: tuple[float, float, float]) -> str:
    rgb = ", ".join(_fmt(channel) for channel in color)
    return f"""
        rel material:binding = </remote/Looks/{look}>
        color3f[] primvars:displayColor = [({rgb})]"""


def _remote_usda(cfg: BatteryInsertionConfig, numbers: dict) -> str:
    """Fixed-base articulation, authored like the measured YAM asset.

    ``chassis`` is the root rigid body and carries ``PhysicsArticulationRootAPI``.
    ``root_joint`` is a ``PhysicsFixedJoint`` from the enclosing ``remote`` Xform
    (not a rigid body, so the world) to the chassis; that is the secured fixture,
    and it does not rely on Isaac Lab's ``fix_root_link``. The plunger links are
    rigid bodies nested under the chassis, as YAM nests its links.
    """
    compartment = cfg.compartment
    offset = compartment.contact_offset_m
    authored_boxes = [box for box in static_boxes(cfg) if box.name not in ("nose", "back_cap", "front_cap")]
    colliders = [_cube(box, box.collision, offset, guide=True) for box in authored_boxes]
    half_z = compartment.size_xyz[2] / 2.0
    collider_extra = f"""
        uniform token physics:approximation = "convexHull"
        float physxCollision:contactOffset = {_fmt(offset)}
        uniform token purpose = "guide"
        color3f[] primvars:displayColor = [({", ".join(_fmt(c) for c in _COLLIDER_COLOR)})]"""
    for name, outline in end_collider_outlines(cfg).items():
        colliders.append(
            _mesh_usda(
                name,
                _prism_mesh(outline, -half_z, half_z),
                "    ",
                collider_extra,
                ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI"),
            )
        )
    visuals = visual_meshes(cfg)
    visual_blocks = [
        _mesh_usda("shell", visuals["shell"], "    ", _bound("black_abs", _LOOKS["black_abs"]["color"]), ("MaterialBindingAPI",)),
        _mesh_usda("positive_contacts", visuals["contacts"], "    ", _bound("nickel", _LOOKS["nickel"]["color"]), ("MaterialBindingAPI",)),
        _mesh_usda("polarity_marks", visuals["marks"], "    ", _bound("moulded_mark", _LOOKS["moulded_mark"]["color"]), ("MaterialBindingAPI",)),
    ]
    spec = shell_spec(cfg)
    ir = MeshData()
    ir.box((spec.nose_x, 0.0, 0.0), (0.0008, 0.010, 0.006))
    visual_blocks.append(_mesh_usda("ir_window", ir, "    ", "\n        color3f[] primvars:displayColor = [(0.35, 0.03, 0.03)]"))
    chassis_body = "\n".join(colliders + visual_blocks).replace(chr(10), chr(10) + "    ")

    bay_min, bay_max = bay_x_bounds(cfg)
    plunger_blocks = []
    joint_blocks = []
    for slot in active_slots(cfg):
        plunger = plunger_box(cfg, slot, 0.0)
        link = plunger_link_name(slot)
        joint = plunger_joint_name(slot)
        side = spring_side(slot)
        # Rotate the joint frame so +joint always compresses the spring, on either end of the bay.
        rot = "(1, 0, 0, 0)" if side > 0.0 else "(0, 0, 0, 1)"
        pad = _cube(Box(joint + "_pad", (0.0, 0.0, 0.0), plunger.size, True, _LOOKS["nickel"]["color"]), True, offset)
        pad = pad.replace("    }", "        rel material:binding = </remote/Looks/nickel>\n    }", 1).replace(
            'prepend apiSchemas = ["PhysicsCollisionAPI"]', 'prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]'
        )
        # Conical coil from the pad's back face to the bay end wall (visual; it rides on the link).
        bay_end = bay_max if side > 0.0 else bay_min
        coil = coil_spring(
            side * plunger.size[0] / 2.0, bay_end - plunger.center[0], 0.0026, 0.0040, 0.00035, 4.0
        )
        coil_usda = _mesh_usda(
            "coil", coil, "        ", _bound("spring_steel", _LOOKS["spring_steel"]["color"]), ("MaterialBindingAPI",)
        )
        plunger_blocks.append(
            f"""        def Xform "{link}" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "MaterialBindingAPI"]
        )
        {{
            float physics:mass = {_fmt(compartment.plunger_mass_kg)}
            rel material:binding:physics = </remote/physics_material>
            double3 xformOp:translate = ({_fmt(plunger.center[0])}, {_fmt(plunger.center[1])}, {_fmt(plunger.center[2])})
            uniform token[] xformOpOrder = ["xformOp:translate"]
{pad.replace(chr(10), chr(10) + "        ").replace("    def", "            def", 1)}
{coil_usda.replace(chr(10), chr(10) + "    ").replace("        def", "            def", 1)}
        }}"""
        )
        joint_blocks.append(
            f"""    def PhysicsPrismaticJoint "{joint}"
    {{
        rel physics:body0 = <../chassis>
        rel physics:body1 = <../chassis/{link}>
        uniform token physics:axis = "X"
        float physics:lowerLimit = 0
        float physics:upperLimit = {_fmt(compartment.plunger_travel_m)}
        point3f physics:localPos0 = ({_fmt(plunger.center[0])}, {_fmt(plunger.center[1])}, {_fmt(plunger.center[2])})
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = {rot}
        quatf physics:localRot1 = {rot}
        bool physics:jointEnabled = true
    }}"""
        )
    header = "\n".join(
        [
            "# Face-down two-AAA TV remote, likeness of ASIN B07T4N7NTX. Not measured CAD.",
            "# Listing: 23.5 x 4.8 x 2.4 cm, black ABS, two AAA. Bay geometry is ESTIMATED.",
            f"# channel_width_m = {numbers['channel_width_m']}",
            f"# lateral_clearance_m = {numbers['lateral_clearance_m']}",
            f"# seated_rim_margin_m = {numbers['seated_rim_margin_m']}",
            f"# bay_length_m = {numbers['bay_length_m']}",
            f"# drop_slack_m = {numbers['drop_slack_m']}",
            f"# spring_preload_m = {numbers['spring_preload_m']}",
            f"# contact_offset_m = {numbers['contact_offset_m']}",
            f"# plunger_travel_m = {compartment.plunger_travel_m}",
            f"# silhouette_length_m = {numbers['silhouette_length_m']}",
            "# Cover removed: the bay is open from above. The spring is at each slot's negative end.",
            "# Fixed base: root_joint (PhysicsFixedJoint) anchors the chassis, which is the articulation root.",
            "# Colliders are guide-purpose (not rendered); the shell, contacts, marks, and coils are visual only.",
            "# Plungers are prismatic joints; +joint compresses the spring. The battery is not parented here.",
            "# single_slot omits slot_b. Joint 0 is the extended pose authored below.",
        ]
    )
    material = _physics_material(
        "physics_material", compartment.static_friction, compartment.dynamic_friction, compartment.restitution
    )
    return f"""#usda 1.0
(
    defaultPrim = "remote"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
    doc = "Face-down two-AAA TV remote with the battery cover removed"
)

{header}
def Xform "remote"
{{
{material}
{_looks_usda("remote")}
    def Xform "chassis" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI", "MaterialBindingAPI", "PhysicsArticulationRootAPI", "PhysxArticulationAPI"]
    )
    {{
        float physics:mass = {_fmt(compartment.mass_kg)}
        rel material:binding:physics = </remote/physics_material>
        int physxRigidBody:solverPositionIterationCount = {compartment.solver_position_iterations}
        int physxRigidBody:solverVelocityIterationCount = {compartment.solver_velocity_iterations}
        bool physxArticulation:enabledSelfCollisions = false
        # Never sleep. A resting remote fell asleep and its spring drives then ignored new
        # targets: on the server a free plunger stayed at 0 under 0.86 N (942249a).
        float physxArticulation:sleepThreshold = 0
        float physxArticulation:stabilizationThreshold = 0
        quatf xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{chassis_body}
{chr(10).join(plunger_blocks)}
    }}
    def PhysicsFixedJoint "root_joint"
    {{
        rel physics:body0 = <..>
        rel physics:body1 = <../chassis>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        bool physics:jointEnabled = true
    }}
{chr(10).join(joint_blocks)}
}}
"""


def _battery_usda(cfg: BatteryInsertionConfig) -> str:
    battery = cfg.battery
    radius = battery.diameter_m / 2.0
    # Local +Z is the positive terminal. The nub is visual and has no collision.
    nub_center = battery.length_m / 2.0 + battery.nub_length_m / 2.0
    material = _physics_material(
        "physics_material", battery.static_friction, battery.dynamic_friction, battery.restitution
    )
    return f"""#usda 1.0
(
    defaultPrim = "battery"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
    doc = "Provisional AAA cell. Collision is the cylinder body. The red nub is visual only."
)

def Xform "battery" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI", "MaterialBindingAPI"]
)
{{
    float physics:mass = {_fmt(battery.mass_kg)}
    rel material:binding:physics = </battery/physics_material>
{material}
    def Cylinder "body" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        uniform token axis = "Z"
        double height = {_fmt(battery.length_m)}
        double radius = {_fmt(radius)}
        color3f[] primvars:displayColor = [(0.72, 0.73, 0.75)]
        float physxCollision:contactOffset = {_fmt(cfg.compartment.contact_offset_m)}
    }}
    def Cylinder "positive_nub"
    {{
        uniform token axis = "Z"
        double height = {_fmt(battery.nub_length_m)}
        double radius = {_fmt(battery.nub_radius_m)}
        double3 xformOp:translate = (0, 0, {_fmt(nub_center)})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [(0.8, 0.05, 0.05)]
    }}
}}
"""


def _marks_usda(cfg: BatteryInsertionConfig) -> str:
    """Visual marks baked in scene coordinates. Spawn this file at the identity pose."""
    top = cfg.table.top_z_m + 0.0005
    remote = cfg.remote_pose.position_xyz
    nose = cfg.compartment.front_face_x - cfg.compartment.silhouette_length_m
    local_center_x = (nose + cfg.compartment.front_face_x) / 2.0
    marks = [
        Box(
            "remote_mark",
            (remote[0] + local_center_x, remote[1], top),
            (cfg.compartment.silhouette_length_m + 0.01, cfg.compartment.size_xyz[1] + 0.01, 0.0004),
            False,
            (0.1, 0.45, 0.2),
        )
    ]
    for name in cell_names(cfg):
        pose = battery_pose(cfg, name)
        marks.append(
            Box(
                f"{name}_mark",
                (pose.position_xyz[0], pose.position_xyz[1], top),
                (cfg.battery.length_m + 0.006, cfg.battery.diameter_m + 0.006, 0.0004),
                False,
                (0.1, 0.35, 0.55),
            )
        )
    cubes = "\n".join(_cube(mark, False) for mark in marks)
    return f"""#usda 1.0
(
    defaultPrim = "marks"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "Visual table marks only. No physics, no collision, no joints."
)

def Xform "marks"
{{
{cubes}
}}
"""
