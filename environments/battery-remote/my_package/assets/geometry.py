"""Procedural remote, battery, and table-mark geometry. No Isaac Sim imports.

The compartment is a C-channel. A plain cylinder can rest on the floor without
starting inside a wall, and the side lips leave a top opening narrower than the
cell so it cannot lift out. The mouth, past ``lip_end_x``, is the full channel
cross-section so the cell enters along the slot. A prismatic plunger at the
mouth pushes a seated cell against the back stop. Nothing welds the cell to
the remote and nothing teleports it into the slot.

The red nub is visual only. Collision is the cylinder body, so the nub does
not fight the plunger or the back wall. Polarity is the cylinder axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from my_package.scoring.config import BatteryInsertionConfig, SlotSpec, active_slots
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
    seating_depth_m: float
    radial_offset_m: float
    positive_axis_xyz: tuple[float, float, float]
    polarity_angle_deg: float
    spring_contact: bool
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


def rest_center_z(cfg: BatteryInsertionConfig) -> float:
    """Center height of a cell sitting on the channel floor."""
    return -cfg.compartment.channel_z_half + cfg.battery.diameter_m / 2.0


def static_boxes(cfg: BatteryInsertionConfig) -> tuple[Box, ...]:
    compartment = cfg.compartment
    outer_y = compartment.size_xyz[1] / 2.0
    outer_z = compartment.size_xyz[2] / 2.0
    x_rear = compartment.rear_face_x
    x_back = compartment.back_face_x
    x_front = compartment.front_face_x
    z_channel = compartment.channel_z_half
    slots = active_slots(cfg)
    body = (0.06, 0.06, 0.07)
    lip_color = (0.14, 0.14, 0.15)
    nose = compartment.front_face_x - compartment.silhouette_length_m
    boxes: list[Box] = [
        _box("nose", (nose, x_rear, -outer_y, outer_y, -outer_z, outer_z), body),
        _box("back_cap", (x_rear, x_back, -outer_y, outer_y, -outer_z, outer_z), body),
        _box("floor", (x_back, x_front, -outer_y, outer_y, -outer_z, -z_channel), body),
        _box("ceiling", (x_back, x_front, -outer_y, outer_y, z_channel, outer_z), body),
    ]
    first_min, first_max = slot_y_bounds(slots[0], compartment.channel_y_half)
    boxes.append(_box("rail_left", (x_back, x_front, -outer_y, first_min, -z_channel, z_channel), body))
    if len(slots) == 1:
        boxes.append(_box("rail_right", (x_back, x_front, first_max, outer_y, -z_channel, z_channel), body))
    else:
        second_min, second_max = slot_y_bounds(slots[1], compartment.channel_y_half)
        boxes.append(_box("spine", (x_back, x_front, first_max, second_min, -z_channel, z_channel), body))
        boxes.append(_box("rail_right", (x_back, x_front, second_max, outer_y, -z_channel, z_channel), body))
    intrusion = compartment.lip_side_intrusion
    for slot in slots:
        ymin, ymax = slot_y_bounds(slot, compartment.channel_y_half)
        boxes.append(
            _box(
                f"{slot.name}_lip_left",
                (x_back, compartment.lip_end_x, ymin, ymin + intrusion, compartment.lip_z_bottom, z_channel),
                lip_color,
            )
        )
        boxes.append(
            _box(
                f"{slot.name}_lip_right",
                (x_back, compartment.lip_end_x, ymax - intrusion, ymax, compartment.lip_z_bottom, z_channel),
                lip_color,
            )
        )
    return tuple(boxes)


def polarity_marks(cfg: BatteryInsertionConfig) -> tuple[Box, ...]:
    """Visual marks on the top face. They have no collision."""
    compartment = cfg.compartment
    outer_z = compartment.size_xyz[2] / 2.0
    mark_z = outer_z + 0.0004
    marks: list[Box] = []
    for slot in active_slots(cfg):
        axis = slot.required_axis_xyz[0]
        positive_x = compartment.seat_center_x + math.copysign(0.012, axis)
        negative_x = compartment.seat_center_x - math.copysign(0.012, axis)
        marks.append(
            _box(
                f"{slot.name}_positive_mark",
                (positive_x - 0.003, positive_x + 0.003, slot.center_y_m - 0.003, slot.center_y_m + 0.003, mark_z, mark_z + 0.0004),
                (0.75, 0.08, 0.08),
                collision=False,
            )
        )
        marks.append(
            _box(
                f"{slot.name}_negative_mark",
                (negative_x - 0.003, negative_x + 0.003, slot.center_y_m - 0.003, slot.center_y_m + 0.003, mark_z, mark_z + 0.0004),
                (0.08, 0.08, 0.08),
                collision=False,
            )
        )
    return tuple(marks)


def likeness_marks(cfg: BatteryInsertionConfig) -> tuple[Box, ...]:
    """Visual cues for a black two-AAA wand. They do not collide.

    The button face is against the table, so the overhead view shows the back
    and the open bay. Side keys and the nose IR window are only there so the
    profile reads as that kind of remote. Positions are provisional.
    """
    compartment = cfg.compartment
    nose = compartment.front_face_x - compartment.silhouette_length_m
    rear = compartment.rear_face_x
    outer_y = compartment.size_xyz[1] / 2.0
    span = rear - nose
    marks = [
        _box(
            "ir_window",
            (nose - 0.0008, nose, -0.006, 0.006, -0.003, 0.003),
            (0.45, 0.08, 0.08),
            collision=False,
        )
    ]
    for index, fraction in enumerate((0.18, 0.34, 0.50, 0.66, 0.82)):
        center_x = nose + span * fraction
        color = (0.55, 0.08, 0.08) if index == 0 else (0.25, 0.25, 0.27)
        for sign, tag in ((-1.0, "n"), (1.0, "p")):
            inner = sign * outer_y
            outer = sign * (outer_y + 0.0008)
            ymin, ymax = (inner, outer) if inner < outer else (outer, inner)
            marks.append(
                _box(
                    f"side_key_{index}_{tag}",
                    (center_x - 0.004, center_x + 0.004, ymin, ymax, -0.006, 0.006),
                    color,
                    collision=False,
                )
            )
    return tuple(marks)


def plunger_box(cfg: BatteryInsertionConfig, slot: SlotSpec, joint_pos_m: float = 0.0) -> Box:
    """Plunger at a prismatic joint position. Zero is the authored extended pose."""
    compartment = cfg.compartment
    size = compartment.plunger_size_xyz
    center = (
        compartment.plunger_center_x + joint_pos_m,
        slot.center_y_m,
        0.0,
    )
    return Box(
        name=f"{slot.name}_plunger",
        center=center,
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
    plunger_joint_m: float = 0.0,
) -> BatteryMeasure:
    """Measure one cell against one slot. Depth is along the channel, not an AABB test."""
    from my_package.scoring.math3d import angle_deg, rotate

    local = _world_point_to_remote(remote_position_xyz, remote_rotation_xyzw, battery_position_xyz)
    positive_world = rotate(battery_rotation_xyzw, (0.0, 0.0, 1.0))
    positive_local = _world_axis_to_remote(remote_rotation_xyzw, positive_world)
    required_world = rotate(remote_rotation_xyzw, slot.required_axis_xyz)
    depth = cfg.compartment.mouth_x - local[0]
    radial = math.hypot(local[1] - slot.center_y_m, local[2])
    plunger = plunger_box(cfg, slot, plunger_joint_m)
    contact = spring_contacts_plunger(
        local,
        positive_local,
        cfg.battery.diameter_m / 2.0,
        cfg.battery.length_m / 2.0,
        plunger,
        cfg.compartment.spring_contact_distance_m,
    )
    return BatteryMeasure(
        seating_depth_m=depth,
        radial_offset_m=radial,
        positive_axis_xyz=positive_world,
        polarity_angle_deg=angle_deg(positive_world, required_world),
        spring_contact=contact,
        local_position_xyz=local,
    )


def _seated_center(cfg: BatteryInsertionConfig, slot: SlotSpec) -> tuple[float, float, float]:
    return (cfg.compartment.seat_center_x, slot.center_y_m, rest_center_z(cfg))


def design_report(cfg: BatteryInsertionConfig) -> dict:
    """Check the provisional clearances. ``ok`` is false when a number contradicts another."""
    problems: list[str] = []
    compartment = cfg.compartment
    diameter = cfg.battery.diameter_m
    radius = diameter / 2.0
    half_length = cfg.battery.length_m / 2.0
    channel_width = 2.0 * compartment.channel_y_half
    top_opening = channel_width - 2.0 * compartment.lip_side_intrusion
    full_depth = compartment.mouth_x - compartment.seat_center_x
    if channel_width <= diameter:
        problems.append(f"channel width {channel_width} is not larger than the cell diameter {diameter}")
    if top_opening >= diameter:
        problems.append(f"top opening {top_opening} must be narrower than the cell diameter {diameter}")
    if compartment.lip_z_bottom >= compartment.channel_z_half:
        problems.append("lip_z_bottom must sit below the channel ceiling")
    if not (compartment.rear_face_x < compartment.back_face_x < compartment.seat_center_x < compartment.lip_end_x < compartment.mouth_x <= compartment.front_face_x):
        problems.append("remote x stations are not in rear < back < seat < lip_end < mouth <= front order")
    outer_x = compartment.front_face_x - compartment.rear_face_x
    nose = compartment.front_face_x - compartment.silhouette_length_m
    if abs(outer_x - compartment.size_xyz[0]) > 1e-9:
        problems.append("front_face_x - rear_face_x must equal the battery-bay length")
    if compartment.silhouette_length_m <= outer_x:
        problems.append("silhouette_length_m must be longer than the battery bay")
    if not nose < compartment.rear_face_x:
        problems.append("the nose must sit entirely past the back of the bay so the mouth stays open")
    if abs(compartment.size_xyz[1] / 2.0 - 0.025) > 1e-9 or abs(compartment.size_xyz[2] / 2.0 - 0.011) > 1e-9:
        problems.append("this shell builder expects the provisional 0.050 by 0.022 remote cross-section")
    back_gap = (compartment.seat_center_x - half_length) - compartment.back_face_x
    if back_gap < 0.0008:
        problems.append(f"seated body is only {back_gap} m from the back stop; need at least 0.0008 m")
    numbers = {
        "channel_width_m": channel_width,
        "lateral_clearance_m": (channel_width - diameter) / 2.0,
        "top_opening_m": top_opening,
        "full_seat_depth_m": full_depth,
        "back_gap_m": back_gap,
        "rest_center_z_m": rest_center_z(cfg),
        "contact_offset_m": compartment.contact_offset_m,
        "silhouette_length_m": compartment.silhouette_length_m,
        "nose_x_m": nose,
    }
    if compartment.contact_offset_m >= numbers["lateral_clearance_m"]:
        problems.append("contact_offset_m is larger than the lateral clearance and would glue the cell")

    boxes = [box for box in static_boxes(cfg) if box.collision]
    for slot in active_slots(cfg):
        center = _seated_center(cfg, slot)
        floor_gaps = []
        other_gaps = []
        for box in boxes:
            gap = cylinder_aabb_gap(center, radius, half_length, *_aabb(box))
            if box.name == "floor":
                floor_gaps.append(gap)
            else:
                other_gaps.append((box.name, gap))
        if not floor_gaps or min(floor_gaps) < -1e-4:
            problems.append(f"{slot.name} seated cell penetrates the floor")
        for name, gap in other_gaps:
            if gap < 0.0002:
                problems.append(f"{slot.name} seated cell clearance to {name} is {gap:.6f} m")
        raised = (center[0], center[1], center[2] + 0.003)
        raised_gaps = [cylinder_aabb_gap(raised, radius, half_length, *_aabb(box)) for box in boxes]
        if min(raised_gaps) >= 0.0:
            problems.append(f"{slot.name} still clears every wall after a 3 mm lift, so the lips do not retain it")
        plunger = plunger_box(cfg, slot, 0.0)
        for box in boxes:
            gap = aabb_gap(*_aabb(plunger), *_aabb(box))
            if gap < 0.0005:
                problems.append(f"{slot.name} plunger clearance to {box.name} is {gap:.6f} m")
        seated_end = (center[0] + half_length, center[1], center[2])
        if point_aabb_distance(seated_end, *_aabb(plunger)) > 0.0:
            problems.append(f"{slot.name} seated positive end is not inside the extended plunger")
        travel = compartment.plunger_travel_m
        overlap = (center[0] + half_length) - (_aabb(plunger)[0][0])
        if overlap <= 0.0 or overlap >= travel:
            problems.append(
                f"{slot.name} plunger overlap {overlap:.6f} m is not inside the {travel} m travel"
            )
        start_x = compartment.front_face_x + half_length + 0.004
        samples = 16
        for index in range(samples + 1):
            x = start_x + (center[0] - start_x) * index / samples
            sample = (x, slot.center_y_m, rest_center_z(cfg))
            for box in boxes:
                gap = cylinder_aabb_gap(sample, radius, half_length, *_aabb(box))
                if box.name == "floor":
                    if gap < -1e-4:
                        problems.append(f"{slot.name} insertion path penetrates the floor at x={x:.4f}")
                        break
                elif gap < 0.0002:
                    problems.append(f"{slot.name} insertion path hits {box.name} at x={x:.4f} (gap {gap:.6f})")
                    break
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
        if between.seating_depth_m >= full_depth - cfg.scoring.axial_tolerance_m:
            problems.append("a cell at the remote origin already counts as fully deep")
    return {"ok": not problems, "problems": problems, "numbers": numbers}


def _fmt(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _cube(box: Box, collision: bool) -> str:
    schemas = 'prepend apiSchemas = ["PhysicsCollisionAPI"]' if collision else ""
    header = f"    def Cube \"{box.name}\" (\n        {schemas}\n    )" if collision else f"    def Cube \"{box.name}\""
    color = ", ".join(_fmt(channel) for channel in box.color)
    contact = ""
    if collision:
        contact = f"\n        float physxCollision:contactOffset = {_fmt(0.0003)}"
    return f"""{header}
    {{
        double size = 1
        double3 xformOp:translate = ({_fmt(box.center[0])}, {_fmt(box.center[1])}, {_fmt(box.center[2])})
        float3 xformOp:scale = ({_fmt(box.size[0])}, {_fmt(box.size[1])}, {_fmt(box.size[2])})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [({color})]{contact}
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


def _remote_usda(cfg: BatteryInsertionConfig, numbers: dict[str, float]) -> str:
    compartment = cfg.compartment
    cubes = "\n".join(
        _cube(box, box.collision) for box in (*static_boxes(cfg), *polarity_marks(cfg), *likeness_marks(cfg))
    )
    plunger_blocks = []
    joint_blocks = []
    for slot in active_slots(cfg):
        plunger = plunger_box(cfg, slot, 0.0)
        link = slot.name.replace("battery_", "slot_") + "_plunger_link"
        joint = slot.name.replace("battery_", "slot_") + "_plunger"
        plunger_blocks.append(
            f"""    def Xform "{link}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = {_fmt(compartment.plunger_mass_kg)}
        double3 xformOp:translate = ({_fmt(plunger.center[0])}, {_fmt(plunger.center[1])}, {_fmt(plunger.center[2])})
        uniform token[] xformOpOrder = ["xformOp:translate"]
{_cube(Box(joint + "_pad", (0.0, 0.0, 0.0), plunger.size, True, plunger.color), True).replace("    def", "        def")}
    }}"""
        )
        joint_blocks.append(
            f"""    def PhysicsPrismaticJoint "{joint}"
    {{
        rel physics:body0 = <../chassis>
        rel physics:body1 = <../{link}>
        uniform token physics:axis = "X"
        float physics:lowerLimit = 0
        float physics:upperLimit = {_fmt(compartment.plunger_travel_m)}
        point3f physics:localPos0 = ({_fmt(plunger.center[0])}, {_fmt(plunger.center[1])}, {_fmt(plunger.center[2])})
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        bool physics:jointEnabled = true
    }}"""
        )
    header = "\n".join(
        [
            "# Provisional procedural TV-remote compartment. Not measured CAD.",
            f"# channel_width_m = {numbers['channel_width_m']}",
            f"# lateral_clearance_m = {numbers['lateral_clearance_m']}",
            f"# top_opening_m = {numbers['top_opening_m']}",
            f"# full_seat_depth_m = {numbers['full_seat_depth_m']}",
            f"# back_gap_m = {numbers['back_gap_m']}",
            f"# contact_offset_m = {numbers['contact_offset_m']}",
            f"# plunger_travel_m = {compartment.plunger_travel_m}",
            f"# silhouette_length_m = {numbers['silhouette_length_m']}",
            "# Likeness of a black two-AAA TV remote. The nose is solid. The mouth at +X stays open.",
            "# The plunger is a prismatic joint. There is no fixed joint and the battery is not parented here.",
            "# single_slot omits slot_b. Joint 0 is the extended pose authored below.",
        ]
    )
    return f"""#usda 1.0
(
    defaultPrim = "remote"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
    doc = "Provisional battery compartment"
)

{header}
def Xform "remote" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{{
    def Xform "chassis" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    )
    {{
        float physics:mass = {_fmt(compartment.mass_kg)}
        float physics:staticFriction = {_fmt(compartment.static_friction)}
        float physics:dynamicFriction = {_fmt(compartment.dynamic_friction)}
        float physics:restitution = {_fmt(compartment.restitution)}
        int physxRigidBody:solverPositionIterationCount = {compartment.solver_position_iterations}
        int physxRigidBody:solverVelocityIterationCount = {compartment.solver_velocity_iterations}
        quatf xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{cubes.replace(chr(10), chr(10) + "    ")}
    }}
{chr(10).join(plunger_blocks)}
{chr(10).join(joint_blocks)}
}}
"""


def _battery_usda(cfg: BatteryInsertionConfig) -> str:
    battery = cfg.battery
    radius = battery.diameter_m / 2.0
    # Local +Z is the positive terminal. The nub is visual and has no collision.
    nub_center = battery.length_m / 2.0 + battery.nub_length_m / 2.0
    return f"""#usda 1.0
(
    defaultPrim = "battery"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
    doc = "Provisional AAA cell. Collision is the cylinder body. The red nub is visual only."
)

def Xform "battery" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)
{{
    float physics:mass = {_fmt(battery.mass_kg)}
    float physics:staticFriction = {_fmt(battery.static_friction)}
    float physics:dynamicFriction = {_fmt(battery.dynamic_friction)}
    float physics:restitution = {_fmt(battery.restitution)}
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
    for slot in active_slots(cfg):
        pose = cfg.battery_a_pose if slot.name == "battery_a" else cfg.battery_b_pose
        marks.append(
            Box(
                f"{slot.name}_mark",
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

