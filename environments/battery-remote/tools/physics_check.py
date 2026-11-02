"""Server-side physics and render check for the battery scene. Prints one JSON report.

Run with the Arena interpreter (starts Kit headless; no streaming, no Arena runner):

    cd /home/ubuntu/workspace/robocurve-scenes/environments/battery-remote
    /home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python tools/physics_check.py
    # physics only, no camera:
    /home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python tools/physics_check.py --no-render

It builds the measured maple table, the generated remote (fixed-base articulation
with two spring plungers), both AAA cells, and both YAM arms held at home, using
the environment's own spawn configs. Then it runs scripted trials. These are not
a policy: the cell pose is driven directly, then released to physics.

1. settle: cells on their marks, nothing touched; drift, fixture motion, plunger rest.
2. insert: for each slot, compress the spring with its drive, lower a correctly
   oriented cell straight into the open slot, then release the drive and the
   cell together and simulate. Reports seating offsets, polarity, plunger
   travel and preload force, residual speed, and the scorer's seated test.
3. disturb: a small lateral velocity kick on each seated cell; is it still seated?
4. pull: an upward velocity on one seated cell; it should leave (nothing welds it).
5. reversed: a cell inserted the wrong way round in slot A; must not count.

Renders (1280x720): overview, robot side, bay close-ups empty and seated, and
the same with collision geometry shown instead of visuals.
Outputs go to outputs/physics_check/<git revision>/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARENA = Path("/home/ubuntu/workspace/IsaacLab-Arena")

# Quaternions (xyzw) that point the cell's +Z (positive terminal) along +X or -X.
POSITIVE_ALONG = {1.0: (0.0, 0.7071067811865475, 0.0, 0.7071067811865476), -1.0: (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)}


def _revision() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _stage(report: dict, key: str, fn):
    try:
        report[key] = fn()
    except Exception as exc:  # keep going so one failure does not hide the rest
        report[key] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=5)}
    return report[key]


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arena-root", type=Path, default=ARENA)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hold-s", type=float, default=2.0, help="Simulated time after each release.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--experience", default=None, help="Kit experience file; default: Isaac Lab's headless rendering one.")
    parser.add_argument("--no-render", action="store_true", help="Physics trials only, no camera.")
    args = parser.parse_args()

    sys.path.insert(0, str(args.arena_root))
    sys.path.insert(0, str(ROOT))
    import torch  # before Kit, as in the launchers

    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(app_parser)
    # This Isaac Lab build has no --enable_cameras on AppLauncher; pass it only if declared.
    known = {action.dest for action in app_parser._actions}
    # --visualizer none alone has no RTX engine, so a Camera would fail at sim.reset.
    kit_files = sorted(str(path) for path in (args.arena_root / "submodules" / "IsaacLab" / "apps").glob("*.kit"))
    experience = args.experience
    if experience is None and not args.no_render:
        experience = next((kit for kit in kit_files if "headless" in kit and "rendering" in kit), None)
    render = not args.no_render and experience is not None
    app_argv = ["--visualizer", "none"]
    if "enable_cameras" in known and render:
        app_argv.append("--enable_cameras")
    if experience and render:
        app_argv += ["--experience", experience]
    # Printed before Kit starts so it survives a crash during startup.
    print("ROBOCURVE_PHYSICS_SETUP", json.dumps({"app_argv": app_argv, "kit_files": kit_files, "render": render}, indent=1), flush=True)
    if not args.no_render and not render:
        print("no headless rendering .kit found; renders skipped (use --experience PATH)", flush=True)
    app = AppLauncher(app_parser.parse_args(app_argv)).app

    revision = _revision()
    out = args.out or ROOT / "outputs" / "physics_check" / revision
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROBOCURVE_GENERATED_DIR", str(out / "assets"))
    report: dict = {"revision": revision, "out": str(out), "app_argv": app_argv, "render": render}

    import isaaclab.sim as sim_utils
    import numpy as np
    from isaaclab.assets import Articulation, RigidObject
    from pxr import Usd, UsdGeom, UsdPhysics

    from my_package.assets.geometry import (
        measure_battery,
        rest_center_z,
        spring_side,
        write_geometry,
    )
    from my_package.assets.spawn import make_battery, make_remote
    from my_package.embodiments.yam_bimanual import _articulation
    from my_package.scoring.action_map import arm_bindings
    from my_package.scoring.config import (
        active_slots,
        battery_pose,
        cell_names,
        default_config_path,
        load_config,
        plunger_joint_name,
    )
    from my_package.scoring.math3d import local_to_world, rotate

    cfg = load_config(default_config_path())
    paths = write_geometry(cfg, Path(os.environ["ROBOCURVE_GENERATED_DIR"]))
    compartment = cfg.compartment
    slots = active_slots(cfg)

    dt = 1.0 / 120.0
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda:0"))
    stage = sim.stage if hasattr(sim, "stage") else sim_utils.get_current_stage()
    device = sim.device

    from isaaclab_arena.assets.background_library import MapleTableRobolab
    from my_package.assets.table import maple_table_usd

    # Arena spawns the table at the origin with no pose (measured by tools/probe_runtime.py).
    for prim_path, spawn_cfg in (
        ("/World/table", sim_utils.UsdFileCfg(usd_path=maple_table_usd(MapleTableRobolab.usd_path))),
        ("/World/dome", sim_utils.DomeLightCfg(intensity=cfg.light_intensity)),
        ("/World/sun", sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0)),
    ):
        spawn_cfg.func(prim_path, spawn_cfg)

    remote_cfg = make_remote(cfg, str(paths["remote"]))._generate_articulation_cfg()
    remote_cfg.prim_path = "/World/remote"
    remote = Articulation(remote_cfg)
    cells = {}
    for name in cell_names(cfg):
        cell_cfg = make_battery(cfg, name, str(paths["battery"]))._generate_rigid_cfg()
        cell_cfg.prim_path = f"/World/{name}"
        cells[name] = RigidObject(cell_cfg)
    left, right = arm_bindings(cfg)
    robots = {
        "RobotLeft": Articulation(_articulation(cfg, "/World/RobotLeft", cfg.robot.left_pose, left)),
        "RobotRight": Articulation(_articulation(cfg, "/World/RobotRight", cfg.robot.right_pose, right)),
    }
    camera = None
    if render:
        # Created only when rendering: a camera init failure aborts sim.reset and the physics with it.
        from isaaclab.sensors import Camera, CameraCfg

        camera = Camera(
            CameraCfg(
                prim_path="/World/render_cam",
                update_period=0.0,
                height=args.height,
                width=args.width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.005, 20.0)),
            )
        )

    sim.reset()
    assets = [remote, *cells.values(), *robots.values()]
    joint_ids = {slot.name: int(remote.find_joints(f"^{plunger_joint_name(slot)}$")[0][0]) for slot in slots}
    targets = {slot.name: 0.0 for slot in slots}
    remote_pose = cfg.remote_pose

    def drive():
        target = _tensor(remote.data.joint_pos).clone()
        for name, joint in joint_ids.items():
            target[0, joint] = targets[name]
        remote.set_joint_position_target(target)
        remote.write_data_to_sim()
        for robot in robots.values():
            robot.set_joint_position_target(_tensor(robot.data.default_joint_pos))
            robot.write_data_to_sim()

    def step(render_frame: bool = False):
        drive()
        sim.step(render=render_frame)
        for asset in assets:
            asset.update(dt)

    def run(seconds: float):
        for _ in range(max(1, int(round(seconds / dt)))):
            step()

    def pose(name):
        cell = cells[name]
        return (
            tuple(_tensor(cell.data.root_pos_w)[0].cpu().tolist()),
            tuple(_tensor(cell.data.root_quat_w)[0].cpu().tolist()),
        )

    def set_cell(name, position, rotation, velocity=(0.0, 0.0, 0.0)):
        cell = cells[name]
        cell.write_root_pose_to_sim(torch.tensor([[*position, *rotation]], device=device, dtype=torch.float32))
        cell.write_root_velocity_to_sim(torch.tensor([[*velocity, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32))

    def remote_state():
        pos = tuple(_tensor(remote.data.root_pos_w)[0].cpu().tolist())
        quat = tuple(_tensor(remote.data.root_quat_w)[0].cpu().tolist())
        return pos, quat

    def plunger(slot_name):
        return float(_tensor(remote.data.joint_pos)[0, joint_ids[slot_name]])

    def fit(name, slot):
        position, rotation = pose(name)
        rpos, rquat = remote_state()
        measured = measure_battery(cfg, slot, position, rotation, rpos, rquat, plunger_joint_m=plunger(slot.name))
        speed = float(torch.linalg.norm(_tensor(cells[name].data.root_lin_vel_w)[0]))
        seated = (
            abs(measured.axial_offset_m) <= cfg.scoring.axial_tolerance_m
            and measured.radial_offset_m <= cfg.scoring.seat_radial_max_m
            and measured.spring_contact is True
            and measured.polarity_angle_deg <= cfg.scoring.polarity_angle_deg
        )
        joint = plunger(slot.name)
        return {
            "axial_offset_mm": round(1000 * measured.axial_offset_m, 3),
            "radial_offset_mm": round(1000 * measured.radial_offset_m, 3),
            "polarity_angle_deg": round(measured.polarity_angle_deg, 2),
            "spring_contact": measured.spring_contact,
            "plunger_compression_mm": round(1000 * joint, 3),
            # Implicit PD drive at target 0: force = stiffness * compression (damping term ~0 at rest).
            "spring_force_n": round(compartment.plunger_stiffness * joint, 4),
            "speed_mm_s": round(1000 * speed, 3),
            "seated_by_scorer_geometry": seated,
            "local_position_mm": [round(1000 * v, 3) for v in measured.local_position_xyz],
        }

    def fixture():
        pos, quat = remote_state()
        return {
            "remote_displacement_mm": round(1000 * math.dist(pos, remote_pose.position_xyz), 4),
            "remote_quat_xyzw": [round(v, 5) for v in quat],
            "is_fixed_base": bool(remote.is_fixed_base) if hasattr(remote, "is_fixed_base") else "attribute missing",
        }

    def reset_scene():
        targets.update({slot.name: 0.0 for slot in slots})
        zeros = torch.zeros_like(_tensor(remote.data.joint_pos))
        remote.write_joint_state_to_sim(zeros, zeros)
        for name in cells:
            start = battery_pose(cfg, name)
            set_cell(name, start.position_xyz, start.rotation_xyzw)
        run(0.5)

    def insert(name, slot, positive_sign):
        """Compress the spring, lower the cell straight down, then release drive and cell together."""
        side = spring_side(slot)
        slack = compartment.plunger_travel_m - compartment.spring_preload_m
        local_drop = (compartment.seat_center_x + side * slack / 2.0, slot.center_y_m, rest_center_z(cfg) + 0.0003)
        rpos, rquat = remote_state()
        targets[slot.name] = compartment.plunger_travel_m * 0.98
        run(0.4)
        rotation = POSITIVE_ALONG[positive_sign]
        start_z = local_drop[2] + 0.03
        steps = int(0.6 / dt)
        for k in range(steps + 1):
            z = start_z + (local_drop[2] - start_z) * k / steps
            set_cell(name, local_to_world(rpos, rquat, (local_drop[0], local_drop[1], z)), rotation)
            step()
        compressed = plunger(slot.name)
        targets[slot.name] = 0.0
        run(args.hold_s)
        held = getattr(remote.data, "joint_pos_target", None)
        target_seen = None if held is None else round(1000 * float(_tensor(held)[0, joint_ids[slot.name]]), 4)
        return {
            "spring_compressed_before_release_mm": round(1000 * compressed, 3),
            "plunger_target_in_sim_mm": target_seen,
            **fit(name, slot),
        }

    def capture(prefix, views):
        if camera is None:
            return "render disabled"
        saved = []
        for view, (eye, target) in views.items():
            camera.set_world_poses_from_view(
                torch.tensor([eye], device=device, dtype=torch.float32),
                torch.tensor([target], device=device, dtype=torch.float32),
            )
            for _ in range(12):
                step(render_frame=True)
                camera.update(dt)
            rgb = _tensor(camera.data.output["rgb"])[0].cpu().numpy()
            from PIL import Image

            path = out / f"{prefix}{view}.png"
            Image.fromarray(np.asarray(rgb[..., :3], dtype=np.uint8)).save(path)
            saved.append(str(path))
        return saved

    rx, ry, rz = remote_pose.position_xyz
    seat_x = rx + compartment.seat_center_x
    views = {
        "overview": ((rx + 0.55, ry - 0.45, rz + 0.40), (rx - 0.05, ry, rz)),
        "robot_side": ((cfg.robot.left_pose.position_xyz[0] - 0.25, 0.0, 0.45), (rx, ry, rz)),
        "bay_close": ((seat_x + 0.09, ry - 0.07, rz + 0.08), (seat_x, ry, rz)),
        "bay_top": ((seat_x, ry + 0.0001, rz + 0.16), (seat_x, ry, rz)),
        "remote_side": ((rx - 0.06, ry - 0.30, rz + 0.06), (rx - 0.06, ry, rz)),
    }

    def settle():
        run(args.hold_s)
        rows = {}
        for name in cells:
            position, rotation = pose(name)
            start = battery_pose(cfg, name)
            axis = rotate(rotation, (0.0, 0.0, 1.0))
            rows[name] = {
                "xy_drift_mm": round(1000 * math.hypot(position[0] - start.position_xyz[0], position[1] - start.position_xyz[1]), 3),
                # Direction of the drift: a rolling cylinder moves across its axis.
                "drift_xy_mm": [round(1000 * (position[i] - start.position_xyz[i]), 3) for i in range(2)],
                "speed_mm_s": round(1000 * float(torch.linalg.norm(_tensor(cells[name].data.root_lin_vel_w)[0])), 3),
                "z_minus_rest_mm": round(1000 * (position[2] - start.position_xyz[2]), 3),
                "axis": [round(v, 4) for v in axis],
            }
        return {"cells": rows, "plunger_rest_mm": {slot.name: round(1000 * plunger(slot.name), 4) for slot in slots}, **fixture()}

    def robot_report():
        """Base-link footprint (geometry owned by the base body only), whole arm at home, colliders."""
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        cx, cy = cfg.table.center_xy
        hx, hy = cfg.table.half_extents_xy

        def owner(prim):
            while prim.IsValid() and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim = prim.GetParent()
            return prim

        result = {}
        for name in robots:
            root = stage.GetPrimAtPath(f"/World/{name}")
            base = next((p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.GetName() == "base" and p.HasAPI(UsdPhysics.RigidBodyAPI)), None)
            colliders = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
            whole = cache.ComputeWorldBound(root).ComputeAlignedRange()
            entry = {
                "collider_count": len(colliders),
                "arm_at_home_bbox": {"min": [round(v, 4) for v in whole.GetMin()], "max": [round(v, 4) for v in whole.GetMax()]},
            }
            if base is not None:
                total = None
                for prim in Usd.PrimRange(base, Usd.TraverseInstanceProxies()):
                    if prim.IsA(UsdGeom.Gprim) and owner(prim) == base:
                        part = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                        if not part.IsEmpty():
                            total = part if total is None else total.UnionWith(part)
                if total is not None:
                    lo, hi = total.GetMin(), total.GetMax()
                    entry["base_bbox"] = {"min": [round(v, 4) for v in lo], "max": [round(v, 4) for v in hi]}
                    entry["base_bottom_minus_table_top_mm"] = round(1000 * (lo[2] - cfg.table.top_z_m), 2)
                    entry["base_inside_table_footprint"] = bool(
                        lo[0] >= cx - hx and hi[0] <= cx + hx and lo[1] >= cy - hy and hi[1] <= cy + hy
                    )
            entry["arm_overhangs_near_edge"] = bool(whole.GetMin()[0] < cx - hx)
            result[name] = entry
        return result

    _stage(report, "robots", robot_report)
    _stage(report, "settle", settle)

    def spring_free_return():
        """Compress each plunger with no cell in the slot, release the target, and watch it return.

        Isolates the drive from the cell: if a free plunger does not return to 0, the drive or
        the joint is wrong; if it does, a seated cell is jamming it.
        """
        def joint_row(slot):
            index = joint_ids[slot.name]
            row = {
                "pos_mm": round(1000 * plunger(slot.name), 4),
                "vel_mm_s": round(1000 * float(_tensor(remote.data.joint_vel)[0, index]), 4),
            }
            for attr in ("joint_pos_target", "applied_torque", "computed_torque"):
                value = getattr(remote.data, attr, None)
                if value is not None:
                    row[attr] = round(float(_tensor(value)[0, index]), 6)
            return row

        result = {}
        for slot in slots:
            targets[slot.name] = compartment.plunger_travel_m * 0.98
        run(0.4)
        result["compressed"] = {slot.name: joint_row(slot) for slot in slots}
        for slot in slots:
            targets[slot.name] = 0.0
        trace = []
        for k in range(int(0.6 / dt)):
            step()
            if k in (0, 1, 2, 5, 10, 20, 40, 70):
                trace.append({"t_s": round((k + 1) * dt, 4), **{slot.name: joint_row(slot) for slot in slots}})
        result["after_release"] = trace
        result["returned_to_zero"] = {slot.name: abs(plunger(slot.name)) < 2e-4 for slot in slots}
        return result

    _stage(report, "spring_free_return", spring_free_return)
    _stage(report, "renders_start", lambda: capture("start_", views))

    names = list(cells)

    def insert_both():
        results = {}
        for name, slot in zip(names, slots):
            results[slot.name] = {"cell": name, **insert(name, slot, slot.required_axis_xyz[0])}
        # Re-measure both after the second insertion settled, so the first is checked for retention too.
        results["after_both"] = {slot.name: fit(name, slot) for name, slot in zip(names, slots)}
        results["fixture"] = fixture()
        return results

    _stage(report, "insert", insert_both)
    _stage(report, "renders_seated", lambda: capture("seated_", {k: views[k] for k in ("bay_close", "bay_top", "overview")}))

    def disturb():
        results = {}
        for name, slot in zip(names, slots):
            set_cell(name, pose(name)[0], pose(name)[1], velocity=(0.0, 0.05, 0.0))
            run(1.0)
            results[slot.name] = fit(name, slot)
        return results

    _stage(report, "disturb_lateral_0.05m_s", disturb)

    def pull():
        name, slot = names[0], slots[0]
        set_cell(name, pose(name)[0], pose(name)[1], velocity=(0.0, 0.0, 0.6))
        run(0.5)
        position, _ = pose(name)
        result = fit(name, slot)
        result["height_above_rest_mm"] = round(1000 * (position[2] - (rz + rest_center_z(cfg))), 2)
        result["left_the_slot"] = not result["seated_by_scorer_geometry"]
        return result

    _stage(report, "pull_up_0.6m_s", pull)

    def reversed_trial():
        reset_scene()
        slot = slots[0]
        result = insert(names[0], slot, -slot.required_axis_xyz[0])
        result["note"] = "inserted reversed; seated_by_scorer_geometry must be false (polarity)"
        return result

    _stage(report, "reversed_slot_a", reversed_trial)

    def collision_overlay():
        reset_scene()
        for name, slot in zip(names, slots):
            insert(name, slot, slot.required_axis_xyz[0])
        shown, hidden = 0, 0
        visual_names = {"shell", "positive_contacts", "polarity_marks", "ir_window", "coil", "positive_nub"}
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World"), Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if not (path.startswith("/World/remote") or path.startswith("/World/battery")):
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Gprim):
                UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
                color = (0.1, 0.6, 0.95) if path.startswith("/World/battery") else (0.95, 0.4, 0.1)
                UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([color])
                shown += 1
            elif prim.GetName() in visual_names:
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden += 1
        files = capture("collision_", {k: views[k] for k in ("bay_close", "bay_top", "remote_side")})
        return {"colliders_shown": shown, "visual_prims_hidden": hidden, "files": files}

    _stage(report, "collision_overlay", collision_overlay)

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("ROBOCURVE_PHYSICS_CHECK_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("ROBOCURVE_PHYSICS_CHECK_END", flush=True)
    app.close()


if __name__ == "__main__":
    main()
