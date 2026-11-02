"""Server-side reach and grasp check for the battery scene with the real YAM arms. One JSON report.

Run with the Arena interpreter (headless Kit, no streaming, no Arena runner):

    cd /home/ubuntu/workspace/robocurve-scenes/environments/battery-remote
    /home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python tools/reach_check.py

Every earlier physics check placed the cells by pose. This one moves the arms. It
builds the same table, mats, remote, cells and arms, then for each target drives
the nearer arm's fingertip midpoint there with damped-least-squares IK on the
gripper body's linear Jacobian (position only):

- above and at each cell, and above each slot's seat
- a grasp attempt on each cell: open, descend, close, lift 5 cm, and check that the
  cell came up with the fingers

It reports the residual position error, joints sitting at a limit, and the gripper
frame's axes in the world at each target, so it is clear whether a top-down grasp
is even oriented sensibly. It is a feasibility probe, not a policy.
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
    except Exception as exc:  # one failed stage must not hide the others
        report[key] = {
            "error": f"{type(exc).__name__}: {exc}",
            # One line, so a grep for "error" also shows where it came from.
            "error_at": " | ".join(line.strip() for line in traceback.format_exc(limit=6).splitlines() if line.strip().startswith(("File", "jac", "return", "dq", "body"))),
        }
    return report[key]


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arena-root", type=Path, default=ARENA)
    parser.add_argument("--use-fabric", action="store_true", help="Simulate with Fabric on, as the Arena env does.")
    parser.add_argument("--approach-m", type=float, default=0.08, help="Height above a target for the approach pose.")
    parser.add_argument("--tolerance-m", type=float, default=0.004, help="Fingertip position error that counts as reached.")
    parser.add_argument("--reach-s", type=float, default=4.0, help="Simulated time allowed per target.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(args.arena_root))
    sys.path.insert(0, str(ROOT))
    import torch  # before Kit, as in the launchers

    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(app_parser)
    app = AppLauncher(app_parser.parse_args(["--visualizer", "none"])).app

    revision = _revision()
    out = args.out or ROOT / "outputs" / "reach_check" / revision
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROBOCURVE_GENERATED_DIR", str(out / "assets"))
    report: dict = {"revision": revision, "out": str(out), "use_fabric": args.use_fabric}

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject

    from my_package.assets.geometry import rest_center_z, write_geometry
    from my_package.assets.spawn import make_battery, make_remote
    from my_package.assets.table import maple_table_usd
    from my_package.embodiments.yam_bimanual import _articulation
    from my_package.scoring.action_map import arm_bindings
    from my_package.scoring.config import active_slots, battery_pose, cell_names, default_config_path, load_config
    from my_package.scoring.math3d import local_to_world

    cfg = load_config(default_config_path())
    paths = write_geometry(cfg, Path(os.environ["ROBOCURVE_GENERATED_DIR"]))
    dt = 1.0 / 120.0
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda:0", use_fabric=args.use_fabric))
    device = sim.device

    from isaaclab_arena.assets.background_library import MapleTableRobolab

    for prim_path, spawn_cfg in (
        ("/World/table", sim_utils.UsdFileCfg(usd_path=maple_table_usd(MapleTableRobolab.usd_path))),
        ("/World/table_marks", sim_utils.UsdFileCfg(usd_path=str(paths["marks"]))),
        ("/World/dome", sim_utils.DomeLightCfg(intensity=cfg.light_intensity)),
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
    bindings = {"left": left, "right": right}
    robots = {
        "left": Articulation(_articulation(cfg, "/World/RobotLeft", cfg.robot.left_pose, left)),
        "right": Articulation(_articulation(cfg, "/World/RobotRight", cfg.robot.right_pose, right)),
    }
    sim.reset()
    assets = [remote, *cells.values(), *robots.values()]

    def index_of(robot, finder, name):
        ids, names = finder(f"^{name}$")
        if len(names) != 1:
            raise KeyError(f"{name!r} not found; have {list(getattr(robot, 'body_names', []))}")
        return int(ids[0])

    arm = {}
    for side, robot in robots.items():
        binding = bindings[side]
        arm[side] = {
            "joints": [index_of(robot, robot.find_joints, j) for j in binding.arm_joints],
            "fingers": [index_of(robot, robot.find_joints, j) for j in binding.finger_joints],
            "gripper": index_of(robot, robot.find_bodies, binding.gripper_body),
            "tips": [index_of(robot, robot.find_bodies, "tip_left"), index_of(robot, robot.find_bodies, "tip_right")],
        }
    targets = {side: _tensor(robot.data.default_joint_pos).clone() for side, robot in robots.items()}
    finger_cmd = {side: cfg.robot.finger_open_m for side in robots}

    def step(n: int = 1):
        for _ in range(n):
            for side, robot in robots.items():
                target = targets[side].clone()
                for j in arm[side]["fingers"]:
                    target[0, j] = finger_cmd[side]
                robot.set_joint_position_target(target)
                robot.write_data_to_sim()
            sim.step(render=False)
            for asset in assets:
                asset.update(dt)

    def raw_jacobian(side):
        view = getattr(robots[side], "root_physx_view", None) or getattr(robots[side], "root_view", None)
        jac = _tensor(view.get_jacobians())
        if not isinstance(jac, torch.Tensor):
            # This build can hand back a Warp/proxy array; list indexing on it fails.
            try:
                import warp as wp

                jac = wp.to_torch(jac)
            except Exception:
                jac = torch.as_tensor(jac, device=device)
        return jac

    # Which Jacobian row/column convention this build uses is decided by measurement
    # (calibrate()), not assumed: the cda9d37 run with the assumed one never converged.
    layout = {}
    measured_jac = {}

    def jacobian_variant(side, body_shift, col_shift):
        jac = raw_jacobian(side)
        body = arm[side]["gripper"] - body_shift
        cols = [j + col_shift for j in arm[side]["joints"]]
        if not (0 <= body < jac.shape[1]) or max(cols) >= jac.shape[3]:
            return None
        jac_body = jac[0, body, :3, :]
        return jac_body[:, torch.tensor(cols, device=jac_body.device)]

    def jacobian(side):
        body_shift, col_shift = layout[side]
        return jacobian_variant(side, body_shift, col_shift)

    def angular_jacobian(side):
        """Angular rows of the analytic Jacobian. They are the same for every point on the gripper,
        so the fingertip lever that spoiled the linear rows does not affect them."""
        body_shift, col_shift = layout[side]
        jac = raw_jacobian(side)
        body = arm[side]["gripper"] - body_shift
        cols = torch.tensor([j + col_shift for j in arm[side]["joints"]], device=jac.device)
        return jac[0, body, 3:6, :][:, cols]

    def gripper_rotation(side):
        x, y, z, w = _tensor(robots[side].data.body_quat_w)[0, arm[side]["gripper"]].tolist()  # xyzw
        return torch.tensor(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            device=device,
        )

    approach_local = {}

    def measure_approach_axis(side):
        """Gripper-frame direction from the gripper origin to the fingertip midpoint, at home."""
        pos = _tensor(robots[side].data.body_pos_w)[0, arm[side]["gripper"]]
        v = gripper_rotation(side).T @ (tip_mid(side) - pos)
        approach_local[side] = v / torch.linalg.norm(v)
        return [round(float(c), 3) for c in approach_local[side]]

    def orientation_error(side):
        """Rotation vector for a pinch across the cell.

        Hard term: the finger axis (gripper local Y) perpendicular to the cell's long axis
        (world X). Weak term (0.3): approach axis toward straight down. The acfa6ce run showed
        the arm cannot point straight down at table height from its base, and forcing it pulled
        the fingertips 27-38 mm off the cell.
        """
        rot = gripper_rotation(side)
        approach = rot @ approach_local[side]
        # The fingers do not move along the gripper's local Y (c54b4eb: local Y was within 2 deg of
        # world Y while the pads were 68 mm apart along X and Z). Use the measured pad-to-pad axis.
        left_pad, right_pad = pad_centres(side)
        finger = right_pad - left_pad
        finger = finger / max(float(torch.linalg.norm(finger)), 1e-6)
        down = torch.tensor([0.0, 0.0, -1.0], device=device)
        # Fingers must close horizontally across the cell: along world +-Y. Only being
        # perpendicular to X let them close vertically, one above the cell and one under it.
        across = torch.tensor([0.0, 1.0 if float(finger[1]) >= 0 else -1.0, 0.0], device=device)
        err = torch.linalg.cross(finger, across) + 0.3 * torch.linalg.cross(approach, down)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(approach @ down)))))
        twist = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(finger @ across))))))
        return err, tilt, twist

    def calibrate(side, delta=0.05):
        """Measure d(tip)/d(joint) by nudging each arm joint, then pick the matching layout."""
        base_targets = targets[side].clone()
        step(int(0.5 / dt))
        p0 = tip_mid(side).clone()
        variants = {(b, c): jacobian_variant(side, b, c) for b in (0, 1) for c in (0, 6)}
        measured = []
        for j in arm[side]["joints"]:
            targets[side][0, j] = base_targets[0, j] + delta
            step(int(0.6 / dt))
            measured.append(((tip_mid(side) - p0) / delta).clone())
            targets[side] = base_targets.clone()
            step(int(0.6 / dt))
        measured = torch.stack(measured, dim=1)  # 3 x 6
        measured_jac[side] = measured.clone()
        scores = {}
        for key, jac in variants.items():
            if jac is None:
                scores[f"body-{key[0]}_cols+{key[1]}"] = None
                continue
            err = torch.linalg.norm(jac - measured) / max(float(torch.linalg.norm(measured)), 1e-9)
            scores[f"body-{key[0]}_cols+{key[1]}"] = round(float(err), 3)
        valid = {k: v for k, v in scores.items() if v is not None}
        best = min(valid, key=valid.get)
        layout[side] = (int(best.split("_")[0].split("-")[1]), int(best.split("+")[1]))
        return {
            "jacobian_shape": list(raw_jacobian(side).shape),
            "relative_error_by_layout": scores,
            "chosen": best,
            "measured_column_norms_m_per_rad": [round(float(v), 3) for v in torch.linalg.norm(measured, dim=0)],
        }

    def quat_to_rot(q):
        x, y, z, w = [float(v) for v in q]  # xyzw, measured on this runtime
        return torch.tensor(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            device=device,
        )

    pad_local = {}

    def measure_pads(side):
        """Contact point near each fingertip, in that finger body's own frame.

        The finger bodies' origins are not at the pads: closed fingers' origins were 72 mm
        apart (b01e670). Local bounds do not depend on the stale world transforms USD keeps
        for articulation links, so this is valid at any time.
        """
        from pxr import Usd, UsdGeom

        stage = sim.stage if hasattr(sim, "stage") else sim_utils.get_current_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide"])
        root = stage.GetPrimAtPath("/World/RobotLeft" if side == "left" else "/World/RobotRight")
        pos = _tensor(robots[side].data.body_pos_w)[0]
        quat = _tensor(robots[side].data.body_quat_w)[0]
        gripper_pos = pos[arm[side]["gripper"]]
        points, report = [], {}
        for k, tip in enumerate(("tip_left", "tip_right")):
            prim = next(p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.GetName() == tip)
            rng = cache.ComputeRelativeBound(prim, prim).ComputeAlignedRange()
            mid = torch.tensor([float(v) for v in rng.GetMidpoint()], device=device)
            half = torch.tensor([float(v) for v in (rng.GetMax() - rng.GetMin())], device=device) / 2.0
            # The fingers are long: steering the mesh centre left the pads ~40 mm above the cell
            # (cc2d125). Use the end of the finger's longest axis that lies farther from the
            # gripper body, stepped back 6 mm onto the contact area.
            axis = int(torch.argmax(half))
            body = arm[side]["tips"][k]
            rot = quat_to_rot(quat[body].tolist())
            ends = []
            for sign in (1.0, -1.0):
                local = mid.clone()
                local[axis] = mid[axis] + sign * (half[axis] - 0.006)
                world = pos[body] + rot @ local
                ends.append((float(torch.linalg.norm(world - gripper_pos)), local))
            tip_local = max(ends, key=lambda item: item[0])[1]
            points.append(tip_local)
            report[tip] = {
                "mesh_centre_mm": [round(1000 * float(v), 1) for v in mid],
                "length_mm": round(2000 * float(half[axis]), 1),
                "contact_point_mm": [round(1000 * float(v), 1) for v in tip_local],
            }
        pad_local[side] = points
        return report

    def pad_centres(side):
        pos = _tensor(robots[side].data.body_pos_w)[0]
        quat = _tensor(robots[side].data.body_quat_w)[0]
        out = []
        for k, body in enumerate(arm[side]["tips"]):
            local = pad_local[side][k] if side in pad_local else torch.zeros(3, device=device)
            out.append(pos[body] + quat_to_rot(quat[body].tolist()) @ local)
        return out

    def tip_mid(side):
        """Grasp point: midpoint of the two finger pads (geometry centres), not the body origins."""
        left_pad, right_pad = pad_centres(side)
        return 0.5 * (left_pad + right_pad)

    def gripper_axes(side):
        quat = _tensor(robots[side].data.body_quat_w)[0, arm[side]["gripper"]].cpu().tolist()
        x, y, z, w = quat  # xyzw, measured on this runtime
        m = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        return {axis: [round(m[r][c], 3) for r in range(3)] for c, axis in enumerate("xyz")}

    def at_limit(side):
        robot = robots[side]
        limits = _tensor(robot.data.soft_joint_pos_limits)[0]
        pos = _tensor(robot.data.joint_pos)[0]
        hits = []
        for j, name in zip(arm[side]["joints"], bindings[side].arm_joints):
            lo, hi = float(limits[j, 0]), float(limits[j, 1])
            if pos[j] <= lo + 0.02 or pos[j] >= hi - 0.02:
                hits.append(name)
        return hits

    def reach(side, goal, seconds=None, orient=False):
        """Closed-loop DLS IK on the fingertip midpoint with a measured, Broyden-updated Jacobian.

        The analytic Jacobian is for the gripper body, not the fingertips, and even the best
        layout was 27% off the measured one (2c545c3), so it never converged. Start from the
        Jacobian measured at home and correct it every step from the observed tip motion.
        """
        goal_t = torch.tensor(goal, device=device, dtype=torch.float32)
        robot = robots[side]
        joints = arm[side]["joints"]
        limits = _tensor(robot.data.soft_joint_pos_limits)[0]
        steps = int((seconds or args.reach_s) / dt)
        damping = 0.02
        jac = (measured_jac[side] if side in measured_jac else jacobian(side)).clone()
        start_residual = float(torch.linalg.norm(goal_t - tip_mid(side)))
        q_prev = _tensor(robot.data.joint_pos)[0, joints].clone()
        p_prev = tip_mid(side).clone()
        for _ in range(steps):
            error = goal_t - tip_mid(side)
            if float(torch.linalg.norm(error)) < args.tolerance_m * 0.5 and (
                not orient or orientation_error(side)[2] < 5.0
            ):
                break
            if orient:
                rot_err, _, _ = orientation_error(side)
                full_jac = torch.cat([jac, 0.05 * angular_jacobian(side)], dim=0)  # 5 cm per rad
                full_err = torch.cat([error, 0.05 * rot_err])
                jjt = full_jac @ full_jac.T + (damping**2) * torch.eye(6, device=device)
                dq = torch.clamp(full_jac.T @ torch.linalg.solve(jjt, full_err), -0.02, 0.02)
            else:
                jjt = jac @ jac.T + (damping**2) * torch.eye(3, device=device)
                dq = torch.clamp(jac.T @ torch.linalg.solve(jjt, error), -0.02, 0.02)
            q_now = _tensor(robot.data.joint_pos)[0, joints]
            for k, j in enumerate(joints):
                # Keep the target within 0.15 rad of the actual joint so it cannot wind up far ahead.
                new = torch.clamp(targets[side][0, j] + dq[k], q_now[k] - 0.15, q_now[k] + 0.15)
                targets[side][0, j] = torch.clamp(new, limits[j, 0], limits[j, 1])
            step()
            q = _tensor(robot.data.joint_pos)[0, joints].clone()
            p = tip_mid(side).clone()
            d_q, d_p = q - q_prev, p - p_prev
            if float(d_q @ d_q) > 1e-8:
                jac = jac + torch.outer(d_p - jac @ d_q, d_q) / float(d_q @ d_q)
            q_prev, p_prev = q, p
        step(30)  # let the drives settle on the last target
        error = goal_t - tip_mid(side)
        residual = float(torch.linalg.norm(error))
        return {
            "goal": [round(v, 4) for v in goal],
            "reached": residual <= args.tolerance_m,
            "start_residual_mm": round(1000 * start_residual, 2),
            "residual_mm": round(1000 * residual, 2),
            "joints_at_limit": at_limit(side),
            "target_minus_actual_rad": [
                round(float(targets[side][0, j] - _tensor(robot.data.joint_pos)[0, j]), 3) for j in joints
            ],
            "gripper_axes_world": gripper_axes(side),
            "approach_tilt_from_down_deg": round(orientation_error(side)[1], 1) if side in approach_local else None,
            "finger_axis_off_world_y_deg": round(orientation_error(side)[2], 1) if side in approach_local else None,
            "joint_pos_rad": [round(float(_tensor(robot.data.joint_pos)[0, j]), 3) for j in joints],
        }

    def nearer_arm(xy):
        return min(robots, key=lambda side: math.dist(xy, (cfg.robot.left_pose if side == "left" else cfg.robot.right_pose).position_xyz[:2]))

    def home(side):
        targets[side] = _tensor(robots[side].data.default_joint_pos).clone()
        finger_cmd[side] = cfg.robot.finger_open_m
        step(int(1.5 / dt))

    step(int(1.0 / dt))
    report["finger_pad_centres_local_mm"] = {side: _stage({}, side, lambda side=side: measure_pads(side)) for side in robots}

    def pad_gaps(side):
        """Pad-centre separation fully open and fully closed on nothing, as a reference for grasps."""
        finger_cmd[side] = cfg.robot.finger_open_m
        step(int(0.5 / dt))
        a, b = pad_centres(side)
        opened = float(torch.linalg.norm(a - b))
        finger_cmd[side] = cfg.robot.finger_closed_m
        step(int(1.0 / dt))
        a, b = pad_centres(side)
        closed = float(torch.linalg.norm(a - b))
        finger_cmd[side] = cfg.robot.finger_open_m
        step(int(0.5 / dt))
        return {"open_mm": round(1000 * opened, 1), "closed_empty_mm": round(1000 * closed, 1)}

    report["pad_gap_reference"] = {side: _stage({}, side, lambda side=side: pad_gaps(side)) for side in robots}
    report["jacobian_calibration"] = {}
    for side in robots:
        # Fixed-base convention as the fallback if calibration itself fails.
        layout[side] = (1, 0)
        _stage(report["jacobian_calibration"], side, lambda side=side: calibrate(side))
    report["approach_axis_in_gripper_frame"] = {side: measure_approach_axis(side) for side in robots}
    report["arms"] = {
        side: {
            "is_fixed_base": bool(getattr(robots[side], "is_fixed_base", None)),
            "tip_mid_at_home": [round(float(v), 4) for v in tip_mid(side).cpu().tolist()],
            "gripper_axes_at_home": gripper_axes(side),
        }
        for side in robots
    }

    def reach_cells():
        rows = {}
        for name in cells:
            pose = battery_pose(cfg, name)
            x, y, z = pose.position_xyz
            side = nearer_arm((x, y))
            rows[name] = {
                "arm": side,
                "above": reach(side, (x, y, z + args.approach_m)),
                "at": reach(side, (x, y, z)),
            }
            home(side)
        return rows

    def reach_slots():
        rows = {}
        rpos, rquat = cfg.remote_pose.position_xyz, cfg.remote_pose.rotation_xyzw
        for slot in active_slots(cfg):
            seat = local_to_world(rpos, rquat, (cfg.compartment.seat_center_x, slot.center_y_m, rest_center_z(cfg)))
            side = nearer_arm(seat[:2])
            rows[slot.name] = {
                "arm": side,
                "above": reach(side, (seat[0], seat[1], seat[2] + args.approach_m)),
                "at_seat_height_plus_2cm": reach(side, (seat[0], seat[1], seat[2] + 0.02)),
            }
            home(side)
        return rows

    def grasp_geometry(side, cell_name):
        """World bounding boxes of both finger bodies and the cell just before closing.

        Says whether the cell actually lies between the finger pads: the IK steers the midpoint
        of the finger bodies' origins, which need not be where the pads are.
        """
        from pxr import Usd, UsdGeom

        stage = sim.stage if hasattr(sim, "stage") else sim_utils.get_current_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide"])
        robot_root = stage.GetPrimAtPath("/World/RobotLeft" if side == "left" else "/World/RobotRight")

        def box(prim):
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            return {"min_mm": [round(1000 * v, 1) for v in rng.GetMin()], "max_mm": [round(1000 * v, 1) for v in rng.GetMax()]}

        # USD keeps articulation links at their spawn pose (b01e670 showed the finger boxes at
        # home), so report the pads from the live body poses instead.
        out = {f"{tip}_pad_mm": [round(1000 * float(v), 1) for v in pad] for tip, pad in zip(("tip_left", "tip_right"), pad_centres(side))}
        cell_pos = _tensor(cells[cell_name].data.root_pos_w)[0]
        out["cell_centre_mm"] = [round(1000 * float(v), 1) for v in cell_pos]
        pos = _tensor(robots[side].data.body_pos_w)[0]
        out["tip_body_origins_mm"] = [[round(1000 * float(v), 1) for v in pos[i]] for i in arm[side]["tips"]]
        out["table_top_mm"] = round(1000 * cfg.table.top_z_m, 1)
        return out

    def reset_cells():
        for cell_name, cell in cells.items():
            start = battery_pose(cfg, cell_name)
            pose = torch.tensor([[*start.position_xyz, *start.rotation_xyzw]], device=device, dtype=torch.float32)
            cell.write_root_pose_to_sim(pose)
            cell.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        step(int(0.3 / dt))

    def grasp_cells():
        rows = {}
        for name, cell in cells.items():
            pose = battery_pose(cfg, name)
            x, y, z = pose.position_xyz
            side = nearer_arm((x, y))
            # The reach stages knocked battery_b ~100 mm off its mat (cc2d125); start each grasp
            # with both cells back on their marks.
            reset_cells()
            finger_cmd[side] = cfg.robot.finger_open_m
            above = reach(side, (x, y, z + args.approach_m), orient=True)
            at = reach(side, (x, y, z), orient=True)
            boxes = _stage({}, "boxes", lambda: grasp_geometry(side, name))
            finger_cmd[side] = cfg.robot.finger_closed_m
            step(int(1.0 / dt))
            fingers = [round(float(_tensor(robots[side].data.joint_pos)[0, j]), 4) for j in arm[side]["fingers"]]
            pads_after = pad_centres(side)
            pad_gap_mm = round(1000 * float(torch.linalg.norm(pads_after[0] - pads_after[1])), 1)
            start_z = float(_tensor(cell.data.root_pos_w)[0, 2])
            lift = reach(side, (x, y, z + 0.05), orient=True)
            end = _tensor(cell.data.root_pos_w)[0].cpu().tolist()
            rise = end[2] - start_z
            rows[name] = {
                "arm": side,
                "approach_reached": above["reached"],
                "grasp_point_reached": at["reached"],
                "grasp_residual_mm": at["residual_mm"],
                "finger_pos_after_close_m": fingers,
                "fingers_stopped_by_cell": all(abs(f - cfg.robot.finger_closed_m) > 0.002 for f in fingers),
                "cell_rise_mm": round(1000 * rise, 1),
                "lifted": rise > 0.03,
                "gripper_axes_at_grasp": at["gripper_axes_world"],
                "before_close": boxes,
                "pad_gap_after_close_mm": pad_gap_mm,
                "cell_diameter_mm": round(1000 * cfg.battery.diameter_m, 1),
                "approach_tilt_from_down_deg": at["approach_tilt_from_down_deg"],
                "finger_axis_off_world_y_deg": at["finger_axis_off_world_y_deg"],
            }
            finger_cmd[side] = cfg.robot.finger_open_m
            step(int(0.5 / dt))
            home(side)
        return rows

    _stage(report, "reach_cells", reach_cells)
    _stage(report, "reach_slots", reach_slots)
    _stage(report, "grasp_cells", grasp_cells)

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("ROBOCURVE_REACH_CHECK_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("ROBOCURVE_REACH_CHECK_END", flush=True)
    app.close()


if __name__ == "__main__":
    main()
