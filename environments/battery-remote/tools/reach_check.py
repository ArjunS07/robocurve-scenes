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
- a robot insertion: grasp a cell, tilt it, dip its negative end into the slot, push the
  spring plunger in, release, and apply the scorer's seated test

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
    parser.add_argument("--grasp-raise-m", type=float, default=0.0035, help="Pinch this far above the cell's axis.")
    parser.add_argument("--insert-tilt-deg", type=float, default=30.0, help="Cell tilt about the finger axis while dipping its spring end into the slot.")
    parser.add_argument("--hold-s", type=float, default=1.5, help="Settle time after releasing the inserted cell.")
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

    from my_package.assets.geometry import measure_battery, rest_center_z, spring_side, write_geometry
    from my_package.assets.spawn import make_battery, make_remote
    from my_package.assets.table import maple_table_usd
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
    orient_goal = {"tilt_y_deg": 0.0, "approach_weight": 0.3}

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
        # Insertion tilts the held cell about world Y (the finger axis) to dip one end into the
        # bay first; otherwise the approach goal is straight down.
        theta = math.radians(orient_goal["tilt_y_deg"])
        down = torch.tensor([-math.sin(theta), 0.0, -math.cos(theta)], device=device)
        # Fingers must close horizontally across the cell: along world +-Y. Only being
        # perpendicular to X let them close vertically, one above the cell and one under it.
        across = torch.tensor([0.0, 1.0 if float(finger[1]) >= 0 else -1.0, 0.0], device=device)
        err = torch.linalg.cross(finger, across) + orient_goal["approach_weight"] * torch.linalg.cross(approach, down)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(approach @ down)))))
        twist = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(finger @ across))))))
        return err, tilt, twist

    analytic = {}

    def lever_jacobian(side, body_shift, col_shift):
        """Fingertip-point Jacobian from the gripper body's: v_tip = v_body + w x r.

        The 27% mismatch (2c545c3) compared the body's linear rows with the tips' motion and
        left out this lever, r ~ 0.2 m from the gripper origin to the pads. Recomputed every
        step, it needs no nudging calibration and no Broyden updates.
        """
        jac = raw_jacobian(side)
        body = arm[side]["gripper"] - body_shift
        cols = [j + col_shift for j in arm[side]["joints"]]
        if not (0 <= body < jac.shape[1]) or max(cols) >= jac.shape[3]:
            return None
        block = jac[0, body][:, torch.tensor(cols, device=jac.device)]
        r = [float(v) for v in tip_mid(side) - _tensor(robots[side].data.body_pos_w)[0, arm[side]["gripper"]]]
        skew = torch.tensor([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]], device=block.device)
        return block[:3] - skew @ block[3:6]

    def calibrate(side, delta=0.05):
        """Measure d(tip)/d(joint) by nudging each arm joint, then pick the matching layout."""
        base_targets = targets[side].clone()
        step(int(0.5 / dt))
        p0 = tip_mid(side).clone()
        variants = {(b, c): jacobian_variant(side, b, c) for b in (0, 1) for c in (0, 6)}
        lever_variants = {(b, c): lever_jacobian(side, b, c) for b in (0, 1) for c in (0, 6)}
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
        lever_scores = {}
        for key, jac in lever_variants.items():
            name = f"lever_body-{key[0]}_cols+{key[1]}"
            lever_scores[name] = None if jac is None else round(float(torch.linalg.norm(jac.to(measured.device) - measured) / max(float(torch.linalg.norm(measured)), 1e-9)), 3)
        lever_valid = {k: v for k, v in lever_scores.items() if v is not None}
        scores.update(lever_scores)
        analytic.pop(side, None)
        if lever_valid:
            lever_best = min(lever_valid, key=lever_valid.get)
            # Drive sag under load makes the measured columns a little short; 25% still means
            # the same Jacobian, while a wrong layout is off by ~100%.
            if lever_valid[lever_best] < 0.25:
                key = (int(lever_best.split("_")[1].split("-")[1]), int(lever_best.split("+")[1]))
                analytic[side] = layout[side] = key
                best = lever_best
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

    def reach(side, goal, seconds=None, orient=False, retries=1):
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
        # Slow integral term that holds gravity sag; it starts at whatever offset holds the arm now.
        bias = (targets[side][0, joints] - q_prev).clone()
        for _ in range(steps):
            error = goal_t - tip_mid(side)
            if float(torch.linalg.norm(error)) < args.tolerance_m * 0.5 and (
                not orient or orientation_error(side)[2] < 5.0
            ):
                break
            if side in analytic:
                jac = lever_jacobian(side, *analytic[side])
            if orient:
                rot_err, _, _ = orientation_error(side)
                full_jac = torch.cat([jac, 0.05 * angular_jacobian(side)], dim=0)  # 5 cm per rad
                full_err = torch.cat([error, 0.05 * rot_err])
                jjt = full_jac @ full_jac.T + (damping**2) * torch.eye(6, device=device)
                dq = torch.clamp(full_jac.T @ torch.linalg.solve(jjt, full_err), -0.1, 0.1)
            else:
                jjt = jac @ jac.T + (damping**2) * torch.eye(3, device=device)
                dq = torch.clamp(jac.T @ torch.linalg.solve(jjt, error), -0.1, 0.1)
            q_now = _tensor(robot.data.joint_pos)[0, joints]
            # Target = actual + this step's correction + a slow integral. Adding the full step to the
            # previous target each tick was an integrator with gain 1 per 8 ms tick against drives
            # that lag ~0.1 s, so it oscillated inside its 0.15 rad window and stopped 9-39 mm off
            # wherever the clock ran out (594e24e), with no joint at a limit.
            bias = torch.clamp(bias + 0.02 * dq, -0.15, 0.15)
            for k, j in enumerate(joints):
                targets[side][0, j] = torch.clamp(q_now[k] + dq[k] + bias[k], limits[j, 0], limits[j, 1])
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
        if residual > args.tolerance_m and retries > 0:
            # The Jacobian measured at home is a poor start far from home: the right arm stopped
            # 39.5 mm off battery_a with its orientation met (be1f60b). Re-measure it here and go
            # again, then restore the home one so later reaches start as before.
            saved_jac, saved_layout = measured_jac.get(side), layout[side]
            try:
                calibrate(side, delta=0.02)
                again = reach(side, goal, seconds, orient, retries - 1)
            finally:
                layout[side] = saved_layout
                if saved_jac is None:
                    measured_jac.pop(side, None)
                else:
                    measured_jac[side] = saved_jac
            again["first_attempt_residual_mm"] = round(1000 * residual, 2)
            again["recalibrated_at_pose"] = True
            return again
        return {
            "jacobian": "analytic_lever" if side in analytic else "measured_broyden",
            "goal": [round(v, 4) for v in goal],
            "reached": residual <= args.tolerance_m,
            "start_residual_mm": round(1000 * start_residual, 2),
            "residual_mm": round(1000 * residual, 2),
            "joints_at_limit": at_limit(side),
            # Drive torque on each arm joint at the end; a value at the effort limit means the
            # joint is saturated and cannot hold its target (sag the IK cannot remove).
            "applied_torque_n_m": [round(float(v), 2) for v in _tensor(robots[side].data.applied_torque)[0, arm[side]["joints"]]],
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

    def grasp(name, side, release=True):
        """Open, descend on the cell with the fingers across it, close, lift 5 cm. One row."""
        cell = cells[name]
        x, y, z = battery_pose(cfg, name).position_xyz
        # The finger ends reach ~6 mm below the contact points, so pads at the cell's axis put
        # them into the 1 mm mat. At +1.5 mm (ff38750) the pads arrived 1.2 mm low and both arms'
        # fingers stalled ~41 mm apart on a 10.5 mm cell (pad-centre gap 18-23 mm; the nut check
        # reads 1.3 mm on its 24 mm flats), dragging on the mat. Flat vertical finger faces touch a
        # lying cylinder at its widest line whatever the pinch height, as long as the faces reach
        # below the axis; +3.5 mm leaves the finger ends ~3 mm over the mat with the axis 2.5 mm
        # above them.
        z = z + args.grasp_raise_m
        finger_cmd[side] = cfg.robot.finger_open_m
        above = reach(side, (x, y, z + args.approach_m), orient=True)
        # Re-measure the Jacobian here, in free air, for the descent. Recalibrating at the cell
        # instead flicked both cells off the table (252e198), so nothing near the cell retries.
        saved_jac, saved_layout = measured_jac.get(side), layout[side]
        try:
            calibrate(side, delta=0.02)
            at = reach(side, (x, y, z), orient=True, retries=0)
            if not at["reached"]:
                # The right arm stopped 22 mm short in X (c9c5606). Back off 3 cm into free air,
                # re-measure there and come down again; never re-measure at the cell.
                first = at["residual_mm"]
                reach(side, (x, y, z + 0.03), orient=True, retries=0)
                calibrate(side, delta=0.02)
                at = dict(reach(side, (x, y, z), orient=True, retries=0), first_attempt_residual_mm=first)
            boxes = _stage({}, "boxes", lambda: grasp_geometry(side, name))
            finger_cmd[side] = cfg.robot.finger_closed_m
            step(int(1.0 / dt))
            fingers = [round(float(_tensor(robots[side].data.joint_pos)[0, j]), 4) for j in arm[side]["fingers"]]
            pads_after = pad_centres(side)
            pad_gap_mm = round(1000 * float(torch.linalg.norm(pads_after[0] - pads_after[1])), 1)
            start_z = float(_tensor(cell.data.root_pos_w)[0, 2])
            after_close = {
                "cell_centre_mm": [round(1000 * float(v), 1) for v in _tensor(cell.data.root_pos_w)[0]],
                "pads_mm": [[round(1000 * float(v), 1) for v in pad] for pad in pads_after],
            }
            lift = reach(side, (x, y, z + 0.05), orient=True, retries=0)
        finally:
            layout[side] = saved_layout
            if saved_jac is not None:
                measured_jac[side] = saved_jac
        rise = float(_tensor(cell.data.root_pos_w)[0, 2]) - start_z
        row = {
            "arm": side,
            "approach_reached": above["reached"],
            "grasp_point_reached": at["reached"],
            "grasp_residual_mm": at["residual_mm"],
            "first_attempt_residual_mm": at.get("first_attempt_residual_mm"),
            "joints_at_limit": at["joints_at_limit"],
            "joint_pos_rad": at["joint_pos_rad"],
            "applied_torque_n_m": at["applied_torque_n_m"],
            "target_minus_actual_rad": at["target_minus_actual_rad"],
            "approach": {k: above.get(k) for k in ("reached", "residual_mm", "target_minus_actual_rad")},
            "finger_pos_after_close_m": fingers,
            "fingers_stopped_by_cell": all(abs(f - cfg.robot.finger_closed_m) > 0.002 for f in fingers),
            "cell_rise_mm": round(1000 * rise, 1),
            "lifted": rise > 0.03,
            "gripper_axes_at_grasp": at["gripper_axes_world"],
            "before_close": boxes,
            "after_close": after_close,
            "lift": {k: lift.get(k) for k in ("reached", "residual_mm", "start_residual_mm", "joints_at_limit")},
            "cell_after_lift_mm": [round(1000 * float(v), 1) for v in _tensor(cell.data.root_pos_w)[0]],
            "pad_gap_after_close_mm": pad_gap_mm,
            "cell_diameter_mm": round(1000 * cfg.battery.diameter_m, 1),
            "approach_tilt_from_down_deg": at["approach_tilt_from_down_deg"],
            "finger_axis_off_world_y_deg": at["finger_axis_off_world_y_deg"],
        }
        if release:
            finger_cmd[side] = cfg.robot.finger_open_m
            step(int(0.5 / dt))
            home(side)
        return row

    def grasp_cells():
        rows = {}
        for name in cells:
            x, y, _ = battery_pose(cfg, name).position_xyz
            side = nearer_arm((x, y))
            # The reach stages knocked battery_b ~100 mm off its mat (cc2d125); start each grasp
            # with both cells back on their marks.
            reset_cells()
            rows[name] = grasp(name, side)
        return rows

    def cell_axis(name):
        """World centre and unit +Z axis (positive terminal) of a cell."""
        quat = _tensor(cells[name].data.root_quat_w)[0].tolist()
        return _tensor(cells[name].data.root_pos_w)[0].clone(), quat_to_rot(quat)[:, 2]

    def insert_cell():
        """Pick a cell with the arm that lifted it, dip its negative end into the slot against the
        spring, push the plunger in, let go so the cell drops level and the spring seats it.

        A level drop does not fit: the bay at rest is 1.5 mm shorter than a cell (spring
        preload), and the fingers cannot follow the cell below the rim (13 mm channel for a
        10.5 mm cell). So tilt the cell about the finger axis, steer its low end, and release
        above the rim. The low end is steered from the live cell pose, so the cell's offset in
        the hand does not matter.
        """
        rows = {}
        lifted = report.get("grasp_cells", {})
        order = [n for n in cells if isinstance(lifted.get(n), dict) and lifted[n].get("lifted")] or list(cells)
        name = order[0]
        side = nearer_arm(battery_pose(cfg, name).position_xyz[:2])
        rpos, rquat = cfg.remote_pose.position_xyz, cfg.remote_pose.rotation_xyzw
        # Cells start positive end toward +X, so pick the slot that wants +X: no 180 deg turn.
        slot = next((s for s in active_slots(cfg) if s.required_axis_xyz[0] > 0), active_slots(cfg)[0])
        side_sign = spring_side(slot)  # +1 when the spring is at the slot's +X end
        joint = int(remote.find_joints(f"^{plunger_joint_name(slot)}$")[0][0])
        half = cfg.battery.length_m / 2.0
        axis_z = rest_center_z(cfg)
        seat_x = cfg.compartment.seat_center_x
        rows.update({"cell": name, "arm": side, "slot": slot.name, "spring_side": side_sign})

        def plunger_mm():
            return round(1000 * float(_tensor(remote.data.joint_pos)[0, joint]), 3)

        def low_end():
            centre, axis = cell_axis(name)
            # The spring end is the negative terminal: -axis from the centre.
            return centre - half * axis

        def cell_tilt_deg():
            _, axis = cell_axis(name)
            return round(math.degrees(math.asin(max(-1.0, min(1.0, float(axis[2]))))), 1)

        def move_low_end(local, passes=2, **kw):
            goal = torch.tensor(local_to_world(rpos, rquat, local), device=device, dtype=torch.float32)
            row = None
            for _ in range(passes):  # steer the tips by the low end's measured error
                tip_goal = tip_mid(side) + (goal - low_end())
                row = reach(side, tuple(float(v) for v in tip_goal), orient=True, retries=0, **kw)
            err = goal - low_end()
            return {
                "low_end_error_mm": [round(1000 * float(v), 1) for v in err],
                "tip_residual_mm": row["residual_mm"],
                "approach_tilt_from_goal_deg": row["approach_tilt_from_down_deg"],
                "cell_tilt_deg": cell_tilt_deg(),
                "plunger_compression_mm": plunger_mm(),
                "joints_at_limit": row["joints_at_limit"],
            }

        reset_cells()
        rows["grasp"] = grasp(name, side, release=False)
        if not rows["grasp"]["lifted"]:
            # 252e198 went on with an empty hand and steered nothing.
            finger_cmd[side] = cfg.robot.finger_open_m
            home(side)
            rows["skipped"] = "grasp did not lift the cell"
            return rows
        # Positive tilt about Y lowers +X; lower the spring end, which is at side_sign * X.
        orient_goal.update(tilt_y_deg=side_sign * args.insert_tilt_deg, approach_weight=1.0)
        try:
            end_x = seat_x + side_sign * (half - 0.0025)  # 1 mm short of the resting plunger face
            push_x = seat_x + side_sign * (half + 0.0020)  # plunger in 3.5 of its 4 mm
            rows["tilt_above"] = move_low_end((end_x, slot.center_y_m, axis_z + 0.03))
            rows["dip"] = move_low_end((end_x, slot.center_y_m, axis_z + 0.001))
            rows["push"] = move_low_end((push_x, slot.center_y_m, axis_z + 0.001))
        finally:
            orient_goal.update(tilt_y_deg=0.0, approach_weight=0.3)
        finger_cmd[side] = cfg.robot.finger_open_m
        step(int(1.0 / dt))
        tips = tip_mid(side)
        reach(side, (float(tips[0]), float(tips[1]), float(tips[2]) + 0.05), orient=True, retries=0)
        home(side)
        step(int(args.hold_s / dt))
        position = tuple(_tensor(cells[name].data.root_pos_w)[0].tolist())
        rotation = tuple(_tensor(cells[name].data.root_quat_w)[0].tolist())
        measured = measure_battery(cfg, slot, position, rotation, rpos, rquat, plunger_joint_m=plunger_mm() / 1000.0)
        speed = float(torch.linalg.norm(_tensor(cells[name].data.root_lin_vel_w)[0]))
        rows["after_release"] = {
            "axial_offset_mm": round(1000 * measured.axial_offset_m, 2),
            "radial_offset_mm": round(1000 * measured.radial_offset_m, 2),
            "polarity_angle_deg": round(measured.polarity_angle_deg, 1),
            "spring_contact": measured.spring_contact,
            "plunger_compression_mm": plunger_mm(),
            "speed_mm_s": round(1000 * speed, 2),
            "cell_tilt_deg": cell_tilt_deg(),
            # The scorer's seated test, as in tools/physics_check.py.
            "seated": bool(
                abs(measured.axial_offset_m) <= cfg.scoring.axial_tolerance_m
                and measured.radial_offset_m <= cfg.scoring.seat_radial_max_m
                and measured.spring_contact is True
                and measured.polarity_angle_deg <= cfg.scoring.polarity_angle_deg
            ),
        }
        return rows

    _stage(report, "reach_cells", reach_cells)
    _stage(report, "reach_slots", reach_slots)
    _stage(report, "grasp_cells", grasp_cells)
    _stage(report, "insert_cell", insert_cell)

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("ROBOCURVE_REACH_CHECK_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("ROBOCURVE_REACH_CHECK_END", flush=True)
    app.close()


if __name__ == "__main__":
    main()
