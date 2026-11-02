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

    def jacobian(side):
        robot = robots[side]
        view = getattr(robot, "root_physx_view", None) or getattr(robot, "root_view", None)
        jac = _tensor(view.get_jacobians())
        if not isinstance(jac, torch.Tensor):
            # This build can hand back a Warp/proxy array; list indexing on it fails.
            try:
                import warp as wp

                jac = wp.to_torch(jac)
            except Exception:
                jac = torch.as_tensor(jac, device=device)
        fixed = bool(getattr(robot, "is_fixed_base", True))
        # Fixed base: the root body has no Jacobian row. Floating base (the YAM arms report
        # is_fixed_base False here even though root_joint anchors them): 6 root columns come first.
        body = arm[side]["gripper"] - (1 if fixed else 0)
        cols = arm[side]["joints"] if fixed else [j + 6 for j in arm[side]["joints"]]
        jac_body = jac[0, body, :3, :]
        return jac_body[:, torch.tensor(cols, device=jac_body.device)]

    def tip_mid(side):
        pos = _tensor(robots[side].data.body_pos_w)[0]
        return 0.5 * (pos[arm[side]["tips"][0]] + pos[arm[side]["tips"][1]])

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

    def reach(side, goal, seconds=None):
        """Closed-loop DLS IK on the fingertip midpoint; returns the residual and diagnostics."""
        goal_t = torch.tensor(goal, device=device, dtype=torch.float32)
        robot = robots[side]
        limits = _tensor(robot.data.soft_joint_pos_limits)[0]
        steps = int((seconds or args.reach_s) / dt)
        damping = 0.05
        error = None
        for _ in range(steps):
            error = goal_t - tip_mid(side)
            if float(torch.linalg.norm(error)) < args.tolerance_m * 0.5:
                break
            jac = jacobian(side)
            jjt = jac @ jac.T + (damping**2) * torch.eye(3, device=device)
            dq = jac.T @ torch.linalg.solve(jjt, error)
            dq = torch.clamp(dq, -0.02, 0.02)
            for k, j in enumerate(arm[side]["joints"]):
                targets[side][0, j] = torch.clamp(targets[side][0, j] + dq[k], limits[j, 0], limits[j, 1])
            step()
        step(30)  # let the drives settle on the last target
        error = goal_t - tip_mid(side)
        residual = float(torch.linalg.norm(error))
        return {
            "goal": [round(v, 4) for v in goal],
            "reached": residual <= args.tolerance_m,
            "residual_mm": round(1000 * residual, 2),
            "joints_at_limit": at_limit(side),
            "gripper_axes_world": gripper_axes(side),
            "joint_pos_rad": [round(float(_tensor(robot.data.joint_pos)[0, j]), 3) for j in arm[side]["joints"]],
        }

    def nearer_arm(xy):
        return min(robots, key=lambda side: math.dist(xy, (cfg.robot.left_pose if side == "left" else cfg.robot.right_pose).position_xyz[:2]))

    def home(side):
        targets[side] = _tensor(robots[side].data.default_joint_pos).clone()
        finger_cmd[side] = cfg.robot.finger_open_m
        step(int(1.5 / dt))

    step(int(1.0 / dt))
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

    def grasp_cells():
        rows = {}
        for name, cell in cells.items():
            pose = battery_pose(cfg, name)
            x, y, z = pose.position_xyz
            side = nearer_arm((x, y))
            finger_cmd[side] = cfg.robot.finger_open_m
            above = reach(side, (x, y, z + args.approach_m))
            at = reach(side, (x, y, z))
            finger_cmd[side] = cfg.robot.finger_closed_m
            step(int(1.0 / dt))
            fingers = [round(float(_tensor(robots[side].data.joint_pos)[0, j]), 4) for j in arm[side]["fingers"]]
            start_z = float(_tensor(cell.data.root_pos_w)[0, 2])
            lift = reach(side, (x, y, z + 0.05))
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
