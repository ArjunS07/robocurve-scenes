"""Server-side robot check for the M16 nut scene with the real YAM arms. One JSON report.

Run with the Arena interpreter (headless Kit, no streaming, no Arena runner):

    cd /home/ubuntu/workspace/robocurve-scenes/environments/atb1-nut
    /home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python tools/reach_check.py

physics_check.py turned the nut with a scripted wrench. This moves the arm nearer the
nut with damped-least-squares IK on the fingertip contact points (the method proven
in battery-remote/tools/reach_check.py) and scores every physics step with the task
scorer:

1. grasp    - top-down, fingers closing across two opposite flats of the nut on its mat; lift 5 cm
2. carry    - nut axis vertical above the bolt tip, coaxial with the bolt
3. lower    - onto the tip until it rests in the lead-in band
4. turn     - wrist (joint6) clockwise from above in strokes; between strokes open, turn the
              wrist back onto the next pair of flats, re-target the nut's current height, close
5. release  - open and back away; seated and released scores 100

If the grasp or carry fails, the turn stage still runs: the nut is placed on the bolt tip by
pose and grasped there, so a pick failure does not hide whether the robot can thread it.
It is a feasibility probe, not a policy. Outputs go to outputs/reach_check/<git revision>/.
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
HEX_PERIOD_DEG = 60.0


def _revision() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "--", "."], cwd=ROOT) != 0
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
            "error_at": " | ".join(line.strip() for line in traceback.format_exc(limit=6).splitlines() if line.strip().startswith("File")),
        }
    return report[key]


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


# ---- pure helpers (no Isaac imports; tested in tests/test_reach_check_tool.py) ----------------


def wrap_deg(angle: float, period: float = 360.0) -> float:
    """Wrap into [-period/2, period/2)."""
    return (angle + period / 2.0) % period - period / 2.0


def hex_flat_phase_deg(points_xy, step_deg: float = 0.25) -> float:
    """Direction of a flat's outward normal, in [0, 60), from points of a hex nut's outline.

    Flat normals are where the outline reaches least far: for each candidate phase take the
    farthest projection onto its six directions, and keep the phase where that is smallest.
    """
    points = [(float(x), float(y)) for x, y in points_xy]
    if len(points) < 6:
        raise ValueError("need the nut's outline points")
    best, best_reach = 0.0, math.inf
    for n in range(int(round(HEX_PERIOD_DEG / step_deg))):
        phase = n * step_deg
        reach = 0.0
        for k in range(6):
            a = math.radians(phase + k * HEX_PERIOD_DEG)
            c, s = math.cos(a), math.sin(a)
            reach = max(reach, max(x * c + y * s for x, y in points))
        if reach < best_reach - 1e-12:
            best, best_reach = phase, reach
    return best


def flats_toward(flat_yaw_deg: float, preferred_deg: float) -> float:
    """The flat-normal direction (flat_yaw + k * 60) closest to a preferred direction, mod 180."""
    return preferred_deg + wrap_deg(flat_yaw_deg - preferred_deg, HEX_PERIOD_DEG)


def choose_wrist_target(j6_now: float, yaw_per_rad: float, delta_deg: float, j6_goal: float, lo: float, hi: float) -> float | None:
    """Wrist angle that puts the fingers on a pair of flats, as close to j6_goal as the limits allow.

    delta_deg is how far (mod 60) the finger axis must turn in world yaw to meet the flats;
    yaw_per_rad is the measured finger yaw change per radian of joint6 (about +-1 when the
    approach is vertical). Returns None when no candidate lies within [lo, hi].
    """
    if abs(yaw_per_rad) < 1e-6:
        raise ValueError("joint6 does not turn the fingers about the vertical")
    candidates = []
    for k in range(-8, 9):
        j6 = j6_now + math.radians(delta_deg + k * HEX_PERIOD_DEG) / yaw_per_rad
        if lo <= j6 <= hi:
            candidates.append(j6)
    return min(candidates, key=lambda j6: abs(j6 - j6_goal)) if candidates else None


def summarize_torques(values) -> dict:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return {"samples": 0}
    mags = sorted(abs(v) for v in values)
    return {
        "samples": len(values),
        "mean_abs_n_m": round(sum(mags) / len(mags), 4),
        "p95_abs_n_m": round(mags[min(len(mags) - 1, int(0.95 * len(mags)))], 4),
        "max_abs_n_m": round(mags[-1], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arena-root", type=Path, default=ARENA)
    parser.add_argument("--use-fabric", action="store_true", help="Simulate with Fabric on, as the Arena env does.")
    parser.add_argument("--approach-m", type=float, default=0.06, help="Height above a target for the approach pose.")
    parser.add_argument("--tolerance-m", type=float, default=0.003, help="Fingertip position error that counts as reached.")
    parser.add_argument("--reach-s", type=float, default=4.0, help="Simulated time allowed per target.")
    parser.add_argument("--grasp-raise-m", type=float, default=0.002, help="Contact points this far above the nut's centre.")
    parser.add_argument("--turn-rate", type=float, default=0.2, help="Wrist turns per second while threading.")
    parser.add_argument("--stroke-deg", type=float, default=180.0, help="Wrist travel per stroke before a regrasp.")
    parser.add_argument("--max-strokes", type=int, default=18)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(args.arena_root))
    sys.path.insert(0, str(ROOT))
    import copy

    import torch  # before Kit, as in the launchers

    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(app_parser)
    app = AppLauncher(app_parser.parse_args(["--visualizer", "none"])).app

    revision = _revision()
    out = args.out or ROOT / "outputs" / "reach_check" / (revision + ("-fabric" if args.use_fabric else ""))
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROBOCURVE_GENERATED_DIR", str(out / "assets"))
    report: dict = {"revision": revision, "out": str(out), "use_fabric": args.use_fabric}

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg

    from my_package.assets.factory import resolve_fastener_assets
    from my_package.assets.geometry import ensure_generated_assets
    from my_package.embodiments.action_map import ARM_JOINTS, FINGER_JOINTS
    from my_package.embodiments.grasp import is_holding, mean_open_fraction
    from my_package.embodiments.yam_bimanual import YamBimanualEmbodiment
    from my_package.scoring.config import bolt_position_xyz, load_config
    from my_package.scoring.math3d import angle_deg, rotate
    from my_package.scoring.scorer import carry_episode_memory, nut_centre_xyz, required_travel_m, step as score_step
    from my_package.scoring.snapshot import NutSnapshot, SceneSnapshot, clear_episode_state
    from my_package.tasks.state_values import quaternion_xyzw, sim_quaternion

    cfg = load_config()
    f = cfg.fastener
    travel = required_travel_m(cfg)
    nut_asset, bolt_asset = resolve_fastener_assets(cfg)
    nut_usd, bolt_usd, fixture_usd = ensure_generated_assets(cfg, nut_asset, bolt_asset)
    fx, fy, fz = cfg.fixture_pose.position_xyz
    tip_z = fz + travel  # top of the exposed thread
    centre_up = f.nut_origin_to_lower_face_m + f.nut_height_m / 2.0
    report["thread"] = {"pitch_mm": f.pitch_m * 1000, "travel_mm": round(travel * 1000, 3), "bolt_tip_z_mm": round(tip_z * 1000, 3)}

    dt = 1.0 / 120.0
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda:0", use_fabric=args.use_fabric))
    device = sim.device
    stage = sim.stage if hasattr(sim, "stage") else sim_utils.get_current_stage()

    from isaaclab_arena.assets.background_library import MapleTableRobolab
    from my_package.assets.table import maple_table_usd

    for prim_path, spawn_cfg in (
        ("/World/table", sim_utils.UsdFileCfg(usd_path=maple_table_usd(MapleTableRobolab.usd_path))),
        ("/World/dome", sim_utils.DomeLightCfg(intensity=2000.0)),
    ):
        spawn_cfg.func(prim_path, spawn_cfg)

    def rigid(name, path, position_xyz, rotation_xyzw):
        return RigidObject(
            RigidObjectCfg(
                prim_path=f"/World/{name}",
                spawn=sim_utils.UsdFileCfg(usd_path=str(path)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(position_xyz), rot=sim_quaternion(rotation_xyzw)),
            )
        )

    fixture = rigid("fixture", fixture_usd, cfg.fixture_pose.position_xyz, cfg.fixture_pose.rotation_xyzw)
    bolt = rigid("bolt", bolt_usd, bolt_position_xyz(cfg), cfg.fixture_pose.rotation_xyzw)
    nut = rigid("nut", nut_usd, cfg.nut_pose.position_xyz, cfg.nut_pose.rotation_xyzw)
    # The environment's own arm configs (spawn with colliders, config gains), re-rooted outside an env.
    embodiment = YamBimanualEmbodiment(cfg)
    robot_paths = {"left": "/World/RobotLeft", "right": "/World/RobotRight"}
    robots = {}
    for side, field in (("left", "left_robot"), ("right", "right_robot")):
        arm_cfg = copy.deepcopy(getattr(embodiment.scene_config, field))
        arm_cfg.prim_path = robot_paths[side]
        robots[side] = Articulation(arm_cfg)
    sim.reset()
    assets = [fixture, bolt, nut, *robots.values()]

    def index_of(finder, name):
        ids, names = finder(f"^{name}$")
        if len(names) != 1:
            raise KeyError(f"{name!r} not found")
        return int(ids[0])

    arm = {}
    for side, robot in robots.items():
        arm[side] = {
            "joints": [index_of(robot.find_joints, j) for j in ARM_JOINTS],
            "fingers": [index_of(robot.find_joints, j) for j in FINGER_JOINTS],
            "gripper": index_of(robot.find_bodies, cfg.robot.gripper_body_name),
            "tips": [index_of(robot.find_bodies, "tip_left"), index_of(robot.find_bodies, "tip_right")],
        }
    targets = {side: _tensor(robot.data.default_joint_pos).clone() for side, robot in robots.items()}
    finger_cmd = {side: cfg.robot.finger_open_m for side in robots}

    # ---- nut state and scoring, every physics step ------------------------------------------
    track = {"yaw": 0.0, "prev": None}
    scorer = {"snapshot": None, "report": None}
    pad_local = {}

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

    def pad_centres(side):
        pos = _tensor(robots[side].data.body_pos_w)[0]
        quat = _tensor(robots[side].data.body_quat_w)[0]
        out_pads = []
        for k, body in enumerate(arm[side]["tips"]):
            local = pad_local[side][k] if side in pad_local else torch.tensor(cfg.robot.finger_pad_offsets_m[k], device=device)
            out_pads.append(pos[body] + quat_to_rot(quat[body].tolist()) @ local)
        return out_pads

    def tip_mid(side):
        """Grasp point: midpoint of the two fingertip contact points, not the body origins."""
        left_pad, right_pad = pad_centres(side)
        return 0.5 * (left_pad + right_pad)

    def finger_yaw_deg(side):
        left_pad, right_pad = pad_centres(side)
        d = right_pad - left_pad
        return math.degrees(math.atan2(float(d[1]), float(d[0])))

    def pad_gap_mm(side):
        a, b = pad_centres(side)
        return round(1000 * float(torch.linalg.norm(a - b)), 1)

    def nut_state():
        quat = quaternion_xyzw(_tensor(nut.data.root_quat_w)[0])
        centre = nut_centre_xyz(cfg, _tensor(nut.data.root_pos_w)[0].cpu().tolist(), quat)
        x_axis = rotate(quat, (1.0, 0.0, 0.0))
        return {
            "centre": centre,
            "quat": quat,
            "yaw_deg": math.degrees(math.atan2(x_axis[1], x_axis[0])),
            "tilt_deg": angle_deg(rotate(quat, (0.0, 0.0, 1.0)), (0.0, 0.0, 1.0)),
            "gap_m": centre[2] - f.nut_height_m / 2.0 - fz,
            "radial_m": math.hypot(centre[0] - fx, centre[1] - fy),
        }

    def holding_by_env_rule(side, centre):
        """The environment's hold rule (sim_bridge): pad midpoint near the nut, fingers part open."""
        fingers = [float(_tensor(robots[side].data.joint_pos)[0, j]) for j in arm[side]["fingers"]]
        fraction = mean_open_fraction(fingers, cfg.robot.finger_open_m, cfg.robot.finger_closed_m)
        distance = float(torch.linalg.norm(tip_mid(side) - torch.tensor(centre, device=device)))
        held = is_holding(
            distance_m=distance,
            open_fraction=fraction,
            radius_m=cfg.grasp.radius_m,
            hold_open_fraction_max=cfg.grasp.hold_open_fraction_max,
            hold_open_fraction_min=cfg.grasp.hold_open_fraction_min,
        )
        return held, fraction, distance

    def score_now():
        state = nut_state()
        if track["prev"] is not None:
            track["yaw"] += wrap_deg(state["yaw_deg"] - track["prev"])
        track["prev"] = state["yaw_deg"]
        holding = any(holding_by_env_rule(side, state["centre"])[0] for side in robots)
        observed = SceneSnapshot(
            nut=NutSnapshot(
                position_xyz=tuple(state["centre"]),
                rotation_xyzw=state["quat"],
                gripper_holding=holding,
                released=not holding,
                off_table=False,
            ),
            fixture_position_xyz=tuple(_tensor(fixture.data.root_pos_w)[0].cpu().tolist()),
            fixture_rotation_xyzw=quaternion_xyzw(_tensor(fixture.data.root_quat_w)[0]),
        )
        previous = scorer["snapshot"]
        observed = clear_episode_state(observed) if previous is None else carry_episode_memory(previous, observed)
        scorer["snapshot"], scorer["report"] = score_step(cfg, observed, dt)
        return state

    def clockwise_turns():
        return -track["yaw"] / 360.0

    def score_summary():
        r = scorer["report"]
        if r is None:
            return None
        t = r.thread
        return {
            "score": r.score,
            "reasons": list(r.hard_reasons),
            "gap_mm": round(t.gap_m * 1000, 3),
            "descent_mm": round(t.descent_m * 1000, 3),
            "thread_turns": round(t.thread_turns, 3),
            "tilt_deg": round(t.tilt_deg, 2),
            "radial_mm": round(t.radial_m * 1000, 3),
            "aligned": t.aligned,
            "engaged": t.engaged,
            "threaded": t.threaded,
            "cross_threaded": t.cross_threaded,
            "ever_lifted": r.ever_lifted,
        }

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
            score_now()

    # ---- arm kinematics (as battery-remote reach_check) --------------------------------------
    def torch_array(value):
        value = _tensor(value)
        if not isinstance(value, torch.Tensor):
            # This build can hand back a Warp/proxy array; list indexing on it fails.
            try:
                import warp as wp

                value = wp.to_torch(value)
            except Exception:
                value = torch.as_tensor(value, device=device)
        return value

    def view_of(side):
        return getattr(robots[side], "root_physx_view", None) or getattr(robots[side], "root_view", None)

    layout = {}
    measured_jac = {}

    def jacobian_variant(side, body_shift, col_shift, rows=slice(0, 3)):
        jac = torch_array(view_of(side).get_jacobians())
        body = arm[side]["gripper"] - body_shift
        cols = [j + col_shift for j in arm[side]["joints"]]
        if not (0 <= body < jac.shape[1]) or max(cols) >= jac.shape[3]:
            return None
        return jac[0, body, rows, :][:, torch.tensor(cols, device=jac.device)]

    def angular_jacobian(side):
        """Angular rows of the analytic Jacobian: the same for every point on the gripper."""
        return jacobian_variant(side, *layout[side], rows=slice(3, 6))

    analytic = {}

    def lever_jacobian(side, body_shift, col_shift):
        """Fingertip-point Jacobian from the gripper body's: v_tip = v_body + w x r.

        The 27% mismatch (2c545c3) compared the body's linear rows with the tips' motion and
        left out this lever, r ~ 0.2 m from the gripper origin to the pads. Recomputed every
        step, it needs no nudging calibration and no Broyden updates.
        """
        jac = torch_array(view_of(side).get_jacobians())
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
            name = f"body-{key[0]}_cols+{key[1]}"
            scores[name] = None if jac is None else round(float(torch.linalg.norm(jac - measured) / max(float(torch.linalg.norm(measured)), 1e-9)), 3)
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
        return {"relative_error_by_layout": scores, "chosen": best}

    def gripper_rotation(side):
        return quat_to_rot(_tensor(robots[side].data.body_quat_w)[0, arm[side]["gripper"]].tolist())

    approach_local = {}

    def measure_approach_axis(side):
        """Gripper-frame direction from the gripper origin to the fingertip midpoint, at home."""
        pos = _tensor(robots[side].data.body_pos_w)[0, arm[side]["gripper"]]
        v = gripper_rotation(side).T @ (tip_mid(side) - pos)
        approach_local[side] = v / torch.linalg.norm(v)
        return [round(float(c), 3) for c in approach_local[side]]

    def orientation_error(side, across=None):
        """Rotation vector: finger axis onto `across` (a horizontal unit vector; None leaves the
        yaw free) and a weak (0.3) pull of the approach axis toward straight down."""
        approach = gripper_rotation(side) @ approach_local[side]
        left_pad, right_pad = pad_centres(side)
        finger = right_pad - left_pad
        finger = finger / max(float(torch.linalg.norm(finger)), 1e-6)
        down = torch.tensor([0.0, 0.0, -1.0], device=device)
        err = 0.3 * torch.linalg.cross(approach, down)
        twist = None
        if across is not None:
            a = torch.tensor([float(across[0]), float(across[1]), 0.0], device=device)
            a = a if float(finger @ a) >= 0 else -a  # the finger axis is a line
            err = err + torch.linalg.cross(finger, a)
            twist = math.degrees(math.acos(max(-1.0, min(1.0, float(finger @ a)))))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(approach @ down)))))
        return err, tilt, twist

    def measure_pads(side):
        """Contact point near each fingertip, in that finger body's own frame (local bounds, so
        the stale USD world transforms of articulation links do not matter)."""
        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide"])
        root = stage.GetPrimAtPath(robot_paths[side])
        pos = _tensor(robots[side].data.body_pos_w)[0]
        quat = _tensor(robots[side].data.body_quat_w)[0]
        gripper_pos = pos[arm[side]["gripper"]]
        points, out_report = [], {}
        for k, tip in enumerate(("tip_left", "tip_right")):
            prim = next(p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.GetName() == tip)
            rng = cache.ComputeRelativeBound(prim, prim).ComputeAlignedRange()
            mid = torch.tensor([float(v) for v in rng.GetMidpoint()], device=device)
            half = torch.tensor([float(v) for v in (rng.GetMax() - rng.GetMin())], device=device) / 2.0
            # End of the finger's longest axis farther from the gripper, 6 mm back onto the pad.
            axis = int(torch.argmax(half))
            body = arm[side]["tips"][k]
            rot = quat_to_rot(quat[body].tolist())
            ends = []
            for sign in (1.0, -1.0):
                local = mid.clone()
                local[axis] = mid[axis] + sign * (half[axis] - 0.006)
                ends.append((float(torch.linalg.norm(pos[body] + rot @ local - gripper_pos)), local))
            tip_local = max(ends, key=lambda item: item[0])[1]
            points.append(tip_local)
            out_report[tip] = {"length_mm": round(2000 * float(half[axis]), 1), "contact_point_mm": [round(1000 * float(v), 1) for v in tip_local]}
        pad_local[side] = points
        return out_report

    def at_limit(side):
        limits = _tensor(robots[side].data.soft_joint_pos_limits)[0]
        pos = _tensor(robots[side].data.joint_pos)[0]
        return [name for j, name in zip(arm[side]["joints"], ARM_JOINTS) if pos[j] <= limits[j, 0] + 0.02 or pos[j] >= limits[j, 1] - 0.02]

    class Servo:
        """Closed-loop DLS IK on the fingertip midpoint with a measured, Broyden-updated Jacobian.

        The analytic Jacobian is for the gripper body, not the fingertips, and never converged in
        the battery runs. Start from the Jacobian measured at home and correct it every step from
        the observed tip motion. `free` selects the arm joints the solve may move (joint6 is left
        out while it turns the nut).
        """

        def __init__(self, side):
            self.side = side
            self.jac = measured_jac[side].clone()
            self.q_prev = self.q().clone()
            self.p_prev = tip_mid(side).clone()

        def q(self):
            return _tensor(robots[self.side].data.joint_pos)[0, arm[self.side]["joints"]]

        def command(self, goal_t, across=None, orient=True, free=range(6)):
            side = self.side
            free = list(free)
            error = goal_t - tip_mid(side)
            base = lever_jacobian(side, *analytic[side]) if side in analytic else self.jac
            jac = base[:, free]
            if orient:
                rot_err, _, _ = orientation_error(side, across)
                full_jac = torch.cat([jac, 0.05 * angular_jacobian(side)[:, free]], dim=0)  # 5 cm per rad
                full_err = torch.cat([error, 0.05 * rot_err])
            else:
                full_jac, full_err = jac, error
            rows = full_jac.shape[0]
            jjt = full_jac @ full_jac.T + (0.02**2) * torch.eye(rows, device=device)
            dq = torch.clamp(full_jac.T @ torch.linalg.solve(jjt, full_err), -0.02, 0.02)
            joints = arm[side]["joints"]
            limits = _tensor(robots[side].data.soft_joint_pos_limits)[0]
            q_now = self.q()
            for k, col in enumerate(free):
                j = joints[col]
                # Keep the target within 0.15 rad of the actual joint so it cannot wind up far ahead.
                new = torch.clamp(targets[side][0, j] + dq[k], q_now[col] - 0.15, q_now[col] + 0.15)
                targets[side][0, j] = torch.clamp(new, limits[j, 0], limits[j, 1])

        def observe(self):
            q, p = self.q().clone(), tip_mid(self.side).clone()
            d_q, d_p = q - self.q_prev, p - self.p_prev
            if float(d_q @ d_q) > 1e-8:
                self.jac = self.jac + torch.outer(d_p - self.jac @ d_q, d_q) / float(d_q @ d_q)
            self.q_prev, self.p_prev = q, p

    def reach(side, goal, seconds=None, across=None, orient=True, servo=None):
        goal_t = torch.tensor([float(v) for v in goal], device=device)
        servo = servo or Servo(side)
        start_residual = float(torch.linalg.norm(goal_t - tip_mid(side)))
        for _ in range(int((seconds or args.reach_s) / dt)):
            if float(torch.linalg.norm(goal_t - tip_mid(side))) < args.tolerance_m * 0.5 and (
                across is None or orientation_error(side, across)[2] < 4.0
            ):
                break
            servo.command(goal_t, across, orient)
            step()
            servo.observe()
        step(30)  # let the drives settle on the last target
        residual = float(torch.linalg.norm(goal_t - tip_mid(side)))
        _, tilt, twist = orientation_error(side, across)
        return {
            "jacobian": "analytic_lever" if side in analytic else "measured_broyden",
            "goal_mm": [round(1000 * float(v), 1) for v in goal],
            "reached": residual <= args.tolerance_m,
            "start_residual_mm": round(1000 * start_residual, 2),
            "residual_mm": round(1000 * residual, 2),
            "joints_at_limit": at_limit(side),
            "approach_tilt_from_down_deg": round(tilt, 1),
            "finger_axis_off_target_deg": None if twist is None else round(twist, 1),
        }

    # ---- measurements ------------------------------------------------------------------------
    step(int(1.0 / dt))
    report["finger_contact_points_local_mm"] = {side: _stage({}, side, lambda side=side: measure_pads(side)) for side in robots}

    def pad_gaps(side):
        finger_cmd[side] = cfg.robot.finger_open_m
        step(int(0.5 / dt))
        opened = pad_gap_mm(side)
        finger_cmd[side] = cfg.robot.finger_closed_m
        step(int(1.0 / dt))
        closed = pad_gap_mm(side)
        finger_cmd[side] = cfg.robot.finger_open_m
        step(int(0.5 / dt))
        return {"open_mm": opened, "closed_empty_mm": closed}

    report["pad_gap_reference"] = {side: _stage({}, side, lambda side=side: pad_gaps(side)) for side in robots}
    report["jacobian_calibration"] = {}
    for side in robots:
        layout[side] = (1, 6)  # the layout the battery runs measured (floating-base report)
        _stage(report["jacobian_calibration"], side, lambda side=side: calibrate(side))
        measure_approach_axis(side)

    def nut_flat_phase():
        """Flat-normal direction in the nut's own frame, from its collision mesh outline."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        root = stage.GetPrimAtPath("/World/nut")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        to_root = cache.GetLocalToWorldTransform(root).GetInverse()
        meshes = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
        colliders = [p for p in meshes if p.HasAPI(UsdPhysics.CollisionAPI)] or meshes
        points = []
        for prim in colliders:
            xf = cache.GetLocalToWorldTransform(prim) * to_root
            for p in UsdGeom.Mesh(prim).GetPointsAttr().Get() or []:
                q = xf.Transform(Gf.Vec3d(*p))
                points.append((q[0], q[1]))
        radii = sorted(math.hypot(x, y) for x, y in points)
        return {
            "phase_deg": hex_flat_phase_deg(points),
            "points": len(points),
            "outline_radius_mm": [round(1000 * radii[0], 2), round(1000 * radii[-1], 2)] if radii else None,
            "prims": [str(p.GetPath()) for p in colliders][:4],
        }

    flat = _stage(report, "nut_flat_phase", nut_flat_phase)
    # Without a measurement, assume flat normals on the nut's local +-Y (the fingers then close
    # along world Y as in the battery grasp, which worked).
    phase_deg = flat["phase_deg"] if isinstance(flat, dict) and "phase_deg" in flat else 90.0

    side = min(robots, key=lambda s: math.dist(cfg.nut_pose.position_xyz[:2], (cfg.robot.left_base_xyz if s == "left" else cfg.robot.right_base_xyz)[:2]))
    report["arm"] = side
    wrist = {"yaw_per_rad": None}
    grip = {"dz": args.grasp_raise_m}

    def across_for_flats(nut_yaw_deg, preferred_deg):
        yaw = math.radians(flats_toward(nut_yaw_deg + phase_deg, preferred_deg))
        return (math.cos(yaw), math.sin(yaw))

    def measure_wrist(nudge=0.2):
        """Finger yaw change per radian of joint6, at the current (near-vertical) pose."""
        j6 = arm[side]["joints"][5]
        start, before = float(targets[side][0, j6]), finger_yaw_deg(side)
        targets[side][0, j6] = start + nudge
        step(int(0.6 / dt))
        moved = wrap_deg(finger_yaw_deg(side) - before, 180.0)
        targets[side][0, j6] = start
        step(int(0.6 / dt))
        wrist["yaw_per_rad"] = math.radians(moved) / nudge
        return {"finger_yaw_deg_per_rad": round(math.degrees(wrist["yaw_per_rad"]), 1), "clockwise_is_joint6": "negative" if wrist["yaw_per_rad"] > 0 else "positive"}

    def reset_nut_to_mat():
        pose = torch.tensor([[*cfg.nut_pose.position_xyz, *sim_quaternion(cfg.nut_pose.rotation_xyzw)]], device=device, dtype=torch.float32)
        nut.write_root_pose_to_sim(pose)
        nut.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        track["prev"] = None
        step(int(0.3 / dt))

    def place_on_tip(gap_m):
        pose = torch.tensor([[fx, fy, fz + gap_m - f.nut_origin_to_lower_face_m, 0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        nut.write_root_pose_to_sim(pose)
        nut.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        track["prev"] = None
        step(int(0.8 / dt))

    def grasp_at(preferred_deg=90.0):
        """Open, approach from above with the fingers across the flats nearest `preferred_deg`,
        descend to the nut's centre (+ grasp_raise), close. Returns the measurements."""
        state = nut_state()
        cx, cy, cz = state["centre"]
        across = across_for_flats(state["yaw_deg"], preferred_deg)
        finger_cmd[side] = cfg.robot.finger_open_m
        goal_above = (cx, cy, cz + args.grasp_raise_m + args.approach_m)
        above = reach(side, goal_above, across=across)
        if not above["reached"]:
            # The home Jacobian is a poor start far from home (battery be1f60b); re-measure in free air.
            calibrate(side, delta=0.02)
            above = dict(reach(side, goal_above, across=across), recalibrated=True)
        wrist_report = measure_wrist() if wrist["yaw_per_rad"] is None else None
        # Descend with a Jacobian measured just above the nut: the home one left the pads 29.7 mm
        # off (252e198). Never recalibrate at the nut itself; the nudges knock it.
        calibrate(side, delta=0.02)
        at = reach(side, (cx, cy, cz + args.grasp_raise_m), across=across)
        offset_before = [round(1000 * float(v), 1) for v in tip_mid(side) - torch.tensor(nut_state()["centre"], device=device)]
        finger_cmd[side] = cfg.robot.finger_closed_m
        step(int(1.0 / dt))
        state = nut_state()
        held, fraction, distance = holding_by_env_rule(side, state["centre"])
        grip["dz"] = float(tip_mid(side)[2]) - state["centre"][2]
        return {
            "across_deg": round(math.degrees(math.atan2(across[1], across[0])), 1),
            "above": above,
            "at": at,
            "wrist": wrist_report,
            "pad_minus_nut_centre_before_close_mm": offset_before,
            "pad_gap_after_close_mm": pad_gap_mm(side),
            "holding_geometrically": pad_gap_mm(side) > report["pad_gap_reference"].get(side, {}).get("closed_empty_mm", 24.0) + 3.0,
            "finger_open_fraction": None if fraction is None else round(fraction, 3),
            "holding_by_env_rule": held,
            "pad_mid_to_nut_centre_mm": round(1000 * distance, 1),
        }

    # ---- stages ------------------------------------------------------------------------------
    def grasp():
        reset_nut_to_mat()
        scorer["snapshot"] = None
        result = grasp_at()
        start_z = nut_state()["centre"][2]
        cx, cy, _ = nut_state()["centre"]
        lift = reach(side, (cx, cy, float(tip_mid(side)[2]) + 0.05), across=None)
        state = nut_state()
        rise = state["centre"][2] - start_z
        result.update(
            {
                "lift": lift,
                "nut_rise_mm": round(1000 * rise, 1),
                "lifted": rise > 0.03,
                "nut_tilt_deg": round(state["tilt_deg"], 1),
                "score": score_summary(),
            }
        )
        return result

    def finger_across():
        left_pad, right_pad = pad_centres(side)
        d = right_pad - left_pad
        n = math.hypot(float(d[0]), float(d[1]))
        return (float(d[0]) / n, float(d[1]) / n)

    def carry():
        if not report.get("grasp", {}).get("lifted"):
            raise RuntimeError("grasp did not lift the nut")
        across = finger_across()  # keep the yaw, level the fingers
        target_centre_z = tip_z + f.nut_height_m / 2.0 + 0.004
        # a5f5537 went straight for a point 3 cm above the tip and ended with the nut tilted 56 deg,
        # 20 mm off the axis and below the tip level: it struck the bolt on the way. Rise first,
        # cross at a safe height, then come down on the axis.
        safe_z = target_centre_z + grip["dz"] + 0.06

        def air_reach(goal):
            """Free-air reach with the nut held; re-measure the Jacobian here and retry on a miss.
            c9c5606 ended these legs 11-41 mm off with the Jacobian measured over the mat."""
            row = reach(side, goal, across=across)
            if not row["reached"]:
                first = row["residual_mm"]
                calibrate(side, delta=0.02)
                row = dict(reach(side, goal, across=across), recalibrated=True, first_residual_mm=first)
            return row

        def nut_reach(centre_goal, passes=2):
            """Steer the nut's centre, not the pads: the nut can sit off-centre or tilted in the grip."""
            goal = torch.tensor(centre_goal, device=device)
            row = None
            for _ in range(passes):
                tip_goal = tip_mid(side) + (goal - torch.tensor(nut_state()["centre"], device=device))
                row = reach(side, tuple(float(v) for v in tip_goal), across=across)
            state = nut_state()
            row["nut_centre_error_mm"] = [round(1000 * (a - b), 1) for a, b in zip(state["centre"], centre_goal)]
            return row

        now = tip_mid(side)
        rise = air_reach((float(now[0]), float(now[1]), max(safe_z, float(now[2]))))
        high = nut_reach((fx, fy, safe_z - grip["dz"]))
        state = nut_state()
        high["nut_tilt_deg"] = round(state["tilt_deg"], 1)
        if state["radial_m"] > 0.004 or state["tilt_deg"] > 10.0:
            # Coming down from here would strike the bolt, as in a5f5537 and c9c5606.
            raise RuntimeError(
                f"nut not over the bolt at safe height: radial {1000 * state['radial_m']:.1f} mm, "
                f"tilt {state['tilt_deg']:.1f} deg, rise residual {rise['residual_mm']} mm, high residual {high['residual_mm']} mm"
            )
        calibrate(side, delta=0.02)  # local Jacobian for the short descent onto the tip
        above = nut_reach((fx, fy, target_centre_z + 0.02))
        over = nut_reach((fx, fy, target_centre_z))
        state = nut_state()
        return {
            "rise": rise,
            "high": high,
            "above_tip": above,
            "over_tip": over,
            "nut_radial_mm": round(1000 * state["radial_m"], 2),
            "nut_tilt_deg": round(state["tilt_deg"], 2),
            "nut_gap_mm": round(1000 * state["gap_m"], 2),
            "held": holding_by_env_rule(side, state["centre"])[0],
            "pad_gap_mm": pad_gap_mm(side),
        }

    def lower():
        carried = report.get("carry", {})
        if "error" in carried or carried.get("nut_radial_mm", 99) > 3.0:
            raise RuntimeError("carry did not bring the nut over the bolt")
        # Lower face 2 mm into the lead-in band: the server sequence dropped 2.45 mm onto the first crest.
        goal_centre_z = tip_z - 0.002 + f.nut_height_m / 2.0
        result = reach(side, (fx, fy, goal_centre_z + grip["dz"]), seconds=3.0, orient=True, across=finger_across())
        step(int(0.5 / dt))
        state = nut_state()
        return {
            "reach": result,
            "nut_gap_mm": round(1000 * state["gap_m"], 3),
            "travel_mm": round(1000 * travel, 3),
            "rests_on_tip": travel - cfg.scoring.thread_lead_in_m - 0.001 <= state["gap_m"] <= travel + 0.001,
            "nut_radial_mm": round(1000 * state["radial_m"], 2),
            "nut_tilt_deg": round(state["tilt_deg"], 2),
            "score": score_summary(),
        }

    def joint6_effort():
        j = arm[side]["joints"][5]
        row = {}
        data = robots[side].data
        for name in ("applied_torque", "computed_torque"):
            value = getattr(data, name, None)
            if value is not None:
                row[name] = float(torch_array(value)[0, j])
        try:
            # What PhysX actually applied at the joint, not the drive's PD estimate.
            row["projected"] = float(torch_array(view_of(side).get_dof_projected_joint_forces())[0, j])
        except Exception:
            pass
        return row

    def turn():
        lowered = report.get("lower", {})
        start = "carried"
        if "error" in lowered or not lowered.get("rests_on_tip"):
            # Pick or carry failed: test threading on its own from a nut placed on the tip.
            start = "placed_on_tip"
            finger_cmd[side] = cfg.robot.finger_open_m
            reach(side, (fx, fy, tip_z + 0.08), across=None)
            place_on_tip(travel - 0.002)
            # A fresh score: the nut on the tip is above the lift clearance, so it counts as lifted.
            scorer["snapshot"] = None
            staged = grasp_at()
        else:
            staged = None
        j6 = arm[side]["joints"][5]
        limits = _tensor(robots[side].data.soft_joint_pos_limits)[0]
        lo, hi = float(limits[j6, 0]) + 0.05, float(limits[j6, 1]) - 0.05
        cw = -1.0 if (wrist["yaw_per_rad"] or 1.0) > 0 else 1.0  # joint6 direction that turns clockwise from above
        stroke = math.radians(args.stroke_deg)
        j6_start = max(lo, min(hi, -cw * stroke / 2.0))
        rate = 2.0 * math.pi * args.turn_rate
        servo = Servo(side)
        first = nut_state()
        first_turns = clockwise_turns()
        strokes, stop = [], None
        nut_lost = lambda s: s["radial_m"] > cfg.scoring.track_radial_m or s["tilt_deg"] > cfg.scoring.track_tilt_deg

        def hold(goal_z, seconds):
            goal = torch.tensor([fx, fy, goal_z], device=device)
            for _ in range(int(seconds / dt)):
                servo.command(goal, None, orient=True, free=range(5))
                step()
                servo.observe()

        def q6_now():
            return float(_tensor(robots[side].data.joint_pos)[0, j6])

        def regrasp():
            """Open, turn the wrist back onto the next pair of flats, re-target the nut's height, close."""
            finger_cmd[side] = cfg.robot.finger_open_m
            hold(nut_state()["centre"][2] + grip["dz"], 0.4)
            state = nut_state()
            delta = wrap_deg(state["yaw_deg"] + phase_deg - finger_yaw_deg(side), HEX_PERIOD_DEG)
            back = choose_wrist_target(q6_now(), wrist["yaw_per_rad"] or 1.0, delta, j6_start, lo, hi)
            if back is None:
                return None
            goal = torch.tensor([fx, fy, state["centre"][2] + grip["dz"]], device=device)
            for _ in range(int(3.0 / dt)):
                if abs(q6_now() - back) < 0.02:
                    break
                tgt = float(targets[side][0, j6])
                targets[side][0, j6] = tgt + max(-0.03, min(0.03, back - tgt))
                servo.command(goal, None, orient=True, free=range(5))
                step()
                servo.observe()
            z = nut_state()["centre"][2] + grip["dz"]
            hold(z, 0.3)
            finger_cmd[side] = cfg.robot.finger_closed_m
            hold(z, 0.8)
            held, fraction, _ = holding_by_env_rule(side, nut_state()["centre"])
            return {
                "joint6_rad": round(q6_now(), 3),
                "finger_off_flats_deg": round(wrap_deg(nut_state()["yaw_deg"] + phase_deg - finger_yaw_deg(side), HEX_PERIOD_DEG), 1),
                "pad_gap_mm": pad_gap_mm(side),
                "holding_by_env_rule": held,
                "finger_open_fraction": None if fraction is None else round(fraction, 3),
                "score": score_summary(),
            }

        cw_limit = hi if cw > 0 else lo
        initial_regrasp = None
        if (cw_limit - q6_now()) * cw < 0.5 * stroke:
            # The grasp left too little wrist travel for a stroke.
            initial_regrasp = regrasp()
        for index in range(args.max_strokes):
            state = nut_state()
            if state["gap_m"] <= cfg.scoring.seat_gap_m * 0.5:
                stop = "seated"
                break
            if nut_lost(state):
                stop = "nut_left_the_bolt"
                break
            # Stroke: wrist clockwise, gripper descending along the helix from the nut's height now.
            z0 = state["centre"][2] + grip["dz"]
            j6_from = q6_now()
            j6_to = max(lo, min(hi, j6_from + cw * stroke))
            gap0, turns0 = state["gap_m"], clockwise_turns()
            efforts, track_err = [], []
            for _ in range(int(3.0 * stroke / rate / dt)):
                q6 = q6_now()
                if (q6 - j6_to) * cw >= -0.01:
                    break
                # Ramp at the turn rate, never more than 0.3 rad ahead of the joint (no windup on a binding nut).
                tgt = float(targets[side][0, j6]) + cw * rate * dt
                tgt = max(q6 - 0.3, min(q6 + 0.3, tgt))
                targets[side][0, j6] = j6_to if (tgt - j6_to) * cw > 0 else tgt
                wrist_turns = abs(q6 - j6_from) / (2.0 * math.pi)
                goal = torch.tensor([fx, fy, z0 - f.pitch_m * wrist_turns], device=device)
                servo.command(goal, None, orient=True, free=range(5))
                step()
                servo.observe()
                efforts.append(joint6_effort())
                track_err.append(float(targets[side][0, j6]) - q6_now())
                state = nut_state()
                if state["gap_m"] <= cfg.scoring.seat_gap_m * 0.5 or nut_lost(state):
                    break
            state = nut_state()
            nut_turned = clockwise_turns() - turns0
            descent = gap0 - state["gap_m"]
            row = {
                "stroke": index,
                "wrist_turns": round(abs(q6_now() - j6_from) / (2.0 * math.pi), 3),
                "nut_turns": round(nut_turned, 3),
                "descent_mm": round(1000 * descent, 3),
                "descent_per_nut_turn_mm": round(1000 * descent / nut_turned, 3) if nut_turned > 0.05 else None,
                "gap_mm": round(1000 * state["gap_m"], 3),
                "tilt_deg": round(state["tilt_deg"], 2),
                "radial_mm": round(1000 * state["radial_m"], 3),
                "pad_gap_mm": pad_gap_mm(side),
                "joint6_tracking_error_rad_max": round(max((abs(e) for e in track_err), default=0.0), 3),
                "score_end_of_stroke": score_summary(),
            }
            for key in ("applied_torque", "projected"):
                row[f"joint6_{key}"] = summarize_torques(e.get(key) for e in efforts)
            if state["gap_m"] <= cfg.scoring.seat_gap_m * 0.5:
                strokes.append(row)
                stop = "seated"
                break
            row["regrasp"] = regrasp()
            strokes.append(row)
            if row["regrasp"] is None:
                stop = "no_wrist_angle_on_flats"
                break
        else:
            stop = "max_strokes"
        state = nut_state()
        turns = clockwise_turns() - first_turns
        descent = first["gap_m"] - state["gap_m"]
        all_applied = [s["joint6_applied_torque"].get("max_abs_n_m") for s in strokes]
        all_projected = [s["joint6_projected"].get("max_abs_n_m") for s in strokes]
        return {
            "start": start,
            "staged_grasp": staged,
            "initial_regrasp": initial_regrasp,
            "stop": stop,
            "strokes_run": len(strokes),
            "nut_turns": round(turns, 3),
            "descent_mm": round(1000 * descent, 3),
            "descent_per_turn_mm": round(1000 * descent / turns, 3) if turns > 0.05 else None,
            "final_gap_mm": round(1000 * state["gap_m"], 3),
            "final_tilt_deg": round(state["tilt_deg"], 2),
            "max_stroke_joint6_applied_torque_n_m": max((v for v in all_applied if v is not None), default=None),
            "max_stroke_joint6_projected_torque_n_m": max((v for v in all_projected if v is not None), default=None),
            "scripted_drive_reference_n_m": {"tracked": 0.05, "threw_the_nut": 0.2},
            "score": score_summary(),
            "strokes": strokes,
        }

    def release():
        finger_cmd[side] = cfg.robot.finger_open_m
        step(int(0.6 / dt))
        x, y, z = (float(v) for v in tip_mid(side))
        away = reach(side, (x, y, z + 0.06), across=None)
        step(int(1.0 / dt))
        state = nut_state()
        held, _, distance = holding_by_env_rule(side, state["centre"])
        return {
            "away": away,
            "nut_gap_mm": round(1000 * state["gap_m"], 3),
            "seated": abs(state["gap_m"]) <= cfg.scoring.seat_gap_m,
            "holding_by_env_rule": held,
            "pad_mid_to_nut_mm": round(1000 * distance, 1),
            "score": score_summary(),
        }

    _stage(report, "grasp", grasp)
    _stage(report, "carry", carry)
    _stage(report, "lower", lower)
    _stage(report, "turn", turn)
    _stage(report, "release", release)

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("ROBOCURVE_NUT_REACH_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("ROBOCURVE_NUT_REACH_END", flush=True)
    app.close()


if __name__ == "__main__":
    main()
