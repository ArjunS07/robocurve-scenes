"""Server-side check of the physical M8 thread. Prints one JSON report.

Run with the Arena interpreter (starts Kit headless; no Arena runner, no robot):

    cd /home/ubuntu/workspace/robocurve-scenes/environments/atb1-nut
    /home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python tools/physics_check.py
    # physics only, no cameras:
    /home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python tools/physics_check.py --no-render

It spawns the maple table, the generated kinematic fixture, and the nut, with the
same USDA the environment uses, at the Arena dt (1/120 s). The robot is replaced
by a scripted wrench on the nut so the thread itself is tested:

1. settle    - the nut rests on the table, the fixture does not move
2. engage    - the nut is placed coaxially just above the stud and dropped; its pins land on the ridge
3. advance   - a clockwise torque turns it N turns; descent per turn must match the 1.25 mm pitch
4. release   - torque off; a partly threaded nut must stay put (no slide to the seat)
5. pull      - an upward force of several times its weight must not pull it off the stud
6. seat      - clockwise again until the collar stops it
7. back_off  - counter-clockwise until the pins leave the ridge, then the nut lifts free

Every trajectory sample is also fed to the task scorer, so the report shows the
score the environment would give (50 while released mid-thread, 100 when seated
and released). Renders (1280x720): overview, close-up mid-thread, and the same
with colliders highlighted. Outputs go to outputs/physics_check/<git revision>/.
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
        dirty = subprocess.call(["git", "diff", "--quiet", "--", "."], cwd=ROOT) != 0
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "."], cwd=ROOT, text=True).strip()
        return rev + ("-dirty" if dirty or untracked else "")
    except Exception:
        return "unknown"


def _stage(report: dict, key: str, fn):
    try:
        report[key] = fn()
    except Exception as exc:  # keep going so one failure does not hide the rest
        report[key] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=5)}
    return report[key]


def _grep_arena(root: Path, needle: str, limit: int = 15) -> list[str]:
    hits = []
    for path in sorted((root / "isaaclab_arena").rglob("*.py")):
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if needle in line:
                hits.append(f"{path.relative_to(root)}:{number}: {line.strip()[:160]}")
                if len(hits) >= limit:
                    return hits
    return hits


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arena-root", type=Path, default=ARENA)
    parser.add_argument("--turns", type=float, default=3.0, help="Clockwise turns in the advance stage.")
    parser.add_argument("--turn-rate", type=float, default=0.5, help="Target turns per second.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
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
    # This Isaac Lab build has no --enable_cameras on AppLauncher (it is an Arena runner flag).
    known = {action.dest for action in app_parser._actions}
    # --visualizer none alone has no RTX engine, so a Camera fails at sim.reset; use the headless rendering experience.
    kit_files = sorted(str(path) for path in (args.arena_root / "submodules" / "IsaacLab" / "apps").glob("*.kit"))
    experience = args.experience
    if experience is None and not args.no_render:
        experience = next((kit for kit in kit_files if "headless" in kit and "rendering" in kit), None)
    app_argv = ["--visualizer", "none"]
    if "enable_cameras" in known and not args.no_render:
        app_argv.append("--enable_cameras")
    if experience:
        app_argv += ["--experience", experience]
    camera_wiring = _grep_arena(args.arena_root, "enable_cameras")
    print("ROBOCURVE_PHYSICS_SETUP", json.dumps({"app_argv": app_argv, "kit_files": kit_files, "arena_camera_wiring": camera_wiring}, indent=1), flush=True)
    render = not args.no_render and experience is not None
    if not args.no_render and experience is None:
        print("no headless rendering .kit found; running physics only (pass --experience PATH to render)", flush=True)
    app = AppLauncher(app_parser.parse_args(app_argv)).app

    revision = _revision()
    out = args.out or ROOT / "outputs" / "physics_check" / revision
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROBOCURVE_GENERATED_DIR", str(out / "assets"))
    report: dict = {"revision": revision, "out": str(out), "app_argv": app_argv, "rendered": render}

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    from my_package.assets.geometry import (
        ensure_generated_assets,
        ridge_start_top_z_m,
        thread_turns_available,
    )
    from my_package.scoring.config import load_config
    from my_package.scoring.math3d import angle_deg, rotate
    from my_package.scoring.scorer import carry_episode_memory, required_travel_m, step as score_step
    from my_package.scoring.snapshot import NutSnapshot, SceneSnapshot, clear_episode_state
    from my_package.tasks.state_values import quaternion_xyzw, sim_quaternion

    cfg = load_config()
    f = cfg.fastener
    travel = required_travel_m(cfg)
    nut_usd, fixture_usd = ensure_generated_assets(cfg)
    report["thread"] = {
        "pitch_mm": f.pitch_m * 1000,
        "travel_mm": travel * 1000,
        "ridge_start_top_mm": ridge_start_top_z_m(cfg) * 1000,
        "ridge_turns": thread_turns_available(cfg),
    }

    dt = 1.0 / 120.0
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda:0"))
    stage = sim.stage if hasattr(sim, "stage") else sim_utils.get_current_stage()
    try:
        from isaaclab_arena.assets.background_library import MapleTableRobolab
        from my_package.assets.table import maple_table_usd

        table_cfg = sim_utils.UsdFileCfg(usd_path=maple_table_usd(MapleTableRobolab.usd_path))
        report["table"] = "maple_table_robolab"
    except Exception as exc:
        table_cfg = sim_utils.GroundPlaneCfg()
        report["table"] = f"ground plane (maple table unavailable: {exc})"
    for prim_path, spawn_cfg in (
        ("/World/table", table_cfg),
        ("/World/dome", sim_utils.DomeLightCfg(intensity=2000.0)),
        ("/World/sun", sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0)),
    ):
        spawn_cfg.func(prim_path, spawn_cfg)
    if isinstance(table_cfg, sim_utils.GroundPlaneCfg):
        stage.GetPrimAtPath("/World/table").GetAttribute("xformOp:translate").Set((0.0, 0.0, cfg.table.top_z_m))

    def rigid(name, path, pose):
        return RigidObject(
            RigidObjectCfg(
                prim_path=f"/World/{name}",
                spawn=sim_utils.UsdFileCfg(usd_path=str(path)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pose.position_xyz), rot=sim_quaternion(pose.rotation_xyzw)),
            )
        )

    fixture = rigid("fixture", fixture_usd, cfg.fixture_pose)
    nut = rigid("nut", nut_usd, cfg.nut_pose)
    camera = None
    if render:
        from isaaclab.sensors import Camera, CameraCfg

        camera = Camera(
            CameraCfg(
                prim_path="/World/render_cam",
                update_period=0.0,
                height=args.height,
                width=args.width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.002, 20.0)),
            )
        )
    sim.reset()
    device = sim.device
    fx, fy, fz = cfg.fixture_pose.position_xyz

    # ---- wrench / drive -------------------------------------------------------------
    drive = {"mode": None}

    def apply_wrench(force_z: float, torque_z: float):
        forces = torch.tensor([[[0.0, 0.0, force_z]]], device=device)
        torques = torch.tensor([[[0.0, 0.0, torque_z]]], device=device)
        if hasattr(nut, "set_external_force_and_torque"):
            try:
                nut.set_external_force_and_torque(forces, torques, is_global=True)
            except TypeError:
                nut.set_external_force_and_torque(forces, torques)
            drive["mode"] = "external_wrench"
            return
        composer = getattr(nut, "permanent_wrench_composer", None)
        if composer is not None:
            composer.set_forces_and_torques(forces=forces, torques=torques, is_global=True)
            drive["mode"] = "wrench_composer"
            return
        raise RuntimeError("no external-wrench API on RigidObject in this Isaac Lab build")

    # ---- state ----------------------------------------------------------------------
    track = {"yaw_unwrapped": 0.0, "prev_yaw": None}
    scorer = {"snapshot": None, "report": None}

    def read():
        pos = _tensor(nut.data.root_pos_w)[0].cpu().tolist()
        quat = quaternion_xyzw(_tensor(nut.data.root_quat_w)[0])
        ang = _tensor(nut.data.root_ang_vel_w)[0].cpu().tolist()
        fpos = _tensor(fixture.data.root_pos_w)[0].cpu().tolist()
        fquat = quaternion_xyzw(_tensor(fixture.data.root_quat_w)[0])
        x_axis = rotate(quat, (1.0, 0.0, 0.0))
        yaw = math.atan2(x_axis[1], x_axis[0])
        if track["prev_yaw"] is not None:
            delta = yaw - track["prev_yaw"]
            delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
            track["yaw_unwrapped"] += delta
        track["prev_yaw"] = yaw
        return {
            "pos": pos,
            "quat": quat,
            "wz": ang[2],
            "gap_m": pos[2] - f.nut_height_m / 2.0 - fz,
            "radial_m": math.hypot(pos[0] - fx, pos[1] - fy),
            "tilt_deg": angle_deg(rotate(quat, (0.0, 0.0, 1.0)), (0.0, 0.0, 1.0)),
            "clockwise_turns": -track["yaw_unwrapped"] / (2.0 * math.pi),
            "fixture_drift_m": math.dist(fpos, cfg.fixture_pose.position_xyz),
            "fixture_quat": fquat,
        }

    def score(state, released: bool):
        observed = SceneSnapshot(
            nut=NutSnapshot(
                position_xyz=tuple(state["pos"]),
                rotation_xyzw=state["quat"],
                gripper_holding=not released,
                released=released,
                off_table=False,
            ),
            fixture_position_xyz=tuple(cfg.fixture_pose.position_xyz[i] + 0.0 for i in range(3)),
            fixture_rotation_xyzw=state["fixture_quat"],
        )
        previous = scorer["snapshot"]
        observed = clear_episode_state(observed) if previous is None else carry_episode_memory(previous, observed)
        scorer["snapshot"], scorer["report"] = score_step(cfg, observed, dt)
        return scorer["report"]

    def advance(seconds: float, *, force_z=0.0, torque_fn=None, released=True, until=None, render_frames=False, sample_every=6):
        samples = []
        for index in range(max(1, int(seconds / dt))):
            state = read()
            torque = torque_fn(state) if torque_fn else 0.0
            apply_wrench(force_z, torque)
            nut.write_data_to_sim()
            sim.step(render=render_frames)
            nut.update(dt)
            fixture.update(dt)
            state = read()
            result = score(state, released)
            if index % sample_every == 0:
                samples.append(
                    {
                        "t": round(index * dt, 3),
                        "cw_turns": round(state["clockwise_turns"], 4),
                        "gap_mm": round(state["gap_m"] * 1000, 4),
                        "tilt_deg": round(state["tilt_deg"], 3),
                        "radial_mm": round(state["radial_m"] * 1000, 4),
                        "score": result.score,
                    }
                )
            if until is not None and until(state):
                break
        apply_wrench(0.0, 0.0)
        nut.write_data_to_sim()
        return read(), samples

    def spin(target_turns_per_s: float):
        target = -2.0 * math.pi * target_turns_per_s  # clockwise is negative about +Z

        def torque(state):
            feedforward = 1.5e-4 * math.copysign(1.0, target)
            return max(-2e-3, min(2e-3, feedforward + 5e-4 * (target - state["wz"])))

        return torque

    def place(gap_m: float, clockwise_deg: float):
        yaw = -math.radians(clockwise_deg)
        pose = torch.tensor(
            [[fx, fy, fz + gap_m + f.nut_height_m / 2.0, *sim_quaternion((0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))]],
            device=device,
        )
        nut.write_root_pose_to_sim(pose)
        nut.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        track["prev_yaw"] = None
        track["yaw_unwrapped"] = yaw
        read()

    def summary(state):
        return {
            "gap_mm": round(state["gap_m"] * 1000, 4),
            "cw_turns": round(state["clockwise_turns"], 4),
            "tilt_deg": round(state["tilt_deg"], 3),
            "radial_mm": round(state["radial_m"] * 1000, 4),
            "fixture_drift_mm": round(state["fixture_drift_m"] * 1000, 4),
            "score": None if scorer["report"] is None else scorer["report"].score,
            "score_reasons": None if scorer["report"] is None else list(scorer["report"].hard_reasons),
        }

    # ---- renders (defined before the trials that call them) --------------------------------------------------------------------
    views = {
        "overview": ((fx + 0.16, fy - 0.16, fz + 0.12), (fx, fy + 0.04, fz)),
        "close_up": ((fx + 0.035, fy - 0.035, fz + 0.022), (fx, fy, fz + 0.006)),
    }

    def capture(prefix, names):
        from PIL import Image
        import numpy as np

        saved = []
        for name in names:
            eye, target = views[name]
            camera.set_world_poses_from_view(
                torch.tensor([eye], device=device, dtype=torch.float32),
                torch.tensor([target], device=device, dtype=torch.float32),
            )
            for _ in range(12):
                sim.step(render=True)
                nut.update(dt)
                camera.update(dt)
            rgb = _tensor(camera.data.output["rgb"])[0].cpu().numpy()
            path = out / f"{prefix}{name}.png"
            Image.fromarray(np.asarray(rgb[..., :3], dtype=np.uint8)).save(path)
            saved.append(str(path))
        return saved

    # ---- trials ---------------------------------------------------------------------
    def settle():
        state, _ = advance(1.0)
        rest = state["pos"][2] - (cfg.table.top_z_m + f.nut_height_m / 2.0)
        return {**summary(state), "nut_center_above_rest_mm": round(rest * 1000, 3)}

    _stage(report, "settle", settle)

    def engage():
        # 20 degrees clockwise past the ridge start, 0.5 mm above where the pins would land.
        start_gap = travel - f.pitch_m * 20.0 / 360.0
        place(start_gap + 0.0005, 20.0)
        placed = read()
        scorer["snapshot"] = None
        # Every step for the first 0.1 s: the 0afbd1a run only kept the last samples, by which
        # time the nut had already dropped ~7.5 mm past the ridge.
        early_state, early = advance(0.1, sample_every=1)
        state, samples = advance(0.65)
        return {
            **summary(state),
            "expected_gap_mm": round(start_gap * 1000, 4),
            "placed_gap_mm": round(placed["gap_m"] * 1000, 4),
            "first_0.1s": early,
            "samples": samples[-4:],
        }

    engaged = _stage(report, "engage", engage)

    def advance_stage():
        before = read()
        goal = before["clockwise_turns"] + args.turns
        timeout = 4.0 * args.turns / args.turn_rate
        state, samples = advance(timeout, torque_fn=spin(args.turn_rate), released=False, until=lambda s: s["clockwise_turns"] >= goal)
        turns = state["clockwise_turns"] - before["clockwise_turns"]
        descent = before["gap_m"] - state["gap_m"]
        # Deviation from the ideal helix along the whole run.
        ideal = [before["gap_m"] - f.pitch_m * (s["cw_turns"] - round(before["clockwise_turns"], 4)) for s in samples]
        deviation = max((abs(s["gap_mm"] / 1000 - g) for s, g in zip(samples, ideal)), default=None)
        return {
            **summary(state),
            "drive_mode": drive["mode"],
            "turns": round(turns, 4),
            "descent_mm": round(descent * 1000, 4),
            "descent_per_turn_mm": round(descent / turns * 1000, 4) if turns > 0.05 else None,
            "max_helix_deviation_mm": None if deviation is None else round(deviation * 1000, 4),
            "max_tilt_deg": max((s["tilt_deg"] for s in samples), default=None),
            "samples": samples[:: max(1, len(samples) // 12)],
        }

    if not isinstance(engaged, dict) or "error" not in engaged:
        _stage(report, "advance", advance_stage)
        def release():
            before = read()
            state, samples = advance(1.5, released=True)
            return {
                **summary(state),
                "gap_change_mm": round((state["gap_m"] - before["gap_m"]) * 1000, 4),
                "turn_change": round(state["clockwise_turns"] - before["clockwise_turns"], 4),
                "retained": abs(state["gap_m"] - before["gap_m"]) < 0.0003,
            }

        _stage(report, "release", release)
        if camera is not None:
            _stage(report, "renders_mid_thread", lambda: capture("", ("overview", "close_up")))

        def pull():
            before = read()
            weight = f.nut_mass_kg * 9.81
            state, _ = advance(0.5, force_z=4.0 * weight, released=True)
            after_state, _ = advance(0.5, released=True)
            play = f.pitch_m - f.ridge_thickness_m - f.pin_thickness_m
            rise = state["gap_m"] - before["gap_m"]
            return {
                "force_n": round(4.0 * weight, 4),
                "rise_mm": round(rise * 1000, 4),
                "groove_play_mm": round(play * 1000, 4),
                "held_on_stud": rise < play + 0.0002,
                "after": summary(after_state),
            }

        _stage(report, "pull", pull)

        def seat():
            state, samples = advance(
                4.0 * (f.turns + 1) / args.turn_rate,
                torque_fn=spin(args.turn_rate),
                released=False,
                until=lambda s: s["gap_m"] <= cfg.scoring.seat_gap_m * 0.5,
            )
            stalled, _ = advance(0.5, torque_fn=spin(args.turn_rate), released=False)
            released_state, _ = advance(1.0, released=True)
            return {
                "reached": summary(state),
                "after_extra_torque": summary(stalled),
                "released": summary(released_state),
                "score_released_seated": scorer["report"].score if scorer["report"] else None,
            }

        _stage(report, "seat", seat)

        def back_off():
            free_gap = travel + f.pitch_m
            state, samples = advance(
                4.0 * (f.turns + 2) / args.turn_rate,
                torque_fn=spin(-args.turn_rate),
                released=False,
                until=lambda s: s["gap_m"] >= free_gap,
            )
            weight = f.nut_mass_kg * 9.81
            lifted, _ = advance(0.6, force_z=2.0 * weight, released=False)
            return {
                "unthreaded": summary(state),
                "lifted": summary(lifted),
                "freed": lifted["gap_m"] > ridge_start_top_z_m(cfg) + 0.002,
            }

        _stage(report, "back_off", back_off)

    def collision_overlay():
        from pxr import Usd, UsdGeom, UsdPhysics

        place(travel * 0.5, 360.0 * 0.5 * f.turns)
        shown, hidden = 0, 0
        for root in ("/World/fixture", "/World/nut"):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    name = prim.GetName()
                    color = (0.95, 0.1, 0.1) if name.startswith("pin_") else (1.0, 0.6, 0.0) if name.startswith("ridge_") else (0.1, 0.6, 0.95)
                    UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([color])
                    shown += 1
                else:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden += 1
        return {"colliders_shown": shown, "visual_only_hidden": hidden, "files": capture("collision_", ("overview", "close_up"))}

    if camera is not None:
        _stage(report, "collision_overlay", collision_overlay)

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("ROBOCURVE_PHYSICS_CHECK_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("ROBOCURVE_PHYSICS_CHECK_END", flush=True)
    app.close()


if __name__ == "__main__":
    main()
