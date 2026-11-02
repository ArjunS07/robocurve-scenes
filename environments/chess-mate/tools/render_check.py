"""Server-side visual and resting check for the chess scene. Prints one JSON report.

Run with the Arena interpreter (starts Kit headless with cameras; no streaming):

    cd /home/ubuntu/workspace/robocurve-scenes/environments/chess-mate
    /home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python tools/render_check.py

It builds the same scene the environment builds, using the same spawn configs:
the maple table, the generated board and 32 pieces, and both YAM arms held at
home. It then settles the physics and reports:

- for every piece: drift from its square, base height against the board top
  (penetration or floating), tilt, and residual speed
- board drift
- each YAM base's bounds against the tabletop and footprint, plus whether the
  YAM asset has any colliders at all
- PNG renders at 1280x720: overview, robot side, top-down, a close view of the
  pieces, a grazing view of the squares (z-fighting check), and the same views
  with collision geometry shown instead of visuals

Outputs go to outputs/render_check/<git revision>/ next to this package.
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

VIEWS = {
    # name: (eye, target) in env coordinates; the board is centered at x=0.55.
    "overview": ((1.30, -0.80, 0.55), (0.50, 0.0, 0.05)),
    "robot_side": ((-0.05, -0.05, 0.42), (0.55, 0.0, 0.02)),
    "top_down": ((0.55, 0.0001, 0.62), (0.55, 0.0, 0.0)),
    "pieces_close_white": ((0.33, -0.20, 0.10), (0.46, -0.02, 0.03)),
    "pieces_close_black": ((0.80, 0.18, 0.11), (0.65, 0.02, 0.03)),
    "squares_grazing": ((0.95, 0.30, 0.045), (0.55, 0.0, 0.007)),
}
OVERLAY_VIEWS = ("overview", "pieces_close_white", "squares_grazing")


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
        report[key] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)}
    return report[key]


def _grep_arena(root: Path, needle: str, limit: int = 15) -> list[str]:
    """Where the installed Arena handles a flag, so a camera failure can be fixed in one round."""
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
    parser.add_argument("--settle-s", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--experience", default=None, help="Kit experience file; default: Isaac Lab's headless rendering one.")
    parser.add_argument("--no-render", action="store_true", help="Physics and robot checks only, no camera.")
    args = parser.parse_args()

    sys.path.insert(0, str(args.arena_root))
    sys.path.insert(0, str(ROOT))
    import torch  # before Kit, as in the launchers

    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(app_parser)
    # This Isaac Lab build has no --enable_cameras on AppLauncher (it is an Arena runner
    # flag); pass it only where it exists.
    known = {action.dest for action in app_parser._actions}
    # The default --visualizer none experience has no RTX engine, so a Camera fails at
    # sim.reset ("Unable to create prim for graph at /Render/PostProcess/SDGPipeline").
    # Isaac Lab ships a headless rendering experience for camera sensors; use it.
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
    # Printed before Kit starts so it survives a crash during startup.
    print("ROBOCURVE_RENDER_SETUP", json.dumps({"app_argv": app_argv, "kit_files": kit_files, "arena_camera_wiring": camera_wiring}, indent=1), flush=True)
    if not args.no_render and experience is None:
        print("no headless rendering .kit found; rerun with --experience PATH or --no-render", flush=True)
    app = AppLauncher(app_parser.parse_args(app_argv)).app

    revision = _revision()
    out = args.out or ROOT / "outputs" / "render_check" / revision
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROBOCURVE_GENERATED_DIR", str(out / "assets"))
    report: dict = {"revision": revision, "out": str(out), "app_argv": app_argv, "arena_camera_wiring": camera_wiring}

    import isaaclab.sim as sim_utils
    import numpy as np
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sensors import Camera, CameraCfg
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdPhysics

    from my_package.assets.geometry import piece_poses, write_geometry
    from my_package.assets.spawn import make_board, make_piece
    from my_package.embodiments.yam_bimanual import _articulation
    from my_package.scoring.action_map import arm_bindings
    from my_package.scoring.config import default_config_path, load_config, puzzles_for
    from my_package.scoring.math3d import angle_deg, rotate

    cfg = load_config(default_config_path())
    puzzle = puzzles_for(cfg)[0]
    paths = write_geometry(cfg, Path(os.environ["ROBOCURVE_GENERATED_DIR"]))

    dt = 1.0 / 120.0
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda:0"))
    stage = sim.stage if hasattr(sim, "stage") else sim_utils.get_current_stage()

    from isaaclab_arena.assets.background_library import MapleTableRobolab

    # Arena spawns the table at the origin with no pose (measured by tools/probe_runtime.py).
    for prim_path, spawn_cfg in (
        ("/World/table", sim_utils.UsdFileCfg(usd_path=MapleTableRobolab.usd_path)),
        # Not Arena's own light assets; dimmer than the first run, which overexposed the board.
        ("/World/dome", sim_utils.DomeLightCfg(intensity=700.0)),
        ("/World/sun", sim_utils.DistantLightCfg(intensity=1200.0, angle=1.0)),
    ):
        spawn_cfg.func(prim_path, spawn_cfg)

    def rigid(obj, name):
        rigid_cfg = obj._generate_rigid_cfg()
        rigid_cfg.prim_path = f"/World/{name}"
        return RigidObject(rigid_cfg)

    board = rigid(make_board(cfg, str(paths["board"])), "board")
    pieces = {}
    for name in sorted(puzzle.homes):
        color, role, _ = name.split("_", 2)
        pieces[name] = rigid(make_piece(cfg, puzzle, name, str(paths[f"{color}_{role}"])), name)
    left, right = arm_bindings(cfg)
    robots = {
        "RobotLeft": Articulation(_articulation(cfg, "/World/RobotLeft", cfg.robot.left_pose, left)),
        "RobotRight": Articulation(_articulation(cfg, "/World/RobotRight", cfg.robot.right_pose, right)),
    }
    camera = None if args.no_render else Camera(
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
    homes = {name: pose for name, pose in ((p.name, p) for p in piece_poses(cfg, puzzle))}

    def hold_robots():
        for robot in robots.values():
            robot.set_joint_position_target(_tensor(robot.data.default_joint_pos))
            robot.write_data_to_sim()

    def step(render: bool = False):
        hold_robots()
        sim.step(render=render)
        for asset in (board, *pieces.values(), *robots.values()):
            asset.update(dt)

    steps = int(args.settle_s / dt)
    peak_speed = {name: 0.0 for name in pieces}
    for index in range(steps):
        step(render=False)
        if index >= steps - int(0.5 / dt):
            for name, piece in pieces.items():
                speed = float(torch.linalg.norm(_tensor(piece.data.root_lin_vel_w)[0]))
                peak_speed[name] = max(peak_speed[name], speed)

    def settle_report():
        board_pos = _tensor(board.data.root_pos_w)[0].cpu().tolist()
        board_top = board_pos[2] + cfg.board.thickness_m / 2.0
        rows = {}
        for name, piece in pieces.items():
            pos = _tensor(piece.data.root_pos_w)[0].cpu().tolist()
            quat = tuple(_tensor(piece.data.root_quat_w)[0].cpu().tolist())
            home = homes[name].position_xyz
            rows[name] = {
                "xy_drift_mm": round(1000 * math.hypot(pos[0] - home[0], pos[1] - home[1]), 3),
                "base_minus_board_top_mm": round(1000 * (pos[2] - board_top), 3),
                "tilt_deg": round(angle_deg(rotate(quat, (0.0, 0.0, 1.0)), (0.0, 0.0, 1.0)), 3),
                "speed_last_0.5s_mm_s": round(1000 * peak_speed[name], 3),
            }
        worst = {key: max(rows.items(), key=lambda item: abs(item[1][key])) for key in next(iter(rows.values()))}
        return {
            "settle_s": args.settle_s,
            "board_drift_mm": round(1000 * math.dist(board_pos, cfg.board_pose.position_xyz), 3),
            "worst": {key: {"piece": name, "value": row[key]} for key, (name, row) in worst.items()},
            "pieces": rows,
        }

    _stage(report, "settle", settle_report)

    def robot_report():
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        table_min_x = cfg.table.center_xy[0] - cfg.table.half_extents_xy[0]
        table_max_x = cfg.table.center_xy[0] + cfg.table.half_extents_xy[0]
        table_min_y = cfg.table.center_xy[1] - cfg.table.half_extents_xy[1]
        table_max_y = cfg.table.center_xy[1] + cfg.table.half_extents_xy[1]
        result = {}
        for name in robots:
            root = stage.GetPrimAtPath(f"/World/{name}")
            base = next((p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.GetName() == "base"), None)
            colliders = [str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
            whole = cache.ComputeWorldBound(root).ComputeAlignedRange()
            entry = {"collider_count": len(colliders), "colliders_sample": colliders[:6]}
            entry["arm_bbox_at_home"] = {"min": list(whole.GetMin()), "max": list(whole.GetMax())}
            entry["arm_past_table_near_edge_mm"] = round(1000 * max(0.0, table_min_x - whole.GetMin()[0]), 1)
            if base is not None:
                # The links are nested under the base prim, so its bound is the whole arm.
                # Measure only meshes whose nearest rigid-body ancestor is the base itself.
                own = []
                for prim in Usd.PrimRange(base, Usd.TraverseInstanceProxies()):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue
                    owner = prim
                    while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                        owner = owner.GetParent()
                    if owner == base:
                        own.append(cache.ComputeWorldBound(prim).ComputeAlignedRange())
                if own:
                    lo = [min(box.GetMin()[i] for box in own) for i in range(3)]
                    hi = [max(box.GetMax()[i] for box in own) for i in range(3)]
                    entry["base_link_bbox"] = {"min": lo, "max": hi}
                    entry["base_bottom_minus_table_top_mm"] = round(1000 * (lo[2] - cfg.table.top_z_m), 2)
                    entry["base_inside_table_footprint"] = bool(
                        lo[0] >= table_min_x and hi[0] <= table_max_x and lo[1] >= table_min_y and hi[1] <= table_max_y
                    )
                else:
                    entry["base_link_bbox"] = "no meshes owned by the base link"
                # Where the arm reaches at home: +X means toward the board.
                gripper = next((p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.GetName() == "gripper"), None)
                if gripper is not None:
                    tip = cache.ComputeWorldBound(gripper).ComputeAlignedRange()
                    entry["gripper_bbox_at_home"] = {"min": list(tip.GetMin()), "max": list(tip.GetMax())}
            entry["lowest_point_minus_table_top_mm"] = round(1000 * (whole.GetMin()[2] - cfg.table.top_z_m), 2)
            result[name] = entry
        return result

    _stage(report, "robots", robot_report)

    def capture(prefix: str, names):
        saved = []
        for name in names:
            eye, target = VIEWS[name]
            camera.set_world_poses_from_view(
                torch.tensor([eye], device=sim.device, dtype=torch.float32),
                torch.tensor([target], device=sim.device, dtype=torch.float32),
            )
            # Several rendered frames so RTX accumulation and the pose update settle.
            for _ in range(12):
                step(render=True)
                camera.update(dt)
            rgb = _tensor(camera.data.output["rgb"])[0].cpu().numpy()
            path = out / f"{prefix}{name}.png"
            Image.fromarray(np.asarray(rgb[..., :3], dtype=np.uint8)).save(path)
            saved.append(str(path))
        return saved

    if camera is not None:
        _stage(report, "renders", lambda: capture("", VIEWS))

    def collision_overlay():
        hidden, shown = 0, 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World"), Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if not (path.startswith("/World/board") or path.startswith("/World/white_") or path.startswith("/World/black_")):
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Gprim):
                gprim = UsdGeom.Gprim(prim)
                UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
                gprim.CreateDisplayColorAttr().Set([(0.95, 0.35, 0.1) if "/board" in path else (0.1, 0.6, 0.95)])
                shown += 1
            elif prim.GetName() in ("visual", "top", "edge", "bottom"):
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden += 1
        return {"colliders_shown": shown, "visual_prims_hidden": hidden, "files": capture("collision_", OVERLAY_VIEWS)}

    if camera is not None:
        _stage(report, "collision_overlay", collision_overlay)

    (out / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("ROBOCURVE_RENDER_CHECK_BEGIN")
    print(json.dumps({key: value for key, value in report.items() if key != "settle"} | {"settle_worst": report.get("settle", {}).get("worst"), "board_drift_mm": report.get("settle", {}).get("board_drift_mm")}, indent=1, default=str))
    print("ROBOCURVE_RENDER_CHECK_END", flush=True)
    app.close()


if __name__ == "__main__":
    main()
