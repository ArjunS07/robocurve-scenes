"""Read-only probe of the installed Arena / Isaac Lab runtime. Prints one JSON report.

Run on the server with the Arena interpreter:

    /home/ubuntu/workspace/IsaacLab-Arena/.venv/bin/python probe_runtime.py

Add ``--sim`` to also start Kit (``--visualizer none``) and read a live rigid body's
``root_quat_w``. Nothing is written to disk and no shared file is edited.

Answers:
- quaternion order of Isaac Lab state arrays and init-state defaults (xyzw vs wxyz)
- maple_table_robolab USD path and its world-space top height
- YAM USDA: articulation roots, joints (type, limits, bodies), rigid bodies, gripper-like links
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

_APP: list = []  # Kit stays open until every probe has run, so pxr stays importable
ARENA = Path("/home/ubuntu/workspace/IsaacLab-Arena")
YAM = Path("/home/ubuntu/workspace/robocurve/assets/yam/yam_source_resolved/yam_source_resolved.usda")


def _try(report: dict, key: str, fn) -> None:
    try:
        report[key] = fn()
    except Exception as exc:  # report every failure instead of stopping
        report[key] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=3)}


def _grep(root: Path, pattern: str, limit: int = 12) -> list[str]:
    regex = re.compile(pattern)
    hits = []
    for path in root.rglob("*.py"):
        try:
            for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{path}:{number}: {line.strip()[:160]}")
                    if len(hits) >= limit:
                        return hits
        except OSError:
            continue
    return hits


def quaternion_static() -> dict:
    """Source-level evidence; no Kit needed."""
    import importlib.util

    out: dict = {}
    spec = importlib.util.find_spec("isaaclab")
    out["isaaclab_location"] = None if spec is None else str(spec.submodule_search_locations)
    try:
        from importlib.metadata import version

        out["isaaclab_version"] = version("isaaclab")
    except Exception as exc:
        out["isaaclab_version"] = repr(exc)
    if spec is not None and spec.submodule_search_locations:
        root = Path(list(spec.submodule_search_locations)[0])
        out["root_quat_w_doc"] = _grep(root / "assets", r"root_quat_w.*(wxyz|xyzw|w, x, y, z|x, y, z, w)")
        out["init_rot_default"] = _grep(root / "assets", r"rot:\s*tuple.*=")
    out["arena_pose_doc"] = _grep(ARENA / "isaaclab_arena" / "utils", r"(xyzw|wxyz)", limit=8)
    return out


def table_info() -> dict:
    out: dict = {"usd_candidates": []}
    for path in ARENA.rglob("*.py"):
        text = path.read_text(errors="ignore")
        if "maple_table_robolab" in text:
            for line in text.splitlines():
                if "usd" in line.lower() and ("maple" in line.lower() or "table" in line.lower()):
                    out["usd_candidates"].append(f"{path}: {line.strip()[:200]}")
    try:
        from pxr import Usd, UsdGeom

        usd_paths = set()
        for item in out["usd_candidates"]:
            usd_paths.update(re.findall(r"[\"']([^\"']+\.usd[a-z]?)[\"']", item))
        boxes = {}
        for usd in sorted(usd_paths):
            if "{" in usd:
                boxes[usd] = "templated path; resolve manually"
                continue
            stage = Usd.Stage.Open(usd)
            if stage is None:
                boxes[usd] = "could not open"
                continue
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            box = cache.ComputeWorldBound(stage.GetDefaultPrim() or stage.GetPseudoRoot()).ComputeAlignedRange()
            boxes[usd] = {
                "min": list(box.GetMin()),
                "max": list(box.GetMax()),
                "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage),
                "upAxis": str(UsdGeom.GetStageUpAxis(stage)),
            }
        out["bounds"] = boxes
    except ImportError:
        out["bounds"] = "pxr not importable outside Kit; rerun with --sim"
    return out


def yam_info() -> dict:
    out: dict = {"path": str(YAM), "exists": YAM.is_file()}
    try:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(YAM))
        roots, joints, bodies = [], [], []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                roots.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(str(prim.GetPath()))
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                entry = {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "body0": [str(t) for t in joint.GetBody0Rel().GetTargets()],
                    "body1": [str(t) for t in joint.GetBody1Rel().GetTargets()],
                }
                for attr in ("physics:lowerLimit", "physics:upperLimit", "physics:axis"):
                    value = prim.GetAttribute(attr).Get() if prim.HasAttribute(attr) else None
                    entry[attr.split(":")[1]] = None if value is None else str(value)
                joints.append(entry)
        out.update(articulation_roots=roots, rigid_bodies=bodies, joints=joints)
        out["gripper_like"] = [b for b in bodies if re.search(r"grip|finger|tcp|ee|hand", b, re.I)]
    except ImportError:
        # USDA is ASCII; fall back to a text scan.
        text = YAM.read_text(errors="ignore")
        out["joint_defs"] = re.findall(r'def (Physics\w*Joint) "([^"]+)"', text)[:40]
        out["xform_names"] = sorted(set(re.findall(r'def Xform "([^"]+)"', text)))[:80]
        out["note"] = "pxr not importable outside Kit; text scan only"
    return out


def quaternion_live() -> dict:
    """Spawn one cuboid with a 90 degree rotation about X authored in USD, then read root_quat_w.

    wxyz reads ~[0.707, 0.707, 0, 0]; xyzw reads ~[0.707, 0, 0, 0.707].
    """
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    app_args = parser.parse_args(["--visualizer", "none"])
    _APP.append(AppLauncher(app_args).app)
    if True:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
        default_rot = tuple(RigidObjectCfg.InitialStateCfg().rot)
        half = 0.7071067811865476
        # Write the same physical rotation in whichever order the cfg default implies.
        rot = (half, half, 0.0, 0.0) if default_rot == (1.0, 0.0, 0.0, 0.0) else (half, 0.0, 0.0, half)
        cfg = RigidObjectCfg(
            prim_path="/World/probe_cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.1, 0.2, 0.3),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=rot),
        )
        cube = RigidObject(cfg)
        sim.reset()
        sim.step()
        cube.update(0.01)
        raw = [round(float(v), 4) for v in cube.data.root_quat_w[0].detach().cpu().tolist()]
        # Extent of the rotated cube along world Z tells the physical rotation independently.
        from pxr import Usd, UsdGeom

        stage = sim_utils.get_current_stage() if hasattr(sim_utils, "get_current_stage") else sim.stage
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        box = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/probe_cube")).ComputeAlignedRange()
        size = [round(float(v), 3) for v in (box.GetMax() - box.GetMin())]
        return {
            "init_state_default_rot": default_rot,
            "written_rot": rot,
            "root_quat_w": raw,
            "world_bbox_size": size,
            "interpretation": "bbox (0.1, 0.3, 0.2) means a real 90 deg turn about X; "
            "then root_quat_w ~[.707,.707,0,0] is wxyz and ~[.707,0,0,.707] is xyzw",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim", action="store_true", help="Also start Kit and read a live root_quat_w.")
    args = parser.parse_args()
    sys.path.insert(0, str(ARENA))
    report: dict = {}
    if args.sim:
        import torch  # noqa: F401  before Kit, same CUDA import-order workaround as the launchers

        _try(report, "quaternion_live", quaternion_live)
    _try(report, "quaternion_static", quaternion_static)
    _try(report, "table", table_info)
    _try(report, "yam", yam_info)
    print("ROBOCURVE_PROBE_BEGIN")
    print(json.dumps(report, indent=1, default=str))
    print("ROBOCURVE_PROBE_END", flush=True)
    for app in _APP:
        app.close()


if __name__ == "__main__":
    main()
