"""Print short summaries of the latest battery and nut reach checks and the Fabric render check.

Plain Python (numpy and Pillow for the render diff); no Kit. Run from anywhere:

    python environments/tools/summarize_checks.py
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

ENVS = Path(__file__).resolve().parents[1]


def latest(pattern: str):
    paths = glob.glob(str(ENVS / pattern))
    return max(paths, key=os.path.getmtime) if paths else None


def pick(row: dict, keys) -> dict:
    return {k: row.get(k) for k in keys}


def battery() -> None:
    print("== 1. BATTERY ==")
    path = latest("battery-remote/outputs/reach_check/*/report.json")
    if not path:
        print("no report")
        return
    print(path)
    r = json.load(open(path))
    print("calibration", {s: (v.get("chosen"), {k: x for k, x in (v.get("relative_error_by_layout") or {}).items() if x is not None and x < 0.6})
                          for s, v in r.get("jacobian_calibration", {}).items() if isinstance(v, dict)})
    keys = ("arm", "grasp_residual_mm", "first_attempt_residual_mm", "joints_at_limit",
            "pad_gap_after_close_mm", "cell_rise_mm", "lifted")
    for name, row in r.get("grasp_cells", {}).items():
        if not isinstance(row, dict) or "error" in row:
            print(name, row)
            continue
        print("grasp", name, pick(row, keys + ("approach_tilt_from_down_deg", "finger_axis_off_world_y_deg", "fingers_stopped_by_cell")))
        print("   before_close", row.get("before_close"))
        print("   after_close", row.get("after_close"), "lift", row.get("lift"), "cell_after_lift", row.get("cell_after_lift_mm"))
        mirrored = row.get("from_mirrored_seed")
        if mirrored:
            print("   mirrored:", mirrored if "error" in mirrored else pick(mirrored, keys))
    ins = r.get("insert_cell", {})
    for k in ("cell", "arm", "slot", "skipped", "tilt_above", "dip", "push", "after_release", "error", "error_at"):
        if k in ins:
            print("insert", k, ins[k])


def nut() -> None:
    print("== 2. NUT ==")
    path = latest("atb1-nut/outputs/reach_check/*/report.json")
    if not path:
        print("no report")
        return
    print(path)
    r = json.load(open(path))
    print("calibration", {s: (v.get("chosen"), {k: x for k, x in (v.get("relative_error_by_layout") or {}).items() if x is not None and x < 0.6})
                          for s, v in r.get("jacobian_calibration", {}).items() if isinstance(v, dict)})
    g = r.get("grasp", {})
    print("grasp", pick(g, ("error", "error_at", "pad_gap_after_close_mm", "finger_open_fraction",
                            "holding_by_env_rule", "nut_rise_mm", "lifted", "nut_tilt_deg")),
          "at", pick(g.get("at", {}), ("residual_mm", "approach_tilt_from_down_deg", "finger_axis_off_target_deg")))
    c = r.get("carry", {})
    print("carry", pick(c, ("error", "nut_radial_mm", "nut_tilt_deg", "nut_gap_mm", "held")),
          {k: pick(c.get(k) or {}, ("residual_mm", "first_residual_mm", "nut_centre_error_mm", "nut_tilt_deg"))
           for k in ("rise", "high", "above_tip", "over_tip")})
    print("lower", pick(r.get("lower", {}), ("error", "nut_gap_mm", "travel_mm", "rests_on_tip", "nut_radial_mm", "nut_tilt_deg")))
    t = r.get("turn", {})
    print("turn", pick(t, ("error", "error_at", "stop", "strokes_run", "nut_turns", "descent_mm", "descent_per_turn_mm",
                           "final_gap_mm", "final_tilt_deg", "max_stroke_joint6_applied_torque_n_m",
                           "max_stroke_joint6_projected_torque_n_m")))
    for s in t.get("strokes", []):
        rg = s.get("regrasp") or {}
        score = (rg.get("score") or s.get("score_end_of_stroke") or {}).get("score")
        print(" ", s.get("stroke"), "nut", s.get("nut_turns"), "mm/turn", s.get("descent_per_nut_turn_mm"),
              "gap", s.get("gap_mm"), "tilt", s.get("tilt_deg"),
              "tau", (s.get("joint6_applied_torque") or {}).get("max_abs_n_m"),
              "open", rg.get("finger_open_fraction"), "score", score)
    e = r.get("release", {})
    print("release", pick(e, ("error", "nut_gap_mm", "seated", "holding_by_env_rule")), e.get("score"))


def fabric() -> None:
    print("== 3. FABRIC ==")
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        print("needs numpy and Pillow:", exc)
        return
    dirs = sorted(glob.glob(str(ENVS / "battery-remote/outputs/physics_check/*")), key=os.path.getmtime)[-2:]
    for d in dirs:
        try:
            a, b = (np.asarray(Image.open(f"{d}/{p}_bay_close.png"), dtype=float) for p in ("start", "seated"))
            print(os.path.basename(d), "start vs seated diff:", round(float(abs(a - b).mean()), 2))
        except Exception as exc:
            print(os.path.basename(d), "no renders:", exc)


if __name__ == "__main__":
    for section in (battery, nut, fabric):
        try:
            section()
        except Exception as exc:  # one broken report must not hide the others
            print(section.__name__, "failed:", type(exc).__name__, exc)
