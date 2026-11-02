"""Optional task-camera snapshots during a launcher run.

Active only when ``ROBOCURVE_CAPTURE_DIR`` is set (``launch_battery_insertion.py
--capture-dir``) and the run has cameras (``--enable-cameras``). At each control
step listed in ``ROBOCURVE_CAPTURE_STEPS`` (default ``1,25,49``) it writes every
task camera's RGB image as ``<camera>_step<NNNN>.png``. Otherwise it does nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

CAMERA_NAMES = ("overhead_cam", "front_cam", "side_cam")
_written: set[tuple[str, int]] = set()
_warned: set[str] = set()


def capture_steps(raw: str | None) -> set[int]:
    text = raw if raw is not None else "1,25,49"
    return {int(part) for part in text.replace(" ", "").split(",") if part}


def maybe_capture(env, step_id: int) -> list[str]:
    directory = os.environ.get("ROBOCURVE_CAPTURE_DIR")
    if not directory or step_id not in capture_steps(os.environ.get("ROBOCURVE_CAPTURE_STEPS")):
        return []
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in CAMERA_NAMES:
        if (name, step_id) in _written:
            continue
        try:
            sensor = env.scene[name]
        except Exception:
            if name not in _warned:
                _warned.add(name)
                print(f"capture: no camera {name!r} in the scene (run with --enable-cameras)")
            continue
        rgb = sensor.data.output["rgb"]
        rgb = rgb.torch if hasattr(rgb, "torch") else rgb
        image = rgb[0].detach().cpu().numpy()[..., :3]
        from PIL import Image

        path = out / f"{name}_step{step_id:04d}.png"
        Image.fromarray(image.astype("uint8")).save(path)
        _written.add((name, step_id))
        saved.append(str(path))
    if saved:
        print(f"capture: wrote {saved}", flush=True)
    return saved
