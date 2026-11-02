"""Save the task cameras' RGB frames at chosen control steps.

Enabled by ``ROBOCURVE_CAPTURE_DIR`` (the launcher's ``--capture-dir``) and only
useful with ``--enable-cameras``. ``ROBOCURVE_CAPTURE_STEPS`` is a comma list of
``env.common_step_counter`` values, default ``1,25,49``. The PNG writer is pure
Python so no image library is needed on the server.
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

DEFAULT_STEPS = "1,25,49"
_DONE: set[tuple[str, int, str]] = set()


def capture_steps(text: str | None = None) -> set[int]:
    raw = DEFAULT_STEPS if text is None else text
    steps = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            value = int(item)
            if value < 0:
                raise ValueError(f"capture step must be >= 0, got {value}")
            steps.add(value)
    return steps


def png_bytes(rows: list[list[tuple[int, int, int]]] | list, width: int, height: int) -> bytes:
    """Encode RGB8 rows (height lists of width (r, g, b) tuples, or a flat bytes object)."""
    if isinstance(rows, (bytes, bytearray)):
        flat = bytes(rows)
        if len(flat) != width * height * 3:
            raise ValueError("flat RGB buffer has the wrong size")
        raw = b"".join(b"\x00" + flat[y * width * 3 : (y + 1) * width * 3] for y in range(height))
    else:
        raw = b"".join(b"\x00" + bytes(channel for pixel in row for channel in pixel[:3]) for row in rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def write_rgb_png(path: Path, image) -> None:
    """``image`` is an HxWx3 or HxWx4 uint8 tensor or array."""
    if hasattr(image, "torch"):
        image = image.torch
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    height, width = int(image.shape[0]), int(image.shape[1])
    flat = image[:, :, :3].astype("uint8").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(flat, width, height))


def maybe_capture(env, cfg) -> list[Path]:
    directory = os.environ.get("ROBOCURVE_CAPTURE_DIR")
    if not directory:
        return []
    step = int(getattr(env, "common_step_counter", -1))
    if step not in capture_steps(os.environ.get("ROBOCURVE_CAPTURE_STEPS")):
        return []
    written = []
    for camera in cfg.cameras:
        key = (directory, step, camera.name)
        if key in _DONE:
            continue
        _DONE.add(key)
        try:
            sensor = env.scene[camera.name]
            image = sensor.data.output["rgb"][0]
        except Exception as exc:  # cameras are only present with --enable-cameras
            print(f"capture: camera {camera.name!r} unavailable at step {step}: {exc}", flush=True)
            continue
        path = Path(directory) / f"step{step:05d}_{camera.name}.png"
        write_rgb_png(path, image)
        print(f"capture: wrote {path}", flush=True)
        written.append(path)
    return written
