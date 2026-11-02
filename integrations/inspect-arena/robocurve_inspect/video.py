"""Export saved Inspect camera frames in a separate process, without starting Isaac."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

CAMERAS = ('overhead', 'front', 'side')


def require_ffmpeg(executable=None):
    """Verify the actual H.264 encoder before a paid rollout starts."""
    encoder = executable or shutil.which('ffmpeg')
    if encoder is None:
        try:
            import imageio_ffmpeg
            encoder = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            pass
    if encoder is None:
        raise RuntimeError('MP4 recording needs FFmpeg with libx264. Install ffmpeg on the server '
                           'or pass --ffmpeg /path/to/ffmpeg; no model calls were made.')
    command = [str(encoder), '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', '16x16', '-r', '15', '-i', 'pipe:0',
               '-frames:v', '1', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
               '-f', 'null', '-']
    try:
        result = subprocess.run(command, input=bytes(16 * 16 * 3), capture_output=True,
                                timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f'Cannot start the MP4 encoder: {exc}') from exc
    if result.returncode:
        raise RuntimeError('FFmpeg H.264 preflight failed: ' + result.stderr.decode(errors='replace')[-1500:])
    return str(encoder)


def _export(run_dir, ffmpeg, fps):
    """Called in the export subprocess: the installed SDK encoder may sys.exit."""
    import math
    import numpy as np
    from inspect_robots.frames import _parse_frame_filename
    from inspect_robots._video import encode_stream

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Video FPS must be positive and finite')
    # Read raw saved frames rather than requiring a finalized eval JSON: this
    # also recovers partial runs after a handled interruption or policy error.
    groups = {}
    for path in sorted((run_dir / 'inspect' / 'frames').rglob('*.npy')):
        identity = _parse_frame_filename(path.name)
        if identity is None:
            continue
        trial, camera, step = identity
        if camera in CAMERAS:
            groups.setdefault((path.parent, trial), {}).setdefault(camera, []).append((step, path))
    if len(groups) != 1:
        raise RuntimeError(f'Expected one recorded trial, found {len(groups)}; raw frames are preserved')
    streams = next(iter(groups.values()))
    missing = set(CAMERAS) - streams.keys()
    if missing:
        raise RuntimeError(f'Cannot record all three views; missing {sorted(missing)}')
    common = set.intersection(*(set(step for step, _ in streams[name]) for name in CAMERAS))
    if not common:
        raise RuntimeError('No complete three-camera frame sets were saved')
    # An interruption can occur between saving cameras. Clip only incomplete
    # sets at export; keep every original frame, and report the omission.
    selected = {name: sorted((step, path) for step, path in streams[name] if step in common)
                for name in CAMERAS}
    unpaired = {name: len(streams[name]) - len(selected[name]) for name in CAMERAS}
    video_dir = run_dir / 'videos'
    video_dir.mkdir(exist_ok=True)
    heights = []
    report = {'status': 'complete', 'fps': fps, 'clock': 'simulation time',
              'camera_order': list(CAMERAS), 'steps': sorted(common),
              'unpaired_frames_excluded': unpaired, 'streams': {}}
    for name in CAMERAS:
        temporary = video_dir / f'{name}.partial.mp4'
        result = encode_stream(selected[name], temporary, fps, ffmpeg)
        if result.error or result.piped != len(common) or result.skipped_empty:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f'{name} MP4 failed: {result.error or "missing/empty frames"}')
        target = video_dir / f'{name}.mp4'
        temporary.replace(target)
        shape = np.load(selected[name][0][1], mmap_mode='r', allow_pickle=False).shape
        heights.append((int(shape[0]) + 1) // 2 * 2)
        report['streams'][name] = {'path': str(target.relative_to(run_dir)), 'frames': result.piped}
    height = max(heights)
    temporary = run_dir / 'rollout.partial.mp4'
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y']
    for name in CAMERAS:
        command += ['-i', str(video_dir / f'{name}.mp4')]
    filters = [f'[{index}:v]pad=iw:{height}:0:0:black,setsar=1[v{index}]' for index in range(3)]
    filters += ['[v0][v1][v2]hstack=inputs=3:shortest=1[out]']
    command += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-an',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    result = subprocess.run(command, capture_output=True, timeout=300, check=False)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError('Combined MP4 failed: ' + result.stderr.decode(errors='replace')[-1500:])
    temporary.replace(run_dir / 'rollout.mp4')
    report['composite'] = 'rollout.mp4'
    (video_dir / 'recording.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def export_recording(run_dir, ffmpeg, fps):
    """Isolate SDK CLI exits and video encoding from Kit and the evaluation."""
    run_dir = Path(run_dir)
    command = [sys.executable, str(Path(__file__).resolve()), '--run-dir', str(run_dir),
               '--ffmpeg', ffmpeg, '--fps', str(fps)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError('MP4 export failed; raw frames are preserved. ' + (result.stderr or result.stdout)[-2000:])
    report = json.loads((run_dir / 'videos' / 'recording.json').read_text())
    print(f'MP4 saved: {run_dir / "rollout.mp4"}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--ffmpeg')
    parser.add_argument('--fps', type=float, required=True)
    args = parser.parse_args()
    try:
        encoder = require_ffmpeg(args.ffmpeg)
        _export(args.run_dir, encoder, args.fps)
    except Exception as exc:
        folder = args.run_dir / 'videos'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'recording.json').write_text(json.dumps({'status': 'failed', 'error': str(exc)}, indent=2) + '\n')
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
