"""Small camera artifacts, using numpy and the standard library only."""
import json
from pathlib import Path
import struct
import zlib
import numpy as np
from .adapter import plain


def write_png(path, frame):
    image = np.asarray(frame)
    height, width, channels = image.shape
    if channels != 3 or image.dtype != np.uint8:
        raise ValueError('PNG writer requires uint8 RGB')
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    raw = b''.join(b'\0' + row.tobytes() for row in image)
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    Path(path).write_bytes(data)


def save_capture(directory, observation, info, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'action_labels': list(info.action_space.semantics.dim_labels),
                'control_hz': info.control_hz, 'state': plain(dict(observation.state)),
                'task_report': plain(report), 'cameras': {}}
    frames = []
    for label, image in observation.images.items():
        write_png(directory / f'{label}.png', image)
        manifest['cameras'][label] = {'shape': list(image.shape), 'dtype': str(image.dtype),
                                     'min': int(image.min()), 'max': int(image.max()),
                                     'std': float(image.std()), 'time': observation.image_times.get(label)}
        frames.append(image)
    height = max(frame.shape[0] for frame in frames)
    sheet = np.full((height, sum(frame.shape[1] for frame in frames), 3), 32, dtype=np.uint8)
    left = 0
    for frame in frames:
        sheet[:frame.shape[0], left:left + frame.shape[1]] = frame
        left += frame.shape[1]
    write_png(directory / 'contact_sheet.png', sheet)
    (directory / 'capture.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    figures = ''.join(f'<figure><img src="{name}.png"><figcaption>{name}</figcaption></figure>'
                      for name in observation.images)
    (directory / 'cameras.html').write_text('<!doctype html><meta charset="utf-8"><title>Arena cameras</title>'
        '<style>body{background:#222;color:white;font:20px sans-serif}main{display:flex;gap:16px;flex-wrap:wrap}'
        'figure{margin:0}img{max-width:30vw;image-rendering:auto}figcaption{padding:8px}</style>'
        '<h1>Actual simulated sensor feeds</h1><main>' + figures + '</main>')
