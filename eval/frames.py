"""Judge-side frame sampling, cached per (split, video_id, N) so all systems
see the same batch. Independent of generation-side extraction."""

import io
import os

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


def _sample_indices(total: int, n: int) -> list[int]:
    n = min(n, total)
    return [int(x) for x in np.linspace(0, total - 1, n)]


def extract_frames(video_path: str, n: int, max_long_side: int, quality: int) -> list[bytes]:
    """Return up to n uniformly-sampled frames as JPEG bytes."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise RuntimeError(f"video has no frames: {video_path}")
    idx = _sample_indices(total, n)
    arr = vr.get_batch(idx).asnumpy()  # [k, H, W, 3]
    out = []
    for frame in arr:
        img = Image.fromarray(frame)
        w, h = img.size
        scale = max_long_side / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        out.append(buf.getvalue())
    return out


def cached_frames(cache_dir: str, split: str, video_id: str, n: int,
                  video_path: str, max_long_side: int, quality: int) -> list[bytes]:
    """Extract-or-load frames; cached as a directory of jpgs. Corrupt/partial
    cache (wrong count) is recomputed, never silently used."""
    d = os.path.join(cache_dir, "frames", f"{split}__{video_id}__n{n}")
    if os.path.isdir(d):
        files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
        if files:
            data = []
            ok = True
            for f in files:
                try:
                    with open(os.path.join(d, f), "rb") as fh:
                        data.append(fh.read())
                except OSError:
                    ok = False
                    break
            if ok and data:
                return data
    os.makedirs(d, exist_ok=True)
    frames = extract_frames(video_path, n, max_long_side, quality)
    for i, b in enumerate(frames):
        tmp = os.path.join(d, f".{i:04d}.jpg.tmp")
        with open(tmp, "wb") as fh:
            fh.write(b)
        os.replace(tmp, os.path.join(d, f"{i:04d}.jpg"))
    return frames
