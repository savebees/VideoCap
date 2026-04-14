import base64
import logging
import math
import os
import subprocess

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    quality: int = 95,
    max_long_side: int = 672,
) -> list[str]:
    """Extract frames from video at target fps. Returns list of frame filenames."""
    os.makedirs(output_dir, exist_ok=True)

    scale_filter = (
        f"fps={fps},"
        f"scale='if(gt(iw,ih),min({max_long_side},iw),-2)'"
        f":'if(gt(ih,iw),min({max_long_side},ih),-2)'"
    )
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", scale_filter,
        "-q:v", str(max(1, min(31, int(32 - quality * 31 / 100)))),
        "-start_number", "0",
        os.path.join(output_dir, "%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    frame_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".jpg"))
    if not frame_files:
        raise RuntimeError(f"No frames extracted from {video_path}")

    logger.info(f"Extracted {len(frame_files)} frames at {fps} fps from {os.path.basename(video_path)}")
    return frame_files


def load_frames_as_base64(
    frame_dir: str,
    fps: float,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[str, int]:
    """Load frames as base64 data URL for vLLM video input."""
    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not frame_files:
        raise RuntimeError(f"No frames found in {frame_dir}")

    if start_time is not None or end_time is not None:
        start_idx = int(math.floor(start_time * fps)) if start_time is not None else 0
        end_idx = int(math.ceil(end_time * fps)) if end_time is not None else len(frame_files)
        start_idx = max(0, start_idx)
        end_idx = min(len(frame_files), end_idx)
        if start_idx >= end_idx:
            start_idx = max(0, end_idx - 1)
        frame_files = frame_files[start_idx:end_idx]

    b64_frames = []
    for fname in frame_files:
        with open(os.path.join(frame_dir, fname), "rb") as f:
            b64_frames.append(base64.b64encode(f.read()).decode("ascii"))

    return "data:video/jpeg;base64," + ",".join(b64_frames), len(b64_frames)


def make_video_frames_content(
    frame_dir: str,
    fps: float,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[dict, int]:
    """Build vLLM-compatible video content dict from pre-extracted frames."""
    data_url, num_frames = load_frames_as_base64(frame_dir, fps, start_time, end_time)
    return {"type": "video_url", "video_url": {"url": data_url}}, num_frames
