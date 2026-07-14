import base64
import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    quality: int = 95,
    max_long_side: int = 672,
) -> list[str]:

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


def build_video_content(
    frame_dir: str,
    fps: float,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[dict, int]:
    """Mux the window's frames into an fps-stamped MJPEG/AVI (-c:v copy, no
    re-encode). The container carries the time base: bare JPEG concatenation
    decodes as 1 fps, so at 6 fps the model would read an 8 s clip as 48 s."""
    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))

    start_idx = 0
    if start_time is not None and end_time is not None:
        start_idx = int(start_time * fps)
        end_idx = int(end_time * fps)
        frame_files = frame_files[start_idx:end_idx]

    if not frame_files:
        raise RuntimeError(
            f"No frames in {frame_dir} for window [{start_time}, {end_time}] at {fps} fps "
            f"(frame index range [{start_idx}:{int((end_time or 0) * fps)}]). "
            f"An empty window means the segment is degenerate or inverted."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_path = os.path.join(tmp_dir, "clip.avi")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", f"{fps}",
            "-start_number", str(start_idx),
            "-i", os.path.join(frame_dir, "%06d.jpg"),
            "-frames:v", str(len(frame_files)),
            "-c:v", "copy",
            "-r", f"{fps}",
            clip_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed: {result.stderr}")
        with open(clip_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")

    data_url = "data:video/avi;base64," + b64
    return {"type": "video_url", "video_url": {"url": data_url}}, len(frame_files)
