"""Step 0: Preprocessing — metadata extraction + frame extraction."""

import json
import logging
import os

from utils.video import get_video_metadata
from utils.frames import extract_frames

logger = logging.getLogger(__name__)


def run_step0(video_path: str, config: dict) -> tuple[dict, str]:
    """Extract video metadata and frames. Cached if outputs exist."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(config["output_dir"], video_id)
    os.makedirs(video_output_dir, exist_ok=True)

    metadata_path = os.path.join(video_output_dir, "metadata.json")
    frame_dir = os.path.join(video_output_dir, "frames")

    frames_exist = (
        os.path.isdir(frame_dir)
        and any(f.endswith(".jpg") for f in os.listdir(frame_dir))
    )
    if os.path.exists(metadata_path) and frames_exist:
        logger.info(f"[Step 0] {video_id}: cached")
        with open(metadata_path) as f:
            return json.load(f), frame_dir

    logger.info(f"[Step 0] {video_id}: extracting metadata + frames")
    meta = get_video_metadata(video_path)
    assert meta["duration"] > 0, f"Invalid duration: {meta['duration']}"

    metadata = {
        "video_id": video_id,
        "file_path": os.path.relpath(video_path),
        **meta,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    extract_frames(
        video_path, frame_dir,
        fps=config.get("video_fps", 1.0),
        quality=config.get("frame_quality", 95),
        max_long_side=config.get("frame_max_long_side", 672),
    )
    return metadata, frame_dir
