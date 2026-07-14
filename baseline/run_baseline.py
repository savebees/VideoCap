"""Naive whole-video caption baseline: the SAME served model as the pipeline
produces one plain caption per clip (no chunking/segmentation/actions), isolating
the pipeline/prompt variable. Self-contained; writes baseline.json next to the
pipeline's annotation.json."""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.video import extract_metadata
from utils.frames import build_video_content
from utils.vllm_client import get_vlm_client, get_extra_body

logger = logging.getLogger(__name__)

# Deliberately minimal so the baseline measures the bare model, not prompt craft.
PROMPT_BASELINE = "Describe this video in detail."


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_uniform_frames(video_path: str, out_dir: str, n: int,
                           quality: int, max_long_side: int) -> tuple[dict, int]:
    """Uniformly sample n frames spanning the whole video. Rate is clamped to the
    source fps, so a clip with fewer than n real frames yields all of them instead
    of ffmpeg duplicating frames. Returns (metadata, n_extracted)."""
    meta = extract_metadata(video_path)
    duration = meta["duration"]
    assert duration > 0, f"Invalid duration: {duration}"
    source_fps = meta["fps"]
    assert source_fps > 0, f"Invalid fps: {source_fps}"

    fps_extract = min(n / duration, source_fps)   # n frames across the clip, never oversampling

    scale_filter = (
        f"fps={fps_extract:.6f},"
        f"scale='if(gt(iw,ih),min({max_long_side},iw),-2)'"
        f":'if(gt(ih,iw),min({max_long_side},ih),-2)'"
    )
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", scale_filter,
        "-q:v", str(max(1, min(31, int(32 - quality * 31 / 100)))),
        "-start_number", "0",
        os.path.join(out_dir, "%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    frames = sorted(f for f in os.listdir(out_dir) if f.endswith(".jpg"))
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    # Trim rounding overflow to exactly n, keeping an evenly-spaced subset.
    if len(frames) > n:
        keep_idx = {round(i * (len(frames) - 1) / (n - 1)) for i in range(n)}
        for j, f in enumerate(frames):
            if j not in keep_idx:
                os.remove(os.path.join(out_dir, f))
        frames = sorted(f for f in os.listdir(out_dir) if f.endswith(".jpg"))

    return meta, len(frames)


def process_video(video_path: str, config: dict, client, force: bool = False) -> str | None:
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(config["output_dir"], video_id)
    out_path = os.path.join(out_dir, "baseline.json")

    if os.path.exists(out_path) and not force:
        logger.info(f"[Baseline] {video_id}: cached, skipping")
        return out_path

    logger.info(f"[Baseline] {video_id}: start")
    timings: dict = {}
    t0 = time.perf_counter()

    n = int(config.get("num_frames", 32))
    tmp_dir = tempfile.mkdtemp(prefix=f"baseline_{video_id}_")
    try:
        t = time.perf_counter()
        meta, num_frames = extract_uniform_frames(
            video_path, tmp_dir, n,
            quality=config.get("frame_quality", 95),
            max_long_side=config.get("frame_max_long_side", 672),
        )
        timings["extract"] = round(time.perf_counter() - t, 2)
        logger.info(f"[Baseline] {video_id}: {num_frames} frames (duration {meta['duration']:.1f}s)")

        video_content, num_frames = build_video_content(tmp_dir, fps=1.0)

        t = time.perf_counter()
        response = client.chat.completions.create(
            model=config["vlm_model"],
            messages=[{"role": "user", "content": [
                video_content, {"type": "text", "text": PROMPT_BASELINE},
            ]}],
            temperature=config.get("vlm_temperature", 0.7),
            top_p=config.get("vlm_top_p", 0.8),
            presence_penalty=config.get("vlm_presence_penalty", 1.5),
            max_tokens=config.get("vlm_max_tokens", 2048),
            extra_body=get_extra_body(config),
        )
        timings["inference"] = round(time.perf_counter() - t, 2)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    caption = _strip_thinking(response.choices[0].message.content or "")
    word_count = len(caption.split())
    timings["total"] = round(time.perf_counter() - t0, 2)

    doc = {
        "video_id": video_id,
        "caption": caption,
        "word_count": word_count,
        "num_frames": num_frames,
        "model": config["vlm_model"],
        "timings_sec": timings,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    logger.info(f"[Baseline] {video_id}: {word_count} words, {num_frames} frames "
                f"(infer={timings['inference']}s) -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Naive whole-video caption baseline")
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--video_dir", type=str, help="Directory containing video files")
    parser.add_argument("--config", type=str, help="Path to baseline config YAML")
    parser.add_argument("--force", action="store_true", help="Re-run even if baseline.json exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)

    if args.video:
        video_paths = [args.video]
    elif args.video_dir:
        video_paths = sorted(
            os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir)
            if f.endswith((".mp4", ".avi", ".mkv", ".mov", ".webm"))
        )
    else:
        parser.error("Either --video or --video_dir is required")

    logger.info(f"[Baseline] Found {len(video_paths)} video(s)")
    client = get_vlm_client(config)
    for vpath in video_paths:
        try:
            process_video(vpath, config, client, force=args.force)
        except Exception as e:
            logger.error(f"[Baseline] {os.path.basename(vpath)} failed: {e}")


if __name__ == "__main__":
    main()
