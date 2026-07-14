"""Naive caption baseline for served third-party VLMs, one model at a time.
Same contract as run_baseline.py. config input_format picks the payload:
"video" = single video_url (Qwen family), "image" = image_url list (InternVL).
Outputs: baseline/results/{model_tag}/{split}/{video_id}.json."""

import argparse
import base64
import json
import logging
import os
import shutil
import sys
import tempfile
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.frames import build_video_content
from utils.vllm_client import get_vlm_client, get_extra_body
from baseline.run_baseline import PROMPT_BASELINE, extract_uniform_frames, _strip_thinking, load_config

logger = logging.getLogger(__name__)


def build_image_content(frame_dir: str) -> tuple[list[dict], int]:
    """Return a list of image_url content items (one per sampled frame)."""
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    items = []
    for fname in frames:
        with open(os.path.join(frame_dir, fname), "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        items.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return items, len(items)


def process_video(video_path: str, config: dict, client, force: bool = False) -> str | None:
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = config["output_dir"]
    out_path = os.path.join(out_dir, f"{video_id}.json")

    if os.path.exists(out_path) and not force:
        logger.info(f"[Baseline] {video_id}: cached, skipping")
        return out_path

    logger.info(f"[Baseline] {video_id}: start")
    timings: dict = {}
    t0 = time.perf_counter()

    n = int(config.get("num_frames", 32))
    input_format = config.get("input_format", "video")
    tmp_dir = tempfile.mkdtemp(prefix=f"mbaseline_{video_id}_")
    try:
        t = time.perf_counter()
        meta, num_frames = extract_uniform_frames(
            video_path, tmp_dir, n,
            quality=config.get("frame_quality", 95),
            max_long_side=config.get("frame_max_long_side", 672),
        )
        timings["extract"] = round(time.perf_counter() - t, 2)
        logger.info(f"[Baseline] {video_id}: {num_frames} frames (duration {meta['duration']:.1f}s)")

        if input_format == "image":
            media_items, num_frames = build_image_content(tmp_dir)
        else:
            video_content, num_frames = build_video_content(tmp_dir, fps=1.0)
            media_items = [video_content]

        t = time.perf_counter()
        response = client.chat.completions.create(
            model=config["vlm_model"],
            messages=[{"role": "user", "content": [
                *media_items, {"type": "text", "text": PROMPT_BASELINE},
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
        "input_format": input_format,
        "timings_sec": timings,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    logger.info(f"[Baseline] {video_id}: {word_count} words, {num_frames} frames "
                f"(infer={timings['inference']}s) -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generalized naive caption baseline")
    parser.add_argument("--video", type=str)
    parser.add_argument("--video_dir", type=str)
    parser.add_argument("--config", type=str, required=True, help="Per-model baseline config YAML")
    parser.add_argument("--force", action="store_true")
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

    logger.info(f"[Baseline] {config['vlm_model']}: {len(video_paths)} video(s)")
    client = get_vlm_client(config)
    for vpath in video_paths:
        try:
            process_video(vpath, config, client, force=args.force)
        except Exception as e:
            logger.error(f"[Baseline] {os.path.basename(vpath)} failed: {e}")


if __name__ == "__main__":
    main()
