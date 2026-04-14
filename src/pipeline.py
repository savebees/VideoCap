"""Main orchestration + CLI for the dense video annotation pipeline."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.preprocess import run_step0
from src.scene_segmentation import run_step1, run_step2, run_step3
from src.atomic_events import run_step4, run_step5
from src.object_detection import run_step6
from utils.vllm_client import get_vlm_client

logger = logging.getLogger(__name__)
PIPELINE_VERSION = "0.3.0"


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "default.yaml",
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_annotation(segments: list[dict], metadata: dict, config: dict,
                    step1_raw_count: int, step2_repairs: int, step5_repairs: int) -> str:
    video_id = metadata["video_id"]
    video_output_dir = os.path.join(config["output_dir"], video_id)

    total_events = sum(len(s.get("events", [])) for s in segments)
    total_objects = sum(len(s.get("objects", [])) for s in segments)
    n = len(segments)

    annotation = {
        "video_id": video_id,
        "duration": metadata["duration"],
        "fps": metadata["fps"],
        "resolution": f"{metadata['width']}x{metadata['height']}",
        "codec": metadata["codec"],
        "file_size_mb": metadata["file_size_mb"],
        "segments": segments,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "models": {"vlm": config["vlm_model"], "detector": "GroundingDINO-Swin-T"},
            "total_segments": n,
            "avg_segment_duration": round(metadata["duration"] / n, 2) if n else 0,
            "total_events": total_events,
            "avg_events_per_segment": round(total_events / n, 2) if n else 0,
            "total_objects": total_objects,
            "step1_raw_segments": step1_raw_count,
            "step2_repairs": step2_repairs,
            "step5_repairs": step5_repairs,
        },
    }

    out_path = os.path.join(video_output_dir, "annotation.json")
    with open(out_path, "w") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved annotation to {out_path}")
    return out_path


def process_video(video_path: str, config: dict) -> str:
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    logger.info(f"{'='*60}\nProcessing: {video_id}\n{'='*60}")

    metadata, frame_dir = run_step0(video_path, config)
    client = get_vlm_client(config)

    # Category A: scene segmentation + description
    segments = run_step1(client, frame_dir, metadata, config)
    step1_raw_count = len(segments)
    segments, step2_repairs = run_step2(segments, metadata["duration"], config, video_id)
    segments = run_step3(client, frame_dir, segments, metadata, config)

    # Category C: atomic events
    segments = run_step4(client, frame_dir, segments, metadata, config)
    segments, step5_repairs = run_step5(segments, metadata, config)

    # Category B: object detection
    segments = run_step6(client, frame_dir, segments, metadata, config)

    return save_annotation(segments, metadata, config, step1_raw_count, step2_repairs, step5_repairs)


def clear_cache(video_path: str, config: dict):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(config["output_dir"], video_id)
    for fname in ("step1_segments.json", "step2_validated.json", "step3_captioned.json",
                  "step4_events.json", "step5_events_validated.json", "step6_objects.json",
                  "annotation.json"):
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            logger.info(f"Removed: {fpath}")


def main():
    parser = argparse.ArgumentParser(description="Dense Video Annotation Pipeline")
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--video_dir", type=str, help="Directory containing video files")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--force", action="store_true", help="Clear cache and re-run")
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

    logger.info(f"Found {len(video_paths)} video(s)")

    for vpath in video_paths:
        if args.force:
            clear_cache(vpath, config)
        process_video(vpath, config)


if __name__ == "__main__":
    main()
