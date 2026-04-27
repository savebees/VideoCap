import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import jsonschema
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.preprocess import run_step0
from src.scene_segmentation import run_step1, run_step2, run_step3
from src.actions import run_step4
from src.object_detection import run_step5
from utils.vllm_client import get_vlm_client

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "default.yaml",
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


@contextmanager
def _timer(timings: dict, key: str):
    t = time.perf_counter()
    yield
    timings[key] = round(time.perf_counter() - t, 2)


def _detect_frame_resolution(config: dict, video_id: str) -> str:
    """Read the first extracted frame jpg and return its 'WxH' string."""
    from PIL import Image
    frame_dir = os.path.join(config["output_dir"], video_id, "frames")
    if not os.path.isdir(frame_dir):
        return "unknown"
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not frames:
        return "unknown"
    with Image.open(os.path.join(frame_dir, frames[0])) as img:
        return f"{img.size[0]}x{img.size[1]}"


def save_annotation(segments: list[dict], objects: list[dict], metadata: dict, config: dict,
                    step1_raw_count: int, step2_repairs: int,
                    timings: dict) -> tuple[str, str]:
    video_id = metadata["video_id"]
    video_output_dir = os.path.join(config["output_dir"], video_id)

    total_actions = sum(len(s.get("actions", [])) for s in segments)
    total_objects = len(objects)
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
            "models": {"vlm": config["vlm_model"]},
            "total_segments": n,
            "avg_segment_duration": round(metadata["duration"] / n, 2) if n else 0,
            "total_actions": total_actions,
            "avg_actions_per_segment": round(total_actions / n, 2) if n else 0,
            "step1_raw_segments": step1_raw_count,
            "step2_repairs": step2_repairs,
            "timings_sec": timings,
        },
    }

    objects_doc = {
        "video_id": video_id,
        "frame_extraction_fps": config.get("video_fps", 1.0),
        "frame_resolution": _detect_frame_resolution(config, video_id),
        "objects": objects,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "models": {"vlm": config["vlm_model"]},
            "total_objects": total_objects,
            "detection_nms_threshold": config.get("detection_nms_threshold", 0.5),
            "detection_confidence_floor": config.get("detection_confidence_floor", 0.3),
        },
    }

    schema_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "schemas",
    )
    annotation_schema_path = os.path.join(schema_dir, "annotation.schema.json")
    objects_schema_path = os.path.join(schema_dir, "objects.schema.json")

    for doc, schema_path, label in [
        (annotation, annotation_schema_path, "annotation"),
        (objects_doc, objects_schema_path, "objects"),
    ]:
        with open(schema_path) as f:
            schema = json.load(f)
        try:
            jsonschema.validate(instance=doc, schema=schema)
        except jsonschema.ValidationError as e:
            logger.warning(f"Schema validation failed for {video_id} ({label}): "
                           f"{e.message} at {list(e.absolute_path)}")

    annotation_path = os.path.join(video_output_dir, "annotation.json")
    objects_path = os.path.join(video_output_dir, "objects.json")
    with open(annotation_path, "w") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)
    with open(objects_path, "w") as f:
        json.dump(objects_doc, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved annotation to {annotation_path}")
    logger.info(f"Saved objects to {objects_path}")
    return annotation_path, objects_path


def process_video(video_path: str, config: dict) -> tuple[str, str]:
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    logger.info(f"{'='*60}\nProcessing: {video_id}\n{'='*60}")

    timings: dict = {}
    video_t0 = time.perf_counter()

    with _timer(timings, "step0"):
        metadata, frame_dir = run_step0(video_path, config)
    client = get_vlm_client(config)

    # scene segmentation + description
    with _timer(timings, "step1"):
        segments = run_step1(client, frame_dir, metadata, config)
    step1_raw_count = len(segments)
    with _timer(timings, "step2"):
        segments, step2_repairs = run_step2(segments, metadata["duration"], config, video_id)
    with _timer(timings, "step3"):
        segments = run_step3(client, frame_dir, segments, metadata, config)

    # actions
    with _timer(timings, "step4"):
        segments = run_step4(client, frame_dir, segments, metadata, config)

    # object detection (per-frame, output saved to a separate file)
    with _timer(timings, "step5"):
        objects = run_step5(client, frame_dir, segments, metadata, config)

    timings["total"] = round(time.perf_counter() - video_t0, 2)
    logger.info(
        f"[{video_id}] timings(s): " +
        " ".join(f"{k}={v}" for k, v in timings.items())
    )

    return save_annotation(segments, objects, metadata, config,
                           step1_raw_count, step2_repairs, timings)


def clear_cache(video_path: str, config: dict):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(config["output_dir"], video_id)
    for fname in ("step1_segments.json", "step2_validated.json", "step3_captioned.json",
                  "step4_actions.json", "step5_objects.json",
                  "annotation.json", "objects.json"):
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
