"""Steps 4-5: VLM atomic event segmentation + boundary validation."""

import copy
import json
import logging
import os

from openai import OpenAI

from prompts import PROMPT_ATOMIC_EVENTS
from utils.frames import build_video_content
from utils.video import parse_json_array

logger = logging.getLogger(__name__)


# ─── Step 4 ───

def _parse_events(raw_text: str) -> list[dict]:
    events = parse_json_array(raw_text)
    for e in events:
        assert all(k in e for k in ("event_id", "start", "end", "description"))
        e["start"], e["end"], e["event_id"] = float(e["start"]), float(e["end"]), int(e["event_id"])
    return events


def _segment_single_scene(client: OpenAI, frame_dir: str, fps: float,
                          scene: dict, config: dict) -> list[dict]:
    video_content, num_frames = build_video_content(
        frame_dir, fps, start_time=scene["start"], end_time=scene["end"])
    prompt = PROMPT_ATOMIC_EVENTS.format(
        num_frames=num_frames, start=scene["start"], end=scene["end"],
        duration=scene["end"] - scene["start"], parent_description=scene["description"])

    max_retries = config.get("max_retries", 3)
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=config["vlm_model"],
            messages=[{"role": "user", "content": [
                video_content, {"type": "text", "text": prompt},
            ]}],
            temperature=config.get("vlm_temperature", 0.0),
            max_tokens=config.get("vlm_max_tokens_step4", 4096),
        )
        raw_text = response.choices[0].message.content or ""
        try:
            return _parse_events(raw_text)
        except (json.JSONDecodeError, AssertionError) as e:
            logger.warning(f"[Step 4] Scene {scene['scene_id']} parse failed (attempt {attempt}): {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"[Step 4] Scene {scene['scene_id']}: failed after {max_retries} retries") from e


def run_step4(client: OpenAI, frame_dir: str, segments: list[dict],
              metadata: dict, config: dict) -> list[dict]:
    """Per-segment VLM calls to divide each scene into atomic events."""
    video_id = metadata["video_id"]
    cache_path = os.path.join(config["output_dir"], video_id, "step4_events.json")
    if os.path.exists(cache_path):
        logger.info(f"[Step 4] {video_id}: cached")
        with open(cache_path) as f:
            return json.load(f)

    segments = copy.deepcopy(segments)
    fps = config.get("video_fps", 1.0)

    for i, seg in enumerate(segments):
        logger.info(f"[Step 4] {video_id}: scene {seg['scene_id']}/{len(segments)} "
                     f"[{seg['start']:.1f}s - {seg['end']:.1f}s]")
        segments[i]["events"] = _segment_single_scene(client, frame_dir, fps, seg, config)
        logger.info(f"[Step 4] Scene {seg['scene_id']}: {len(segments[i]['events'])} events")

    with open(cache_path, "w") as f:
        json.dump(segments, f, indent=2)
    logger.info(f"[Step 4] {video_id}: saved")
    return segments


# ─── Step 5 ───

def _validate_events(events: list[dict], scene_start: float, scene_end: float,
                     min_dur: float) -> tuple[list[dict], int]:
    """Validate and repair event boundaries."""
    events = copy.deepcopy(events)
    repairs = 0

    if not events:
        return [{"event_id": 1, "start": scene_start, "end": scene_end,
                 "description": "(no description)"}], 1

    events.sort(key=lambda e: e["start"])

    # Snap endpoints
    if abs(events[0]["start"] - scene_start) > 1e-6:
        events[0]["start"] = scene_start
        repairs += 1
    if abs(events[-1]["end"] - scene_end) > 1e-6:
        events[-1]["end"] = scene_end
        repairs += 1

    # Fix gaps/overlaps
    for i in range(len(events) - 1):
        diff = events[i + 1]["start"] - events[i]["end"]
        if abs(diff) < 1e-6:
            continue
        repairs += 1
        if abs(diff) < 0.5:
            mid = (events[i]["end"] + events[i + 1]["start"]) / 2
            events[i]["end"] = mid
            events[i + 1]["start"] = mid
        else:
            events[i]["end"] = events[i + 1]["start"]

    # Merge micro-events
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(events):
            dur = events[i]["end"] - events[i]["start"]
            if dur < min_dur and len(events) > 1:
                if i == 0:
                    ni = 1
                elif i == len(events) - 1:
                    ni = i - 1
                else:
                    left = events[i - 1]["end"] - events[i - 1]["start"]
                    right = events[i + 1]["end"] - events[i + 1]["start"]
                    ni = i - 1 if left <= right else i + 1

                if ni < i:
                    events[ni]["end"] = events[i]["end"]
                    events[ni]["description"] += " " + events[i]["description"]
                else:
                    events[ni]["start"] = events[i]["start"]
                    events[ni]["description"] = events[i]["description"] + " " + events[ni]["description"]
                events.pop(i)
                changed = True
                repairs += 1
            else:
                i += 1

    # Final cleanup
    for i in range(len(events) - 1):
        if abs(events[i + 1]["start"] - events[i]["end"]) > 1e-6:
            mid = (events[i]["end"] + events[i + 1]["start"]) / 2
            events[i]["end"] = mid
            events[i + 1]["start"] = mid
    events[0]["start"] = scene_start
    events[-1]["end"] = scene_end

    for idx, e in enumerate(events, start=1):
        e["event_id"] = idx

    return events, repairs


def run_step5(segments: list[dict], metadata: dict, config: dict) -> tuple[list[dict], int]:
    """Atomic event boundary validation per scene."""
    video_id = metadata["video_id"]
    cache_path = os.path.join(config["output_dir"], video_id, "step5_events_validated.json")
    if os.path.exists(cache_path):
        logger.info(f"[Step 5] {video_id}: cached")
        with open(cache_path) as f:
            data = json.load(f)
        return data["segments"], data["repairs"]

    segments = copy.deepcopy(segments)
    min_dur = config.get("min_event_duration", 1.0)
    total_repairs = 0

    for seg in segments:
        seg["events"], repairs = _validate_events(seg["events"], seg["start"], seg["end"], min_dur)
        total_repairs += repairs

    with open(cache_path, "w") as f:
        json.dump({"segments": segments, "repairs": total_repairs}, f, indent=2)
    logger.info(f"[Step 5] {video_id}: {total_repairs} repairs")
    return segments, total_repairs
