"""VLM action annotation per scene. Allows overlap and gaps."""

import copy
import json
import logging
import os
import re

from openai import OpenAI

from prompts import get_prompts
from utils.frames import build_video_content
from utils.video import parse_json_array
from utils.vllm_client import get_extra_body

logger = logging.getLogger(__name__)

# Slack allowed on a clip-relative timestamp before it counts as out of range: the
# VLM rounds to one decimal, so a boundary event can land just past the clip's end.
TIMESTAMP_TOLERANCE = 0.15


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_actions(raw_text: str, clip_duration: float | None) -> list[dict]:
    """Parse the VLM's action array.

    ``clip_duration`` is set only for prompt sets declaring ACTIONS_TIMESTAMPS =
    "relative", where the VLM reports times on the clip's own scale (0.0 = first
    frame). Those are range-checked against the clip here — a value outside it is a
    parse failure and retries — and the caller adds the scene offset. Prompt sets
    asking for absolute times pass None and are parsed exactly as before.
    """
    actions = parse_json_array(raw_text)
    for a in actions:
        assert all(k in a for k in ("action_id", "start", "end", "subject", "description"))
        a["start"], a["end"], a["action_id"] = float(a["start"]), float(a["end"]), int(a["action_id"])
        if clip_duration is None:
            continue
        assert -TIMESTAMP_TOLERANCE <= a["start"] < a["end"] <= clip_duration + TIMESTAMP_TOLERANCE, (
            f"action [{a['start']}, {a['end']}] outside clip [0.0, {clip_duration:.1f}]"
        )
        a["start"] = max(0.0, a["start"])
        a["end"] = min(clip_duration, a["end"])
    return actions


def _segment_single_scene(client: OpenAI, frame_dir: str, fps: float,
                          scene: dict, config: dict, surveillance: bool = False) -> list[dict]:
    video_content, num_frames = build_video_content(
        frame_dir, fps, start_time=scene["start"], end_time=scene["end"])
    clip_duration = scene["end"] - scene["start"]
    prompts = get_prompts(config.get("dataset_type"))
    template = prompts.PROMPT_ACTIONS_SURVEILLANCE if surveillance else prompts.PROMPT_ACTIONS
    prompt = template.format(
        num_frames=num_frames, start=scene["start"], end=scene["end"],
        duration=clip_duration, parent_description=scene["description"])

    relative = getattr(prompts, "ACTIONS_TIMESTAMPS", "absolute") == "relative"

    max_retries = config.get("max_retries", 3)
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=config["vlm_model"],
            messages=[{"role": "user", "content": [
                video_content, {"type": "text", "text": prompt},
            ]}],
            temperature=config.get("vlm_temperature_actions", 0.2),
            top_p=config.get("vlm_top_p", 0.8),
            presence_penalty=config.get("vlm_presence_penalty_actions", 0.0),
            max_tokens=config.get("vlm_max_tokens_actions", 4096),
            extra_body=get_extra_body(config),
        )
        raw_text = _strip_thinking(response.choices[0].message.content or "")
        try:
            actions = _parse_actions(raw_text, clip_duration if relative else None)
            if relative:
                for a in actions:
                    a["start"] = round(a["start"] + scene["start"], 1)
                    a["end"] = round(a["end"] + scene["start"], 1)
            return actions
        except (json.JSONDecodeError, AssertionError) as e:
            logger.warning(f"[Actions] Scene {scene['scene_id']} parse failed (attempt {attempt}): {e}")
            if attempt == max_retries:
                logger.warning(
                    f"[Actions] Scene {scene['scene_id']}: no parseable actions after "
                    f"{max_retries} retries; recording empty actions and continuing")
                return []


def run_actions(client: OpenAI, frame_dir: str, segments: list[dict],
                metadata: dict, config: dict, surveillance: bool = False) -> list[dict]:
    """Per-scene VLM calls to annotate visual actions. Allows overlap and gaps."""
    video_id = metadata["video_id"]
    cache_path = os.path.join(config["output_dir"], video_id, "segments_with_actions.json")
    if os.path.exists(cache_path):
        logger.info(f"[Actions] {video_id}: cached")
        with open(cache_path) as f:
            return json.load(f)

    segments = copy.deepcopy(segments)
    fps = config.get("video_fps", 1.0)
    min_dur = config.get("min_action_duration", 0.5)

    for i, seg in enumerate(segments):
        logger.info(f"[Actions] {video_id}: scene {seg['scene_id']}/{len(segments)} "
                     f"[{seg['start']:.1f}s - {seg['end']:.1f}s]")
        actions = _segment_single_scene(client, frame_dir, fps, seg, config, surveillance=surveillance)
        actions = [a for a in actions if a["end"] - a["start"] >= min_dur]
        for idx, a in enumerate(actions, start=1):
            a["action_id"] = idx
        segments[i]["actions"] = actions
        logger.info(f"[Actions] Scene {seg['scene_id']}: {len(actions)} actions")

    with open(cache_path, "w") as f:
        json.dump(segments, f, indent=2)
    logger.info(f"[Actions] {video_id}: saved")
    return segments
