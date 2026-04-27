"""Step 4: VLM action annotation per scene. Allows overlap and gaps."""

import copy
import json
import logging
import os
import re

from openai import OpenAI

from prompts import PROMPT_ACTIONS
from utils.frames import build_video_content
from utils.video import parse_json_array
from utils.vllm_client import get_extra_body

logger = logging.getLogger(__name__)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_actions(raw_text: str) -> list[dict]:
    actions = parse_json_array(raw_text)
    for a in actions:
        assert all(k in a for k in ("action_id", "start", "end", "subject", "description"))
        a["start"], a["end"], a["action_id"] = float(a["start"]), float(a["end"]), int(a["action_id"])
    return actions


def _segment_single_scene(client: OpenAI, frame_dir: str, fps: float,
                          scene: dict, config: dict) -> list[dict]:
    video_content, num_frames = build_video_content(
        frame_dir, fps, start_time=scene["start"], end_time=scene["end"])
    prompt = PROMPT_ACTIONS.format(
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
            top_p=config.get("vlm_top_p", 0.8),
            presence_penalty=config.get("vlm_presence_penalty", 0.0),
            max_tokens=config.get("vlm_max_tokens_step4", 4096),
            extra_body=get_extra_body(config),
        )
        raw_text = _strip_thinking(response.choices[0].message.content or "")
        try:
            return _parse_actions(raw_text)
        except (json.JSONDecodeError, AssertionError) as e:
            logger.warning(f"[Step 4] Scene {scene['scene_id']} parse failed (attempt {attempt}): {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"[Step 4] Scene {scene['scene_id']}: failed after {max_retries} retries") from e


def run_step4(client: OpenAI, frame_dir: str, segments: list[dict],
              metadata: dict, config: dict) -> list[dict]:
    """Per-scene VLM calls to annotate visual actions. Allows overlap and gaps."""
    video_id = metadata["video_id"]
    cache_path = os.path.join(config["output_dir"], video_id, "step4_actions.json")
    if os.path.exists(cache_path):
        logger.info(f"[Step 4] {video_id}: cached")
        with open(cache_path) as f:
            return json.load(f)

    segments = copy.deepcopy(segments)
    fps = config.get("video_fps", 1.0)
    min_dur = config.get("min_action_duration", 0.5)

    for i, seg in enumerate(segments):
        logger.info(f"[Step 4] {video_id}: scene {seg['scene_id']}/{len(segments)} "
                     f"[{seg['start']:.1f}s - {seg['end']:.1f}s]")
        actions = _segment_single_scene(client, frame_dir, fps, seg, config)
        actions = [a for a in actions if a["end"] - a["start"] >= min_dur]
        for idx, a in enumerate(actions, start=1):
            a["action_id"] = idx
        segments[i]["actions"] = actions
        logger.info(f"[Step 4] Scene {seg['scene_id']}: {len(actions)} actions")

    with open(cache_path, "w") as f:
        json.dump(segments, f, indent=2)
    logger.info(f"[Step 4] {video_id}: saved")
    return segments
