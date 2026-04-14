"""Steps 1-3: VLM scene segmentation, boundary validation, dense captioning."""

import copy
import json
import logging
import os

from openai import OpenAI

from prompts import PROMPT_SEGMENT, PROMPT_CAPTION, PROMPT_CAPTION_RETRY
from utils.frames import make_video_frames_content
from utils.video import parse_json_array

logger = logging.getLogger(__name__)


# ─── Step 1 ───

def _parse_segments(raw_text: str) -> list[dict]:
    segments = parse_json_array(raw_text)
    for s in segments:
        assert all(k in s for k in ("scene_id", "start", "end", "brief"))
        s["start"], s["end"], s["scene_id"] = float(s["start"]), float(s["end"]), int(s["scene_id"])
    return segments


def run_step1(client: OpenAI, frame_dir: str, metadata: dict, config: dict) -> list[dict]:
    """Single VLM call to segment the entire video into scenes."""
    video_id = metadata["video_id"]
    cache_path = os.path.join(config["output_dir"], video_id, "step1_segments.json")

    if os.path.exists(cache_path):
        logger.info(f"[Step 1] {video_id}: cached")
        with open(cache_path) as f:
            return json.load(f)

    fps = config.get("video_fps", 1.0)
    video_content, num_frames = make_video_frames_content(frame_dir, fps)
    prompt = PROMPT_SEGMENT.format(num_frames=num_frames, duration=metadata["duration"])

    max_retries = config.get("max_retries", 3)
    for attempt in range(1, max_retries + 1):
        logger.info(f"[Step 1] {video_id}: segmentation (attempt {attempt})")
        response = client.chat.completions.create(
            model=config["vlm_model"],
            messages=[{"role": "user", "content": [
                video_content, {"type": "text", "text": prompt},
            ]}],
            temperature=config.get("vlm_temperature", 0.0),
            max_tokens=config.get("vlm_max_tokens_step1", 4096),
        )
        raw_text = response.choices[0].message.content or ""
        try:
            segments = _parse_segments(raw_text)
            break
        except (json.JSONDecodeError, AssertionError) as e:
            logger.warning(f"[Step 1] Parse failed (attempt {attempt}): {e}")
            if attempt == max_retries:
                raise RuntimeError(f"[Step 1] Failed after {max_retries} retries") from e

    with open(cache_path, "w") as f:
        json.dump(segments, f, indent=2)
    logger.info(f"[Step 1] {video_id}: {len(segments)} segments")
    return segments


# ─── Step 2 ───

def run_step2(segments: list[dict], duration: float, config: dict, video_id: str) -> tuple[list[dict], int]:
    """Programmatic boundary repair: snap endpoints, fix gaps/overlaps, merge short segments."""
    cache_path = os.path.join(config["output_dir"], video_id, "step2_validated.json")
    if os.path.exists(cache_path):
        logger.info(f"[Step 2] {video_id}: cached")
        with open(cache_path) as f:
            data = json.load(f)
        return data["segments"], data["repairs"]

    segments = copy.deepcopy(segments)
    repairs = 0
    min_dur = config.get("min_segment_duration", 3.0)

    segments.sort(key=lambda s: s["start"])

    if abs(segments[0]["start"]) > 1e-6:
        segments[0]["start"] = 0.0
        repairs += 1
    if abs(segments[-1]["end"] - duration) > 1e-6:
        segments[-1]["end"] = duration
        repairs += 1

    # Fix gaps/overlaps
    for i in range(len(segments) - 1):
        diff = segments[i + 1]["start"] - segments[i]["end"]
        if abs(diff) < 1e-6:
            continue
        repairs += 1
        if abs(diff) < 0.5:
            mid = (segments[i]["end"] + segments[i + 1]["start"]) / 2
            segments[i]["end"] = mid
            segments[i + 1]["start"] = mid
        else:
            segments[i]["end"] = segments[i + 1]["start"]

    # Merge short segments
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(segments):
            seg_dur = segments[i]["end"] - segments[i]["start"]
            if seg_dur < min_dur and len(segments) > 1:
                if i == 0:
                    ni = 1
                elif i == len(segments) - 1:
                    ni = i - 1
                else:
                    left = segments[i - 1]["end"] - segments[i - 1]["start"]
                    right = segments[i + 1]["end"] - segments[i + 1]["start"]
                    ni = i - 1 if left <= right else i + 1

                if ni < i:
                    segments[ni]["end"] = segments[i]["end"]
                    segments[ni]["brief"] += " " + segments[i]["brief"]
                else:
                    segments[ni]["start"] = segments[i]["start"]
                    segments[ni]["brief"] = segments[i]["brief"] + " " + segments[ni]["brief"]
                segments.pop(i)
                changed = True
                repairs += 1
            else:
                i += 1

    # Final cleanup
    for i in range(len(segments) - 1):
        if abs(segments[i + 1]["start"] - segments[i]["end"]) > 1e-6:
            mid = (segments[i]["end"] + segments[i + 1]["start"]) / 2
            segments[i]["end"] = mid
            segments[i + 1]["start"] = mid
    segments[0]["start"] = 0.0
    segments[-1]["end"] = duration

    for idx, s in enumerate(segments, start=1):
        s["scene_id"] = idx

    with open(cache_path, "w") as f:
        json.dump({"segments": segments, "repairs": repairs}, f, indent=2)
    logger.info(f"[Step 2] {video_id}: {len(segments)} segments, {repairs} repairs")
    return segments, repairs


# ─── Step 3 ───

def _build_prefix(descriptions: list[dict], max_prev: int = 3) -> str:
    if not descriptions:
        return ""
    recent = descriptions[-max_prev:]
    lines = ["=== Previously described scenes (DO NOT repeat this information) ==="]
    for d in recent:
        lines.append(f"Scene {d['scene_id']} [{d['start']:.1f}s - {d['end']:.1f}s]:\n{d['description']}")
    lines.append("=== End of previous scenes ===")
    return "\n\n".join(lines)


def _vlm_call(client: OpenAI, video_content: dict, prompt_text: str, config: dict) -> str:
    response = client.chat.completions.create(
        model=config["vlm_model"],
        messages=[{"role": "user", "content": [
            video_content, {"type": "text", "text": prompt_text},
        ]}],
        temperature=config.get("vlm_temperature", 0.0),
        max_tokens=config.get("vlm_max_tokens_step3", 2048),
    )
    return response.choices[0].message.content.strip()


def run_step3(client: OpenAI, frame_dir: str, segments: list[dict],
              metadata: dict, config: dict) -> list[dict]:
    """Dense captioning with prefix context. Sequential per-segment VLM calls."""
    video_id = metadata["video_id"]
    cache_path = os.path.join(config["output_dir"], video_id, "step3_captioned.json")
    if os.path.exists(cache_path):
        logger.info(f"[Step 3] {video_id}: cached")
        with open(cache_path) as f:
            return json.load(f)

    segments = copy.deepcopy(segments)
    fps = config.get("video_fps", 1.0)
    max_prev = config.get("prefix_context_num", 3)
    min_words = config.get("min_description_words", 100)
    max_retries = config.get("max_retries_short_desc", 2)

    completed = [{"scene_id": s["scene_id"], "start": s["start"],
                   "end": s["end"], "description": s.get("brief", "")} for s in segments]

    for i, seg in enumerate(segments):
        logger.info(f"[Step 3] {video_id}: scene {seg['scene_id']}/{len(segments)} "
                     f"[{seg['start']:.1f}s - {seg['end']:.1f}s]")

        prefix = _build_prefix(completed[:i], max_prev)
        video_content, num_frames = make_video_frames_content(
            frame_dir, fps, start_time=seg["start"], end_time=seg["end"])

        prompt = PROMPT_CAPTION.format(prefix_context=prefix, num_frames=num_frames,
                                        start=seg["start"], end=seg["end"])
        description = _vlm_call(client, video_content, prompt, config)

        word_count = len(description.split())
        for retry in range(max_retries):
            if word_count >= min_words:
                break
            logger.warning(f"[Step 3] Scene {seg['scene_id']}: {word_count} words, retry {retry + 1}")
            retry_prompt = PROMPT_CAPTION_RETRY.format(
                word_count=word_count, previous_attempt=description, prefix_context=prefix)
            description = _vlm_call(client, video_content, retry_prompt, config)
            word_count = len(description.split())

        if word_count < min_words:
            logger.warning(
                f"[Step 3] Scene {seg['scene_id']}: {word_count} words after {max_retries} retries "
                f"(min={min_words}); keeping best result and continuing")

        segments[i]["description"] = description
        segments[i]["word_count"] = word_count
        completed[i]["description"] = description

    with open(cache_path, "w") as f:
        json.dump(segments, f, indent=2)
    logger.info(f"[Step 3] {video_id}: {len(segments)} scenes captioned")
    return segments
