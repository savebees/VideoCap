"""VLM scene segmentation, boundary validation, and dense captioning."""

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


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Scene segmentation

def _parse_segments(raw_text: str) -> list[dict]:
    segments = parse_json_array(raw_text)
    for s in segments:
        assert all(k in s for k in ("scene_id", "start", "end", "brief"))
        s["start"], s["end"], s["scene_id"] = float(s["start"]), float(s["end"]), int(s["scene_id"])
    return segments


def _vlm_segment_call(client: OpenAI, video_content: dict, prompt: str,
                      config: dict, tag: str) -> list[dict]:
    """Single VLM segmentation call with parse + retry. Returns parsed segments."""
    max_retries = config.get("max_retries", 3)
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        logger.info(f"{tag} (attempt {attempt})")
        response = client.chat.completions.create(
            model=config["vlm_model"],
            messages=[{"role": "user", "content": [
                video_content, {"type": "text", "text": prompt},
            ]}],
            temperature=config.get("vlm_temperature_segmentation", 0.2),
            top_p=config.get("vlm_top_p", 0.8),
            presence_penalty=config.get("vlm_presence_penalty_segmentation", 0.5),
            max_tokens=config.get("vlm_max_tokens_segmentation", 4096),
            extra_body=get_extra_body(config),
        )
        raw_text = _strip_thinking(response.choices[0].message.content or "")
        try:
            return _parse_segments(raw_text)
        except (json.JSONDecodeError, AssertionError) as e:
            logger.warning(f"{tag} parse failed (attempt {attempt}): {e}")
            last_exc = e
    raise RuntimeError(f"{tag} failed after {max_retries} retries") from last_exc


def run_segmentation(client: OpenAI, frame_dir: str, metadata: dict, config: dict) -> list[dict]:
    """Single-pass scene segmentation."""
    video_id = metadata["video_id"]
    cache_path = os.path.join(config["output_dir"], video_id, "segments.json")

    if os.path.exists(cache_path):
        logger.info(f"[Segmentation] {video_id}: cached")
        with open(cache_path) as f:
            return json.load(f)

    fps = config.get("video_fps", 1.0)
    prompts = get_prompts(config.get("dataset_type"))
    video_content, num_frames = build_video_content(frame_dir, fps)
    prompt = prompts.PROMPT_SEGMENT.format(
        num_frames=num_frames,
        duration=metadata["duration"],
    )
    segments = _vlm_segment_call(
        client, video_content, prompt, config,
        tag=f"[Segmentation] {video_id}",
    )

    with open(cache_path, "w") as f:
        json.dump(segments, f, indent=2)
    logger.info(f"[Segmentation] {video_id}: {len(segments)} segments")
    return segments


def run_fixed_chunk_segmentation(client: OpenAI, frame_dir: str,
                                 metadata: dict, config: dict) -> list[dict]:
    """Long/surveillance path: deterministic boundaries with per-chunk briefs."""
    video_id = metadata["video_id"]
    cache_path = os.path.join(config["output_dir"], video_id, "segments.json")
    if os.path.exists(cache_path):
        logger.info(f"[Segmentation] {video_id}: cached")
        with open(cache_path) as f:
            return json.load(f)

    duration = metadata["duration"]
    chunk_sec = float(config.get("long_video_chunk_sec", 120))
    fps = float(config.get("video_fps", 1.0))
    frame_interval = 1.0 / fps
    prompt_template = get_prompts(config.get("dataset_type")).PROMPT_BRIEF
    segments: list[dict] = []
    start = 0.0
    scene_id = 1
    while start < duration - 1e-6:
        end = min(start + chunk_sec, duration)
        # a tail shorter than one frame interval has no frame of its own; absorb it
        if duration - end < frame_interval:
            end = duration
        video_content, num_frames = build_video_content(
            frame_dir, fps, start_time=start, end_time=end)
        prompt = prompt_template.format(
            num_frames=num_frames, start=start, end=end)
        brief = _vlm_brief_call(
            client, video_content, prompt, config,
            tag=f"[Segmentation] {video_id}: scene {scene_id} brief",
        )
        segments.append({
            "scene_id": scene_id,
            "start": round(start, 1),
            "end": round(end, 1),
            "brief": brief,
        })
        start = end
        scene_id += 1

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(segments, f, indent=2)
    logger.info(f"[Segmentation] {video_id}: {len(segments)} fixed {chunk_sec:.0f}s chunks (fixed-chunk mode)")
    return segments


# Brief annotation

def _vlm_brief_call(client: OpenAI, video_content: dict, prompt: str,
                    config: dict, tag: str) -> str:
    """Generate the brief for one deterministic scene chunk."""
    logger.info(tag)
    response = client.chat.completions.create(
        model=config["vlm_model"],
        messages=[{"role": "user", "content": [
            video_content, {"type": "text", "text": prompt},
        ]}],
        temperature=config.get("vlm_temperature_segmentation", 0.2),
        top_p=config.get("vlm_top_p", 0.8),
        presence_penalty=config.get("vlm_presence_penalty_segmentation", 0.5),
        max_tokens=config.get("vlm_max_tokens_brief", 256),
        extra_body=get_extra_body(config),
    )
    brief = _strip_thinking(response.choices[0].message.content or "")
    if not brief:
        raise ValueError(f"{tag}: empty brief")
    return brief


# Boundary validation

def run_boundary_validation(segments: list[dict], duration: float, config: dict, video_id: str) -> tuple[list[dict], int]:
    """Programmatic boundary repair."""
    cache_path = os.path.join(config["output_dir"], video_id, "validated.json")
    if os.path.exists(cache_path):
        logger.info(f"[Validation] {video_id}: cached")
        with open(cache_path) as f:
            data = json.load(f)
        return data["segments"], data["repairs"]

    segments = copy.deepcopy(segments)
    repairs = 0

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
    logger.info(f"[Validation] {video_id}: {len(segments)} segments, {repairs} repairs")
    return segments, repairs


# Dense captioning

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
        temperature=config.get("vlm_temperature_captioning", 0.7),
        top_p=config.get("vlm_top_p", 0.8),
        presence_penalty=config.get("vlm_presence_penalty_captioning", 1.0),
        max_tokens=config.get("vlm_max_tokens_captioning", 2048),
        extra_body=get_extra_body(config),
    )
    return _strip_thinking(response.choices[0].message.content or "")


def run_captioning(client: OpenAI, frame_dir: str, segments: list[dict],
                   metadata: dict, config: dict, surveillance: bool = False) -> list[dict]:
    """Dense captioning with prefix context."""
    video_id = metadata["video_id"]
    cache_path = os.path.join(config["output_dir"], video_id, "captions.json")
    if os.path.exists(cache_path):
        logger.info(f"[Captioning] {video_id}: cached")
        with open(cache_path) as f:
            return json.load(f)

    segments = copy.deepcopy(segments)
    fps = config.get("video_fps", 1.0)
    max_prev = config.get("prefix_context_num", 3)
    prompts = get_prompts(config.get("dataset_type"))
    if surveillance:
        min_words = config.get("min_description_words_surveillance", 30)
        caption_template = prompts.PROMPT_CAPTION_SURVEILLANCE
        # Dataset-specific surveillance retry if defined, else the generic one.
        retry_template = getattr(prompts, "PROMPT_CAPTION_RETRY_SURVEILLANCE",
                                 prompts.PROMPT_CAPTION_RETRY)
    else:
        min_words = config.get("min_description_words", 100)
        caption_template = prompts.PROMPT_CAPTION
        retry_template = prompts.PROMPT_CAPTION_RETRY
    max_retries = config.get("max_retries_short_desc", 2)

    completed = [{"scene_id": s["scene_id"], "start": s["start"],
                   "end": s["end"], "description": s.get("brief", "")} for s in segments]

    for i, seg in enumerate(segments):
        logger.info(f"[Captioning] {video_id}: scene {seg['scene_id']}/{len(segments)} "
                     f"[{seg['start']:.1f}s - {seg['end']:.1f}s]")

        prefix = _build_prefix(completed[:i], max_prev)
        video_content, num_frames = build_video_content(
            frame_dir, fps, start_time=seg["start"], end_time=seg["end"])

        prompt = caption_template.format(prefix_context=prefix, num_frames=num_frames,
                                          start=seg["start"], end=seg["end"])
        description = _vlm_call(client, video_content, prompt, config)

        word_count = len(description.split())
        for retry in range(max_retries):
            if word_count >= min_words:
                break
            logger.warning(f"[Captioning] Scene {seg['scene_id']}: {word_count} words, retry {retry + 1}")
            retry_prompt = retry_template.format(
                word_count=word_count, previous_attempt=description,
                prefix_context=prefix, min_words=min_words)
            description = _vlm_call(client, video_content, retry_prompt, config)
            word_count = len(description.split())

        if word_count < min_words:
            logger.warning(
                f"[Captioning] Scene {seg['scene_id']}: {word_count} words after {max_retries} retries "
                f"(min={min_words}); keeping best result and continuing")

        segments[i]["description"] = description
        segments[i]["word_count"] = word_count
        completed[i]["description"] = description

    with open(cache_path, "w") as f:
        json.dump(segments, f, indent=2)
    logger.info(f"[Captioning] {video_id}: {len(segments)} scenes captioned")
    return segments
