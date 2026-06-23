"""Per-frame object detection via Qwen3.6-VL native grounding."""

import base64
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from PIL import Image

from prompts import get_prompts
from utils.video import parse_json_array
from utils.vllm_client import get_extra_body

logger = logging.getLogger(__name__)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _iou(box_a: list, box_b: list) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _box_area(box: list) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _greedy_suppress(dets: list[dict], iou_threshold: float) -> list[dict]:
    """Greedy NMS over boxes pre-sorted by descending area: keep each surviving
    box and suppress later boxes overlapping it beyond iou_threshold (so the
    larger box wins, since Qwen3-VL grounding emits no confidence)."""
    kept = []
    suppressed = [False] * len(dets)
    for i in range(len(dets)):
        if suppressed[i]:
            continue
        kept.append(dets[i])
        for j in range(i + 1, len(dets)):
            if not suppressed[j] and _iou(dets[i]["bbox"], dets[j]["bbox"]) > iou_threshold:
                suppressed[j] = True
    return kept


def _intra_frame_nms(detections: list[dict], iou_threshold: float,
                     cross_label_threshold: float) -> list[dict]:
    """Within a single frame, drop duplicate boxes for the same physical object.

    There is NO cross-frame dedup; each frame is deduplicated independently in
    two passes, both keeping the larger (more complete) box:
      1. Same-label: same label + IoU > iou_threshold is a duplicate.
      2. Cross-label: near-identical boxes (IoU > cross_label_threshold) are the
         same object even when their labels drifted ("man in black jacket" vs
         "person in dark jacket"). The high threshold preserves genuinely
         distinct overlapping objects (a person and the bag they hold).
    """
    if not detections:
        return []

    # Pass 1 — per-label dedup.
    by_label: dict[str, list[dict]] = {}
    for det in detections:
        by_label.setdefault(det["label"], []).append(det)
    kept = []
    for dets in by_label.values():
        dets.sort(key=lambda d: _box_area(d["bbox"]), reverse=True)
        kept.extend(_greedy_suppress(dets, iou_threshold))

    # Pass 2 — cross-label dedup of near-identical boxes.
    kept.sort(key=lambda d: _box_area(d["bbox"]), reverse=True)
    return _greedy_suppress(kept, max(iou_threshold, cross_label_threshold))


def _frame_to_data_url(frame_path: str) -> str:
    """Read a single jpg and produce a data:image URL for the OpenAI-style API."""
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _build_subjects_hint(scene_actions: list[dict]) -> str:
    """Produce the optional subjects-hint snippet from a scene's actions, or empty string."""
    if not scene_actions:
        return ""
    subjects = sorted({a["subject"] for a in scene_actions if a.get("subject")})
    if not subjects:
        return ""
    return (
        f"\nSubjects already identified as performing actions in this scene: "
        f"{', '.join(subjects)}.\n"
        f"When detecting people, reuse these subject phrasings if they match."
    )


def _scene_for_frame(timestamp: float, segments: list[dict]) -> dict | None:
    """Find which segment a frame timestamp falls into. Returns None if outside all segments."""
    for seg in segments:
        if seg["start"] <= timestamp < seg["end"]:
            return seg
    # Edge case: timestamp == last segment's end
    if segments and abs(timestamp - segments[-1]["end"]) < 1e-6:
        return segments[-1]
    return None


def _detect_one_frame(client: OpenAI, frame_path: str, scene: dict,
                      config: dict) -> list[dict]:
    """Run Qwen3.6 detection on a single frame; return raw detections with pixel-space bbox."""
    img = Image.open(frame_path)
    frame_w, frame_h = img.size

    prompts = get_prompts(config.get("dataset_type"))
    template = prompts.PROMPT_DETECT_OBJECTS
    fmt = {"parent_description": scene["description"]}
    # subjects_hint exists only in the general template, not the NWPU one.
    if "{subjects_hint}" in template:
        fmt["subjects_hint"] = _build_subjects_hint(scene.get("actions", []))
    prompt = template.format(**fmt)
    image_url = _frame_to_data_url(frame_path)

    max_retries = config.get("max_retries", 3)
    nms_threshold = config.get("detection_nms_threshold", 0.5)
    cross_label_threshold = config.get("detection_cross_label_iou_threshold", 0.9)
    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=config["vlm_model"],
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]}],
            temperature=config.get("vlm_temperature_detection", 0.0),
            top_p=config.get("vlm_top_p", 0.8),
            presence_penalty=config.get("vlm_presence_penalty_detection", 0.0),
            max_tokens=config.get("vlm_max_tokens_detection", 2048),
            extra_body=get_extra_body(config),
        )
        raw = _strip_thinking(response.choices[0].message.content or "")
        try:
            items = parse_json_array(raw)
            results = []
            for item in items:
                # Qwen3-VL native grounding emits only label + bbox_2d (no
                # confidence — matching the official grounding format).
                if "label" not in item or "bbox_2d" not in item:
                    continue
                bb = item["bbox_2d"]
                if len(bb) != 4:
                    continue
                # Rescale 0-1000 normalized to pixel coordinates of THIS frame jpg.
                x1 = round(float(bb[0]) / 1000 * frame_w)
                y1 = round(float(bb[1]) / 1000 * frame_h)
                x2 = round(float(bb[2]) / 1000 * frame_w)
                y2 = round(float(bb[3]) / 1000 * frame_h)
                # Clamp + sanity check
                x1, x2 = max(0, min(x1, frame_w)), max(0, min(x2, frame_w))
                y1, y2 = max(0, min(y1, frame_h)), max(0, min(y2, frame_h))
                if x2 <= x1 or y2 <= y1:
                    continue
                results.append({
                    "label": str(item["label"]).strip(),
                    "bbox": [x1, y1, x2, y2],
                })
            deduped = _intra_frame_nms(results, nms_threshold, cross_label_threshold)
            if len(deduped) < len(results):
                logger.info(
                    f"[Detection] {os.path.basename(frame_path)}: "
                    f"intra-frame NMS dropped {len(results) - len(deduped)} duplicate(s)"
                )
            return deduped
        except (json.JSONDecodeError, AssertionError, ValueError, TypeError) as e:
            logger.warning(f"[Detection] {os.path.basename(frame_path)} parse failed (attempt {attempt}): {e}")
            if attempt == max_retries:
                logger.error(f"[Detection] {os.path.basename(frame_path)}: giving up after {max_retries} retries")
                return []
    return []


def run_detection(client: OpenAI, frame_dir: str, segments: list[dict],
                  metadata: dict, config: dict) -> list[dict]:
    """Per-frame Qwen3.6 object detection. Returns flat list of detections."""
    video_id = metadata["video_id"]
    cache_path = os.path.join(config["output_dir"], video_id, "detections.json")
    if os.path.exists(cache_path):
        logger.info(f"[Detection] {video_id}: cached")
        with open(cache_path) as f:
            return json.load(f)

    fps = config.get("video_fps", 1.0)
    concurrency = max(1, int(config.get("detection_concurrency", 8)))
    # detection_nms_threshold is read directly inside _detect_one_frame for intra-frame NMS.

    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not frame_files:
        raise RuntimeError(f"No frames in {frame_dir}")

    def _process_frame(fname: str) -> list[dict]:
        """Detect one frame. Returns detections tagged with frame metadata."""
        frame_index = int(os.path.splitext(fname)[0])
        frame_timestamp = round(frame_index / fps, 2)
        scene = _scene_for_frame(frame_timestamp, segments)
        if scene is None:
            logger.warning(f"[Detection] frame {frame_index} (t={frame_timestamp}s) outside all segments; skipping")
            return []

        frame_path = os.path.join(frame_dir, fname)
        detections = _detect_one_frame(client, frame_path, scene, config)
        out = [
            {
                "frame_index": frame_index,
                "frame_timestamp": frame_timestamp,
                "scene_id": scene["scene_id"],
                **d,
            }
            for d in detections
        ]
        logger.info(f"[Detection] {video_id}: frame {frame_index} -> {len(detections)} dets")
        return out

    raw_detections: list[dict] = []
    logger.info(f"[Detection] {video_id}: {len(frame_files)} frames, concurrency={concurrency}")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for dets in pool.map(_process_frame, frame_files):
            raw_detections.extend(dets)

    raw_detections.sort(key=lambda d: (d["frame_index"], -_box_area(d["bbox"])))
    final = [{
        "object_id": idx,
        "frame_index": d["frame_index"],
        "frame_timestamp": d["frame_timestamp"],
        "scene_id": d["scene_id"],
        "label": d["label"],
        "bbox": d["bbox"],
    } for idx, d in enumerate(raw_detections, start=1)]

    with open(cache_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    logger.info(f"[Detection] {video_id}: {len(final)} detections (post intra-frame NMS), saved")
    return final
