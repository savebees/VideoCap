"""Step 6: Object detection using Grounding DINO."""

import copy
import json
import logging
import math
import os
import re

from openai import OpenAI
from PIL import Image

from prompts import PROMPT_EXTRACT_OBJECTS
from utils.video import parse_json_array

logger = logging.getLogger(__name__)


def _get_frame_indices(start: float, end: float, fps: float, strategy: str) -> list[int]:
    """Return frame indices for a segment. 'all' uses every 1fps frame."""
    start_idx = int(math.floor(start * fps))
    end_idx = int(math.ceil(end * fps))
    if strategy == "all":
        return list(range(start_idx, end_idx))
    # middle: single frame
    return [(start_idx + end_idx - 1) // 2]


def _cross_frame_nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    """
    Cross-frame NMS: deduplicate the same object detected across multiple frames.
    For detections with the same label and IoU > threshold, keep the highest confidence.
    """
    if not detections:
        return []

    # Group by label
    by_label: dict[str, list[dict]] = {}
    for det in detections:
        by_label.setdefault(det["label"], []).append(det)

    kept = []
    for label, dets in by_label.items():
        dets.sort(key=lambda d: d["confidence"], reverse=True)
        suppressed = [False] * len(dets)

        for i in range(len(dets)):
            if suppressed[i]:
                continue
            kept.append(dets[i])
            for j in range(i + 1, len(dets)):
                if suppressed[j]:
                    continue
                if _iou(dets[i]["bbox"], dets[j]["bbox"]) > iou_threshold:
                    suppressed[j] = True

    kept.sort(key=lambda d: (d["frame_index"], -d["confidence"]))
    for idx, det in enumerate(kept, start=1):
        det["object_id"] = idx
    return kept


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


def _extract_object_prompts_vlm(client: OpenAI, description: str, config: dict) -> list[str]:
    """Use VLM text-only call to extract object noun phrases from caption."""
    response = client.chat.completions.create(
        model=config["vlm_model"],
        messages=[{"role": "user", "content": PROMPT_EXTRACT_OBJECTS.format(description=description)}],
        temperature=0.0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content or ""
    try:
        result = parse_json_array(raw)
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            return result
    except (json.JSONDecodeError, AssertionError):
        pass

    # Fallback: regex extraction
    phrases = set()
    for match in re.finditer(
        r'\b(?:a|an|the)\s+(?:\w+\s+){0,3}\w+(?:\s+(?:in|with|wearing)\s+(?:\w+\s+){0,2}\w+)?',
        description.lower(),
    ):
        phrase = re.sub(r'^(?:a|an|the)\s+', '', match.group(0).strip())
        if len(phrase.split()) >= 2:
            phrases.add(phrase)
    return list(phrases)[:20]


def _load_grounding_dino(config: dict):
    from groundingdino.util.inference import load_model
    return load_model(config["grounding_dino_config"], config["grounding_dino_weights"])


def _run_grounding_dino(model, image_path: str, text_prompts: list[str],
                        box_threshold: float, text_threshold: float,
                        orig_w: int, orig_h: int,
                        frame_w: int, frame_h: int) -> list[dict]:
    from groundingdino.util.inference import predict
    import groundingdino.datasets.transforms as T

    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image, None)

    text_prompt = " . ".join(text_prompts) + " ."
    boxes, logits, phrases = predict(
        model=model, image=image_tensor, caption=text_prompt,
        box_threshold=box_threshold, text_threshold=text_threshold,
    )

    scale_x = orig_w / frame_w if frame_w > 0 else 1
    scale_y = orig_h / frame_h if frame_h > 0 else 1

    detections = []
    for box, logit, phrase in zip(boxes, logits, phrases):
        cx, cy, w, h = box.tolist()
        detections.append({
            "label": phrase,
            "bbox": [round((cx - w/2) * frame_w * scale_x), round((cy - h/2) * frame_h * scale_y),
                     round((cx + w/2) * frame_w * scale_x), round((cy + h/2) * frame_h * scale_y)],
            "confidence": round(float(logit), 4),
        })
    return detections


def run_step6(client: OpenAI, frame_dir: str, segments: list[dict],
              metadata: dict, config: dict) -> list[dict]:
    """Object detection: extract prompts, run Grounding DINO on all frames, NMS dedup."""
    video_id = metadata["video_id"]
    cache_path = os.path.join(config["output_dir"], video_id, "step6_objects.json")
    if os.path.exists(cache_path):
        logger.info(f"[Step 6] {video_id}: cached")
        with open(cache_path) as f:
            return json.load(f)

    segments = copy.deepcopy(segments)
    fps = config.get("video_fps", 1.0)
    box_threshold = config.get("detection_box_threshold", 0.35)
    text_threshold = config.get("detection_text_threshold", 0.25)
    nms_threshold = config.get("detection_nms_threshold", 0.5)
    strategy = config.get("keyframe_strategy", "all")

    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not frame_files:
        raise RuntimeError(f"No frames in {frame_dir}")
    sample = Image.open(os.path.join(frame_dir, frame_files[0]))
    frame_w, frame_h = sample.size

    model = _load_grounding_dino(config)

    for i, seg in enumerate(segments):
        logger.info(f"[Step 6] {video_id}: scene {seg['scene_id']}/{len(segments)}")

        text_prompts = _extract_object_prompts_vlm(client, seg["description"], config)
        if not text_prompts:
            segments[i]["objects"] = []
            continue

        frame_indices = _get_frame_indices(seg["start"], seg["end"], fps, strategy)
        raw_detections = []

        for kf_idx in frame_indices:
            frame_path = os.path.join(frame_dir, f"{kf_idx:06d}.jpg")
            if not os.path.exists(frame_path):
                continue
            for det in _run_grounding_dino(model, frame_path, text_prompts,
                                           box_threshold, text_threshold,
                                           metadata["width"], metadata["height"],
                                           frame_w, frame_h):
                raw_detections.append({"frame_index": kf_idx, **det})

        segments[i]["objects"] = _cross_frame_nms(raw_detections, nms_threshold)
        logger.info(f"[Step 6] Scene {seg['scene_id']}: "
                     f"{len(raw_detections)} raw -> {len(segments[i]['objects'])} after NMS")

    with open(cache_path, "w") as f:
        json.dump(segments, f, indent=2)
    logger.info(f"[Step 6] {video_id}: saved")
    return segments
