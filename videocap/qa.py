"""Derive grounded question-answer examples from accepted VideoCap records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from videocap.schema import validate_record

_GLOBAL_TASKS = {
    "short": ("video_summary", "What is the overall content of the video?"),
    "main_object": (
        "action_recognition",
        "What actions does the main subject perform throughout the video?",
    ),
    "background": (
        "scene_transition",
        "How does the setting or background change throughout the video?",
    ),
    "camera": (
        "camera_understanding",
        "How do the camera shots and viewpoints change throughout the video?",
    ),
    "detailed": (
        "temporal_reasoning",
        "How do the events unfold from beginning to end?",
    ),
}


def derive_qa(record: Mapping[str, Any], *, split: str = "unspecified") -> dict[str, Any]:
    """Create deterministic QA examples with annotation-level provenance."""

    validate_record(record)
    video_id = record["video_id"]
    all_event_ids = [event["event_id"] for event in record["events"]]
    examples: list[dict[str, Any]] = []
    for dimension, (task, question) in _GLOBAL_TASKS.items():
        examples.append(
            {
                "qa_id": f"{video_id}__global_{dimension}",
                "task": task,
                "question": question,
                "answer": record["captions"][dimension],
                "provenance": {"event_ids": all_event_ids, "evidence_frames_ms": []},
            }
        )
    for event in record["events"]:
        examples.append(
            {
                "qa_id": f"{video_id}__{event['event_id']}__understanding",
                "task": "event_understanding",
                "question": f"What happens from {event['start_ms']} ms to {event['end_ms']} ms?",
                "answer": event["caption"],
                "provenance": {
                    "event_ids": [event["event_id"]],
                    "evidence_frames_ms": event["evidence_frames_ms"],
                },
            }
        )
        examples.append(
            {
                "qa_id": f"{video_id}__{event['event_id']}__grounding",
                "task": "temporal_grounding",
                "question": f"When does this event occur: {event['caption']}",
                "answer": f"From {event['start_ms']} ms to {event['end_ms']} ms.",
                "provenance": {
                    "event_ids": [event["event_id"]],
                    "evidence_frames_ms": event["evidence_frames_ms"],
                },
            }
        )
    return {
        "schema_version": "videocap-qa/v0.1",
        "video_id": video_id,
        "split": split,
        "examples": examples,
    }


__all__ = ["derive_qa"]
