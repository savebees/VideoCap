"""Derive grounded question-answer examples from accepted VideoCap records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from videocap.schema import validate_record

_GLOBAL_QUESTIONS = {
    "short": "What happens in this video?",
    "main_object": "Who or what is central to the video, and what do they do?",
    "background": "Where does the video take place, and what is visible in the background?",
    "camera": "How is the video filmed?",
    "detailed": "Describe the video in detail and in chronological order.",
}


def derive_qa(record: Mapping[str, Any], *, split: str = "unspecified") -> dict[str, Any]:
    """Create deterministic QA examples with annotation-level provenance."""

    validate_record(record)
    video_id = record["video_id"]
    all_event_ids = [event["event_id"] for event in record["events"]]
    examples: list[dict[str, Any]] = []
    for dimension, question in _GLOBAL_QUESTIONS.items():
        examples.append(
            {
                "qa_id": f"{video_id}__global_{dimension}",
                "task": f"global_{dimension}",
                "question": question,
                "answer": record["captions"][dimension],
                "provenance": {"event_ids": all_event_ids, "evidence_frames_ms": []},
            }
        )
    for event in record["events"]:
        examples.append(
            {
                "qa_id": f"{video_id}__{event['event_id']}",
                "task": "temporal_event",
                "question": f"What happens from {event['start_ms']} ms to {event['end_ms']} ms?",
                "answer": event["caption"],
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
