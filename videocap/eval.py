"""Deterministic reference-based evaluation for VideoCap annotations."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from videocap.schema import validate_record
from videocap.structured import DIMENSIONS

DEFAULT_THRESHOLDS = {
    "global_caption_f1": 0.5,
    "event_boundary_iou": 0.5,
    "event_caption_f1": 0.5,
    "temporal_coverage_delta": 0.2,
    "consistency": 0.5,
    "hallucination": 0.5,
}


def _tokens(text: str) -> Counter[str]:
    return Counter(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def _overlap(candidate: Counter[str], reference: Counter[str]) -> int:
    return sum((candidate & reference).values())


def _precision(candidate: Counter[str], reference: Counter[str]) -> float:
    count = sum(candidate.values())
    return _overlap(candidate, reference) / count if count else 1.0


def _f1(candidate: str, reference: str) -> float:
    candidate_tokens = _tokens(candidate)
    reference_tokens = _tokens(reference)
    overlap = _overlap(candidate_tokens, reference_tokens)
    candidate_count = sum(candidate_tokens.values())
    reference_count = sum(reference_tokens.values())
    total = candidate_count + reference_count
    return 2 * overlap / total if total else 1.0


def _interval_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    intersection = max(
        0,
        min(left["end_ms"], right["end_ms"])
        - max(left["start_ms"], right["start_ms"]),
    )
    union = max(left["end_ms"], right["end_ms"]) - min(left["start_ms"], right["start_ms"])
    return intersection / union if union else 0.0


def _match_events(
    candidates: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, int, float], ...]:
    pairs = sorted(
        (
            (_interval_iou(candidate, reference), candidate_index, reference_index)
            for candidate_index, candidate in enumerate(candidates)
            for reference_index, reference in enumerate(references)
        ),
        reverse=True,
    )
    used_candidates: set[int] = set()
    used_references: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, candidate_index, reference_index in pairs:
        if iou <= 0:
            break
        if candidate_index in used_candidates or reference_index in used_references:
            continue
        used_candidates.add(candidate_index)
        used_references.add(reference_index)
        matches.append((candidate_index, reference_index, iou))
    return tuple(matches)


def _coverage(events: Sequence[Mapping[str, Any]], duration_ms: int) -> float:
    intervals = sorted((event["start_ms"], event["end_ms"]) for event in events)
    covered = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            covered += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return (covered + end - start) / duration_ms


def evaluate_record(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Evaluate one annotation against a reference and apply acceptance thresholds."""

    validate_record(candidate)
    validate_record(reference)
    if candidate["video_id"] != reference["video_id"]:
        raise ValueError("candidate and reference video_id must match")
    if candidate["duration_ms"] != reference["duration_ms"]:
        raise ValueError("candidate and reference duration_ms must match")

    caption_scores = {
        dimension: _f1(candidate["captions"][dimension], reference["captions"][dimension])
        for dimension in DIMENSIONS
    }
    matches = _match_events(candidate["events"], reference["events"])
    boundary_iou = sum(match[2] for match in matches) / max(
        len(candidate["events"]), len(reference["events"])
    )
    event_caption_f1 = sum(
        _f1(
            candidate["events"][candidate_index]["caption"],
            reference["events"][reference_index]["caption"],
        )
        for candidate_index, reference_index, _ in matches
    ) / max(len(candidate["events"]), len(reference["events"]))

    candidate_event_text = " ".join(event["caption"] for event in candidate["events"])
    candidate_all_text = " ".join((*candidate["captions"].values(), candidate_event_text))
    reference_all_text = " ".join(
        (*reference["captions"].values(), *(event["caption"] for event in reference["events"]))
    )
    candidate_coverage = _coverage(candidate["events"], candidate["duration_ms"])
    reference_coverage = _coverage(reference["events"], reference["duration_ms"])
    metrics = {
        "global_caption_f1": sum(caption_scores.values()) / len(caption_scores),
        "event_boundary_iou": boundary_iou,
        "event_caption_f1": event_caption_f1,
        "temporal_coverage": candidate_coverage,
        "temporal_coverage_delta": abs(candidate_coverage - reference_coverage),
        "consistency": _precision(
            _tokens(candidate["captions"]["detailed"]),
            _tokens(candidate_event_text),
        ),
        "hallucination": 1.0 - _precision(_tokens(candidate_all_text), _tokens(reference_all_text)),
    }
    accepted = all(
        metrics[name] <= threshold
        if name in {"temporal_coverage_delta", "hallucination"}
        else metrics[name] >= threshold
        for name, threshold in thresholds.items()
    )
    return {
        "video_id": candidate["video_id"],
        "accepted": accepted,
        "caption_f1_by_dimension": caption_scores,
        "matched_events": len(matches),
        "candidate_events": len(candidate["events"]),
        "reference_events": len(reference["events"]),
        "metrics": metrics,
    }


def evaluate_dataset(
    candidates: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    split: str = "unspecified",
    thresholds: Mapping[str, float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Evaluate an aligned dataset and return per-video and macro metrics."""

    candidate_by_id = {record["video_id"]: record for record in candidates}
    reference_by_id = {record["video_id"]: record for record in references}
    if len(candidate_by_id) != len(candidates) or len(reference_by_id) != len(references):
        raise ValueError("video_id values must be unique")
    if candidate_by_id.keys() != reference_by_id.keys():
        raise ValueError("candidate and reference video_id sets must match")
    records = [
        evaluate_record(candidate_by_id[video_id], reference_by_id[video_id], thresholds=thresholds)
        for video_id in sorted(candidate_by_id)
    ]
    metric_names = tuple(records[0]["metrics"])
    aggregate = {
        name: sum(record["metrics"][name] for record in records) / len(records)
        for name in metric_names
    }
    return {
        "schema_version": "videocap-eval/v0.1",
        "split": split,
        "thresholds": dict(thresholds),
        "videos": len(records),
        "accepted": sum(record["accepted"] for record in records),
        "aggregate": aggregate,
        "records": records,
    }


__all__ = ["DEFAULT_THRESHOLDS", "evaluate_dataset", "evaluate_record"]
