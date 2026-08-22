"""Strict intermediate contracts for the VideoCap grounding flow.

The public dense-caption protocol is intentionally small.  This module keeps the
production graph explicit so processing windows are never confused with semantic
event windows and every model stage can be inspected independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


DIMENSIONS = ("short", "main_object", "background", "camera", "detailed")


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _timestamp(value: Any, field_name: str, duration_ms: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0 or value > duration_ms:
        raise ValueError(f"{field_name} must be within [0, {duration_ms}]")
    return value


def _evidence(values: Sequence[int], start_ms: int, end_ms: int) -> tuple[int, ...]:
    result = tuple(values)
    if tuple(sorted(set(result))) != result:
        raise ValueError("evidence_frames_ms must be sorted and unique")
    if any(timestamp < start_ms or timestamp > end_ms for timestamp in result):
        raise ValueError("evidence frame is outside its interval")
    return result


@dataclass(frozen=True)
class ProcessingWindow:
    window_id: str
    start_ms: int
    end_ms: int
    evidence_frames_ms: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_id", _text(self.window_id, "window_id"))
        if self.end_ms <= self.start_ms:
            raise ValueError("processing window end_ms must be greater than start_ms")
        object.__setattr__(self, "evidence_frames_ms", _evidence(self.evidence_frames_ms, self.start_ms, self.end_ms))

    def to_dict(self) -> dict[str, Any]:
        return {"window_id": self.window_id, "start_ms": self.start_ms, "end_ms": self.end_ms, "evidence_frames_ms": list(self.evidence_frames_ms)}


@dataclass(frozen=True)
class WindowCaption:
    window_id: str
    captions: Mapping[str, str]
    evidence_frames_ms: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_id", _text(self.window_id, "window_id"))
        if set(self.captions) != set(DIMENSIONS):
            raise ValueError(f"window caption dimensions must be exactly {DIMENSIONS}")
        normalized = {key: _text(self.captions[key], f"captions.{key}") for key in DIMENSIONS}
        object.__setattr__(self, "captions", normalized)
        evidence = tuple(self.evidence_frames_ms)
        for timestamp in evidence:
            if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
                raise ValueError("window caption evidence_frames_ms must contain non-negative integers")
        if tuple(sorted(set(evidence))) != evidence:
            raise ValueError("window caption evidence_frames_ms must be sorted and unique")
        object.__setattr__(self, "evidence_frames_ms", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {"window_id": self.window_id, "captions": dict(self.captions), "evidence_frames_ms": list(self.evidence_frames_ms)}


@dataclass(frozen=True)
class EventCandidate:
    candidate_id: str
    source_window_ids: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        if not self.source_window_ids:
            raise ValueError("event candidate must reference at least one processing window")
        object.__setattr__(self, "source_window_ids", tuple(_text(item, "source_window_id") for item in self.source_window_ids))
        object.__setattr__(self, "description", _text(self.description, "description"))

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "source_window_ids": list(self.source_window_ids), "description": self.description}


@dataclass(frozen=True)
class EventCluster:
    cluster_id: str
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cluster_id", _text(self.cluster_id, "cluster_id"))
        if not self.candidate_ids:
            raise ValueError("event cluster must contain at least one candidate")
        object.__setattr__(self, "candidate_ids", tuple(_text(item, "candidate_id") for item in self.candidate_ids))

    def to_dict(self) -> dict[str, Any]:
        return {"cluster_id": self.cluster_id, "candidate_ids": list(self.candidate_ids)}


@dataclass(frozen=True)
class EventWindow:
    event_id: str
    cluster_id: str
    start_ms: int
    end_ms: int
    evidence_frames_ms: tuple[int, ...] = field(default_factory=tuple)

    def validate(self, duration_ms: int) -> None:
        object.__setattr__(self, "event_id", _text(self.event_id, "event_id"))
        object.__setattr__(self, "cluster_id", _text(self.cluster_id, "cluster_id"))
        object.__setattr__(self, "start_ms", _timestamp(self.start_ms, "start_ms", duration_ms))
        object.__setattr__(self, "end_ms", _timestamp(self.end_ms, "end_ms", duration_ms))
        if self.end_ms <= self.start_ms:
            raise ValueError("event window end_ms must be greater than start_ms")
        object.__setattr__(self, "evidence_frames_ms", _evidence(self.evidence_frames_ms, self.start_ms, self.end_ms))

    def to_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "cluster_id": self.cluster_id, "start_ms": self.start_ms, "end_ms": self.end_ms, "evidence_frames_ms": list(self.evidence_frames_ms)}


@dataclass(frozen=True)
class EventCaption:
    event: EventWindow
    captions: Mapping[str, str]

    def __post_init__(self) -> None:
        if set(self.captions) != set(DIMENSIONS):
            raise ValueError(f"event caption dimensions must be exactly {DIMENSIONS}")
        object.__setattr__(self, "captions", {key: _text(self.captions[key], f"captions.{key}") for key in DIMENSIONS})

    def to_dict(self) -> dict[str, Any]:
        return {**self.event.to_dict(), "captions": dict(self.captions)}


class WindowCaptioner(Protocol):
    def __call__(self, sample: Any, window: ProcessingWindow) -> Mapping[str, Any]: ...


class CandidateGenerator(Protocol):
    def __call__(self, sample: Any, windows: Sequence[ProcessingWindow], captions: Sequence[WindowCaption]) -> Sequence[Mapping[str, Any]]: ...


class Clusterer(Protocol):
    def __call__(self, sample: Any, candidates: Sequence[EventCandidate]) -> Sequence[Mapping[str, Any]]: ...


class EventWindowProposer(Protocol):
    def __call__(self, sample: Any, cluster: EventCluster, candidates: Sequence[EventCandidate], windows: Sequence[ProcessingWindow]) -> Mapping[str, Any]: ...


class BoundaryReviewer(Protocol):
    def __call__(self, sample: Any, event: EventWindow) -> Mapping[str, Any]: ...


class EventCaptionRefiner(Protocol):
    def __call__(self, sample: Any, event: EventWindow, windows: Sequence[WindowCaption]) -> Mapping[str, Any]: ...


class GlobalCaptionMerger(Protocol):
    def __call__(self, sample: Any, events: Sequence[EventCaption]) -> Mapping[str, str]: ...


class QualityFilter(Protocol):
    def __call__(self, sample: Any, events: Sequence[EventCaption], global_captions: Mapping[str, str]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class StructuredAdapterBundle:
    caption_window: WindowCaptioner | None = None
    generate_event_candidates: CandidateGenerator | None = None
    cluster_event_candidates: Clusterer | None = None
    propose_event_window: EventWindowProposer | None = None
    review_event_boundary: BoundaryReviewer | None = None
    refine_event_caption: EventCaptionRefiner | None = None
    merge_global_caption: GlobalCaptionMerger | None = None
    quality_filter: QualityFilter | None = None

    def require(self, name: str) -> Any:
        adapter = getattr(self, name)
        if not callable(adapter):
            raise RuntimeError(f"videocap stage '{name}' has no adapter")
        return adapter
