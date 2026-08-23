"""Data structures shared by the VideoCap stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DIMENSIONS = ("short", "main_object", "background", "camera", "detailed")


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _ordered_timestamps(
    values: Sequence[int],
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> tuple[int, ...]:
    timestamps = tuple(values)
    if any(
        isinstance(timestamp, bool) or not isinstance(timestamp, int) for timestamp in timestamps
    ):
        raise ValueError("timestamps must be integers")
    if tuple(sorted(set(timestamps))) != timestamps:
        raise ValueError("timestamps must be sorted and unique")
    if start_ms is not None and any(timestamp < start_ms for timestamp in timestamps):
        raise ValueError("timestamp is outside its interval")
    if end_ms is not None and any(timestamp > end_ms for timestamp in timestamps):
        raise ValueError("timestamp is outside its interval")
    return timestamps


@dataclass(frozen=True)
class VideoSample:
    video_id: str
    video_path: Path
    duration_ms: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _non_empty(self.video_id, "video_id"))
        if Path(self.video_id).name != self.video_id or self.video_id in {".", ".."}:
            raise ValueError("video_id must be safe to use as a directory name")
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, int):
            raise ValueError("duration_ms must be an integer")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")


@dataclass(frozen=True)
class ProcessingWindow:
    window_id: str
    start_ms: int
    end_ms: int
    evidence_frames_ms: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.end_ms <= self.start_ms:
            raise ValueError("processing window must have positive duration")
        object.__setattr__(self, "window_id", _non_empty(self.window_id, "window_id"))
        object.__setattr__(
            self,
            "evidence_frames_ms",
            _ordered_timestamps(
                self.evidence_frames_ms,
                start_ms=self.start_ms,
                end_ms=self.end_ms,
            ),
        )
        if not self.evidence_frames_ms:
            raise ValueError("processing window must contain evidence frames")

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "evidence_frames_ms": list(self.evidence_frames_ms),
        }


@dataclass(frozen=True)
class WindowCaption:
    window_id: str
    captions: Mapping[str, str]
    evidence_frames_ms: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_id", _non_empty(self.window_id, "window_id"))
        if tuple(self.captions) != DIMENSIONS:
            raise ValueError(f"captions must contain {DIMENSIONS} in order")
        object.__setattr__(
            self,
            "captions",
            {name: _non_empty(self.captions[name], name) for name in DIMENSIONS},
        )
        object.__setattr__(
            self,
            "evidence_frames_ms",
            _ordered_timestamps(self.evidence_frames_ms, start_ms=0),
        )
        if not self.evidence_frames_ms:
            raise ValueError("window caption must retain its evidence frames")

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "captions": dict(self.captions),
            "evidence_frames_ms": list(self.evidence_frames_ms),
        }


@dataclass(frozen=True)
class EventProposal:
    event_id: str
    source_window_ids: tuple[str, ...]
    short_caption: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if not self.source_window_ids:
            raise ValueError("event proposal must reference a processing window")
        if self.end_ms <= self.start_ms:
            raise ValueError("event proposal must have positive duration")
        object.__setattr__(self, "event_id", _non_empty(self.event_id, "event_id"))
        source_window_ids = tuple(
            _non_empty(window_id, "source_window_id") for window_id in self.source_window_ids
        )
        if len(set(source_window_ids)) != len(source_window_ids):
            raise ValueError("event proposal must not repeat a processing window")
        object.__setattr__(self, "source_window_ids", source_window_ids)
        object.__setattr__(self, "short_caption", _non_empty(self.short_caption, "short_caption"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_window_ids": list(self.source_window_ids),
            "short_caption": self.short_caption,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }


@dataclass(frozen=True)
class EventWindow:
    event_id: str
    start_ms: int
    end_ms: int
    evidence_frames_ms: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.end_ms <= self.start_ms:
            raise ValueError("event window must have positive duration")
        object.__setattr__(self, "event_id", _non_empty(self.event_id, "event_id"))
        object.__setattr__(
            self,
            "evidence_frames_ms",
            _ordered_timestamps(
                self.evidence_frames_ms,
                start_ms=self.start_ms,
                end_ms=self.end_ms,
            ),
        )
        if self.evidence_frames_ms != (self.start_ms, self.end_ms):
            raise ValueError("event evidence must contain its exact start and end")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "evidence_frames_ms": list(self.evidence_frames_ms),
        }


@dataclass(frozen=True)
class EventCaption:
    event: EventWindow
    caption: str
    caption_frames_ms: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "caption", _non_empty(self.caption, "caption"))
        object.__setattr__(
            self,
            "caption_frames_ms",
            _ordered_timestamps(
                self.caption_frames_ms,
                start_ms=self.event.start_ms,
                end_ms=self.event.end_ms,
            ),
        )
        if not self.caption_frames_ms or (
            self.caption_frames_ms[0] != self.event.start_ms
            or self.caption_frames_ms[-1] != self.event.end_ms
        ):
            raise ValueError("event caption frames must include its exact start and end")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.event.to_dict(),
            "caption": self.caption,
            "caption_frames_ms": list(self.caption_frames_ms),
        }


__all__ = [
    "DIMENSIONS",
    "EventCaption",
    "EventProposal",
    "EventWindow",
    "ProcessingWindow",
    "VideoSample",
    "WindowCaption",
]
