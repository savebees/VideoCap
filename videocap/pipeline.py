"""The fixed VideoCap annotation pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from videocap.adapters.llm import LLM
from videocap.adapters.vlm import VLM, uniform_timestamps
from videocap.config import PipelineConfig
from videocap.io import atomic_write_json, atomic_write_jsonl
from videocap.schema import validate_record
from videocap.structured import ProcessingWindow, VideoSample


def processing_windows(
    duration_ms: int,
    config: PipelineConfig,
) -> tuple[ProcessingWindow, ...]:
    """Split a video into overlapping half-open processing windows."""

    step_ms = config.window_ms - config.overlap_ms
    windows: list[ProcessingWindow] = []
    start_ms = 0
    while start_ms < duration_ms:
        end_ms = min(start_ms + config.window_ms, duration_ms)
        evidence = uniform_timestamps(
            start_ms,
            end_ms - 1,
            config.evidence_frames,
        )
        windows.append(
            ProcessingWindow(
                window_id=f"W{len(windows) + 1:04d}",
                start_ms=start_ms,
                end_ms=end_ms,
                evidence_frames_ms=evidence,
            )
        )
        if end_ms == duration_ms:
            break
        start_ms += step_ms
    return tuple(windows)


class VideoCap:
    version = "0.2.0"

    def __init__(self, vlm: VLM, llm: LLM, config: PipelineConfig) -> None:
        self.vlm = vlm
        self.llm = llm
        self.config = config

    def process(self, sample: VideoSample, output_dir: Path) -> Mapping[str, Any]:
        """Annotate one video and retain each inspectable stage."""

        output_dir.mkdir(parents=True, exist_ok=False)
        windows = processing_windows(sample.duration_ms, self.config)
        atomic_write_jsonl(
            output_dir / "processing_windows.jsonl",
            (window.to_dict() for window in windows),
        )

        window_captions = tuple(self.vlm.caption_window(sample, window) for window in windows)
        atomic_write_jsonl(
            output_dir / "window_captions.jsonl",
            (caption.to_dict() for caption in window_captions),
        )

        proposals = self.llm.propose_events(windows, window_captions)
        if not proposals:
            raise ValueError("LLM returned no event proposals")
        atomic_write_jsonl(
            output_dir / "event_proposals.jsonl",
            (proposal.to_dict() for proposal in proposals),
        )

        events = tuple(
            self.vlm.review_event_boundary(sample, proposal, windows) for proposal in proposals
        )
        if any(event.end_ms >= sample.duration_ms for event in events):
            raise ValueError("event boundary falls outside the video")
        atomic_write_jsonl(
            output_dir / "event_boundaries.jsonl",
            (event.to_dict() for event in events),
        )

        event_captions = tuple(self.vlm.caption_event(sample, event) for event in events)
        atomic_write_jsonl(
            output_dir / "event_captions.jsonl",
            (caption.to_dict() for caption in event_captions),
        )

        global_captions = self.llm.merge_global_caption(
            windows,
            window_captions,
            event_captions,
        )
        atomic_write_json(
            output_dir / "global_caption.json",
            {"captions": dict(global_captions)},
        )

        record = {
            "schema_version": "videocap/v0.2",
            "video_id": sample.video_id,
            "duration_ms": sample.duration_ms,
            "captions": dict(global_captions),
            "events": [
                {
                    **caption.event.to_dict(),
                    "caption": caption.caption,
                }
                for caption in event_captions
            ],
        }
        validate_record(record)
        return record


__all__ = ["VideoCap", "processing_windows"]
