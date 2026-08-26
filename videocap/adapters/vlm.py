"""Visual stages backed by an OpenAI-compatible multimodal model."""

from __future__ import annotations

import base64
import math
import re
import subprocess
from collections.abc import Mapping, Sequence

from videocap.adapters import ChatClient
from videocap.adapters.llm import parse_dimensions
from videocap.config import VLMConfig
from videocap.prompts import (
    EVENT_CAPTION_PROMPT,
    build_coarse_event_boundary_prompt,
    build_fine_event_boundary_prompt,
    build_window_caption_prompt,
)
from videocap.structured import (
    EventCaption,
    EventProposal,
    EventWindow,
    ProcessingWindow,
    VideoSample,
    WindowCaption,
)


def uniform_timestamps(start_ms: int, end_ms: int, count: int) -> tuple[int, ...]:
    """Return ``count`` timestamps including both interval endpoints."""

    if end_ms <= start_ms or count < 2:
        raise ValueError("timestamp sampling requires a positive interval and at least two frames")
    return tuple(
        start_ms + round(index * (end_ms - start_ms) / (count - 1)) for index in range(count)
    )


def coarse_boundary_frames(
    event: EventProposal,
    windows: Sequence[ProcessingWindow],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    by_id = {window.window_id: window for window in windows}
    selected = tuple(by_id[window_id] for window_id in event.source_window_ids)

    start_ms, end_ms = selected[0].start_ms, selected[-1].end_ms
    frames = tuple(range(start_ms, end_ms, 1_000))
    if len(frames) < 2:
        frames = uniform_timestamps(start_ms, end_ms - 1, 2)
    elif len(frames) > 24:
        frames = uniform_timestamps(start_ms, end_ms - 1, 24)
    elif frames[-1] != end_ms - 1:
        frames = (*frames[:-1], end_ms - 1) if len(frames) == 24 else (*frames, end_ms - 1)
    return frames, frames, frames


def fine_boundary_frames(
    event: EventProposal,
    coarse_start_ms: int,
    coarse_end_ms: int,
    coarse_frames: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    def neighborhood(center_ms: int) -> tuple[int, ...]:
        index = coarse_frames.index(center_ms)
        adjacent_gaps = []
        if index:
            adjacent_gaps.append(center_ms - coarse_frames[index - 1])
        if index + 1 < len(coarse_frames):
            adjacent_gaps.append(coarse_frames[index + 1] - center_ms)
        radius_ms = max(2_000, *adjacent_gaps)
        start_ms = max(event.start_ms, center_ms - radius_ms)
        end_ms = min(event.end_ms - 1, center_ms + radius_ms)
        if end_ms <= start_ms:
            return (start_ms,)
        count = min(12, end_ms - start_ms + 1)
        return uniform_timestamps(start_ms, end_ms, count)

    start_frames = neighborhood(coarse_start_ms)
    end_frames = neighborhood(coarse_end_ms)
    return tuple(sorted(set(start_frames + end_frames))), start_frames, end_frames


def parse_boundary(
    text: str,
    *,
    start_frames: Sequence[int],
    end_frames: Sequence[int],
    coarse: bool,
) -> tuple[int, int]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if coarse:
        if lines == ["STATUS: INCONSISTENT", "START_MS: NONE", "END_MS: NONE"]:
            raise ValueError("visual evidence does not support a coherent event")
        if len(lines) != 3 or lines[0] != "STATUS: OK":
            raise ValueError("coarse boundary response has invalid format")
        lines = lines[1:]
    if len(lines) != 2:
        raise ValueError("boundary response must contain START_MS and END_MS")
    start_match = re.fullmatch(r"START_MS: (\d+)", lines[0])
    end_match = re.fullmatch(r"END_MS: (\d+)", lines[1])
    if not start_match or not end_match:
        raise ValueError("boundary response must contain integer timestamps")
    start_ms, end_ms = int(start_match.group(1)), int(end_match.group(1))
    if start_ms not in start_frames or end_ms not in end_frames:
        raise ValueError(
            "boundary timestamps must be selected from the supplied frames; "
            f"got start={start_ms}, end={end_ms}, "
            f"start_choices={tuple(start_frames)}, end_choices={tuple(end_frames)}"
        )
    if end_ms <= start_ms:
        raise ValueError("event end must be greater than start")
    return start_ms, end_ms


class VLM:
    def __init__(self, config: VLMConfig) -> None:
        self.chat = ChatClient(config)
        self.frame_height = config.frame_height
        self._frames: dict[tuple[str, int], str] = {}

    def _image(self, sample: VideoSample, timestamp_ms: int) -> str:
        key = (str(sample.video_path), timestamp_ms)
        if key not in self._frames:
            # Container duration can extend a few milliseconds past the final decodable frame.
            seek_ms = min(timestamp_ms, max(0, sample.duration_ms - 100))
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{seek_ms / 1000:.3f}",
                "-i",
                str(sample.video_path),
                "-frames:v",
                "1",
                "-vf",
                f"scale=-2:{self.frame_height}",
                "-f",
                "image2",
                "-c:v",
                "mjpeg",
                "-q:v",
                "6",
                "pipe:1",
            ]
            frame = subprocess.run(command, capture_output=True, check=True).stdout
            self._frames[key] = "data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii")
        return self._frames[key]

    def _complete(
        self,
        sample: VideoSample,
        prompt: str,
        timestamps: Sequence[int],
        roles: Mapping[int, str] | None = None,
        *,
        max_tokens: int = 1500,
    ) -> str:
        content: list[Mapping[str, object]] = [{"type": "text", "text": prompt}]
        for timestamp in timestamps:
            label = roles[timestamp] if roles else "EVIDENCE"
            content.extend(
                (
                    {"type": "text", "text": f"[FRAME role={label} timestamp_ms={timestamp}]"},
                    {"type": "image_url", "image_url": {"url": self._image(sample, timestamp)}},
                )
            )
        return self.chat.complete(
            [{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )

    def caption_window(self, sample: VideoSample, window: ProcessingWindow) -> WindowCaption:
        captions = parse_dimensions(
            self._complete(sample, build_window_caption_prompt(), window.evidence_frames_ms)
        )
        return WindowCaption(window.window_id, captions, window.evidence_frames_ms)

    def review_event_boundary(
        self,
        sample: VideoSample,
        event: EventProposal,
        windows: Sequence[ProcessingWindow],
    ) -> EventWindow:
        by_id = {window.window_id: window for window in windows}
        selected = tuple(by_id[window_id] for window_id in event.source_window_ids)
        coarse_frames, coarse_start, coarse_end = coarse_boundary_frames(event, windows)
        start_set, end_set = set(coarse_start), set(coarse_end)
        coarse_roles = {
            timestamp: (
                "START_END_BOUNDARY"
                if timestamp in start_set and timestamp in end_set
                else "START_BOUNDARY"
                if timestamp in start_set
                else "END_BOUNDARY"
                if timestamp in end_set
                else "CONTINUITY_ANCHOR"
            )
            for timestamp in coarse_frames
        }
        coarse = parse_boundary(
            self._complete(
                sample,
                build_coarse_event_boundary_prompt(
                    event_caption=event.short_caption,
                    windows=selected,
                ),
                coarse_frames,
                coarse_roles,
            ),
            start_frames=coarse_start,
            end_frames=coarse_end,
            coarse=True,
        )

        fine_frames, fine_start, fine_end = fine_boundary_frames(
            event,
            coarse[0],
            coarse[1],
            coarse_frames,
        )
        fine_roles = {
            timestamp: "+".join(
                role
                for role, members in (
                    ("START_NEIGHBORHOOD", fine_start),
                    ("END_NEIGHBORHOOD", fine_end),
                )
                if timestamp in members
            )
            for timestamp in fine_frames
        }
        start_ms, end_ms = parse_boundary(
            self._complete(
                sample,
                build_fine_event_boundary_prompt(
                    event_caption=event.short_caption,
                    coarse_start_ms=coarse[0],
                    coarse_end_ms=coarse[1],
                ),
                fine_frames,
                fine_roles,
            ),
            start_frames=fine_start,
            end_frames=fine_end,
            coarse=False,
        )
        return EventWindow(event.event_id, start_ms, end_ms, (start_ms, end_ms))

    def caption_event(self, sample: VideoSample, event: EventWindow) -> EventCaption:
        duration_ms = event.end_ms - event.start_ms
        count = min(24, max(8, math.ceil(duration_ms / 1_000)))
        timestamps = uniform_timestamps(event.start_ms, event.end_ms, count)
        caption = self._complete(sample, EVENT_CAPTION_PROMPT, timestamps, max_tokens=1200)
        return EventCaption(event, caption, timestamps)


__all__ = [
    "VLM",
    "coarse_boundary_frames",
    "fine_boundary_frames",
    "parse_boundary",
    "uniform_timestamps",
]
