"""Prompt for synthesizing final video-level five-dimensional captions."""

from __future__ import annotations

from collections.abc import Sequence

from videocap.structured import EventCaption, ProcessingWindow, WindowCaption

GLOBAL_CAPTION_PROMPT = """Synthesize the final video-level captions from the temporally grounded
events and the ordered processing-window captions below.

The event document provides the most reliable account of what happens and in what order. The
processing-window document provides complementary observations about the main subjects,
backgrounds, camera behavior, and fine visual details. Processing windows overlap, so combine
repeated observations into one consistent account.

Produce five complementary captions for the entire video:

short: Summarize the central content of the complete video in one concise sentence.
main_object: Describe the principal subject or group across the video, including visually supported
attributes, actions, positions, movements, and meaningful changes over time.
background: Describe the environments across the video, including relevant objects, settings,
weather, time, and visible transitions between environments.
camera: Describe the video's camera movement, framing, viewpoint, shot changes, and other visible
cinematographic behavior.
detailed: Write a coherent chronological account of the video's beginning, development, and
ending. Use the grounded events as its narrative structure and enrich them with consistent details
from the processing windows.

Keep the five fields mutually consistent and grounded in the supplied documents.

<EVENTS>
{event_document}
</EVENTS>

<PROCESSING_WINDOWS>
{window_document}
</PROCESSING_WINDOWS>

Return the five captions in this exact order and format:
[short]
<caption>
[main_object]
<caption>
[background]
<caption>
[camera]
<caption>
[detailed]
<caption>
"""


def _timestamp(timestamp_ms: int) -> str:
    hours, remainder = divmod(timestamp_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _event_block(event: EventCaption) -> str:
    return (
        f"[EVENT {event.event.event_id} | {_timestamp(event.event.start_ms)} - "
        f"{_timestamp(event.event.end_ms)}]\n"
        f"Caption: {_one_line(event.caption)}"
    )


def _window_block(window: ProcessingWindow, caption: WindowCaption) -> str:
    if caption.window_id != window.window_id:
        raise ValueError(
            f"window caption {caption.window_id!r} does not match {window.window_id!r}"
        )
    return (
        f"[WINDOW {window.window_id} | {_timestamp(window.start_ms)} - "
        f"{_timestamp(window.end_ms)}]\n"
        f"Short: {_one_line(caption.captions['short'])}\n"
        f"Main object: {_one_line(caption.captions['main_object'])}\n"
        f"Background: {_one_line(caption.captions['background'])}\n"
        f"Camera: {_one_line(caption.captions['camera'])}\n"
        f"Detailed: {_one_line(caption.captions['detailed'])}"
    )


def build_global_caption_prompt(
    *,
    windows: Sequence[ProcessingWindow],
    window_captions: Sequence[WindowCaption],
    events: Sequence[EventCaption],
) -> str:
    if not windows:
        raise ValueError("global caption synthesis requires processing windows")
    if len(windows) != len(window_captions):
        raise ValueError("processing windows and window captions must have equal length")
    if not events:
        raise ValueError("global caption synthesis requires grounded events")
    event_document = "\n\n".join(_event_block(event) for event in events)
    window_document = "\n\n".join(
        _window_block(window, caption)
        for window, caption in zip(windows, window_captions, strict=True)
    )
    return GLOBAL_CAPTION_PROMPT.format(
        event_document=event_document,
        window_document=window_document,
    )


__all__ = ["GLOBAL_CAPTION_PROMPT", "build_global_caption_prompt"]
