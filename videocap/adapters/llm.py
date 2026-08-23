"""Text-only stages backed by an OpenAI-compatible chat model."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from videocap.adapters import ChatClient
from videocap.config import ModelConfig
from videocap.prompts import build_event_proposal_prompt, build_global_caption_prompt
from videocap.structured import (
    DIMENSIONS,
    EventCaption,
    EventProposal,
    ProcessingWindow,
    WindowCaption,
)


def parse_dimensions(text: str) -> dict[str, str]:
    """Parse the five labelled caption fields used throughout VideoCap."""

    headings = list(re.finditer(r"^\[([^\]]+)]\s*$", text, re.MULTILINE))
    if tuple(match.group(1) for match in headings) != DIMENSIONS:
        raise ValueError(f"caption response must contain {DIMENSIONS} in order")
    captions: dict[str, str] = {}
    for index, (dimension, heading) in enumerate(zip(DIMENSIONS, headings, strict=True)):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        caption = text[heading.end() : end].strip()
        if not caption:
            raise ValueError(f"caption response has an empty {dimension} field")
        captions[dimension] = caption
    return captions


def parse_event_proposals(
    text: str,
    windows: Sequence[ProcessingWindow],
) -> tuple[EventProposal, ...]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or len(lines) % 4:
        raise ValueError("event response must contain complete four-line blocks")

    positions = {window.window_id: index for index, window in enumerate(windows)}
    proposals: list[EventProposal] = []
    for offset in range(0, len(lines), 4):
        marker, window_line, caption_line, closing = lines[offset : offset + 4]
        if marker != "EVENT" or closing != "END_EVENT":
            raise ValueError("event response has invalid block markers")
        if not window_line.startswith("WINDOWS: ") or not caption_line.startswith("CAPTION: "):
            raise ValueError("event response must include WINDOWS and CAPTION")

        window_ids = tuple(
            item.strip() for item in window_line.removeprefix("WINDOWS: ").split(",")
        )
        try:
            selected_positions = [positions[window_id] for window_id in window_ids]
        except KeyError as exc:
            raise ValueError(f"event references unknown window {exc.args[0]}") from exc
        if selected_positions != list(
            range(selected_positions[0], selected_positions[0] + len(selected_positions))
        ):
            raise ValueError("event windows must be consecutive and ordered")

        selected = [windows[index] for index in selected_positions]
        proposals.append(
            EventProposal(
                event_id=f"event_{len(proposals):04d}",
                source_window_ids=window_ids,
                short_caption=caption_line.removeprefix("CAPTION: "),
                start_ms=selected[0].start_ms,
                end_ms=selected[-1].end_ms,
            )
        )
    return tuple(proposals)


class LLM:
    def __init__(self, config: ModelConfig) -> None:
        self.chat = ChatClient(config)

    def propose_events(
        self,
        windows: Sequence[ProcessingWindow],
        captions: Sequence[WindowCaption],
    ) -> tuple[EventProposal, ...]:
        prompt = build_event_proposal_prompt(windows=windows, captions=captions)
        text = self.chat.complete([{"role": "user", "content": prompt}], max_tokens=1800)
        return parse_event_proposals(text, windows)

    def merge_global_caption(
        self,
        windows: Sequence[ProcessingWindow],
        window_captions: Sequence[WindowCaption],
        events: Sequence[EventCaption],
    ) -> Mapping[str, str]:
        prompt = build_global_caption_prompt(
            windows=windows,
            window_captions=window_captions,
            events=events,
        )
        return parse_dimensions(
            self.chat.complete([{"role": "user", "content": prompt}], max_tokens=2500)
        )


__all__ = ["LLM", "parse_dimensions", "parse_event_proposals"]
