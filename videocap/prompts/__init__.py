"""Versioned prompt templates used by VideoCap."""

from videocap.prompts.event_caption import (
    COARSE_EVENT_BOUNDARY_PROMPT,
    EVENT_CAPTION_PROMPT,
    EVENT_PROPOSAL_PROMPT,
    FINE_EVENT_BOUNDARY_PROMPT,
    build_coarse_event_boundary_prompt,
    build_event_proposal_prompt,
    build_fine_event_boundary_prompt,
)
from videocap.prompts.global_caption import GLOBAL_CAPTION_PROMPT, build_global_caption_prompt
from videocap.prompts.window_caption import build_window_caption_prompt

__all__ = [
    "COARSE_EVENT_BOUNDARY_PROMPT",
    "EVENT_CAPTION_PROMPT",
    "EVENT_PROPOSAL_PROMPT",
    "FINE_EVENT_BOUNDARY_PROMPT",
    "GLOBAL_CAPTION_PROMPT",
    "build_coarse_event_boundary_prompt",
    "build_event_proposal_prompt",
    "build_fine_event_boundary_prompt",
    "build_global_caption_prompt",
    "build_window_caption_prompt",
]
