"""Window-caption prompt inspired by AuroraCap's VDC taxonomy.

Source: https://github.com/wenhaochai/aurora
"""

from __future__ import annotations

WINDOW_CAPTION_PROMPT = """You are given an ordered sequence of frames from one continuous
video segment. Describe only what the frames support; do not infer content outside this
segment.

Write five complementary captions:

short: Summarize the main visible content in one concise sentence.
main_object: Describe the main subject, its visible attributes, actions, position, and movement.
background: Describe the setting, relevant objects, lighting, weather, and visible changes.
camera: Describe framing, viewpoint, shot changes, pans, zooms, and camera movement.
detailed: Give a chronological account of the visible segment in at least three sentences.

Return only these fields in this exact order:
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


def build_window_caption_prompt() -> str:
    return WINDOW_CAPTION_PROMPT


__all__ = [
    "WINDOW_CAPTION_PROMPT",
    "build_window_caption_prompt",
]
