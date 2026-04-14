# ─── Step 1: VLM Scene Segmentation ───

PROMPT_SEGMENT = """You are given {num_frames} frames (sampled at 1 fps) from a video that is {duration:.1f} seconds long.

YOUR TASK: Divide this video into coherent scene segments. A scene is a continuous sequence of frames sharing the same visual setting, activity, or subject.

GUIDELINES:
- Target ~5-15 seconds per segment. Shorter segments are acceptable for rapid scene changes.
- Each segment must have clear visual coherence — same location, same activity, same camera angle.
- Place boundaries at visual transitions: location changes, camera cuts, subject changes, or major activity shifts.

OUTPUT FORMAT: A JSON array, and nothing else:
[
  {{"scene_id": 1, "start": 0.0, "end": ..., "brief": "1-2 sentence summary"}},
  {{"scene_id": 2, "start": ..., "end": ..., "brief": "1-2 sentence summary"}},
  ...
]

RULES:
- FULL COVERAGE: First segment starts at 0.0, last segment ends at {duration:.1f}. No gaps allowed.
- CONTINUITY: Each segment's start must exactly equal the previous segment's end.
- TIMESTAMPS: Use seconds with one decimal place.
- BRIEF: 1-2 sentence summary of what happens in the segment.
- Output ONLY the JSON array. No explanation, no markdown fences."""


# ─── Step 3: Dense Captioning with Prefix Context ───

PROMPT_CAPTION = """You are a professional video annotator. Describe this video clip in detail.

{prefix_context}

You are given {num_frames} frames (1 fps) from a video clip [{start:.1f}s - {end:.1f}s].

Your description MUST cover ALL of the following five dimensions:
1. SHORT SUMMARY: A 1-2 sentence overview of what happens
2. BACKGROUND & ENVIRONMENT: The setting, lighting, weather, indoor/outdoor details
3. MAIN OBJECTS & CHARACTERS: Who/what is present, their appearance, clothing, distinguishing attributes
4. CAMERA MOVEMENT: Static, panning, zooming, handheld, tracking, etc.
5. DETAILED TEMPORAL EVENTS: Every action and state change in chronological order

Rules:
- Your description must be at least 100 words.
- Be specific: "a woman in a navy blue apron standing at a marble countertop" not "a person in a kitchen".
- If previous scenes are listed above, DO NOT re-describe objects, settings, or characters already mentioned there. Focus on what is NEW or CHANGED.
- Output ONLY the description text. Do not wrap it in JSON or add any other formatting."""


PROMPT_CAPTION_RETRY = """Your previous description of this video clip was too short ({word_count} words).
Here is what you wrote:
---
{previous_attempt}
---

Please rewrite and EXPAND this description to at least 100 words. Add more visual details about the environment, character actions, object appearances, and camera work.

{prefix_context}

Output ONLY the expanded description text."""


# ─── Step 4: Atomic Event Segmentation ───

PROMPT_ATOMIC_EVENTS = """You are given {num_frames} frames (1 fps) from a video clip [{start:.1f}s - {end:.1f}s], which is {duration:.1f} seconds long.

This clip belongs to the following scene:
"{parent_description}"

YOUR TASK: Divide this clip into atomic events — short sub-segments (typically 2-5 seconds each) where each segment captures exactly ONE action, motion, or visual state.

Examples of atomic events:
- "A woman picks up a knife from the counter."
- "The camera pans left across a mountain range."
- "A man sits down on a bench."
- "Text overlay appears showing the university logo."
- "The person waves at the camera while smiling."

RULES:
- FULL COVERAGE: Events must cover [{start:.1f}s - {end:.1f}s] with no gaps. First event starts at {start:.1f}s, last event ends at {end:.1f}s.
- CONTINUITY: Each event's start must exactly equal the previous event's end.
- DURATION: Each event should be 1-5 seconds. If a single action takes longer than 5 seconds (e.g., a person talking continuously), it is acceptable to have one longer event rather than artificially splitting it.
- ONE ACTION PER EVENT: Each event should describe exactly one atomic action or state. Not "she picks up a knife and starts chopping" — that is two events.
- DESCRIPTIONS: One sentence per event. Be specific and concrete. Use present tense.
- TIMESTAMPS: Use ABSOLUTE timestamps relative to the original video.

Output a JSON array and nothing else:
[
  {{"event_id": 1, "start": {start:.1f}, "end": ..., "description": "..."}},
  ...
]"""


# ─── Step 6: Object Prompt Extraction (text-only VLM call) ───

PROMPT_EXTRACT_OBJECTS = """Extract all visually distinct objects and characters mentioned in the following scene description. Return them as short noun phrases suitable for an object detection model.

Scene description:
"{description}"

Rules:
- Include people with their distinguishing attributes (e.g., "woman in navy blazer", "man with glasses").
- Include significant objects (e.g., "wooden podium", "laptop computer", "whiteboard").
- Do NOT include abstract concepts, actions, or scene-level descriptions.
- Do NOT include camera movements or lighting descriptions.
- Each phrase should be 1-5 words, specific enough for visual detection.

Output a JSON array of strings, and nothing else:
["object phrase 1", "object phrase 2", ...]"""
