# ─── Step 1: VLM Scene Segmentation ───

PROMPT_SEGMENT = """You are given {num_frames} frames (sampled at 1 fps) from a video that is {duration:.1f} seconds long.

YOUR TASK: Divide this video into major scene segments. A "scene" is a continuous sequence where the overall setting, location, and primary subject remain the same. Minor changes like camera pans, zooms, or a person shifting position do NOT constitute a new scene.

WHEN TO CUT:
- The location or environment changes (e.g., indoors → outdoors, office → street)
- The primary subject changes completely (e.g., speaker A → speaker B at a different location)
- There is a hard edit cut to a visually distinct shot

WHEN NOT TO CUT:
- Camera movement within the same location (pan, tilt, zoom, tracking)
- Minor activity changes by the same subject in the same location
- Lighting or color shifts without a location change
- Text overlays or captions appearing over the same scene

GUIDELINES:
- You MUST cover the ENTIRE video from 0.0 to {duration:.1f}. Do NOT stop early.

OUTPUT FORMAT: A JSON array, and nothing else:
[
  {{"scene_id": 1, "start": 0.0, "end": ..., "brief": "1-2 sentence description"}},
  ...
]

RULES:
- FULL COVERAGE: First segment starts at 0.0, last segment ends at {duration:.1f}. No gaps.
- CONTINUITY: Each segment's start must exactly equal the previous segment's end.
- TIMESTAMPS: Seconds with one decimal place.
- BRIEF: Direct description of what happens. Do NOT start with labels like "Short summary:", "Scene:", or "Brief:". Just write the content.
- Output ONLY the JSON array. No explanation, no markdown fences."""


# ─── Step 3: Dense Captioning with Prefix Context ───

PROMPT_CAPTION = """You are a professional video annotator. Describe this video clip in rich, specific detail.

{prefix_context}

You are given {num_frames} frames (1 fps) from a video clip [{start:.1f}s - {end:.1f}s].

Your description must weave together the following aspects into a single flowing narrative:
- What happens overall
- The setting and environment (location, lighting, indoor/outdoor, atmosphere)
- Who and what is present (people's appearance, clothing, distinguishing features; key objects)
- How the camera behaves (static, panning, zooming, handheld, tracking)
- The sequence of actions and state changes over time

CRITICAL FORMAT RULES:
- Write as continuous prose paragraphs. No section headings. No numbered lists. No bold labels. No markdown.
- Do NOT start with labels like "Summary:", "Description:", or "1.". Begin directly with the content.
- Minimum 100 words.

CONTENT RULES:
- Be specific and concrete: "a woman in a navy blue apron at a marble countertop" not "a person in a kitchen".
- If previous scenes are listed above, do NOT re-describe objects, settings, or characters already mentioned. Focus on what is NEW or CHANGED.
- Describe ONLY what is visible. Do NOT speculate, infer, or guess. Avoid phrases like "possibly", "might be", "suggests", "implies", "appears to be a [venue/event type]". 
- DO NOT fabricate or imagine any details that are not directly visible in the frames. If you cannot see it clearly, do not mention it.

Output ONLY the description text."""


PROMPT_CAPTION_RETRY = """Your previous description of this video clip was too short ({word_count} words).
Here is what you wrote:
---
{previous_attempt}
---

Please rewrite and EXPAND this description to at least 100 words. Add more visual details about the environment, character actions, object appearances, and camera work.

{prefix_context}

Output ONLY the expanded description text."""


# ─── Step 4: Action Annotation ───

PROMPT_ACTIONS = """You are given {num_frames} frames (1 fps) from a video clip [{start:.1f}s - {end:.1f}s], which is {duration:.1f} seconds long.

This clip belongs to the following scene:
"{parent_description}"

YOUR TASK: Annotate the visible actions happening in this clip. An action is a coherent motion or state performed by ONE specific visual subject.

CORE RULES:
- ONE SUBJECT PER ACTION. If two people are doing things at the same time, create two separate actions, one for each.
- Actions CAN overlap in time. Two concurrent actions by different subjects should have overlapping time windows.
- Actions do NOT need to cover the entire clip. Quiet moments with no notable action can be left unannotated.
- Only annotate VISUALLY SALIENT subjects: people or objects in the foreground, performing clear actions, or occupying significant screen space. Ignore background crowds and peripheral activity.
- DO NOT annotate camera movements (pans, zooms, cuts) as actions.
- Let each action's duration match the natural length of the motion. Do not split a continuous action artificially, and do not merge separate actions.

FORBIDDEN — NO SEMANTIC SPECULATION:
- Describe ONLY what is visually observable.
- Do NOT use verbs like "explains", "describes", "introduces", "states", "argues", "suggests", "implies".
- Use visual verbs: "gestures", "walks", "turns head", "smiles", "picks up", "sits down", "raises hand", "looks at".
- If a person is clearly speaking, say "speaks to the camera" or "speaks while gesturing" — do NOT summarize what they are saying.

SUBJECT FIELD:
- A short noun phrase identifying the subject (e.g., "woman in red shirt", "man with cowboy hat", "young boy in teal shirt", "buffet line").
- Be specific enough that the subject is uniquely identifiable within the clip.

OUTPUT FORMAT: A JSON array, and nothing else:
[
  {{"action_id": 1, "start": ..., "end": ..., "subject": "...", "description": "..."}},
  {{"action_id": 2, "start": ..., "end": ..., "subject": "...", "description": "..."}},
  ...
]

RULES:
- TIMESTAMPS: Use ABSOLUTE timestamps relative to the original video, with one decimal place. Must be within [{start:.1f}, {end:.1f}].
- DESCRIPTIONS: One sentence, present tense, specific and concrete.
- Output ONLY the JSON array. No explanation, no markdown fences."""


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
