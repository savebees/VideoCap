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

FORBIDDEN — NO SEMANTIC SPECULATION
- Describe ONLY what is visually observable.
- Do NOT use verbs like "explains", "describes", "introduces", "states", "argues", "suggests", "implies".
- Use visual verbs: "gestures", "walks", "turns head", "smiles", "picks up", "sits down", "raises hand", "looks at".
- If a person is clearly speaking, say "speaks to the camera" or "speaks while gesturing". Do NOT summarize what they are saying.

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


# ─── Step 5: Per-frame Object Detection (VLM with vision) ───

PROMPT_DETECT_OBJECTS = """You are given a single frame from a video clip.

This frame belongs to the following scene:
"{parent_description}"
{subjects_hint}

TASK: Detect the KEY objects and people in this frame, and provide a tight bounding box for each.

WHAT COUNTS AS KEY (in priority order):
1. People actively performing the action described in the scene above.
2. Objects those people directly interact with (held, touched, used).
3. Large central objects that define the setting, such as a podium, table, or projector screen.

EXCLUDE:
- Background or edge people not central to the action, including partially cropped or distant figures.
- Decorative or structural surfaces filling the frame, such as tablecloths, walls, floors, ceilings, or sky.
- Body parts in isolation, text overlays, logos, and watermarks.
- Objects occupying less than about 5% of the frame area, unless they are the focus of the action.

ONE OBJECT = ONE BOX = ONE LABEL.
Do NOT output overlapping boxes for the same physical object under different names. If a milk carton could be called either "milk carton" or "beverage container", pick the more specific one and output it ONCE.

LABELS:
- Short noun phrase, 1 to 6 words, concrete (NOT "presentation" or "discussion").
- When the scene description names an object or person, stay close to that wording. Keep the key identifying attributes such as clothing color, role, or position, but you may shorten a long phrase. For example, "a woman with shoulder-length blonde hair wearing a striped shirt" becomes "woman in striped shirt".
- Distinguish multiple instances by visible attributes, for example "man in red shirt" versus "man in blue shirt".

BOUNDING BOX:
- Format [x1, y1, x2, y2], normalized to a 0 to 1000 scale (top-left origin).
- Tightly enclose the object. For a person, span head to feet, or to the crop or occlusion edge.

CONFIDENCE: float in [0, 1]. Be honest, and use a value below 0.5 when guessing.

OUTPUT: a JSON array, nothing else. No markdown fences, no explanation.
[
  {{"label": "...", "bbox_2d": [x1, y1, x2, y2], "confidence": 0.92}}
]

Be selective. If you find yourself listing many objects, you are likely including non-key items. Recheck and keep only what is essential to the action.
If nothing key is present (transition or blank frame), output []."""
