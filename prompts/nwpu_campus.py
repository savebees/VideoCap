"""NWPU Campus dataset prompts.

NWPU Campus is fixed-camera outdoor surveillance footage from 43 campus
locations (pedestrians, cyclists, vehicles). Within a single video the camera is
fixed: no pans, zooms, cuts, or scene changes. These prompts tune the generic
prompts for that setting. Anything not defined here falls back to ``prompts.general``.

A template only declares the placeholders it uses; the detection code fills
subjects_hint solely when present, so this surveillance prompt omits it:
- PROMPT_CAPTION_SURVEILLANCE:       {prefix_context} {num_frames} {start} {end}
- PROMPT_CAPTION_RETRY_SURVEILLANCE: {word_count} {previous_attempt} {min_words} {prefix_context}
- PROMPT_ACTIONS_SURVEILLANCE:       {num_frames} {start} {end} {duration}
- PROMPT_DETECT_OBJECTS:             {parent_description}
"""

# ─── Dense Captioning ───

PROMPT_CAPTION_SURVEILLANCE = """You are annotating footage from a FIXED outdoor surveillance camera on a university campus.

{prefix_context}

You are given {num_frames} frames (1 fps) from this fixed campus camera, clip [{start:.1f}s - {end:.1f}s].

CONTEXT:
- The camera is fixed: no pans, zooms, cuts, or shot changes. The location does NOT change between clips.
- This is an outdoor campus scene (e.g. a walkway, road, plaza, building entrance, parking area, lawn). Typical subjects are pedestrians, cyclists, and vehicles.
- Activity is often sparse, and many clips have little or no movement. That is normal. If little is happening, a short description is correct and expected. Do NOT invent activity to fill space.

Cover these aspects in continuous prose:
- The setting: type of campus area, paths/roads, surrounding buildings or structures, vegetation, lighting.
- Any people, cyclists, or vehicles visible: appearance, clothing, what they are doing, the direction they move, and where they enter or exit the frame.
- Notable static elements that ground the location (signage, doors, benches, lamp posts, parked bicycles or vehicles, road markings).

CRITICAL FORMAT RULES:
- Write as continuous prose. No section headings. No bullet lists. No markdown.
- Do NOT start with labels like "Summary:" or "Description:". Begin directly with the content.

CONTENT RULES:
- Be specific and concrete: "a man in a black jacket cycles from left to right along the walkway" not "a person moves".
- Describe only what you actually observe. If you do not see activity in part of the clip, simply do not describe activity there; you do not need to state that the scene is empty or static.
- If previous clips above already described the static setting, do NOT re-describe the setting. Still describe any people, cyclists, or vehicles present in THIS clip, even if similar subjects appeared before.
- Describe ONLY what is visible. Do NOT speculate about intent, identity, or whether behavior is normal/abnormal. Avoid "appears to", "seems to", "might be".
- DO NOT fabricate. If you cannot see it clearly, do not mention it.

Output ONLY the description text."""


# Campus surveillance retry: the clip may genuinely be near-empty, so this MUST NOT
# push the model to invent activity to hit the word count — that is the failure mode
# of reusing the generic expand-to-N-words retry on sparse fixed-camera footage. It
# asks for a lower bar ({min_words}) and routes any expansion into the static setting.
PROMPT_CAPTION_RETRY_SURVEILLANCE = """Your previous description of this campus surveillance clip was a bit short ({word_count} words).
Here is what you wrote:
---
{previous_attempt}
---

Please rewrite to at least {min_words} words. This is a FIXED outdoor campus camera and the clip may genuinely have little or no activity (an empty walkway, road, or plaza) — that is normal and acceptable.

- Do NOT invent pedestrians, cyclists, vehicles, motion, or events that are not visible just to reach the word count. Fabrication is worse than a short description.
- If activity is sparse, reach the length by describing the STATIC setting more fully instead: the type of campus area and its layout, paths/roads, surrounding buildings, vegetation, signage, parked bicycles or vehicles, lighting, and time-of-day cues.
- If the clip is truly empty, a brief honest statement that the area stays empty with no pedestrians or vehicles is a fully acceptable answer even if it stays short.

{prefix_context}

Output ONLY the description text."""


# ─── Action Annotation ───

PROMPT_ACTIONS_SURVEILLANCE = """You are given {num_frames} frames (1 fps) from a FIXED outdoor campus surveillance camera, clip [{start:.1f}s - {end:.1f}s], duration {duration:.1f}s.

YOUR TASK: Annotate visible actions by people, cyclists, and vehicles in this campus clip. An action is a coherent motion or state performed by ONE specific visual subject.

CORE RULES:
- ONE SUBJECT PER ACTION. Two people moving at the same time → two separate actions.
- Actions CAN overlap in time, and gaps are fine. Not every second needs an action.
- Subjects frequently enter and leave the frame mid-clip. Clip the action's start/end to the window when the subject is actually visible.
- The camera is FIXED. Do NOT annotate camera movement (there is none). Do NOT treat parallax as an action.
- If the clip is empty (no people, cyclists, or vehicles doing anything), output an empty array [].
- Only annotate VISUALLY SALIENT subjects. Ignore tiny distant figures and irrelevant background (e.g., foliage moving in wind, far-off pedestrians barely visible).

REPORT FINE PHYSICAL INTERACTIONS
- Report close-range interactions between a person and an object or another
  person as their own separate actions, even when brief or subtle: reaching
  into a bag, pocket, or container; taking or removing an object; handing or
  passing an object; picking an item up from the ground; pushing, pulling, or
  grabbing another person.
- Describe these by what the hands and bodies actually do. Do NOT collapse them
  into a single high-level summary such as "helps", "assists", or "checks on".

FORBIDDEN — NO SEMANTIC SPECULATION
- Describe ONLY what is visually observable. Do NOT judge whether behavior is normal or abnormal.
- If a person is speaking, say "speaks while gesturing"; do NOT guess what they are saying.

SUBJECT FIELD:
- Short noun phrase identifying the subject (e.g., "man in dark jacket", "woman carrying a backpack", "cyclist in red helmet", "white sedan", "person in white shirt").
- Be specific enough to disambiguate from other subjects in the clip.

OUTPUT FORMAT: A JSON array, and nothing else:
[
  {{"action_id": 1, "start": ..., "end": ..., "subject": "...", "description": "..."}},
  ...
]

RULES:
- TIMESTAMPS: ABSOLUTE timestamps relative to the original video, one decimal place. Must be within [{start:.1f}, {end:.1f}].
- DESCRIPTIONS: One sentence, present tense, specific and concrete.
- Output ONLY the JSON array. No explanation, no markdown fences. If nothing happens, output []."""


# ─── Per-frame Object Detection ───

PROMPT_DETECT_OBJECTS = """You are given a single frame from a FIXED outdoor campus surveillance camera.

This frame belongs to the following scene:
"{parent_description}"

TASK: Detect EVERY clearly visible person in this frame, plus salient vehicles and the objects people are using. Put a tight bounding box around each.

ALWAYS DETECT — do NOT skip these:
- EVERY person who is clearly visible, in ANY posture or activity: walking, running, cycling, standing still, sitting, squatting, crouching, bending, leaning, or interacting with someone or something. A person does NOT need to be "doing" anything — a person who is merely standing or sitting STILL gets a box.
- Cyclists together with the bicycle they are riding, and any vehicle that is moving or being entered, exited, or loaded.
- An object a person is actively holding or using (bag, cart, box, umbrella, phone).

ONLY SKIP:
- Figures so tiny, blurry, or deep in the background that you genuinely cannot tell they are a person.
- Pure scenery and structure: roads, walkways, walls, lawns, sky, building facades, closed doors.
- Static background clutter that nobody is using: parked bicycles, parked cars, benches, bins, lamp posts, racks. (A bike being ridden or pushed IS detected; an empty parked bike is NOT.)
- Text overlays, timestamps, logos, watermarks, body parts in isolation.

GROUPING — applies to INANIMATE clusters only, NEVER to people you can tell apart:
- If THREE OR MORE near-identical inanimate objects are parked/piled/racked together (a rack of parked bikes, a row of parked cars), draw ONE box with a count-style label like "row of parked bicycles", or skip them as background.
- Box each clearly distinguishable person SEPARATELY. Only when a crowd is so dense that individuals cannot be separated may you use a single "group of N people" box for the inseparable part.

ONE SUBJECT = ONE BOX = ONE LABEL. Do not output overlapping boxes for the same person or object under different names.

LIMIT: up to about 15 boxes. Prefer boxing every distinct visible person over staying under the limit; only if there are clearly more than ~15 people, box the most prominent individuals and cover the rest with one "group of people" box.

LABELS:
- Short noun phrase, 1 to 6 words, concrete: "man in pink t-shirt", "person sitting on steps", "cyclist", "white sedan", "person with backpack".
- Distinguish people by visible attributes (clothing color, posture): "man in black t-shirt" vs "man in white shirt".

BOUNDING BOX:
- Format [x1, y1, x2, y2], normalized to a 0 to 1000 scale (top-left origin).
- Tightly enclose the subject. For a person, span head to feet, or to the crop/occlusion edge.

OUTPUT: a JSON array, nothing else. No markdown fences, no explanation.
[
  {{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}
]

Output [] ONLY when there are genuinely no visible people, cyclists, or vehicles (a truly empty walkway, road, or plaza). If even ONE person is visible anywhere in the frame, the array MUST NOT be empty."""
