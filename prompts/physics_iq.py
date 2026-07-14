"""Physics-IQ prompts: fixed-camera tabletop physics experiments (8 s clips).
Subjects are objects and substances. Falls back to prompts.general.
PROMPT_ACTIONS deliberately omits {parent_description} so caption phrasing
cannot contaminate action annotations.
"""

# Scene segmentation

PROMPT_SEGMENT = """You are given {num_frames} frames sampled at a uniform rate from a video that is {duration:.1f} seconds long.

This video is a continuous recording of a tabletop physics experiment from a fixed camera. The setting stays constant; what changes over time is the physical state of the scene.

YOUR TASK: Divide this video into segments that follow the phases of the physical experiment. A segment is a stretch of the video in which the physical state of the scene stays qualitatively the same.

WHEN TO CUT:
- A distinct physical event begins: an object is released or starts to move, a liquid starts to pour, a flame ignites, a collision sets new objects in motion.
- One physical process completes and a different one begins (a ball finishes rolling and a struck object starts to topple).
- The event finishes and the scene settles into a resting state.

WHEN TO KEEP ONE SEGMENT:
- A single continuous process stays in one segment for its whole duration, even if several objects move as part of it.
- Small residual motion (an object wobbling before coming to rest) belongs to the segment of the process that caused it.

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
- BRIEF: Direct description of the physical state or event in the segment. Do NOT start with labels like "Scene:" or "Brief:". Just write the content.
- Output ONLY the JSON array. No explanation, no markdown fences."""


# Dense captioning

PROMPT_CAPTION = """You are annotating footage of a controlled tabletop physics experiment recorded by a FIXED camera.

{prefix_context}

You are given {num_frames} frames sampled at a uniform rate from this clip [{start:.1f}s to {end:.1f}s].

The footage shows physical events: objects fall, roll, collide, tip over, swing, or bounce; liquids pour, spill, flow, or mix; materials burn, melt, or deform; light, shadows, or magnets interact with objects. This clip covers one phase of the experiment and may show the setup at rest, an event in progress, or the scene after the event.

Cover these aspects in continuous prose:
- The setup visible in this clip: the surface and background, each visible object and substance, their colors, shapes, apparent materials, sizes relative to each other, and how they are arranged (stacked, suspended, leaning, contained).
- Any motion or state change during this clip, in temporal order: what starts to move, in which direction, what it touches or collides with, and every visible change of state (tipping, breaking, spreading, mixing, igniting, deforming). If nothing moves in this clip, describe the arrangement only.
- Where each thing that moved ends up.

CRITICAL FORMAT RULES:
- Write as continuous prose. No section headings. No bullet lists. No markdown.
- Do NOT start with labels like "Summary:" or "Description:". Begin directly with the content.

CONTENT RULES:
- Be specific and concrete: "the steel sphere drops onto the raised end of the wooden plank, and the opposite end of the plank flips upward, launching the cork into the air".
- Do NOT describe stillness or absence of change. Never write that something "remains", "stays", "is static/stationary", or that "nothing moves/happens". For a motionless object, state only where it is and what it looks like: "six colored dominoes stand upright in a row".
- End a motion at its visible endpoint ("the ball rolls to a stop against the block"); do not add that things stay that way afterwards.
- Never close with a summary sentence about the scene as a whole ("no further motion occurs", "everything is as before"). End after the last concrete detail.
- If previous scenes are listed above, do NOT re-describe the setup already covered. Focus on what is NEW or CHANGED in this clip.
- Describe materials and objects by their visible appearance. If you cannot identify an object, describe its shape and color instead of guessing its identity.
- Describe ONLY what is visible in the frames. Do NOT name physical laws, forces, or principles. Do NOT write "due to gravity", "demonstrating momentum", "because of magnetism", or similar explanations. Report the motion itself, not its cause.
- Do NOT describe what will happen after the last frame or what happened before the first frame.
- If a hand appears, describe exactly what it does (releases, pushes, cuts, pours) without guessing intent.
- Do NOT fabricate. Avoid "appears to", "seems to", "might be". If you cannot see it clearly, do not mention it.

Output ONLY the description text."""


PROMPT_CAPTION_RETRY = """Your previous description of this physics experiment clip was too short ({word_count} words).
Here is what you wrote:
---
{previous_attempt}
---

Please rewrite and EXPAND this description to at least {min_words} words.

- Reach the length by adding VISIBLE detail only: the appearance, color, material, and arrangement of each object in the setup; the exact order in which things move; the path, direction, and contact points of each moving object; every intermediate state change; and the final position of each object.
- Do NOT invent objects, motion, or outcomes that are not visible in the frames. Fabrication is worse than a short description.
- Do NOT pad with explanations of physical laws or causes. Report only what can be seen.
- Do NOT pad by stating that things remain still or unchanged; add visible detail instead.

{prefix_context}

Output ONLY the expanded description text."""


# Actions

# The VLM sees a 0-based subclip; it reports clip-local times and code adds the
# offset (asking for absolute times made it do the addition inconsistently).
ACTIONS_TIMESTAMPS = "relative"

PROMPT_ACTIONS = """You are given {num_frames} frames sampled at a uniform rate from a FIXED camera recording of a tabletop physics experiment. The frames form one clip that is {duration:.1f} seconds long.

TIME IS RELATIVE TO THIS CLIP: the first frame you see is 0.0 seconds and the last is {duration:.1f} seconds. Report every timestamp on that scale. Do NOT add any offset and do NOT try to place events on the timeline of the original video — that is done for you afterwards.

YOUR TASK: Annotate the visible physical actions in this clip. An action is one coherent motion or state change performed or undergone by ONE specific visual subject.

SUBJECTS ARE OBJECTS AND SUBSTANCES:
- Typical subjects: "red ball", "wooden block", "stream of water", "candle flame", "metal sphere", "sheet of paper".
- A hand that enters the frame is also a valid subject: "hand" or "hand holding scissors".

WHAT COUNTS AS AN ACTION — THE DISPLACEMENT TEST:
- Before you write down any action, apply this test to its subject: compare the frame at the action's start with the frame at its end. Has that subject's position, orientation, or physical state visibly CHANGED between those two frames? If yes, it is an action. If it looks the same in both, it is NOT an action and must not appear in the output.
- Describing where something merely IS is never an action. "hangs suspended from the frame", "sits on the table", "stands upright", "rests against the wall", "remains stationary", "stays still" all fail the displacement test. A stationary ramp, track, stand, or backdrop piece is scenery: it belongs in the scene description, never in this list — no matter how prominent it looks.
- No rewording makes stillness an action. "is positioned on", "is held by", "dangles from", "occupies the center" are the same failure with different verbs. Before output, re-check every entry: if its subject looks identical in the entry's first and last frame, DELETE the entry. An output with fewer, real actions beats one padded with state descriptions.
- Now, among the subjects that DO pass the test, miss none of them. Every object or substance that visibly moves must be the subject of at least one action. A ball that rolls, a domino that topples, a liquid that spreads — leaving any of them out is a failure.
- If nothing in the clip moves, the correct output is [].
- ONE SUBJECT PER ACTION. If a ball hits a block and both move, that is two actions: one for the ball, one for the block.
- Chains of cause and effect become sequences of actions: the hand releases the ball, the ball rolls down the ramp, the ball strikes the wooden block, the block tips over. Annotate EVERY link of the chain, not just the first mover.
- If three or more near-identical items topple or move as one connected wave (a row of dominoes, a stack of blocks), annotate the wave as ONE action whose subject is the group, e.g. "row of colored dominoes": start when the first item begins to move, end when the last item comes to rest. Do not silently omit them because they look like scenery.
- Actions CAN overlap in time, and gaps are fine. Not every moment needs an action.
- The camera is FIXED. Do NOT annotate camera movement.
- If nothing moves or changes in the clip, output an empty array [].

TIME RESOLUTION — LOCALIZE, DO NOT COPY THE CLIP BOUNDS:
- Each frame you are given carries its own timestamp, and consecutive frames are a fraction of a second apart. You can and MUST resolve events to a fraction of a second.
- An action's start is the timestamp of the frame where the motion visibly BEGINS. Its end is the timestamp of the frame where it visibly STOPS. Read them off the frames.
- Do NOT reuse the clip's own bounds 0.0 and {duration:.1f} as an action's start and end. Copying the clip bounds instead of localizing the event is the single most common failure here. An action that spans the whole clip is almost always wrong.
- Do NOT round to whole seconds. Most events in this footage are short: an impact, a release, a bounce, or a topple typically lasts 0.2 to 1.0 seconds. Values like 1.3, 2.7, 4.2 are expected and correct; a list of actions that all start and end on whole seconds is wrong.
- A long, genuinely continuous process (a liquid pouring for the entire clip) may legitimately span the clip. That is the exception, not the rule.

USE PHYSICAL, VISUAL VERBS:
- Good: "falls", "drops", "rolls", "slides", "tips over", "collides with", "bounces off", "swings", "rotates", "pours", "spills", "flows", "spreads", "sinks", "floats", "ignites", "melts", "deforms", "snaps", "comes to rest".
- Do NOT name laws, forces, or causes: never write "due to gravity", "demonstrates inertia", "is attracted by magnetic force". If a magnet visibly pulls an object, write "slides toward the magnet".
- Do NOT guess intent for hands: write "hand releases the ball", not "hand prepares to demonstrate the experiment".
- Do NOT describe outcomes beyond the visible frames.

SUBJECT FIELD:
- Short noun phrase identifying the subject by visible appearance (e.g., "orange billiard ball", "tall glass of water", "hand in blue glove").
- Be specific enough to disambiguate from other subjects in the clip.

OUTPUT FORMAT: A JSON array, and nothing else:
[
  {{"action_id": 1, "start": ..., "end": ..., "subject": "...", "description": "..."}},
  ...
]

RULES:
- TIMESTAMPS: RELATIVE to this clip, one decimal place, taken from the frames where the motion starts and stops. Every value must lie within [0.0, {duration:.1f}], and start must be strictly less than end.
- DESCRIPTIONS: One sentence, present tense, specific and concrete, stating the motion and what it contacts or produces.
- Output ONLY the JSON array. No explanation, no markdown fences. If nothing happens, output []."""


# Per-frame object detection

PROMPT_DETECT_OBJECTS = """You are given a single frame from a FIXED camera recording of a tabletop physics experiment.

This frame belongs to the following scene:
"{parent_description}"

TASK: Detect the objects that take part in the physical experiment, and provide a tight bounding box for each. The subjects of this footage are the experiment objects and substances.

ALWAYS DETECT:
- Every object involved in the physical event: objects that move or will move (balls, blocks, dominoes, toys), objects that are struck, pushed, or knocked over, and objects that receive something (a container being filled, a surface being splashed).
- Structures the event runs on or through: ramps, tracks, levers, strings, stands, scales.
- Visible substances that act in the event: a stream or puddle of liquid, a flame, smoke, sand. Box the visible extent of the substance.
- A hand or tool if present in the frame.
- Small objects count. A marble, a magnet, a coin, or a match can be the most important object in the frame. Detect key objects even if they are small.

ONLY SKIP:
- The table surface, backdrop, walls, and floor as a whole.
- Fixed studio elements not involved in the event (lighting, clamps holding the backdrop).
- Text overlays, timestamps, logos, watermarks.
- Objects that are completely outside the event and never interact with anything.

GROUPING:
- If THREE OR MORE near-identical items form one connected unit (a row of dominoes, a stack of blocks, a cluster of marbles), draw ONE box around the whole unit with a count-style label such as "row of dominoes" or "stack of wooden blocks". Do NOT box each item separately.
- Otherwise, ONE OBJECT = ONE BOX = ONE LABEL. Do NOT output overlapping boxes for the same physical object under different names.

LIMIT: output AT MOST 10 boxes.

LABELS:
- Short noun phrase, 1 to 6 words, describing visible appearance: "red billiard ball", "clear glass jar", "row of black dominoes", "metal ramp", "hand in white glove".
- If you cannot identify an object, label its shape and color: "small silver sphere", not a guessed identity.
- When the scene description names an object, stay close to that wording.

BOUNDING BOX:
- Format [x1, y1, x2, y2], normalized to a 0 to 1000 scale (top-left origin).
- Tightly enclose the object, or its visible part at a crop or occlusion edge.

OUTPUT: a JSON array, nothing else. No markdown fences, no explanation.
[
  {{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}
]

Output [] only if the frame is blank or contains no experiment objects at all."""