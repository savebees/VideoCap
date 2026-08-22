"""Prompt design inspired by AuroraCap's VDC benchmark.

Source: https://github.com/wenhaochai/aurora
"""

from __future__ import annotations

import random


SHORT_CAPTION_PROMPTS = [
    "Write a one-sentence summary of the video.",
    "Summarize the video in one concise sentence.",
    "Provide a brief description of the video in one sentence.",
    "Describe the main action in the video in one sentence.",
    "What is the video about? Summarize it in one sentence.",
    "In one sentence, summarize the key visual elements of the video.",
    "Provide a one-sentence summary that captures the main subject and action in the video.",
    "Write a concise one-sentence description that encapsulates the essence of the video.",
    "Describe the main theme or action of the video in a single sentence.",
    "What is happening in the video? Provide a one-sentence summary.",
    "Given these frames, write a brief one-sentence summary that captures the essence of the video's visual and artistic style.",
    "Summarize the key visual and thematic elements of the video in one concise sentence.",
    "Provide a one-sentence description that highlights the main subject and action depicted in the video.",
    "In one sentence, describe the primary visual and artistic elements of the video.",
    "Write a concise one-sentence summary that encapsulates the main action and visual style of the video.",
    "Briefly one-sentence Summary of the visual, Photographic and artistic style.",
]

MAIN_OBJECT_CAPTION_PROMPTS = [
    "Description of the main subject actions or status sequence. This suggests including the main subjects (person, object, animal, or none) and their attributes, their action, their position, and movements during the video frames.",
    "Describe the main subject's actions and movements.",
    "What is the main object doing in these frames?",
    "Summarize the primary subject's attributes and actions.",
    "Describe the main subject's position and movements.",
    "What actions does the main object take in these frames?",
    "Describe the main subject, including their attributes and movements throughout the video.",
    "Provide a detailed description of the main object's actions and positions in these frames.",
    "Summarize the main subject's actions, attributes, and movements during the video.",
    "Describe the primary subject's movements and actions in detail.",
    "What are the main object's attributes and how do they move throughout the video?",
    "Given these equally spaced frames, provide a comprehensive description of the main subject, including their attributes, actions, positions, and movements.",
    "Describe the primary object or subject in the video, detailing their attributes, actions, positions, and movements in these frames.",
    "Based on these frames, provide a detailed description of the main subject, including their attributes, actions, positions, and how they navigate through the video.",
    "Using these frames, describe the main subject's attributes, actions, and movements, detailing their positions and how they interact with the environment.",
    "Provide an elaborate description of the main object in the video, covering their attributes, actions, positions, and movements as shown in these frames.",
]

BACKGROUND_CAPTION_PROMPTS = [
    "The images are given containing equally spaced video frames.Summary of the background. This should also include the objects, location, weather, and time.",
    "Describe the background, including objects, location, weather, and time.",
    "Summarize the background setting of the video based on these frames.",
    "What is the environment like in these frames?",
    "Describe the location and weather in these frames.",
    "What background objects and settings are visible in these frames?",
    "Summarize the background of the video, including details about the location, objects, weather, and time.",
    "Describe the environment shown in these frames, covering objects, location, weather, and time.",
    "Provide a detailed background description based on these frames, mentioning objects, location, weather, and time.",
    "Explain the setting of the video, focusing on the background elements like objects, location, weather, and time.",
    "Describe the overall environment in these frames, including details about objects, location, weather, and time.",
    "Given these equally spaced frames, provide a comprehensive background description, covering the objects, location, weather, and time.",
    "Imagine the environment from these frames and write a detailed description of the background, including objects, location, weather, and time.",
    "Based on these frames, describe the setting in detail, mentioning the objects present, the specific location, the weather conditions, and the time of day.",
    "Provide an elaborate background description based on these frames, covering all aspects of the environment such as objects, location, weather, and time.",
    "Using these frames as a reference, give a thorough description of the background, including details about the objects, location, weather, and time.",
]

CAMERA_CAPTION_PROMPTS = [
    "Summary of the view shot, camera movement and changes in shooting angles in the sequence of video frames.",
    "Describe the camera movements in these frames.",
    "What are the camera angles and movements throughout the video?",
    "Summarize the camera actions and perspectives.",
    "Describe any camera zooms, pans, or angle changes.",
    "What camera movements are present in these frames?",
    "Describe the camera's movements, including pans, zooms, and angle changes in these frames.",
    "Summarize the camera actions and changes in shooting angles during the video.",
    "Provide a detailed description of the camera's movements and perspectives.",
    "Describe the camera's actions and how it follows the main subject.",
    "What are the camera movements and angle shifts in these frames?",
    "Given these equally spaced frames, provide a comprehensive description of the camera's movements, including any pans, zooms, and changes in shooting angles.",
    "Describe the camera's movements and angles in detail, explaining how it follows the main subject and changes perspectives.",
    "Based on these frames, provide a detailed description of the camera's actions, including any pans, zooms, angle shifts, and how it captures the scene.",
    "Using these frames, describe the camera's movements, including its tracking of the main subject, changes in angles, and any zooms or pans.",
    "Provide an elaborate description of the camera movements, covering pans, zooms, and changes in shooting angles as shown in these frames.",
]

DETAILED_CAPTION_PROMPTS = [
    "Describe this video segment based on the sequence of frames in more than three sentences.",
    "What is happening in this video segment? Provide a detailed description in more than three sentences.",
    "Explain this video segment using the sequence of frames in at least three sentences.",
    "Imagine the action represented by this frame sequence and describe the segment in detail in more than three sentences.",
    "Based on these frames, provide a detailed narrative of this video segment in more than three sentences.",
    "Describe the events in this video segment in at least three sentences.",
    "Visualize this segment from the frames and explain what is happening in more than three sentences.",
    "Describe the sequence of events depicted by these frames in a detailed manner.",
    "Given these frames, provide a detailed description of the segment, including the setting, subjects, and actions, in more than three sentences.",
    "Write a comprehensive description of what happens across this frame sequence, describing its beginning, middle, and end in at least three sentences.",
    "Using these frames as a reference, provide a thorough description of this segment's actions and visual details in more than three sentences.",
    "Based on this sequence of frames, describe the segment in detail, mentioning important context, movements, and transitions in more than three sentences.",
    "Describe this frame sequence elaborately, covering the visible storyline, visual elements, and notable features in at least three sentences.",
    "What are the main events and visual details in this video segment? Explain them in more than three sentences.",
    "How does the visible action progress through this frame sequence? Provide a detailed answer in at least three sentences.",
    "Describe the scene, subjects, interactions, and changes shown across this segment in more than three sentences.",
    "What story is conveyed by this sequence of frames? Describe only the visible segment in at least three sentences.",
    "Provide a faithful, detailed account of the visible actions and setting in this video segment in more than three sentences.",
    "Describe the visual progression from the first frame to the last frame in at least three sentences.",
    "Explain the context, actions, and visual transitions shown in this segment in more than three sentences.",
    "Write a detailed multi-sentence caption for this local video segment based on the supplied frames.",
    "Describe the visible scene and its progression in this frame sequence using more than three sentences.",
    "Summarize the local narrative and important visual details of this segment in at least three sentences.",
    "Give a detailed description of the subjects, setting, actions, and changes visible in this video segment.",
]


_PROMPT_POOLS = {
    "short": SHORT_CAPTION_PROMPTS,
    "main_object": MAIN_OBJECT_CAPTION_PROMPTS,
    "background": BACKGROUND_CAPTION_PROMPTS,
    "camera": CAMERA_CAPTION_PROMPTS,
    "detailed": DETAILED_CAPTION_PROMPTS,
}

WINDOW_CAPTION_DIMENSIONS = ("short", "main_object", "background", "camera", "detailed")


def build_window_caption_prompt() -> str:
    """Build one VDC-style multi-dimension prompt in the canonical order."""

    sections = []
    for dimension in WINDOW_CAPTION_DIMENSIONS:
        instruction = random.choice(_PROMPT_POOLS[dimension])
        sections.append(f"[{dimension}]\n{instruction}")
    return (
        "You are given a consecutive sequence of frames from one segment of a video. "
        "Describe only what is visually supported by this sequence and its visible changes. "
        "Do not infer content outside this segment.\n\n"
        "Please answer the following questions in this exact order: short, "
        "main_object, background, camera, detailed. For each answer, put the "
        "dimension label on its own line using the required square brackets, "
        "then write the caption below it. Do not add any other headings, "
        "commentary, markdown, or explanation.\n\n"
        + "\n\n".join(sections)
    )


__all__ = [
    "BACKGROUND_CAPTION_PROMPTS",
    "CAMERA_CAPTION_PROMPTS",
    "DETAILED_CAPTION_PROMPTS",
    "MAIN_OBJECT_CAPTION_PROMPTS",
    "SHORT_CAPTION_PROMPTS",
    "WINDOW_CAPTION_DIMENSIONS",
    "build_window_caption_prompt",
]
