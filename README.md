<h1 align="center">VideoCap</h1>

<p align="center">
  English | <a href="docs/README.zh-CN.md"><ins>简体中文</ins></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/dense--video--annotator-2563eb?style=flat-square" alt="dense-video-annotator">
  <img src="https://img.shields.io/badge/video--eval-2563eb?style=flat-square" alt="video-eval">
  <img src="https://img.shields.io/badge/video--qa-2563eb?style=flat-square" alt="video-qa">
</p>

VideoCap generates dense video annotations with five descriptions—summary, subject, background, camera, and details—plus timestamped semantic events, while video-eval and video-qa support QA generation and annotation quality evaluation for model training.


## 📣 Demo

<p align="center">
  <a href="assets/big-buck-bunny-demo.mp4">
    <img src="assets/big-buck-bunny-demo.gif" width="480" alt="Big Buck Bunny wakes in his burrow and explores a sunny meadow">
  </a>
  <br>
  <sub>Copyright © 2008 Blender Foundation · <a href="https://peach.blender.org/about/">CC BY 3.0</a></sub>
</p>

### VideoCap output

```json
{
  "schema_version": "videocap/v0.2",
  "video_id": "bbb-0038-0078",
  "duration_ms": 40000,
  "captions": {
    "short": "Big Buck Bunny wakes inside a shaded burrow, steps into a bright meadow, pauses to take in the morning, smells a cluster of white flowers, and notices a purple butterfly.",
    "main_object": "The large gray rabbit slowly raises his head from the burrow, crawls into the sunlight, sits upright, stretches his arms and back, scans the meadow with a relaxed smile, bends toward white blossoms, and turns to follow a purple butterfly.",
    "background": "The sequence moves from a dark, grass-lined burrow beneath a broad tree into a sunlit meadow bordered by rocks, leafy trees, rolling green hills, white daisies, purple flowers, and a clear blue sky; small birds and insects animate the otherwise calm landscape.",
    "camera": "It opens with a static wide view of the burrow, moves into a close-up as the rabbit wakes, cuts to medium and low-angle shots as he emerges and stretches, then alternates between close-ups of his face and the flowers, an over-the-shoulder meadow view, and a high-angle shot tracking his attention toward the butterfly.",
    "detailed": "A quiet wide shot holds on the dark burrow beneath the tree. The rabbit is barely visible at first, then raises his head into the light, opens his eyes, and looks toward the entrance. The rabbit crawls out of the burrow, settles on the grass beside the opening, and slowly stretches his arms, shoulders, and back in the warm sunlight. Now fully upright, he tilts his face toward the sky, breathes in, and surveys the open meadow with a calm smile as the camera moves between low-angle and close-up views. He turns toward a patch of white flowers, leans in, and smells the blossoms while the edit cuts from an over-the-shoulder view to a close-up of his face among the petals. A purple butterfly flutters beside the flowers. The rabbit notices it, shifts his attention across the meadow, and moves after it as the sequence ends on a high-angle view."
  },
  "events": [
    {"event_id": "event_0000", "start_ms": 0, "end_ms": 7000, "evidence_frames_ms": [1000, 6000], "caption": "A quiet wide shot holds on the dark burrow beneath the tree. The rabbit is barely visible at first, then raises his head into the light, opens his eyes, and looks toward the entrance."},
    {"event_id": "event_0001", "start_ms": 7000, "end_ms": 16000, "evidence_frames_ms": [8000, 15000], "caption": "The rabbit crawls out of the burrow, settles on the grass beside the opening, and slowly stretches his arms, shoulders, and back in the warm sunlight."},
    {"event_id": "event_0002", "start_ms": 16000, "end_ms": 25000, "evidence_frames_ms": [17000, 24000], "caption": "Now fully upright, he tilts his face toward the sky, breathes in, and surveys the open meadow with a calm smile as the camera moves between low-angle and close-up views."},
    {"event_id": "event_0003", "start_ms": 25000, "end_ms": 33000, "evidence_frames_ms": [26000, 32000], "caption": "He turns toward a patch of white flowers, leans in, and smells the blossoms while the edit cuts from an over-the-shoulder view to a close-up of his face among the petals."},
    {"event_id": "event_0004", "start_ms": 33000, "end_ms": 40000, "evidence_frames_ms": [34000, 39000], "caption": "A purple butterfly flutters beside the flowers. The rabbit notices it, shifts his attention across the meadow, and moves after it as the sequence ends on a high-angle view."}
  ]
}
```

### video-qa

Generated QA examples:

```json
[
  {
    "qa_id": "bbb-0038-0078__global_main_object",
    "task": "action_recognition",
    "question": "What actions does the main subject perform throughout the video?",
    "answer": "The large gray rabbit slowly raises his head from the burrow, crawls into the sunlight, sits upright, stretches his arms and back, scans the meadow with a relaxed smile, bends toward white blossoms, and turns to follow a purple butterfly.",
    "provenance": {"event_ids": ["event_0000", "event_0001", "event_0002", "event_0003", "event_0004"], "evidence_frames_ms": []}
  },
  {
    "qa_id": "bbb-0038-0078__global_background",
    "task": "scene_transition",
    "question": "How does the setting or background change throughout the video?",
    "answer": "The sequence moves from a dark, grass-lined burrow beneath a broad tree into a sunlit meadow bordered by rocks, leafy trees, rolling green hills, white daisies, purple flowers, and a clear blue sky; small birds and insects animate the otherwise calm landscape.",
    "provenance": {"event_ids": ["event_0000", "event_0001", "event_0002", "event_0003", "event_0004"], "evidence_frames_ms": []}
  },
  {
    "qa_id": "bbb-0038-0078__global_detailed",
    "task": "temporal_reasoning",
    "question": "How do the events unfold from beginning to end?",
    "answer": "A quiet wide shot holds on the dark burrow beneath the tree. The rabbit is barely visible at first, then raises his head into the light, opens his eyes, and looks toward the entrance. The rabbit crawls out of the burrow, settles on the grass beside the opening, and slowly stretches his arms, shoulders, and back in the warm sunlight. Now fully upright, he tilts his face toward the sky, breathes in, and surveys the open meadow with a calm smile as the camera moves between low-angle and close-up views. He turns toward a patch of white flowers, leans in, and smells the blossoms while the edit cuts from an over-the-shoulder view to a close-up of his face among the petals. A purple butterfly flutters beside the flowers. The rabbit notices it, shifts his attention across the meadow, and moves after it as the sequence ends on a high-angle view.",
    "provenance": {"event_ids": ["event_0000", "event_0001", "event_0002", "event_0003", "event_0004"], "evidence_frames_ms": []}
  },
  {
    "qa_id": "bbb-0038-0078__event_0003__grounding",
    "task": "temporal_grounding",
    "question": "When does this event occur: He turns toward a patch of white flowers, leans in, and smells the blossoms while the edit cuts from an over-the-shoulder view to a close-up of his face among the petals.",
    "answer": "From 25000 ms to 33000 ms.",
    "provenance": {"event_ids": ["event_0003"], "evidence_frames_ms": [26000, 32000]}
  }
]
```

### video-eval

video-eval evaluates annotation quality against reviewed references using five-dimension caption F1, temporal event IoU, event-caption F1, temporal coverage, global-event consistency, and reference-supported hallucination.

```json
{
  "video_id": "bbb-0038-0078",
  "accepted": true,
  "caption_f1_by_dimension": {
    "short": 0.706,
    "main_object": 0.585,
    "background": 0.580,
    "camera": 0.527,
    "detailed": 0.943
  },
  "matched_events": 5,
  "candidate_events": 5,
  "reference_events": 5,
  "metrics": {
    "global_caption_f1": 0.668,
    "event_boundary_iou": 0.902,
    "event_caption_f1": 0.930,
    "temporal_coverage": 1.0,
    "temporal_coverage_delta": 0.100,
    "consistency": 1.0,
    "hallucination": 0.242
  }
}
```

## 🚀 Quick start

VideoCap requires Python 3.10+, `ffmpeg`, and OpenAI-compatible VLM and LLM endpoints. Clone the repository, then install it with `uv`:

```bash
git clone https://github.com/savebees/VideoCap.git
cd VideoCap
uv sync --locked
```

Or use a standard virtual environment and `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Set the providers and models in [`configs/videocap.json`](configs/videocap.json), export the API key named by `api_key_env`, and run:

```bash
export SILICONFLOW_API_KEY="..."
uv run videocap run videos.jsonl \
  --config configs/videocap.json \
  --output-root runs
```

When using `pip`, replace `uv run videocap` with `videocap` inside the activated environment.

For Claude Code, Codex, or another coding agent, paste:

> Configure VideoCap in this repository using my OpenAI-compatible VLM and LLM providers and models, keep all API keys in environment variables, generate `videos.jsonl` from my video directory, and run a one-video smoke test.

## 📦 Dataset preparation

Generate a manifest directly from a video directory; the helper scans recursively and reads each duration with `ffprobe`:

```bash
uv run python scripts/prepare_dataset.py /path/to/videos --output videos.jsonl
```

The generated UTF-8 JSONL contains one video per line:

```json
{"video_id":"video_001","video_path":"videos/video_001.mp4","duration_ms":68320}
{"video_id":"video_002","video_path":"videos/video_002.mp4","duration_ms":124500}
```

Relative paths are resolved from the manifest directory. Each `video_id` must be unique, each file must exist, and `duration_ms` is expressed in milliseconds.

## TODO

- [ ] Complete video-eval with configurable reference-based metrics and reproducible dataset-level quality reports.
- [ ] Complete video-qa with grounded question-answer generation.

## 🤝 Acknowledgement

We are grateful to the following open-source projects that inspired the design of VideoCap.

- [AuroraCap](https://github.com/wenhaochai/aurora): The VDC five-dimensional video-caption taxonomy and prompt design.
- [MVBench](https://github.com/OpenGVLab/Ask-Anything): The capability-oriented taxonomy for temporal video understanding tasks.
- [TempCompass](https://github.com/llyx97/TempCompass): The temporal-perception dimensions and QA task design.

## License

VideoCap is released under the [MIT License](LICENSE).
