# VideoCap

[简体中文](README.zh-CN.md)

VideoCap turns local videos into two complementary annotation layers: a five-dimensional caption for the complete video and temporally grounded captions for coherent events. It keeps the production graph fixed and inspectable, so processing windows, model evidence, event boundaries, and final annotations remain easy to audit.

## Pipeline

```text
video manifest
  -> overlapping processing windows
  -> VLM: five-dimensional caption for each window
  -> LLM: event proposals from short + main_object captions
  -> VLM: coarse and fine event-boundary review
  -> VLM: caption each grounded event
  -> LLM: merge events and window evidence into global captions
  -> final JSONL + stage artifacts
```

Processing windows are model-input units, not annotation boundaries. One LLM call groups consecutive windows into semantic events; a VLM then selects visually supported timestamps and captions only the accepted interval. The global merge uses grounded events as its chronological backbone and window captions for subject, setting, camera, and fine-detail evidence.

## Requirements

- Python 3.10 or newer
- `ffmpeg` available on `PATH`
- an OpenAI-compatible Chat Completions endpoint that accepts `image_url` data URIs

Install the project with `uv`:

```bash
git clone https://github.com/savebees/VideoCap.git
cd VideoCap
uv sync --locked
```

Standard editable installation also works:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Configuration

All runtime settings live in one file: [`configs/videocap.json`](configs/videocap.json). The VLM and LLM sections use the same small OpenAI-compatible contract but may point to different providers or models.

```json
{
  "pipeline": {
    "window_ms": 24000,
    "overlap_ms": 2000,
    "evidence_frames": 8,
    "output_name": "annotations.jsonl"
  },
  "vlm": {
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key_env": "SILICONFLOW_API_KEY",
    "model": "Qwen/Qwen3.6-35B-A3B",
    "frame_height": 460,
    "timeout_sec": 120,
    "max_retries": 2,
    "extra_body": {"enable_thinking": false}
  },
  "llm": {
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key_env": "SILICONFLOW_API_KEY",
    "model": "Qwen/Qwen3.6-35B-A3B",
    "timeout_sec": 120,
    "max_retries": 2,
    "extra_body": {"enable_thinking": false}
  }
}
```

API keys are read only from the named environment variables. The resolved run configuration therefore contains endpoint and model provenance without storing credentials.

## Video Manifest

The input is UTF-8 JSONL with one video per line:

```json
{"video_id":"video_001","video_path":"videos/video_001.mp4","duration_ms":68320}
{"video_id":"video_002","video_path":"videos/video_002.mp4","duration_ms":124500,"metadata":{"split":"demo"}}
```

Relative video paths are resolved from the manifest directory. Video IDs must be unique, files must exist, and durations are expressed in milliseconds.

## Run

```bash
export SILICONFLOW_API_KEY="..."
uv run videocap run videos.jsonl \
  --config configs/videocap.json \
  --output-root runs
```

Each execution creates a new immutable run directory:

```text
runs/<run_id>/
├── annotations.jsonl
├── config.json
├── failures.jsonl
├── manifest.json
├── summary.json
└── stages/<video_id>/
    ├── processing_windows.jsonl
    ├── window_captions.jsonl
    ├── event_proposals.jsonl
    ├── event_boundaries.jsonl
    ├── event_captions.jsonl
    ├── global_caption.json
    └── failure.json              # only when this video fails
```

The run manifest records the dataset and configuration hashes, VideoCap version, Git state, and creation time. A failed video is recorded explicitly and does not erase successful records from the same run.

## Output

```json
{
  "schema_version": "videocap/v0.2",
  "video_id": "video_001",
  "duration_ms": 68320,
  "captions": {
    "short": "A man prepares and serves a cooked dish.",
    "main_object": "A man prepares ingredients, cooks them, and carries the finished plate.",
    "background": "The activity takes place in a kitchen and adjoining dining area.",
    "camera": "Mostly static medium shots follow the activity from the counter to the table.",
    "detailed": "A man prepares ingredients, cooks them, plates the food, and carries it to a table."
  },
  "events": [
    {
      "event_id": "event_0000",
      "start_ms": 1750,
      "end_ms": 51250,
      "evidence_frames_ms": [1750, 51250],
      "caption": "A man prepares ingredients, cooks them, and plates the finished food."
    }
  ]
}
```

The public schema is packaged at [`videocap/schemas/videocap.schema.json`](videocap/schemas/videocap.schema.json). Prompt templates are kept separate from model adapters under [`videocap/prompts`](videocap/prompts), while provider-independent VLM and LLM implementations live in [`videocap/adapters`](videocap/adapters). The five caption dimensions follow the public VDC prompt taxonomy from [AuroraCap](https://github.com/wenhaochai/aurora).

## Development

```bash
uv sync --locked
uv run ruff check .
uv run pytest
uv build
```

See [`TODO.md`](TODO.md) for work that is deliberately outside the current release.

## License

VideoCap is released under the [MIT License](LICENSE).
