# VideoCap

VideoCap is the single production flow in this repository. It turns a
local video manifest into multi-dimensional captions, temporal events, retained
intermediate artifacts, and a dataset-named final JSONL file. There is no recipe
selection layer.

VideoCap is the caption-production member of a planned video data stack.
`VideoEval` will provide caption and event quality evaluation, and `VideoQA`
will derive question-answer and task labels from the generated annotations.

## Run Configuration

The runtime configuration keeps VideoCap settings under `pipeline_config`:

```json
{
  "pipeline_config": {
    "adapter_factory": "videocap.adapters.siliconflow:build_from_config",
    "max_duration_ms": 24000,
    "overlap_ms": 2000,
    "evidence_frame_count": 8,
    "final_output_name": "VDC_5.jsonl",
    "provider": {
      "base_url": "https://api.siliconflow.cn/v1",
      "api_key_env": "SILICONFLOW_API_KEY",
      "vlm_model": "Qwen/Qwen3.6-35B-A3B",
      "llm_model": "Qwen/Qwen3.6-35B-A3B",
      "local_frame_mode": "data_uri",
      "frame_height": 460
    }
  }
}
```

The default processing profile is a 24-second window with 2 seconds of overlap,
which gives a 22-second step. Eight evidence timestamps are evenly spaced over
the inclusive window interval. Local frames are resized to 460 pixels in
height while preserving aspect ratio.

## VideoCap Stages

For each video, VideoCap writes deterministic processing windows and window
captions, then generates event candidates, clusters adjacent candidates,
proposes coarse event windows, reviews boundaries, refines event captions,
merges global captions, and applies the quality filter. A processing window is
the range shown to the VLM; it is not itself an event label.

The global `captions` object contains the five dimensions `short`,
`main_object`, `background`, `camera`, and `detailed`. The `events` array holds
zero or more temporal events, each with an ID, cluster ID, interval, evidence
timestamps, and an event caption. Event IDs join boundary review, refinement,
and quality filtering artifacts.

The VDC-inspired window prompts live in
`videocap/prompts/window_caption.py`. The five dimension prompts
are combined into one VLM request in the fixed order `short`, `main_object`,
`background`, `camera`, `detailed`, so the same evidence frames are not sent
five times.

## Artifacts

The runner writes a `pipeline_work/` directory containing per-video stage
artifacts under `output/<video_id>/` and the final dataset-named JSONL file under
`final/`. Run-level provenance is stored in `run_manifest.json`; its component
field is `pipeline`, with the value `videocap` and version `0.1`.

Run the pipeline with:

```bash
videocap run videos.jsonl --config configs/videocap/siliconflow.json --output-root runs
```

The API key is read from the configured environment variable and is never
written to the resolved configuration or project files.
