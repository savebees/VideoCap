# Production recipes

A recipe is the unit that turns a dataset into normalized dense captions. It may
contain multiple models and data-engineering stages; users select the recipe,
not a low-level VLM adapter. Every recipe receives the complete sample list, one
isolated working directory, and its runtime configuration, then returns one
success or failure outcome per video.

## NVIDIA Cosmos Curator

The built-in `cosmos-curator` recipe targets NVIDIA's official split-and-caption
video pipeline. It is an external integration rather than a vendored fork. The
adapter was developed against upstream revision
`975b68910f23067fb2391e68368d5f7cd8cf64ce`; production environments should pin
and record the exact revision or release they install.

### Runtime prerequisite

The package itself remains installable on Python 3.10+, but the current upstream
Cosmos Curator runtime is a separate environment. At the audited revision it
pins Python 3.13 in Pixi, vLLM 0.24, Torch 2.11, and CUDA 13.0.2. The official
container/Pixi setup is therefore the supported execution environment; an older
host vLLM environment is not silently treated as compatible. The adapter emits
a clear missing-command or upstream-exit error and supports `import` mode when
the production job is run elsewhere.

The heavyweight environment is opt-in:

```bash
# Core installation: no CUDA, Torch, vLLM, or Cosmos dependencies.
uv sync

# Only when the Cosmos Curator run recipe is needed.
uv run dva env check cosmos-curator
uv run dva env install cosmos-curator
```

Conda is intentionally not required. It is useful for teams that already have
a CUDA fleet managed with Conda, but making it a prerequisite would add a large
bootstrap step and would still not guarantee compatibility with the upstream
CUDA, Torch, vLLM, and Ray versions. The installer downloads a fixed Pixi
binary and the pinned Cosmos Curator source into the DVA cache, then asks the
official `pixi.toml` to create its environment. No shell profile is modified,
and the core package environment stays small. Users who already manage Pixi or
Conda environments can skip installation and use `import` mode, or point
`environment_path` at an existing official Cosmos environment.

The installer stores the pinned upstream checkout and Pixi environment under
`~/.cache/dense-video-annotator/cosmos-curator` by default. Use `--path` to place
it on a large local disk. The command is deliberately separate from `uv sync`,
so users who only import existing outputs or run evaluation never download the
GPU stack. `dva env check` is read-only and returns a non-zero status until the
environment is complete.

For a shared installation, set `DVA_COSMOS_CURATOR_ROOT` to the same environment
root and omit `environment_path` from individual run configs.

In `run` mode, the adapter verifies all video paths, writes one batch input list
and Cosmos configuration, invokes the official
`cosmos_curator.pipelines.video.run_pipeline` entry point without a shell, and
retains stdout/stderr. Adapter-owned arguments guarantee per-clip metadata and
caption generation. Additional official split, filter, captioning, and resource
arguments go under `recipe_config.cosmos_args`.

```json
{
  "recipe_config": {
    "mode": "run",
    "upstream_revision": "975b68910f23067fb2391e68368d5f7cd8cf64ce",
    "caption_field": "qwen_caption",
    "cosmos_args": {
      "splitting_algorithm": "transnetv2",
      "captioning_window_size": 256,
      "caption_quality_flags_enabled": true,
      "caption_quality_stats_enabled": true
    }
  }
}
```

By default the adapter runs
`python -m cosmos_curator.pipelines.video.run_pipeline {config_path}`. In the
official Pixi environment, set `recipe_config.environment_path` to the root
created by the installer; the adapter then constructs the Pixi command for you.
`command` can also be configured explicitly as an array such as
`["pixi", "run", "--as-is", "video-pipeline", "{config_path}"]`. Supported
placeholders are `{config_path}`, `{input_manifest_path}`,
`{input_video_path}`, and `{output_path}`. Commands are token arrays and are
never interpreted by a shell.

`import` mode converts a completed local Cosmos output without launching it:

```json
{
  "recipe_config": {
    "mode": "import",
    "output_path": "/data/cosmos-output",
    "caption_field": "qwen_caption",
    "upstream_revision": "your-pinned-revision"
  }
}
```

This mode is appropriate when Cosmos runs through Slurm, NVCF, Kubernetes, or a
separate container. S3 output must currently be synchronized locally before
conversion. The converter reads `metas/v0/*.json`, adds each window's
clip-relative time to the clip start time, rounds safely to integer
milliseconds, clamps intervals to video duration, and sorts captions
deterministically. If `caption_field` is omitted, exactly one enhanced field is
preferred over exactly one raw caption field; ambiguous metadata fails visibly.

To add another mature production integration, implement
`DenseCaptionRecipe.produce(samples, work_dir, config)` and register the class
with `@RECIPES.register("recipe-name")`. The method should preserve upstream
metadata and errors in `DenseCaptionOutcome.metadata`, while emitting only the
stable `DenseCaptionPrediction` protocol as its prediction.

## Strict structured grounding

`structured-grounding` is the recipe for the full long-video graph. A processing
window is only the range shown to the VLM; it is not an event label. The recipe
creates deterministic fixed windows from `max_duration_ms` and `overlap_ms`,
keeps evidence frame timestamps, asks a VLM adapter for five caption dimensions
(`short`, `main_object`, `background`, `camera`, `detailed`), then runs candidate
generation, semantic clustering, coarse timing, boundary review, event
refinement, global merging, and quality filtering. The boundary review adapter
may return multiple events, which is the explicit split operation.

All eight adapters are mandatory: `caption_window`, `generate_event_candidates`,
`cluster_event_candidates`, `propose_event_window`, `review_event_boundary`,
`refine_event_caption`, `merge_global_caption`, and `quality_filter`. They can be
provided to `StructuredGroundingRecipe` in Python or configured as importable
`module:attribute` paths. Each per-video directory under
`recipe_work/output/<video_id>/` contains the corresponding JSONL stage
artifacts plus `global_caption.json` and `quality_report.json`. The final,
public-shaped artifact is separate: one JSON object per video is written to
`recipe_config.final_output_name`, for example
`recipe_work/final/VDC_5.jsonl` for a five-video run or
`recipe_work/final/VDC_1k.jsonl` for the complete VDC benchmark.

The recipe deliberately has no fallback path. Empty candidates or clusters,
unknown references, incomplete caption dimensions, failed boundary review, empty
empty global caption dimensions, and invalid quality-filter IDs are errors recorded in `failure.json`
and surfaced through the standard run failure artifact. The structured recipe
does not export the legacy `dense-caption/v0.1` protocol. A final record has a
top-level `captions` object with the five VDC dimensions and an `events` array;
each event contains its `event_id`, `cluster_id`, time interval, evidence
timestamps, and one event-level `caption`. The same `event_id` is the join key
across boundary review, event refinement, and quality filtering.

`evidence_frames_ms` is deterministic for processing windows: the configured
`evidence_frame_count` timestamps are evenly spaced across the inclusive window
interval, including its start and end. Event evidence timestamps are proposed by
the boundary-review adapter and validated against the final event interval.

### Vyce OpenAI-compatible provider

The built-in `dense_video_annotator.adapters.vyce:build_from_config` factory
connects the structured recipe to an OpenAI-compatible endpoint. It maps the
window caption, boundary review, and event refinement stages to
`nemotron-vision`; candidate generation, clustering, coarse timing, global
merging, and other text-only stages use `gpt-5.6`. The API key is read only from
the environment variable named by `provider.api_key_env` (default
`VYCE_API_KEY`).

The adapter sends `image_url` messages. Configure a stable `frame_url` for a
smoke test or a `frame_url_template` backed by your frame service for real
videos. The provider used in development rejected base64 data URIs, so the
adapter intentionally raises when neither URL option is configured instead of
claiming that the VLM saw local video frames. Provider HTTP errors, rate limits,
and malformed JSON responses are propagated to the per-video failure artifact.
URL templates may use `{video_id}`, `{video_path}`, `{timestamp_ms}`,
`{start_ms}`, and `{end_ms}`. The last two let a media service return the exact
processing or event clip instead of making the VLM receive the whole video.

Requests are spaced with `min_request_interval_sec` (default 2 seconds). HTTP
429 responses are retried only up to `max_retries` (default 2), honoring
`Retry-After` when supplied and otherwise using exponential backoff from
`retry_backoff_sec`. Once that retry budget is exhausted, the provider error is
propagated and no partial caption is returned.
