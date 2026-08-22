# VideoCap

VideoCap is a production-oriented video understanding flow for creating
global multi-dimensional captions,
temporal event annotations, and future caption-grounded QA labels. The same
outputs can support dataset production, model training, and caption quality
evaluation.

VideoCap is the production flow in this repository. It uses a fixed
window graph rather than a selectable recipe layer:

```text
video
  -> 24s processing windows with 2s overlap and 8 evidence frames
  -> one VLM request for the five caption dimensions
  -> adjacent-window event candidates
  -> semantic clustering
  -> coarse event windows
  -> boundary review and split/merge
  -> event caption refinement
  -> global caption merge
  -> quality filtering
  -> intermediate artifacts + final dataset-named JSONL
```

VideoCap is the caption-production member of a planned video data stack:
`VideoEval` will cover caption and event quality evaluation, while `VideoQA`
will derive question-answer and task labels from the generated annotations.

## Install

Python 3.10 or newer is supported. The core install remains small; model
providers and their credentials are configured only when a run needs them.

```bash
pip install -e .
```

## Run VideoCap

Use the SiliconFlow example as a starting point. Set the API key in the shell,
then run a local JSONL video manifest:

```bash
export SILICONFLOW_API_KEY="..."
videocap run videos.jsonl \
  --config configs/videocap/siliconflow.json \
  --output-root runs
```

The example uses `Qwen/Qwen3.6-35B-A3B`, 24-second windows, 2 seconds of
overlap, 8 evidence frames, and 460-pixel frame height. The key is read from
the environment and is not written to run configuration files.

The final output is written under `pipeline_work/final/` using the configured
dataset name, for example `VDC_5.jsonl`. Intermediate JSONL and JSON artifacts
are retained under `pipeline_work/output/<video_id>/` so every stage can be
audited.

## Output Shape

Each final record represents one video. The top-level `captions` object contains
the five VDC dimensions: `short`, `main_object`, `background`, `camera`, and
`detailed`. The `events` array contains one object per accepted event, with its
`event_id`, `cluster_id`, `start_ms`, `end_ms`, `evidence_frames_ms`, and event
caption. A video can contain multiple events.

The window prompt implementation is in
`videocap/prompts/window_caption.py`. Its prompt pools are
inspired by AuroraCap's VDC benchmark:
https://github.com/wenhaochai/aurora. All five questions are sent in one VLM
request in the fixed order `short`, `main_object`, `background`, `camera`,
`detailed`, so evidence frames are not retransmitted five times.

## Evaluation

The built-in metrics are independent of VideoCap production. `temporal`
reports temporal IoU and proposal precision/recall/F1; `lexical-caption` is a
transparent token-overlap baseline; `soda` is an optional external-command
adapter. The `activitynet-captions` dataset adapter remains available for
metric development and validation.

```bash
videocap components
```

## Development

```bash
uv run pytest
```

The project keeps the public protocol and schemas small, while retaining every
model stage behind explicit adapter boundaries. Provider-specific credentials,
video caches, run outputs, and temporary files are excluded from the public
source tree.
