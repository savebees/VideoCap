# Dense Video Annotator

## Goal

Automatically generate multi-level dense annotations for short videos (3-120s), outputting structured JSON. Annotations cover four dimensions: scene segmentation, dense captioning, atomic events, and object detection.

## Annotation Structure

Each video produces one `annotation.json`:

```
video_id / duration / fps / resolution / codec / file_size_mb
segments[]
  |- scene_id, start, end, brief       <- scene boundaries + summary
  |- description, word_count            <- dense caption (>=100 words)
  |- events[]                           <- atomic events
  |     +-- event_id, start, end, description
  +-- objects[]                         <- object detections
        +-- object_id, label, bbox, confidence, frame_index
metadata
  +-- pipeline_version, models, statistics, repair counts
```

Output schema: `schemas/annotation.schema.json`.

## 7-Step Pipeline

| Step | Type | Input | What It Does | Cache File |
|------|------|-------|--------------|------------|
| 0 | ffprobe + ffmpeg | Video file | Extract metadata + sample frames at target fps | `metadata.json` + `frames/` |
| 1 | VLM (single call) | All frames | Segment video into scene segments | `step1_segments.json` |
| 2 | Programmatic | Step 1 output | Boundary repair: snap endpoints, fix gaps/overlaps, merge short segments | `step2_validated.json` |
| 3 | VLM (per-segment) | Segment frames + prefix context | Dense captioning, auto-retry if < 100 words | `step3_captioned.json` |
| 4 | VLM (per-segment) | Segment frames + description | Subdivide each scene into atomic events | `step4_events.json` |
| 5 | Programmatic | Step 4 output | Event boundary repair, same logic as step 2 | `step5_events_validated.json` |
| 6 | VLM + Grounding DINO | Description + frame images | Extract noun phrases from caption -> per-frame detection -> cross-frame NMS dedup | `step6_objects.json` |

Each step has its own cache file for resumability. Use `--force` to clear cache and re-run.

## Models

| Model | Purpose | Serving |
|-------|---------|---------|
| Qwen2.5-VL-32B-Instruct | Scene segmentation, captioning, event segmentation, object phrase extraction | vLLM (OpenAI-compatible API) |
| Grounding DINO (Swin-T) | Open-set object detection | Local PyTorch inference |

## Directory Structure

```
configs/default.yaml          Single config file
scripts/start_vllm.sh         vLLM launch script (env vars for different machines)
prompts.py                    5 prompt templates
preprocess/preprocess.py      Step 0
src/
  pipeline.py                 Main orchestration + CLI entry point
  scene_segmentation.py       Steps 1-3
  atomic_events.py            Steps 4-5
  object_detection.py         Step 6
utils/
  video.py                    ffprobe metadata + JSON parsing
  frames.py                   ffmpeg frame extraction + base64 encoding
  vllm_client.py              OpenAI client wrapper
schemas/annotation.schema.json
data/videos/                  Input videos
data/output/{video_id}/       Per-video output directory
```

## Key Design Decisions

**Prefix Context**: In step 3, the previous N scene descriptions are prepended to the prompt to avoid redundant descriptions of the same characters/settings. Adapted from ShareGPT4Video.

**Boundary Repair** (Steps 2 & 5): VLM-generated timestamps are imprecise. Programmatic repair includes: endpoint snapping, gap/overlap fixing (< 0.5s gaps use midpoint, otherwise align forward), and merging short segments. All float comparisons use 1e-6 tolerance. Repair counts are persisted with cache so they are not lost on cache hits.

**Object Detection Pipeline**: First uses a text-only VLM call to extract noun phrases from the scene description, then runs Grounding DINO on all 1fps frames, and finally applies cross-frame NMS (same label + IoU > threshold -> keep highest confidence) for deduplication.

**Frame Encoding Format**: `data:video/jpeg;base64,{b64_frame1},{b64_frame2},...` wrapped in `{"type": "video_url", "video_url": {"url": ...}}`. This is the standard vLLM format for Qwen2.5-VL.

**JSON Parsing**: VLM output may contain markdown code fences or trailing text. `parse_json_array()` strips code fence markers and extracts the first complete JSON array using bracket balancing.

## Server Deployment

- Platform: AutoDL (A800-80GB x2)
- Model storage: `/root/autodl-tmp/models/`
- Project directory: `/root/dense-video-annotator/`
- Data directory: symlink `data/ -> /root/autodl-tmp/data/`
- Grounding DINO: symlink `models/grounding_dino/ -> /root/autodl-tmp/models/AI-ModelScope/GroundingDINO/`
- Start vLLM: `TP_SIZE=2 GPU_DEVICES=0,1 bash scripts/start_vllm.sh`
- Local and server code are identical, synced via rsync (excluding ref/, data/, models/Qwen*)

## Usage

```bash
# Single video
python src/pipeline.py --video data/videos/example.mp4

# Batch
python src/pipeline.py --video_dir data/videos

# Force re-run
python src/pipeline.py --video data/videos/example.mp4 --force

# Custom config
python src/pipeline.py --video data/videos/example.mp4 --config configs/default.yaml
```

## Config Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| video_fps | 1.0 | Frame extraction rate |
| frame_max_long_side | 672 | Max long side of extracted frames |
| min_segment_duration | 3.0 | Minimum scene duration (seconds); shorter segments are merged |
| min_event_duration | 1.0 | Minimum event duration (seconds) |
| prefix_context_num | 3 | Number of previous scenes in step 3 prefix context |
| min_description_words | 100 | Minimum word count for dense captions |
| detection_box_threshold | 0.35 | Grounding DINO box confidence threshold |
| detection_text_threshold | 0.25 | Grounding DINO text matching threshold |
| detection_nms_threshold | 0.5 | Cross-frame NMS IoU threshold |
| keyframe_strategy | "all" | Detection strategy: "all" = every frame, "middle" = middle frame only |
| max_retries | 3 | Max VLM call retries |
| vlm_temperature | 0.0 | VLM temperature |
