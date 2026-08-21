# Dense Video Annotator

Dense Video Annotator is a production-oriented system for generating and
evaluating dense video captions. It connects video datasets to mature captioning
recipes, normalizes their outputs, and measures caption quality through
independent metrics. The current focus is caption production and evaluation;
question answering and other semantic annotations are downstream extensions.

## Install

Python 3.10+ is required. Both uv and ordinary pip installations are supported:

```bash
uv sync
uv run dva components
python -m pip install -e '.[dev]'
```

The lockfile is checked in and CI exercises Python 3.10, 3.11, and 3.12.

## Dense-caption protocol

Every recipe emits the same small, model-independent protocol:

```json
{"schema_version":"dense-caption/v0.1","video_id":"video_001","duration_ms":173000,"captions":[{"caption_id":"c_0001","start_ms":12400,"end_ms":26700,"text":"A woman places a red cup beside the laptop.","evidence_frames_ms":[]}]}
```

Intervals may overlap or leave gaps. Recipe/model provenance, quality flags, and
execution metadata stay in run artifacts rather than polluting predictions.

## Industrial production recipe

The first production integration is `cosmos-curator`, an adapter around
[NVIDIA Cosmos Curator](https://github.com/NVIDIA/cosmos-curator). Cosmos Curator
performs the real batch pipeline—video discovery, shot splitting, transcoding,
windowed VLM captioning, optional filters, caption-quality flags, and failure
recovery. This system invokes its official configuration entry point once for
the whole dataset and converts its per-clip metadata into
`dense-caption/v0.1`. No Cosmos Curator source is copied here.

A local JSONL manifest is the default dataset entrance:

```json
{"video_id":"video_001","video_path":"/data/videos/video_001.mp4","duration_ms":173000}
```

Run it from an official Cosmos Curator Pixi/container environment in which this
package is also installed. Supply your own recipe configuration and dataset
manifest:

```bash
dva run videos.jsonl --recipe cosmos-curator \
  --config /path/to/recipe.json --output-root runs
```

The package supports Python 3.10+, while the audited Cosmos Curator environment
is separate and currently uses its official Pixi/container stack (Python 3.13,
vLLM 0.24, Torch 2.11, CUDA 13). An older local VLM environment is not assumed
to be compatible.

The large environment is installed only on demand:

```bash
uv run dva env check cosmos-curator
uv run dva env install cosmos-curator --path /data/envs/dense-video-cosmos
```

Conda is not a prerequisite. The command downloads a fixed Pixi executable and
the audited Cosmos Curator revision into the selected cache path, then creates
the environment from the upstream `pixi.toml`. This keeps the normal package
install small and avoids modifying shell profiles. Existing official Pixi or
Conda-managed environments can still be used by setting
`recipe_config.environment_path`, and teams that run Cosmos elsewhere can use
`import` mode without installing the GPU stack locally.

Users who only validate datasets, evaluate imported outputs, or use another
recipe do not need Pixi, CUDA, Torch, or vLLM installed by this project.

If Cosmos Curator is executed separately through a cluster, cloud job, or
container, set `recipe_config.mode` to `import` and point `output_path` at its
locally available output directory. Detailed configuration and extension
boundaries are in [docs/recipes.md](docs/recipes.md).

## Strict structured grounding

For long videos where a processing chunk must not be mistaken for an event, the
`structured-grounding` recipe executes fixed processing windows, five-dimensional
VLM captions, adjacent-window event candidates, semantic clustering, coarse event
windows, dense boundary review, event refinement, global consistency merging, and
quality filtering. Every stage is an explicit adapter and every intermediate
stage artifact is retained under `recipe_work/output/<video_id>/`, while one
final JSONL record per successful video is written to a dataset-named file such
as `recipe_work/final/VDC_5.jsonl` or `recipe_work/final/VDC_1k.jsonl`. Missing adapters, empty
collections, malformed dimensions, and invalid intervals fail visibly; there is
no fallback to chunk captions. The final record contains an `events` array, so a
video with several events is represented by several objects linked by their
unique `event_id`.
The top-level `captions` object keeps the five VDC dimensions (`short`,
`main_object`, `background`, `camera`, `detailed`); each event carries its
temporal evidence and one event-level `caption`.

The tested Vyce model pairing is `nemotron-vision` for visual stages and
`gpt-5.6` for text reasoning stages. Use a recipe configuration containing this
provider setup and provide `VYCE_API_KEY` in the process environment. The provider must be able
to fetch every evidence image or supported video URL from `frame_url`,
`frame_url_template`, `media_url`, or `media_url_template`; a local
filesystem path and a base64 data URI are not silently treated as visual input.

## Evaluation

The built-in `temporal` metric reports mean temporal IoU, greedy proposal
precision/recall/F1, and F1 at tIoU 0.3, 0.5, and 0.7. `lexical-caption` is a
transparent token-overlap wiring baseline, not a semantic quality claim. `soda`
is an optional external-command adapter, so the third-party implementation and
license remain managed upstream. Multiple references are supported and each
metric remains independent from the production recipe.

The `activitynet-captions` adapter remains available for compatibility and metric
development, but it is not the planned representative release test dataset. It
can validate official ActivityNet Captions files without running production:

```bash
dva activitynet-validate val_1.json val_2.json --video-manifest videos.json
```

## Standard artifacts

Every run writes `predictions.jsonl`, `per_sample.jsonl`, `summary.json`,
`summary.csv`, `failures.jsonl`, `run_manifest.json`, and
`resolved_config.json`. The manifest records task and recipe versions, dataset
fingerprint, configuration hash, seed, and Git state. Batch-level working files
and upstream logs are retained in `recipe_work/` for diagnosis and resumption.

The implementation is under [dense_video_annotator/](dense_video_annotator/).
Recipe details and extension boundaries are documented in
[docs/recipes.md](docs/recipes.md).

## License

MIT. See [LICENSE](LICENSE). External recipes and metrics retain their own
licenses and citation requirements.
