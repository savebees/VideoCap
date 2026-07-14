# Caption Evaluation (`eval/`)

Compares the pipeline against every baseline over all clips of a dataset.
Reads their **output files only**; errors raise. Comparison unit: one
`(split, video_id)` clip × one system → one plain-text caption.

- **pipeline**: `annotation.json` assembled as dense captions + action sentences
  in time order, actions globally deduplicated (`systems._assemble_pipeline`).
- **a3b / model baselines**: the `caption` field of their output JSON.
- All captions pass through the shared `systems.normalize_caption`.

## Metrics

| column | meaning |
|---|---|
| `precision` | supported / (supported + contradicted), pooled. `not_verifiable` excluded |
| `recall` | covered salient items / all salient items |
| `F1` | harmonic mean of precision and recall — the headline number |
| `HR_faithfulness` | mean per-video hallucination rate (contradicted share) |
| `OR_omission` | mean per-video omission rate |
| `sup/con_per_video` | mean supported / contradicted claims per video |
| `not_verifiable_rate` | claims the frames can neither confirm nor deny |
| richness | length, below-floor ratio, corpus vocab, POS ratios, TTR (spaCy) |
| `clip_sim` | CLIP ViT-L/14 caption↔video cosine (`clip_score.py`, separate pass) |
| `metaclip2_sim` | Meta CLIP 2 ViT-H/14 cosine (`metaclip2_score.py`, optional) |

Flow per clip: judge builds a caption-independent salient list (shared across
systems); per system the text LLM decomposes the caption into atomic typed
claims, the visual judge verdicts each claim against 48 uniform frames,
negation claims get a separate denser pass, the text LLM scores salient
coverage, spaCy scores richness.

## Judge

Both the visual judge and the text steps use **Gemma3-27B-it** — an independent
family from every system under test (Qwen / InternVL / LLaVA), so no
self-preference. Two replicas so text never contends with the judge:
`serve_judge.sh` (:8010, GPUs 0-3) and `serve_text.sh` (:8011, GPUs 4-7).

Gemma3 emits pseudo-JSON under pressure; `llm_client.parse_json_strict` and
small claim batches (`judge_claim_batch`) recover per-item, never per-unit.

## Known limitation: temporal scope

Segment captions are concatenated without their time ranges, so a sentence true
only within its segment can be judged false against the whole clip. Measured
cost: ~9% of pipeline contradictions on NWPU, ~40% on Physics-IQ (every clip
there has a static phase). Kept as-is for methodology stability; caption
prompts avoid stillness assertions to reduce it.

## Running

```bash
# NWPU (+ youtube)
bash eval/serve_judge.sh &   # then serve_text.sh
python eval/run_eval.py --config eval/config.yaml

# Physics-IQ (checks completeness, serves judges, evals, tears down)
bash eval/run_physics_iq.sh

# CLIP column (GPU or CPU; independent of the judge run)
python eval/clip_score.py --config <cfg> --model-path <clip-vit-large-patch14>
python eval/clip_score.py --config <cfg> --model-path <...> --merge
```

`run_eval.py` needs spacy (pins numpy<2) — keep it in a separate env from the
vLLM servers (numpy 2.x) and let them talk over HTTP; see `run_physics_iq.sh`.

## Outputs & caching

Metrics land in `results/{dataset}/metrics/`: `summary.csv` (rows = systems)
plus `{system}/per_video.jsonl` with per-claim verdicts and evidence.

Every `(system, split, video_id)` unit is cached atomically under `eval/cache/`
(`_complete` sentinel; partial files recompute). Reruns skip finished units;
adding a system computes only its units; `--force` recomputes everything.
Judge frames are cached per `(split, video_id, n)` and shared across systems.
