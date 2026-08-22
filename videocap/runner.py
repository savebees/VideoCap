"""One reproducible dense-caption production and evaluation run."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from videocap.builtins import load_builtin_components
from videocap.contracts import ComponentRef, DatasetRef, RunManifest
from videocap.io import atomic_write_csv, atomic_write_json, atomic_write_jsonl
from videocap.pipeline import VideoCap
from videocap.protocols import (
    DenseCaptionDataset,
    DenseCaptionMetric,
    DenseCaptionOutcome,
    DenseCaptionProducer,
)
from videocap.registry import METRICS, TASKS
from videocap.schema import validate_document


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    manifest: RunManifest
    summary: Mapping[str, Any]


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("configuration object keys must be strings")
            normalized[key] = _normalize_config_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("configuration floats must be finite")
        return value
    raise TypeError(f"configuration value is not reproducibly serializable: {type(value).__name__}")


def _canonical_config(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_config(config).encode("utf-8")).hexdigest()


def _redact_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: (
                "<redacted>"
                if key.lower() in {"api_key", "apikey", "token", "password", "secret"}
                else _redact_config(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    return value


def _git_state() -> tuple[str | None, bool]:
    repository = Path(__file__).resolve().parents[1]
    commit_result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_result.returncode != 0:
        return None, False
    status_result = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return commit_result.stdout.strip(), bool(status_result.stdout.strip())


def _new_run_id(task_name: str, pipeline_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{task_name}__{pipeline_name}__{timestamp}__{uuid.uuid4().hex[:8]}"


def _flatten_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "run_id": summary["run_id"],
        "n_samples": summary["n_samples"],
        "n_succeeded": summary["n_succeeded"],
        "n_failed": summary["n_failed"],
        "run_duration_sec": summary["run_duration_sec"],
    }
    for metric_name, values in summary["metrics"].items():
        for key, value in values.items():
            row[f"{metric_name}/{key}"] = value
    return row


def run_dense_caption(
    dataset: DenseCaptionDataset,
    output_root: str | Path,
    *,
    task_name: str = "temporal-dense-caption",
    producer: DenseCaptionProducer | None = None,
    metric_names: Sequence[str] = ("exact-structure",),
    config: Mapping[str, Any] | None = None,
    seed: int = 0,
    run_id: str | None = None,
) -> RunResult:
    load_builtin_components()
    task = TASKS.create(task_name)
    producer = producer or VideoCap()
    metrics: list[DenseCaptionMetric] = [METRICS.create(name) for name in metric_names]  # type: ignore[list-item]
    run_config = _normalize_config_value(dict(config or {}))
    samples = tuple(dataset)
    sample_by_id = {sample.video_id: sample for sample in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError("dataset contains duplicate video_id values")

    resolved_run_id = run_id or _new_run_id(task.name, producer.name)
    run_dir = Path(output_root).expanduser().resolve() / resolved_run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    git_commit, git_dirty = _git_state()
    manifest = RunManifest(
        run_id=resolved_run_id,
        created_at=created_at,
        task=ComponentRef(task.name, task.version),
        pipeline=ComponentRef(producer.name, producer.version),
        dataset=DatasetRef(dataset.name, dataset.version, dataset.fingerprint()),
        config_sha256=_config_sha256(run_config),
        seed=seed,
        git_dirty=git_dirty,
        git_commit=git_commit,
    )
    validate_document(manifest.to_dict(), "run_manifest")
    atomic_write_json(run_dir / "resolved_config.json", _redact_config(run_config))
    atomic_write_json(run_dir / "run_manifest.json", manifest.to_dict())

    predictions: list[Mapping[str, Any]] = []
    artifacts: list[Mapping[str, Any]] = []
    n_succeeded = 0
    per_sample: list[Mapping[str, Any]] = []
    failures: list[Mapping[str, Any]] = []
    metric_samples: dict[str, list[Mapping[str, float]]] = {
        metric.name: [] for metric in metrics
    }
    run_started = time.perf_counter()
    work_dir = run_dir / "pipeline_work"
    try:
        raw_outcomes = producer.produce(samples, work_dir, run_config)
        outcomes: dict[str, DenseCaptionOutcome] = {}
        for outcome in raw_outcomes:
            if not isinstance(outcome, DenseCaptionOutcome):
                raise TypeError("pipeline outputs must be DenseCaptionOutcome instances")
            if outcome.video_id not in sample_by_id:
                raise ValueError(f"pipeline returned unknown video_id: {outcome.video_id}")
            if outcome.video_id in outcomes:
                raise ValueError(f"pipeline returned duplicate video_id: {outcome.video_id}")
            outcomes[outcome.video_id] = outcome
    except Exception as exc:
        outcomes = {
            sample.video_id: DenseCaptionOutcome(
                video_id=sample.video_id,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            for sample in samples
        }

    for sample in samples:
        outcome = outcomes.get(sample.video_id)
        if outcome is None:
            outcome = DenseCaptionOutcome(
                video_id=sample.video_id,
                error_type="MissingPipelineOutput",
                message="pipeline returned no outcome for this video",
            )
        production_metadata = _normalize_config_value(dict(outcome.metadata))
        latency_sec = (
            round(outcome.latency_sec, 6) if outcome.latency_sec is not None else None
        )
        if outcome.artifact is not None:
            try:
                validate_document(outcome.artifact, "videocap")
                artifacts.append(dict(outcome.artifact))
                n_succeeded += 1
                per_sample.append(
                    {
                        "video_id": sample.video_id,
                        "status": "succeeded",
                        "latency_sec": latency_sec,
                        "metrics": {},
                        "production": production_metadata,
                    }
                )
            except Exception as exc:
                failures.append({"video_id": sample.video_id, "error_type": type(exc).__name__, "message": str(exc)})
                per_sample.append({"video_id": sample.video_id, "status": "failed", "latency_sec": latency_sec, "metrics": {}, "production": production_metadata})
            continue

        if outcome.prediction is None:
            failure = {
                "video_id": sample.video_id,
                "error_type": outcome.error_type,
                "message": outcome.message,
            }
            failures.append(failure)
            per_sample.append(
                {
                    "video_id": sample.video_id,
                    "status": "failed",
                    "latency_sec": latency_sec,
                    "metrics": {},
                    "production": production_metadata,
                }
            )
            continue

        try:
            prediction = outcome.prediction
            if prediction.video_id != sample.video_id:
                raise ValueError(
                    f"prediction video_id {prediction.video_id!r} does not match "
                    f"sample {sample.video_id!r}"
                )
            task.validate(prediction)
            serialized = prediction.to_dict()

            scores: dict[str, Mapping[str, float]] = {}
            if sample.references:
                for metric in metrics:
                    result = dict(metric.score_references(prediction, sample.references))
                    scores[metric.name] = result
            for metric_name, result in scores.items():
                metric_samples[metric_name].append(result)
            predictions.append(serialized)
            n_succeeded += 1
            per_sample.append(
                {
                    "video_id": sample.video_id,
                    "status": "succeeded",
                    "latency_sec": latency_sec,
                    "metrics": scores,
                    "production": production_metadata,
                }
            )
        except Exception as exc:  # one bad sample must be visible without hiding the rest
            failure = {
                "video_id": sample.video_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            failures.append(failure)
            per_sample.append(
                {
                    "video_id": sample.video_id,
                    "status": "failed",
                    "latency_sec": latency_sec,
                    "metrics": {},
                    "production": production_metadata,
                }
            )

    aggregated = {
        metric.name: dict(metric.aggregate(metric_samples[metric.name]))
        for metric in metrics
        if metric_samples[metric.name]
    }
    summary = {
        "run_id": resolved_run_id,
        "n_samples": len(samples),
        "n_succeeded": n_succeeded,
        "n_failed": len(failures),
        "run_duration_sec": round(time.perf_counter() - run_started, 6),
        "metrics": aggregated,
    }
    atomic_write_jsonl(run_dir / "predictions.jsonl", predictions)
    if artifacts:
        atomic_write_jsonl(run_dir / "artifacts.jsonl", artifacts)
    atomic_write_jsonl(run_dir / "per_sample.jsonl", per_sample)
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_csv(run_dir / "summary.csv", _flatten_summary(summary))
    atomic_write_jsonl(run_dir / "failures.jsonl", failures)
    return RunResult(run_dir=run_dir, manifest=manifest, summary=summary)
