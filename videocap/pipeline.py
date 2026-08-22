"""VideoCap's fixed-window and event-grounded production flow."""

from __future__ import annotations

import importlib
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from videocap.io import atomic_write_json, atomic_write_jsonl
from videocap.protocols import DenseCaptionOutcome, DenseCaptionProducer, DenseCaptionSample
from videocap.schema import validate_document
from videocap.structured import (
    EventCandidate,
    EventCaption,
    EventCluster,
    EventWindow,
    ProcessingWindow,
    StructuredAdapterBundle,
    WindowCaption,
)


def _config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("pipeline_config", {})
    if not isinstance(value, Mapping):
        raise TypeError("pipeline_config must be an object")
    return value


def _load_symbol(path: str) -> Any:
    if ":" not in path:
        raise ValueError(f"adapter path must use module:attribute syntax: {path!r}")
    module_name, attribute = path.split(":", 1)
    if not module_name or not attribute:
        raise ValueError(f"adapter path must use module:attribute syntax: {path!r}")
    value: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        value = getattr(value, part)
    if isinstance(value, type):
        value = value()
    if not callable(value):
        raise TypeError(f"adapter {path!r} is not callable")
    return value


def _windows(duration_ms: int, max_duration_ms: int, overlap_ms: int, evidence_count: int) -> tuple[ProcessingWindow, ...]:
    if max_duration_ms <= 0:
        raise ValueError("max_duration_ms must be positive")
    if overlap_ms < 0 or overlap_ms >= max_duration_ms:
        raise ValueError("overlap_ms must be in [0, max_duration_ms)")
    if evidence_count <= 0:
        raise ValueError("evidence_frame_count must be positive")
    step = max_duration_ms - overlap_ms
    result: list[ProcessingWindow] = []
    start = 0
    index = 0
    while start < duration_ms:
        end = min(start + max_duration_ms, duration_ms)
        if end <= start:
            raise ValueError("fixed window generation produced an empty interval")
        if evidence_count == 1:
            evidence = (start + (end - start) // 2,)
        else:
            evidence = tuple(start + round(i * (end - start) / (evidence_count - 1)) for i in range(evidence_count))
        result.append(ProcessingWindow(f"pw_{index:04d}", start, end, evidence))
        if end == duration_ms:
            break
        start += step
        index += 1
    return tuple(result)


def _window_caption(window: ProcessingWindow, raw: Mapping[str, Any]) -> WindowCaption:
    if not isinstance(raw, Mapping):
        raise TypeError(f"window caption for {window.window_id} must be an object")
    captions = raw.get("captions")
    if not isinstance(captions, Mapping):
        raise ValueError(f"window caption for {window.window_id} must contain captions object")
    evidence = raw.get("evidence_frames_ms", list(window.evidence_frames_ms))
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise TypeError("window caption evidence_frames_ms must be an array")
    # Providers commonly serialize integral millisecond timestamps as JSON
    # floats. Accept only those lossless values; fractional timestamps remain
    # invalid so malformed temporal grounding is still visible.
    normalized = WindowCaption(
        window.window_id,
        captions,
        tuple(_integer(item, "evidence_frames_ms") for item in evidence),
    )
    if tuple(sorted(set(normalized.evidence_frames_ms))) != normalized.evidence_frames_ms:
        raise ValueError(f"window caption evidence for {window.window_id} must be sorted and unique")
    if any(timestamp < window.start_ms or timestamp > window.end_ms for timestamp in normalized.evidence_frames_ms):
        raise ValueError(f"window caption evidence for {window.window_id} is outside processing window")
    return normalized


def _candidate(raw: Mapping[str, Any]) -> EventCandidate:
    if not isinstance(raw, Mapping):
        raise TypeError("event candidate must be an object")
    source = raw.get("source_window_ids")
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise TypeError("event candidate source_window_ids must be an array")
    return EventCandidate(str(raw.get("candidate_id", "")), tuple(source), raw.get("description", ""))


def _cluster(raw: Mapping[str, Any]) -> EventCluster:
    if not isinstance(raw, Mapping):
        raise TypeError("event cluster must be an object")
    candidates = raw.get("candidate_ids")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("event cluster candidate_ids must be an array")
    return EventCluster(str(raw.get("cluster_id", "")), tuple(candidates))


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"{field_name} must be an integer")


def _event(raw: Mapping[str, Any], duration_ms: int, cluster_id: str, event_id: str) -> EventWindow:
    if not isinstance(raw, Mapping):
        raise TypeError("event window must be an object")
    evidence = raw.get("evidence_frames_ms", [])
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise TypeError("event window evidence_frames_ms must be an array")
    event = EventWindow(
        str(raw.get("event_id", event_id)),
        str(raw.get("cluster_id", cluster_id)),
        _integer(raw.get("start_ms"), "start_ms"),
        _integer(raw.get("end_ms"), "end_ms"),
        tuple(_integer(item, "evidence_frames_ms") for item in evidence),
    )
    event.validate(duration_ms)
    return event


class VideoCap(DenseCaptionProducer):
    name = "videocap"
    version = "0.1"

    def __init__(self, adapters: StructuredAdapterBundle | None = None) -> None:
        self.adapters = adapters

    def _adapters(self, cfg: Mapping[str, Any]) -> StructuredAdapterBundle:
        if self.adapters is not None:
            return self.adapters
        factory_path = cfg.get("adapter_factory")
        if isinstance(factory_path, str) and factory_path.strip():
            factory = _load_symbol(factory_path)
            bundle = factory(cfg)
            if not isinstance(bundle, StructuredAdapterBundle):
                raise TypeError("dense-video-annotation adapter_factory must return StructuredAdapterBundle")
            return bundle
        paths = cfg.get("adapters")
        if not isinstance(paths, Mapping):
            raise RuntimeError("dense-video-annotation requires pipeline_config.adapters")
        fields = {
            "caption_window": "caption_window",
            "generate_event_candidates": "generate_event_candidates",
            "cluster_event_candidates": "cluster_event_candidates",
            "propose_event_window": "propose_event_window",
            "review_event_boundary": "review_event_boundary",
            "refine_event_caption": "refine_event_caption",
            "merge_global_caption": "merge_global_caption",
            "quality_filter": "quality_filter",
        }
        loaded = {}
        for field, config_key in fields.items():
            value = paths.get(config_key)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(f"videocap stage '{config_key}' has no adapter")
            loaded[field] = _load_symbol(value)
        return StructuredAdapterBundle(**loaded)

    def produce(self, samples: Sequence[DenseCaptionSample], work_dir: Path, config: Mapping[str, Any]) -> Sequence[DenseCaptionOutcome]:
        cfg = _config(config)
        max_duration = cfg.get("max_duration_ms", 24_000)
        overlap = cfg.get("overlap_ms", 2_000)
        evidence_count = cfg.get("evidence_frame_count", 8)
        final_output_name = cfg.get("final_output_name", "annotations.jsonl")
        if not isinstance(final_output_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.jsonl", final_output_name):
            raise ValueError("pipeline_config.final_output_name must be a safe .jsonl filename")
        if isinstance(max_duration, bool) or not isinstance(max_duration, int):
            raise ValueError("pipeline_config.max_duration_ms must be an integer")
        if isinstance(overlap, bool) or not isinstance(overlap, int):
            raise ValueError("pipeline_config.overlap_ms must be an integer")
        if isinstance(evidence_count, bool) or not isinstance(evidence_count, int):
            raise ValueError("pipeline_config.evidence_frame_count must be an integer")
        adapters = self._adapters(cfg)
        for name in ("caption_window", "generate_event_candidates", "cluster_event_candidates", "propose_event_window", "review_event_boundary", "refine_event_caption", "merge_global_caption", "quality_filter"):
            adapters.require(name)
        output_dir = work_dir / "output"
        final_dir = work_dir / "final"
        work_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir()
        final_dir.mkdir()
        outcomes: list[DenseCaptionOutcome] = []
        final_records: list[Mapping[str, Any]] = []
        for sample in samples:
            started = time.perf_counter()
            sample_dir = output_dir / sample.video_id
            sample_dir.mkdir(parents=True, exist_ok=False)
            try:
                result = self._produce_one(sample, sample_dir, adapters, max_duration, overlap, evidence_count)
                artifact, metadata = result
                final_records.append(artifact)
                outcomes.append(DenseCaptionOutcome(sample.video_id, artifact=artifact, metadata=metadata, latency_sec=time.perf_counter() - started))
            except Exception as exc:
                atomic_write_json(sample_dir / "failure.json", {"error_type": type(exc).__name__, "message": str(exc)})
                outcomes.append(DenseCaptionOutcome(sample.video_id, error_type=type(exc).__name__, message=str(exc), metadata={"stage_failure": True}, latency_sec=time.perf_counter() - started))
        atomic_write_jsonl(final_dir / final_output_name, final_records)
        return tuple(outcomes)

    def _produce_one(self, sample: DenseCaptionSample, directory: Path, adapters: StructuredAdapterBundle, max_duration: int, overlap: int, evidence_count: int) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        windows = _windows(sample.duration_ms, max_duration, overlap, evidence_count)
        atomic_write_jsonl(directory / "processing_windows.jsonl", (window.to_dict() for window in windows))
        captions = tuple(_window_caption(window, adapters.require("caption_window")(sample, window)) for window in windows)
        atomic_write_jsonl(directory / "window_captions.jsonl", (item.to_dict() for item in captions))
        raw_candidates = adapters.require("generate_event_candidates")(sample, windows, captions)
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)) or not raw_candidates:
            raise ValueError("event candidate generation returned no candidates")
        candidates = tuple(_candidate(item) for item in raw_candidates)
        candidate_ids = [item.candidate_id for item in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("event candidate IDs must be unique")
        window_ids = {window.window_id for window in windows}
        if any(source not in window_ids for candidate in candidates for source in candidate.source_window_ids):
            raise ValueError("event candidate references an unknown processing window")
        atomic_write_jsonl(directory / "event_candidates.jsonl", (item.to_dict() for item in candidates))
        raw_clusters = adapters.require("cluster_event_candidates")(sample, candidates)
        if not isinstance(raw_clusters, Sequence) or isinstance(raw_clusters, (str, bytes)) or not raw_clusters:
            raise ValueError("event clustering returned no clusters")
        clusters = tuple(_cluster(item) for item in raw_clusters)
        cluster_ids = [item.cluster_id for item in clusters]
        if len(set(cluster_ids)) != len(cluster_ids):
            raise ValueError("event cluster IDs must be unique")
        candidate_id_set = set(candidate_ids)
        clustered_ids = [candidate_id for cluster in clusters for candidate_id in cluster.candidate_ids]
        if any(candidate_id not in candidate_id_set for candidate_id in clustered_ids):
            raise ValueError("event cluster references an unknown candidate")
        if set(clustered_ids) != candidate_id_set or len(clustered_ids) != len(candidate_id_set):
            raise ValueError("event clusters must cover each candidate exactly once")
        atomic_write_jsonl(directory / "event_clusters.jsonl", (item.to_dict() for item in clusters))
        proposed: list[EventWindow] = []
        for index, cluster in enumerate(clusters):
            raw = adapters.require("propose_event_window")(sample, cluster, candidates, windows)
            proposed.append(_event(raw, sample.duration_ms, cluster.cluster_id, f"event_{index:04d}"))
        atomic_write_jsonl(directory / "coarse_event_windows.jsonl", (item.to_dict() for item in proposed))
        reviewed: list[EventWindow] = []
        for event in proposed:
            raw = adapters.require("review_event_boundary")(sample, event)
            if not isinstance(raw, Mapping):
                raise TypeError(f"boundary review for {event.event_id} must return an object")
            raw_events = raw.get("events")
            if raw_events is None:
                raw_events = [raw]
            if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)) or not raw_events:
                raise ValueError(f"boundary review for {event.event_id} returned no events")
            for split_index, item in enumerate(raw_events):
                reviewed.append(_event(item, sample.duration_ms, event.cluster_id, f"{event.event_id}_{split_index:02d}"))
        if not reviewed:
            raise ValueError("boundary review returned no event windows")
        atomic_write_jsonl(directory / "boundary_reviews.jsonl", (item.to_dict() for item in reviewed))
        event_captions: list[EventCaption] = []
        for event in reviewed:
            raw = adapters.require("refine_event_caption")(sample, event, captions)
            if not isinstance(raw, Mapping):
                raise TypeError(f"event refine for {event.event_id} must return an object")
            event_captions.append(EventCaption(event, raw.get("captions", {})))
        atomic_write_jsonl(directory / "event_captions.jsonl", (item.to_dict() for item in event_captions))
        raw_global_captions = adapters.require("merge_global_caption")(sample, tuple(event_captions))
        if not isinstance(raw_global_captions, Mapping):
            raise TypeError("global caption merge must return a five-dimension object")
        global_captions = dict(WindowCaption("global", raw_global_captions, ()).captions)
        quality = adapters.require("quality_filter")(sample, tuple(event_captions), global_captions)
        if not isinstance(quality, Mapping):
            raise TypeError("quality filter must return an object")
        accepted = quality.get("accepted_event_ids")
        if not isinstance(accepted, Sequence) or isinstance(accepted, (str, bytes)):
            raise ValueError("quality filter must return accepted_event_ids array")
        accepted_ids = set(accepted)
        event_ids = {item.event.event_id for item in event_captions}
        if not accepted_ids.issubset(event_ids):
            raise ValueError("quality filter references an unknown event")
        atomic_write_json(directory / "global_caption.json", {"captions": global_captions})
        atomic_write_json(directory / "quality_report.json", dict(quality))
        structured_result = {
            "schema_version": "videocap/v0.1",
            "video_id": sample.video_id,
            "duration_ms": sample.duration_ms,
            "captions": global_captions,
            "events": [
                {
                    **item.event.to_dict(),
                    "caption": item.captions["detailed"],
                }
                for item in event_captions
                if item.event.event_id in accepted_ids
            ],
            "quality": dict(quality),
        }
        validate_document(structured_result, "videocap")
        metadata = {"processing_windows": [item.to_dict() for item in windows], "window_captions": [item.to_dict() for item in captions], "event_candidates": [item.to_dict() for item in candidates], "event_clusters": [item.to_dict() for item in clusters], "coarse_event_windows": [item.to_dict() for item in proposed], "boundary_reviews": [item.to_dict() for item in reviewed], "event_captions": [item.to_dict() for item in event_captions], "global_captions": global_captions, "quality": dict(quality)}
        return structured_result, metadata
