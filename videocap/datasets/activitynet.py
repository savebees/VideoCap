"""ActivityNet Captions adapter for official and converted annotation files."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from videocap.contracts import DenseCaptionPrediction, TemporalCaption
from videocap.protocols import DenseCaptionDataset, DenseCaptionSample
from videocap.registry import DATASETS


def _milliseconds(seconds: Any, field: str) -> int:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError(f"{field} must be numeric seconds")
    if seconds < 0:
        raise ValueError(f"{field} must be non-negative")
    return int(round(float(seconds) * 1000))


def _annotation_paths(source: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
    values = (source,) if isinstance(source, (str, Path)) else tuple(source)
    if not values:
        raise ValueError("at least one ActivityNet annotation file is required")
    paths = tuple(Path(value).expanduser().resolve() for value in values)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"ActivityNet annotations do not exist: {path}")
    return paths


def _split_matches(requested: str | None, actual: str | None) -> bool:
    if not requested:
        return True
    requested = requested.lower()
    actual = (actual or "").lower()
    aliases = {"val": "validation", "validate": "validation"}
    requested = aliases.get(requested, requested)
    actual = aliases.get(actual, actual)
    return actual == requested or (requested == "validation" and actual.startswith("val_"))


@DATASETS.register("activitynet-captions")
class ActivityNetCaptionsDataset(DenseCaptionDataset):
    """Stream ActivityNet Captions samples, merging references across input files.

    Official files are direct video mappings whose records contain ``duration``,
    ``timestamps`` and ``sentences``. A converted ``database`` representation with
    ``annotations`` is accepted as well. Passing both ``val_1.json`` and
    ``val_2.json`` creates two references for each shared video.
    """

    name = "activitynet-captions"
    version = "0.2"

    def __init__(
        self,
        annotations: str | Path | Sequence[str | Path],
        *,
        video_manifest: str | Path | Mapping[str, str] | None = None,
        split: str | None = None,
        video_root: str | Path | None = None,
        require_video_files: bool = True,
        max_samples: int | None = None,
    ) -> None:
        self.annotation_paths = _annotation_paths(annotations)
        self.split = split
        self.video_root = (
            Path(video_root).expanduser().resolve()
            if video_root
            else self.annotation_paths[0].parent
        )
        self.require_video_files = require_video_files
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.max_samples = max_samples
        self.video_manifest = self._load_manifest(video_manifest)
        self._references = self._load_references()

    @staticmethod
    def _load_manifest(source: str | Path | Mapping[str, str] | None) -> dict[str, str]:
        if source is None:
            return {}
        if isinstance(source, Mapping):
            return {str(key): str(value) for key, value in source.items()}
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"video manifest does not exist: {path}")
        if path.suffix.lower() == ".jsonl":
            result: dict[str, str] = {}
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not {"video_id", "video_path"} <= record.keys():
                    raise ValueError(f"invalid video manifest record at {path}:{line_number}")
                result[str(record["video_id"])] = str(record["video_path"])
            return result
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("video manifest JSON must map video ids to paths")
        return {str(key): str(value) for key, value in document.items()}

    def _load_references(self) -> dict[str, list[tuple[DenseCaptionPrediction, str]]]:
        references: dict[str, list[tuple[DenseCaptionPrediction, str]]] = defaultdict(list)
        for path in self.annotation_paths:
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError(f"ActivityNet annotation root must be an object: {path}")
            database = document.get("database", document)
            if not isinstance(database, dict):
                raise ValueError(f"ActivityNet database must be an object: {path}")
            inferred_split = path.stem.lower()
            for raw_video_id, record in database.items():
                if not isinstance(record, dict):
                    raise ValueError(f"record {raw_video_id!r} in {path} must be an object")
                actual_split = record.get("subset") or inferred_split
                if not _split_matches(self.split, str(actual_split)):
                    continue
                video_id = str(raw_video_id)
                prediction = self._parse_reference(video_id, record, path)
                references[video_id].append((prediction, path.name))
        return dict(references)

    @staticmethod
    def _parse_reference(
        video_id: str,
        record: Mapping[str, Any],
        source: Path,
    ) -> DenseCaptionPrediction:
        duration_ms = _milliseconds(record.get("duration"), f"{video_id}.duration")
        if duration_ms <= 0:
            raise ValueError(f"{video_id}.duration must be positive")

        if "timestamps" in record or "sentences" in record:
            intervals = record.get("timestamps")
            sentences = record.get("sentences")
            if not isinstance(intervals, list) or not isinstance(sentences, list):
                raise ValueError(f"{video_id} timestamps and sentences must be arrays")
            if len(intervals) != len(sentences):
                raise ValueError(f"{video_id} timestamps and sentences must have equal length")
            raw_annotations = [
                {"segment": interval, "sentence": sentence}
                for interval, sentence in zip(intervals, sentences)
            ]
        else:
            raw_annotations = record.get("annotations")
            if not isinstance(raw_annotations, list):
                raise ValueError(
                    f"{video_id} in {source} must contain timestamps/sentences or annotations"
                )

        captions: list[TemporalCaption] = []
        for index, annotation in enumerate(raw_annotations, 1):
            if not isinstance(annotation, dict):
                raise ValueError(f"{video_id} annotation {index} must be an object")
            segment = annotation.get("segment")
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                raise ValueError(f"{video_id} annotation {index} segment must be [start, end]")
            start_ms = _milliseconds(segment[0], f"{video_id}.segment.start")
            end_ms = min(_milliseconds(segment[1], f"{video_id}.segment.end"), duration_ms)
            if end_ms <= start_ms:
                raise ValueError(f"{video_id} annotation {index} has an empty segment")
            text = annotation.get("sentence") or annotation.get("caption")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{video_id} annotation {index} sentence must be non-empty")
            captions.append(
                TemporalCaption(f"c_{index:04d}", start_ms, end_ms, text.strip())
            )
        return DenseCaptionPrediction(video_id, duration_ms, tuple(captions))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in self.annotation_paths:
            content = path.read_bytes()
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        digest.update(json.dumps(self.video_manifest, sort_keys=True).encode("utf-8"))
        digest.update((self.split or "").encode("utf-8"))
        digest.update(str(self.max_samples or "").encode("ascii"))
        return digest.hexdigest()

    def __iter__(self) -> Iterator[DenseCaptionSample]:
        video_ids = sorted(self._references)
        if self.max_samples is not None:
            video_ids = video_ids[: self.max_samples]
        for video_id in video_ids:
            entries = self._references[video_id]
            references = tuple(prediction for prediction, _ in entries)
            duration_ms = references[0].duration_ms
            if any(reference.duration_ms != duration_ms for reference in references[1:]):
                raise ValueError(f"references disagree on duration for {video_id}")
            raw_path = self.video_manifest.get(video_id, f"{video_id}.mp4")
            video_path = Path(raw_path).expanduser()
            if not video_path.is_absolute():
                video_path = (self.video_root / video_path).resolve()
            if self.require_video_files and not video_path.is_file():
                raise ValueError(f"video does not exist: {video_path}")
            yield DenseCaptionSample(
                video_id=video_id,
                video_path=video_path,
                duration_ms=duration_ms,
                references=references,
                metadata={
                    "source": "activitynet-captions",
                    "reference_files": [source for _, source in entries],
                },
            )
