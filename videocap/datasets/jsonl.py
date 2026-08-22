"""Local, streaming JSONL dataset adapter for videos and optional references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from videocap.contracts import DenseCaptionPrediction
from videocap.protocols import DenseCaptionDataset, DenseCaptionSample
from videocap.registry import DATASETS


@DATASETS.register("local-jsonl")
class LocalJSONLDataset(DenseCaptionDataset):
    name = "local-jsonl"
    version = "0.1"

    def __init__(self, path: str | Path, *, require_video_files: bool = True) -> None:
        self.path = Path(path).expanduser().resolve()
        self.require_video_files = require_video_files
        if not self.path.is_file():
            raise FileNotFoundError(f"JSONL dataset does not exist: {self.path}")

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def __iter__(self) -> Iterator[DenseCaptionSample]:
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    yield self._parse_record(record)
                except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                    raise ValueError(
                        f"invalid dataset record at {self.path}:{line_number}: {exc}"
                    ) from exc

    def _parse_record(self, record: dict[str, Any]) -> DenseCaptionSample:
        if not isinstance(record, dict):
            raise TypeError("record must be an object")
        video_path = Path(record["video_path"]).expanduser()
        if not video_path.is_absolute():
            video_path = (self.path.parent / video_path).resolve()
        if self.require_video_files and not video_path.is_file():
            raise ValueError(f"video does not exist: {video_path}")

        raw_references = record.get("references")
        if raw_references is None and record.get("reference") is not None:
            raw_references = [record["reference"]]
        if raw_references is None:
            references = ()
        else:
            if not isinstance(raw_references, list):
                raise ValueError("references must be an array")
            references = tuple(DenseCaptionPrediction.from_dict(item) for item in raw_references)
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        return DenseCaptionSample(
            video_id=record["video_id"],
            video_path=video_path,
            duration_ms=record["duration_ms"],
            references=references,
            metadata=metadata,
        )
