"""Local JSONL video manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from videocap.structured import VideoSample


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def load_manifest(
    path: str | Path,
    *,
    require_video_files: bool = True,
) -> tuple[VideoSample, ...]:
    manifest = Path(path).expanduser().resolve()
    samples: list[VideoSample] = []
    video_ids: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("record must be an object")
                video_path = Path(record["video_path"]).expanduser()
                if not video_path.is_absolute():
                    video_path = (manifest.parent / video_path).resolve()
                if require_video_files and not video_path.is_file():
                    raise FileNotFoundError(f"video does not exist: {video_path}")
                metadata = record.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise TypeError("metadata must be an object")
                sample = VideoSample(
                    video_id=record["video_id"],
                    video_path=video_path,
                    duration_ms=record["duration_ms"],
                    metadata=metadata,
                )
                if sample.video_id in video_ids:
                    raise ValueError(f"duplicate video_id: {sample.video_id}")
                video_ids.add(sample.video_id)
                samples.append(sample)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid manifest record at {manifest}:{line_number}: {exc}"
                ) from exc
    if not samples:
        raise ValueError(f"manifest contains no videos: {manifest}")
    return tuple(samples)


__all__ = ["load_manifest", "manifest_sha256"]
