"""Versioned, model-agnostic contracts for temporal dense captions and runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


DENSE_CAPTION_SCHEMA_VERSION = "dense-caption/v0.1"
RUN_MANIFEST_SCHEMA_VERSION = "run-manifest/v0.2"
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


def _require_non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class TemporalCaption:
    """One factual caption grounded to a half-open-ish interval in milliseconds.

    Captions in the same prediction may overlap and may leave temporal gaps. Evidence
    frames are optional, but when supplied they must fall inside the caption interval.
    """

    caption_id: str
    start_ms: int
    end_ms: int
    text: str
    evidence_frames_ms: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "caption_id", _require_non_empty(self.caption_id, "caption_id"))
        object.__setattr__(self, "text", _require_non_empty(self.text, "text"))
        _require_int(self.start_ms, "start_ms")
        _require_int(self.end_ms, "end_ms")
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")

        evidence = tuple(self.evidence_frames_ms)
        for timestamp in evidence:
            _require_int(timestamp, "evidence_frames_ms item")
            if not self.start_ms <= timestamp <= self.end_ms:
                raise ValueError(
                    f"evidence frame {timestamp} is outside "
                    f"[{self.start_ms}, {self.end_ms}]"
                )
        if tuple(sorted(set(evidence))) != evidence:
            raise ValueError("evidence_frames_ms must be sorted and unique")
        object.__setattr__(self, "evidence_frames_ms", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "caption_id": self.caption_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "evidence_frames_ms": list(self.evidence_frames_ms),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TemporalCaption":
        try:
            return cls(
                caption_id=data["caption_id"],
                start_ms=data["start_ms"],
                end_ms=data["end_ms"],
                text=data["text"],
                evidence_frames_ms=tuple(data.get("evidence_frames_ms", ())),
            )
        except KeyError as exc:
            raise ValueError(f"temporal caption missing field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class DenseCaptionPrediction:
    """The complete output of the temporal dense caption task for one video."""

    video_id: str
    duration_ms: int
    captions: tuple[TemporalCaption, ...]
    schema_version: str = DENSE_CAPTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _require_non_empty(self.video_id, "video_id"))
        _require_int(self.duration_ms, "duration_ms", minimum=1)
        if self.schema_version != DENSE_CAPTION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported dense caption schema: {self.schema_version!r}; "
                f"expected {DENSE_CAPTION_SCHEMA_VERSION!r}"
            )

        captions = tuple(self.captions)
        ids: set[str] = set()
        for caption in captions:
            if not isinstance(caption, TemporalCaption):
                raise ValueError("captions must contain TemporalCaption instances")
            if caption.caption_id in ids:
                raise ValueError(f"duplicate caption_id: {caption.caption_id}")
            if caption.end_ms > self.duration_ms:
                raise ValueError(
                    f"caption {caption.caption_id} ends after video duration: "
                    f"{caption.end_ms} > {self.duration_ms}"
                )
            ids.add(caption.caption_id)
        object.__setattr__(self, "captions", captions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "duration_ms": self.duration_ms,
            "captions": [caption.to_dict() for caption in self.captions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DenseCaptionPrediction":
        try:
            raw_captions = data["captions"]
            if not isinstance(raw_captions, list):
                raise ValueError("captions must be an array")
            return cls(
                schema_version=data["schema_version"],
                video_id=data["video_id"],
                duration_ms=data["duration_ms"],
                captions=tuple(TemporalCaption.from_dict(item) for item in raw_captions),
            )
        except KeyError as exc:
            raise ValueError(f"dense caption prediction missing field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class ComponentRef:
    name: str
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_empty(self.name, "component name"))
        object.__setattr__(self, "version", _require_non_empty(self.version, "component version"))

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComponentRef":
        try:
            return cls(name=data["name"], version=data["version"])
        except KeyError as exc:
            raise ValueError(f"component reference missing field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class DatasetRef(ComponentRef):
    fingerprint: str

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "fingerprint", _require_non_empty(self.fingerprint, "fingerprint"))

    def to_dict(self) -> dict[str, str]:
        return {**super().to_dict(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetRef":
        try:
            return cls(
                name=data["name"],
                version=data["version"],
                fingerprint=data["fingerprint"],
            )
        except KeyError as exc:
            raise ValueError(f"dataset reference missing field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class RunManifest:
    """Provenance kept separate from model predictions."""

    run_id: str
    created_at: str
    task: ComponentRef
    pipeline: ComponentRef
    dataset: DatasetRef
    config_sha256: str
    seed: int
    git_dirty: bool
    git_commit: str | None = None
    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_non_empty(self.run_id, "run_id"))
        object.__setattr__(self, "config_sha256", _require_non_empty(self.config_sha256, "config_sha256"))
        if not _SHA256_RE.fullmatch(self.config_sha256):
            raise ValueError("config_sha256 must be a lowercase SHA-256 hex digest")
        _require_int(self.seed, "seed")
        if not isinstance(self.git_dirty, bool):
            raise ValueError("git_dirty must be a boolean")
        if self.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported run manifest schema: {self.schema_version!r}; "
                f"expected {RUN_MANIFEST_SCHEMA_VERSION!r}"
            )
        try:
            parsed = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("created_at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        if self.git_commit is not None:
            object.__setattr__(self, "git_commit", _require_non_empty(self.git_commit, "git_commit"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "task": self.task.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "dataset": self.dataset.to_dict(),
            "config_sha256": self.config_sha256,
            "seed": self.seed,
            "git_dirty": self.git_dirty,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunManifest":
        try:
            return cls(
                schema_version=data["schema_version"],
                run_id=data["run_id"],
                created_at=data["created_at"],
                task=ComponentRef.from_dict(data["task"]),
                pipeline=ComponentRef.from_dict(data["pipeline"]),
                dataset=DatasetRef.from_dict(data["dataset"]),
                config_sha256=data["config_sha256"],
                seed=data["seed"],
                git_dirty=data["git_dirty"],
                git_commit=data.get("git_commit"),
            )
        except KeyError as exc:
            raise ValueError(f"run manifest missing field: {exc.args[0]}") from exc
