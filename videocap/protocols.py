"""Stable extension points for dense caption production and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from videocap.contracts import DenseCaptionPrediction


@dataclass(frozen=True)
class DenseCaptionSample:
    video_id: str
    video_path: Path
    duration_ms: int
    reference: DenseCaptionPrediction | None = None
    references: tuple[DenseCaptionPrediction, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("sample video_id must be a non-empty string")
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, int):
            raise ValueError("sample duration_ms must be an integer")
        if self.duration_ms <= 0:
            raise ValueError("sample duration_ms must be positive")
        if self.reference is not None:
            if self.reference.video_id != self.video_id:
                raise ValueError("sample and reference video_id must match")
            if self.reference.duration_ms != self.duration_ms:
                raise ValueError("sample and reference duration_ms must match")
        refs = tuple(self.references)
        if self.reference is not None and not refs:
            refs = (self.reference,)
        for reference in refs:
            if not isinstance(reference, DenseCaptionPrediction):
                raise ValueError("references must contain DenseCaptionPrediction instances")
            if reference.video_id != self.video_id:
                raise ValueError("sample and reference video_id must match")
            if reference.duration_ms != self.duration_ms:
                raise ValueError("sample and reference duration_ms must match")
        if self.reference is None and refs:
            object.__setattr__(self, "reference", refs[0])
        object.__setattr__(self, "references", refs)


class DenseCaptionTask(ABC):
    name: str
    version: str

    @abstractmethod
    def validate(self, prediction: DenseCaptionPrediction) -> None:
        """Raise when a pipeline output violates this task contract."""


@dataclass(frozen=True)
class DenseCaptionOutcome:
    """One video's result from the VideoCap production pipeline."""

    video_id: str
    prediction: DenseCaptionPrediction | None = None
    artifact: Mapping[str, Any] | None = None
    error_type: str | None = None
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    latency_sec: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("outcome video_id must be a non-empty string")
        succeeded = self.prediction is not None or self.artifact is not None
        failed = self.error_type is not None
        if succeeded == failed:
            raise ValueError("outcome must contain either a prediction/artifact or an error")
        if self.prediction is not None and self.artifact is not None:
            raise ValueError("outcome cannot contain both prediction and artifact")
        if self.prediction is not None and self.prediction.video_id != self.video_id:
            raise ValueError("outcome and prediction video_id must match")
        if self.artifact is not None:
            artifact_video_id = self.artifact.get("video_id")
            if artifact_video_id != self.video_id:
                raise ValueError("outcome and artifact video_id must match")
        if failed and (not isinstance(self.message, str) or not self.message.strip()):
            raise ValueError("failed outcome must include a non-empty message")
        if self.latency_sec is not None and self.latency_sec < 0:
            raise ValueError("outcome latency_sec must be non-negative")


class DenseCaptionProducer(ABC):
    name: str
    version: str

    @abstractmethod
    def produce(
        self,
        samples: Sequence[DenseCaptionSample],
        work_dir: Path,
        config: Mapping[str, Any],
    ) -> Sequence[DenseCaptionOutcome]:
        """Produce captions for a dataset without applying business acceptance policy."""


class DenseCaptionDataset(ABC):
    name: str
    version: str

    @abstractmethod
    def __iter__(self) -> Iterator[DenseCaptionSample]:
        pass

    @abstractmethod
    def fingerprint(self) -> str:
        """Return a stable content fingerprint for run reproducibility."""


class DenseCaptionMetric(ABC):
    name: str
    version: str

    @abstractmethod
    def score(
        self,
        prediction: DenseCaptionPrediction,
        reference: DenseCaptionPrediction,
    ) -> Mapping[str, float]:
        """Score one prediction/reference pair."""

    def score_references(
        self,
        prediction: DenseCaptionPrediction,
        references: Sequence[DenseCaptionPrediction],
    ) -> Mapping[str, float]:
        """Score against multiple references, retaining the best value per field."""
        if not references:
            return {}
        scores = [dict(self.score(prediction, reference)) for reference in references]
        keys = sorted({key for score in scores for key in score})
        return {
            key: max(score[key] for score in scores if key in score)
            for key in keys
        }

    def aggregate(self, samples: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
        if not samples:
            return {}
        keys = sorted({key for sample in samples for key in sample})
        return {
            key: sum(sample[key] for sample in samples if key in sample)
            / sum(1 for sample in samples if key in sample)
            for key in keys
        }
