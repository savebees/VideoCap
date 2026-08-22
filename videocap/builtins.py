"""Built-in task, dataset, and metric registrations."""

from __future__ import annotations

from typing import Mapping

from videocap.contracts import DenseCaptionPrediction
from videocap.datasets import activitynet, jsonl  # noqa: F401
from videocap.metrics import lexical, soda, temporal  # noqa: F401
from videocap.protocols import (
    DenseCaptionMetric,
    DenseCaptionTask,
)
from videocap.registry import METRICS, TASKS
from videocap.schema import validate_document


class TemporalDenseCaptionTask(DenseCaptionTask):
    name = "temporal-dense-caption"
    version = "0.1"

    def validate(self, prediction: DenseCaptionPrediction) -> None:
        if not isinstance(prediction, DenseCaptionPrediction):
            raise TypeError("temporal dense caption output must be DenseCaptionPrediction")
        validate_document(prediction.to_dict(), "dense_caption")


class ExactStructureMetric(DenseCaptionMetric):
    name = "exact-structure"
    version = "0.1"

    def score(
        self,
        prediction: DenseCaptionPrediction,
        reference: DenseCaptionPrediction,
    ) -> Mapping[str, float]:
        return {"exact_match": float(prediction.to_dict() == reference.to_dict())}


TASKS.add(TemporalDenseCaptionTask.name, TemporalDenseCaptionTask())
METRICS.add(ExactStructureMetric.name, ExactStructureMetric())


def load_builtin_components() -> None:
    """Importing this module registers the built-ins; this is an explicit hook."""
