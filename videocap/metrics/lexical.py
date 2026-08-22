"""Deterministic lexical caption matching for offline benchmark smoke tests.

This is intentionally a transparent baseline metric, not a replacement for
CIDEr/METEOR/SPICE or a semantic judge. It makes benchmark wiring testable without
downloading model checkpoints or relying on a network service.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Mapping

from videocap.contracts import DenseCaptionPrediction
from videocap.protocols import DenseCaptionMetric
from videocap.registry import METRICS

_TOKEN_RE = re.compile(r"[\w]+(?:['-][\w]+)*", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _f1(prediction: list[str], reference: list[str]) -> tuple[float, float, float]:
    if not prediction and not reference:
        return 1.0, 1.0, 1.0
    if not prediction or not reference:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(prediction) & Counter(reference)).values())
    precision = overlap / len(prediction)
    recall = overlap / len(reference)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@METRICS.register("lexical-caption")
class LexicalCaptionMetric(DenseCaptionMetric):
    """Best-reference caption token overlap, reported independently of timing."""

    name = "lexical-caption"
    version = "0.1"

    def score(
        self,
        prediction: DenseCaptionPrediction,
        reference: DenseCaptionPrediction,
    ) -> Mapping[str, float]:
        predicted = " ".join(caption.text for caption in prediction.captions)
        expected = " ".join(caption.text for caption in reference.captions)
        precision, recall, f1 = _f1(_tokens(predicted), _tokens(expected))
        return {
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1,
        }
