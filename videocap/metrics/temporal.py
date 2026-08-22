"""Reference-based temporal proposal metrics for dense captions."""

from __future__ import annotations

from typing import Mapping

from videocap.contracts import DenseCaptionPrediction, TemporalCaption
from videocap.protocols import DenseCaptionMetric
from videocap.registry import METRICS


def temporal_iou(left: TemporalCaption, right: TemporalCaption) -> float:
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    return intersection / union if union else 0.0


def _greedy_matches(prediction: DenseCaptionPrediction, reference: DenseCaptionPrediction, threshold: float) -> list[float]:
    candidates = sorted(
        (temporal_iou(pred, ref), pred.caption_id, ref.caption_id)
        for pred in prediction.captions
        for ref in reference.captions
        if temporal_iou(pred, ref) >= threshold
    )
    used_pred: set[str] = set()
    used_ref: set[str] = set()
    matches: list[float] = []
    for score, pred_id, ref_id in reversed(candidates):
        if pred_id not in used_pred and ref_id not in used_ref:
            used_pred.add(pred_id)
            used_ref.add(ref_id)
            matches.append(score)
    return matches


@METRICS.register("temporal")
class TemporalMetric(DenseCaptionMetric):
    name = "temporal"
    version = "0.1"

    def __init__(self, thresholds: tuple[float, ...] = (0.3, 0.5, 0.7)) -> None:
        self.thresholds = thresholds

    def score(self, prediction: DenseCaptionPrediction, reference: DenseCaptionPrediction) -> Mapping[str, float]:
        n_pred, n_ref = len(prediction.captions), len(reference.captions)
        result: dict[str, float] = {"mean_iou": 0.0, "proposal_precision": 0.0, "proposal_recall": 0.0, "proposal_f1": 0.0}
        all_ious = [max((temporal_iou(pred, ref) for ref in reference.captions), default=0.0) for pred in prediction.captions]
        result["mean_iou"] = sum(all_ious) / n_pred if n_pred else 0.0
        for threshold in self.thresholds:
            matches = len(_greedy_matches(prediction, reference, threshold))
            precision = matches / n_pred if n_pred else (1.0 if n_ref == 0 else 0.0)
            recall = matches / n_ref if n_ref else (1.0 if n_pred == 0 else 0.0)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            key = f"f1@{threshold:g}"
            result[key] = f1
        matches = len(_greedy_matches(prediction, reference, self.thresholds[0]))
        result["proposal_precision"] = matches / n_pred if n_pred else (1.0 if n_ref == 0 else 0.0)
        result["proposal_recall"] = matches / n_ref if n_ref else (1.0 if n_pred == 0 else 0.0)
        p, r = result["proposal_precision"], result["proposal_recall"]
        result["proposal_f1"] = 2 * p * r / (p + r) if p + r else 0.0
        return result
