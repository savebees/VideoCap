"""Built-in dataset adapters."""

from dense_video_annotator.datasets.activitynet import ActivityNetCaptionsDataset
from dense_video_annotator.datasets.jsonl import LocalJSONLDataset

__all__ = ["ActivityNetCaptionsDataset", "LocalJSONLDataset"]
