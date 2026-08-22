"""Built-in dataset adapters."""

from videocap.datasets.activitynet import ActivityNetCaptionsDataset
from videocap.datasets.jsonl import LocalJSONLDataset

__all__ = ["ActivityNetCaptionsDataset", "LocalJSONLDataset"]
