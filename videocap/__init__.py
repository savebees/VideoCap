"""Public contracts and the VideoCap production flow."""

from videocap.contracts import (
    DENSE_CAPTION_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    ComponentRef,
    DatasetRef,
    DenseCaptionPrediction,
    RunManifest,
    TemporalCaption,
)
from videocap.protocols import (
    DenseCaptionDataset,
    DenseCaptionMetric,
    DenseCaptionOutcome,
    DenseCaptionProducer,
    DenseCaptionSample,
    DenseCaptionTask,
)
from videocap.structured import (
    DIMENSIONS,
    EventCandidate,
    EventCaption,
    EventCluster,
    EventWindow,
    ProcessingWindow,
    StructuredAdapterBundle,
    WindowCaption,
)
from videocap.pipeline import VideoCap

__all__ = [
    "DENSE_CAPTION_SCHEMA_VERSION",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "ComponentRef",
    "DatasetRef",
    "DenseCaptionPrediction",
    "DenseCaptionDataset",
    "DenseCaptionMetric",
    "DenseCaptionOutcome",
    "DenseCaptionProducer",
    "DenseCaptionSample",
    "DenseCaptionTask",
    "RunManifest",
    "TemporalCaption",
    "DIMENSIONS",
    "ProcessingWindow",
    "WindowCaption",
    "EventCandidate",
    "EventCluster",
    "EventWindow",
    "EventCaption",
    "StructuredAdapterBundle",
    "VideoCap",
]

__version__ = "0.1.0"
