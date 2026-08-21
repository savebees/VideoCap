"""Public contracts for the Dense Video Annotator system."""

from dense_video_annotator.contracts import (
    DENSE_CAPTION_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    ComponentRef,
    DatasetRef,
    DenseCaptionPrediction,
    RunManifest,
    TemporalCaption,
)
from dense_video_annotator.protocols import (
    DenseCaptionDataset,
    DenseCaptionMetric,
    DenseCaptionOutcome,
    DenseCaptionRecipe,
    DenseCaptionSample,
    DenseCaptionTask,
)
from dense_video_annotator.structured import (
    DIMENSIONS,
    EventCandidate,
    EventCaption,
    EventCluster,
    EventWindow,
    ProcessingWindow,
    StructuredAdapterBundle,
    WindowCaption,
)

__all__ = [
    "DENSE_CAPTION_SCHEMA_VERSION",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "ComponentRef",
    "DatasetRef",
    "DenseCaptionPrediction",
    "DenseCaptionDataset",
    "DenseCaptionMetric",
    "DenseCaptionOutcome",
    "DenseCaptionRecipe",
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
]

__version__ = "0.1.0"
