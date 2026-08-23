"""Event-grounded video captioning with OpenAI-compatible models."""

from videocap.config import Config, ModelConfig, PipelineConfig, VLMConfig
from videocap.pipeline import VideoCap
from videocap.structured import (
    DIMENSIONS,
    EventCaption,
    EventProposal,
    EventWindow,
    ProcessingWindow,
    VideoSample,
    WindowCaption,
)

__version__ = "0.2.0"

__all__ = [
    "Config",
    "DIMENSIONS",
    "EventCaption",
    "EventProposal",
    "EventWindow",
    "ModelConfig",
    "PipelineConfig",
    "ProcessingWindow",
    "VLMConfig",
    "VideoCap",
    "VideoSample",
    "WindowCaption",
    "__version__",
]
