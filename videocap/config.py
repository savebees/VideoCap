"""Typed configuration for the VideoCap pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """One OpenAI-compatible model endpoint."""

    base_url: str
    api_key_env: str
    model: str
    timeout_sec: float = 120.0
    max_retries: int = 2
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("base_url", "api_key_env", "model"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelConfig:
        return cls(
            base_url=str(data["base_url"]).rstrip("/"),
            api_key_env=str(data["api_key_env"]),
            model=str(data["model"]),
            timeout_sec=float(data.get("timeout_sec", 120)),
            max_retries=int(data.get("max_retries", 2)),
            extra_body=dict(data.get("extra_body", {})),
        )


@dataclass(frozen=True)
class VLMConfig(ModelConfig):
    frame_height: int = 460

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.frame_height <= 0:
            raise ValueError("frame_height must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VLMConfig:
        return cls(
            base_url=str(data["base_url"]).rstrip("/"),
            api_key_env=str(data["api_key_env"]),
            model=str(data["model"]),
            timeout_sec=float(data.get("timeout_sec", 120)),
            max_retries=int(data.get("max_retries", 2)),
            extra_body=dict(data.get("extra_body", {})),
            frame_height=int(data.get("frame_height", 460)),
        )


@dataclass(frozen=True)
class PipelineConfig:
    window_ms: int = 24_000
    overlap_ms: int = 2_000
    evidence_frames: int = 8
    output_name: str = "annotations.jsonl"

    def __post_init__(self) -> None:
        if self.window_ms <= 0:
            raise ValueError("pipeline.window_ms must be positive")
        if not 0 <= self.overlap_ms < self.window_ms:
            raise ValueError("pipeline.overlap_ms must be smaller than window_ms")
        if self.evidence_frames < 2:
            raise ValueError("pipeline.evidence_frames must be at least 2")
        if Path(self.output_name).name != self.output_name or not self.output_name.endswith(
            ".jsonl"
        ):
            raise ValueError("pipeline.output_name must be a plain .jsonl filename")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineConfig:
        return cls(
            window_ms=int(data.get("window_ms", 24_000)),
            overlap_ms=int(data.get("overlap_ms", 2_000)),
            evidence_frames=int(data.get("evidence_frames", 8)),
            output_name=str(data.get("output_name", "annotations.jsonl")),
        )


@dataclass(frozen=True)
class Config:
    pipeline: PipelineConfig
    vlm: VLMConfig
    llm: ModelConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Config:
        return cls(
            pipeline=PipelineConfig.from_dict(data.get("pipeline", {})),
            vlm=VLMConfig.from_dict(data["vlm"]),
            llm=ModelConfig.from_dict(data["llm"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> Config:
        document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("configuration root must be a JSON object")
        return cls.from_dict(document)

    def to_dict(self) -> dict[str, Any]:
        def model(config: ModelConfig) -> dict[str, Any]:
            return {
                "base_url": config.base_url,
                "api_key_env": config.api_key_env,
                "model": config.model,
                "timeout_sec": config.timeout_sec,
                "max_retries": config.max_retries,
                "extra_body": dict(config.extra_body),
            }

        vlm = model(self.vlm)
        vlm["frame_height"] = self.vlm.frame_height
        return {
            "pipeline": {
                "window_ms": self.pipeline.window_ms,
                "overlap_ms": self.pipeline.overlap_ms,
                "evidence_frames": self.pipeline.evidence_frames,
                "output_name": self.pipeline.output_name,
            },
            "vlm": vlm,
            "llm": model(self.llm),
        }


__all__ = ["Config", "ModelConfig", "PipelineConfig", "VLMConfig"]
