"""SiliconFlow configuration entry point for VideoCap.

SiliconFlow is OpenAI-compatible, so the implementation is shared with the
existing HTTP adapter. The provider-specific factory only supplies explicit
defaults and keeps credentials in environment variables.
"""

from __future__ import annotations

from typing import Any, Mapping

from videocap.adapters.vyce import (
    OpenAICompatibleChatClient,
    VyceStructuredAdapters,
)
from videocap.structured import StructuredAdapterBundle


class SiliconFlowStructuredAdapters(VyceStructuredAdapters):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        provider = cfg.get("provider", {})
        if not isinstance(provider, Mapping):
            raise TypeError("pipeline_config.provider must be an object")
        normalized = dict(cfg)
        normalized["provider"] = {
            "base_url": "https://api.siliconflow.cn/v1",
            "api_key_env": "SILICONFLOW_API_KEY",
            "vlm_model": "Qwen/Qwen3.6-35B-A3B",
            "llm_model": "Qwen/Qwen3.6-35B-A3B",
            "enable_thinking": False,
            **dict(provider),
        }
        super().__init__(normalized)


def build_from_config(cfg: Mapping[str, Any]) -> StructuredAdapterBundle:
    adapter = SiliconFlowStructuredAdapters(cfg)
    return StructuredAdapterBundle(
        caption_window=adapter.caption_window,
        generate_event_candidates=adapter.generate_event_candidates,
        cluster_event_candidates=adapter.cluster_event_candidates,
        propose_event_window=adapter.propose_event_window,
        review_event_boundary=adapter.review_event_boundary,
        refine_event_caption=adapter.refine_event_caption,
        merge_global_caption=adapter.merge_global_caption,
        quality_filter=adapter.quality_filter,
    )


__all__ = ["OpenAICompatibleChatClient", "SiliconFlowStructuredAdapters", "build_from_config"]
