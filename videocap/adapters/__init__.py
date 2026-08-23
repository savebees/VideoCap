"""Shared OpenAI-compatible client used by the VLM and LLM adapters."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI

from videocap.config import ModelConfig


class ChatClient:
    def __init__(self, config: ModelConfig) -> None:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key environment variable: {config.api_key_env}")
        self.model = config.model
        self.extra_body = dict(config.extra_body)
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_sec,
            max_retries=config.max_retries,
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=0,
            extra_body=self.extra_body or None,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("model returned an empty response")
        return content.strip()


__all__ = ["ChatClient"]
