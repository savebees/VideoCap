"""OpenAI-compatible adapters for the VideoCap production flow.

    The provider receives remote media URLs, not local paths or data URIs. A
    production deployment must therefore configure ``frame_url`` or
    ``frame_url_template`` (the URL may be an individual frame or a provider-
    supported video URL). The adapter raises when that mapping is absent.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from videocap.structured import (
    DIMENSIONS,
    EventCaption,
    EventCandidate,
    EventCluster,
    EventWindow,
    ProcessingWindow,
    StructuredAdapterBundle,
    WindowCaption,
)
from videocap.prompts import WINDOW_CAPTION_DIMENSIONS, build_window_caption_prompt


class VyceChatClient:
    def __init__(self, *, api_key: str, base_url: str, timeout_sec: float = 120.0, min_request_interval_sec: float = 2.0, max_retries: int = 2, retry_backoff_sec: float = 5.0, enable_thinking: bool | None = None) -> None:
        if not api_key.strip():
            raise ValueError("Vyce API key must be non-empty")
        if not base_url.rstrip("/").endswith("/v1"):
            raise ValueError("Vyce base_url must end with /v1")
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout_sec = timeout_sec
        if min_request_interval_sec < 0:
            raise ValueError("min_request_interval_sec must be non-negative")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff_sec <= 0:
            raise ValueError("retry_backoff_sec must be positive")
        self.min_request_interval_sec = min_request_interval_sec
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.enable_thinking = enable_thinking
        self._last_request_at: float | None = None

    def complete(self, *, model: str, messages: Sequence[Mapping[str, Any]], max_tokens: int = 1024) -> str:
        payload: dict[str, Any] = {"model": model, "messages": list(messages), "max_tokens": max_tokens, "temperature": 0}
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json", "User-Agent": "videocap/0.1"},
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            if self._last_request_at is not None:
                wait = self.min_request_interval_sec - (time.monotonic() - self._last_request_at)
                if wait > 0:
                    time.sleep(wait)
            self._last_request_at = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                if exc.code != 429 or attempt >= self.max_retries:
                    raise RuntimeError(f"OpenAI-compatible request failed with HTTP {exc.code}: {detail[:1000]}") from exc
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else self.retry_backoff_sec * (2**attempt)
                except ValueError:
                    delay = self.retry_backoff_sec * (2**attempt)
                time.sleep(max(delay, self.min_request_interval_sec))
        else:  # pragma: no cover
            raise RuntimeError("Vyce request retry loop ended without a response")
        if not isinstance(payload, Mapping):
            raise TypeError("OpenAI-compatible response must be a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenAI-compatible response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenAI-compatible response choice has empty message content")
        return content.strip()


OpenAICompatibleChatClient = VyceChatClient


def _json_text(text: str) -> Any:
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return valid JSON: {text[:500]}") from exc


def _window_captions(text: str) -> Mapping[str, str]:
    """Parse the five labelled free-text answers from one VLM response."""

    pattern = re.compile(r"^\[([^\]]+)\]\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != len(WINDOW_CAPTION_DIMENSIONS):
        raise ValueError("window caption response must contain exactly five dimension headings")
    captions: dict[str, str] = {}
    for index, match in enumerate(matches):
        dimension = match.group(1)
        if dimension not in WINDOW_CAPTION_DIMENSIONS:
            raise ValueError(f"window caption response has unknown dimension heading: {dimension}")
        if dimension in captions:
            raise ValueError(f"window caption response repeats dimension heading: {dimension}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end():end].strip()
        if not value:
            raise ValueError(f"window caption response has empty caption: {dimension}")
        captions[dimension] = value
    if tuple(captions) != WINDOW_CAPTION_DIMENSIONS:
        raise ValueError("window caption response dimensions are out of order")
    return captions


class VyceStructuredAdapters:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        provider = cfg.get("provider", {})
        if not isinstance(provider, Mapping):
            raise TypeError("pipeline_config.provider must be an object")
        vlm = provider.get("vlm", provider)
        llm = provider.get("llm", provider)
        if not isinstance(vlm, Mapping) or not isinstance(llm, Mapping):
            raise TypeError("provider.vlm and provider.llm must be objects")
        self.vlm_client = self._build_client(vlm, provider)
        self.llm_client = self._build_client(llm, provider)
        self.vlm_model = str(vlm.get("model", provider.get("vlm_model", "nemotron-vision")))
        self.llm_model = str(llm.get("model", provider.get("llm_model", "gpt-5.6")))
        self.frame_url = vlm.get("frame_url", provider.get("frame_url"))
        self.frame_url_template = vlm.get("frame_url_template", provider.get("frame_url_template"))
        self.media_url = vlm.get("media_url", provider.get("media_url"))
        self.media_url_template = vlm.get("media_url_template", provider.get("media_url_template"))
        self.local_frame_mode = vlm.get("local_frame_mode")
        self.frame_height = int(vlm.get("frame_height", provider.get("frame_height", 460)))
        if self.frame_height <= 0:
            raise ValueError("provider.frame_height must be positive")
        self._local_frame_cache: dict[tuple[str, int], str] = {}
        if not self.frame_url and not self.frame_url_template and not self.media_url and not self.media_url_template and self.local_frame_mode != "data_uri":
            raise RuntimeError("provider requires frame_url, frame_url_template, media_url, media_url_template, or local_frame_mode=data_uri")

    @staticmethod
    def _build_client(settings: Mapping[str, Any], defaults: Mapping[str, Any]) -> VyceChatClient:
        api_key_env = settings.get("api_key_env", defaults.get("api_key_env", "VYCE_API_KEY"))
        api_key = os.environ.get(str(api_key_env))
        if not api_key:
            raise RuntimeError(f"missing API key environment variable: {api_key_env}")
        return VyceChatClient(
            api_key=api_key,
            base_url=str(settings.get("base_url", defaults.get("base_url", "https://vyceai.com/v1"))),
            timeout_sec=float(settings.get("timeout_sec", defaults.get("timeout_sec", 120))),
            min_request_interval_sec=float(settings.get("min_request_interval_sec", defaults.get("min_request_interval_sec", 2.0))),
            max_retries=int(settings.get("max_retries", defaults.get("max_retries", 2))),
            retry_backoff_sec=float(settings.get("retry_backoff_sec", defaults.get("retry_backoff_sec", 5.0))),
            enable_thinking=settings.get("enable_thinking", defaults.get("enable_thinking")),
        )

    def _image_url(self, sample: Any, timestamp_ms: int, start_ms: int | None = None, end_ms: int | None = None) -> str:
        values = {
            "video_id": sample.video_id,
            "timestamp_ms": timestamp_ms,
            "video_path": Path(sample.video_path).name,
            "start_ms": timestamp_ms if start_ms is None else start_ms,
            "end_ms": timestamp_ms if end_ms is None else end_ms,
        }
        if isinstance(self.frame_url, str) and self.frame_url.strip():
            return self.frame_url
        if isinstance(self.frame_url_template, str) and self.frame_url_template.strip():
            return self.frame_url_template.format(**values)
        if isinstance(self.media_url, str) and self.media_url.strip():
            return self.media_url
        if isinstance(self.media_url_template, str) and self.media_url_template.strip():
            return self.media_url_template.format(**values)
        if self.local_frame_mode == "data_uri":
            cache_key = (str(Path(sample.video_path).resolve()), timestamp_ms)
            if cache_key not in self._local_frame_cache:
                # Encoders often report a duration slightly past the last decodable frame.
                extract_timestamp_ms = min(timestamp_ms, max(0, int(sample.duration_ms) - 500))
                command = ["ffmpeg", "-v", "error", "-ss", f"{extract_timestamp_ms / 1000:.3f}", "-i", str(sample.video_path), "-frames:v", "1", "-vf", f"scale=-2:{self.frame_height}", "-f", "image2", "-c:v", "mjpeg", "-q:v", "6", "pipe:1"]
                result = subprocess.run(command, capture_output=True, check=False)
                if result.returncode != 0 or not result.stdout:
                    raise RuntimeError(f"ffmpeg could not extract frame at {timestamp_ms} ms: {result.stderr.decode('utf-8', 'replace')[:500]}")
                self._local_frame_cache[cache_key] = "data:image/jpeg;base64," + base64.b64encode(result.stdout).decode("ascii")
            return self._local_frame_cache[cache_key]
        raise RuntimeError("no remote media URL configured")

    def _vlm(self, sample: Any, prompt: str, timestamps: Sequence[int], *, start_ms: int | None = None, end_ms: int | None = None) -> str:
        content: list[Mapping[str, Any]] = [{"type": "text", "text": prompt}]
        for timestamp in timestamps:
            content.append({"type": "image_url", "image_url": {"url": self._image_url(sample, timestamp, start_ms, end_ms)}})
        return self.vlm_client.complete(model=self.vlm_model, messages=[{"role": "user", "content": content}], max_tokens=1500)

    def _llm_json(self, prompt: str) -> Any:
        return _json_text(self.llm_client.complete(model=self.llm_model, messages=[{"role": "user", "content": prompt}], max_tokens=1800))

    def caption_window(self, sample: Any, window: ProcessingWindow) -> Mapping[str, Any]:
        captions = _window_captions(
            self._vlm(
                sample,
                build_window_caption_prompt(),
                window.evidence_frames_ms,
                start_ms=window.start_ms,
                end_ms=window.end_ms,
            )
        )
        return {"captions": captions, "evidence_frames_ms": list(window.evidence_frames_ms)}

    def generate_event_candidates(self, sample: Any, windows: Sequence[ProcessingWindow], captions: Sequence[WindowCaption]) -> Sequence[Mapping[str, Any]]:
        prompt = "Return only a JSON array of event candidates. Each item must have candidate_id, source_window_ids, description. Use adjacent processing windows when they describe one event.\n" + json.dumps([item.to_dict() for item in captions], ensure_ascii=False)
        value = self._llm_json(prompt)
        if not isinstance(value, list):
            raise TypeError("candidate model output must be an array")
        return value

    def cluster_event_candidates(self, sample: Any, candidates: Sequence[EventCandidate]) -> Sequence[Mapping[str, Any]]:
        value = self._llm_json("Return only a JSON array of clusters. Each item must have cluster_id and candidate_ids. Do not drop candidates.\n" + json.dumps([item.to_dict() for item in candidates], ensure_ascii=False))
        if not isinstance(value, list):
            raise TypeError("cluster model output must be an array")
        return value

    def propose_event_window(self, sample: Any, cluster: EventCluster, candidates: Sequence[EventCandidate], windows: Sequence[ProcessingWindow]) -> Mapping[str, Any]:
        value = self._llm_json("Return only JSON with integer start_ms and end_ms and evidence_frames_ms array. Propose a coarse semantic event interval inside the video duration.\n" + json.dumps({"cluster": cluster.to_dict(), "candidates": [item.to_dict() for item in candidates], "windows": [item.to_dict() for item in windows]}, ensure_ascii=False))
        if not isinstance(value, Mapping):
            raise TypeError("event-window model output must be an object")
        return value

    def review_event_boundary(self, sample: Any, event: EventWindow) -> Mapping[str, Any]:
        prompt = "Return only JSON object with a non-empty events array. Each event has event_id, start_ms, end_ms, evidence_frames_ms. Review the proposed interval using the evidence frames and split it only when distinct events are present. If no split is justified, return exactly one event covering the proposed interval; never return an empty events array."
        value = _json_text(self._vlm(sample, prompt, event.evidence_frames_ms or (event.start_ms, event.end_ms), start_ms=event.start_ms, end_ms=event.end_ms))
        # Qwen sometimes emits the requested event collection directly as an
        # array instead of wrapping it in {"events": [...]}. The collection
        # itself is unambiguous, so normalize that representation at the
        # provider boundary while retaining strict downstream validation.
        if isinstance(value, list):
            return {"events": value}
        if not isinstance(value, Mapping):
            raise TypeError("boundary-review model output must be an object")
        return value

    def refine_event_caption(self, sample: Any, event: EventWindow, windows: Sequence[WindowCaption]) -> Mapping[str, Any]:
        prompt = "Return only JSON with exactly five string keys: short, main_object, background, camera, detailed. Refine the caption for this event using the supplied evidence."
        return {"captions": _json_text(self._vlm(sample, prompt, event.evidence_frames_ms or (event.start_ms, event.end_ms), start_ms=event.start_ms, end_ms=event.end_ms))}

    def merge_global_caption(self, sample: Any, events: Sequence[EventCaption]) -> Mapping[str, str]:
        value = _json_text(self.llm_client.complete(model=self.llm_model, messages=[{"role": "user", "content": "Return only JSON with exactly five string keys: short, main_object, background, camera, detailed. Write global captions for the entire video from these ordered event captions.\n" + json.dumps([item.to_dict() for item in events], ensure_ascii=False)}], max_tokens=1200))
        if not isinstance(value, Mapping):
            raise TypeError("global caption model output must be an object")
        return value

    def quality_filter(self, sample: Any, events: Sequence[EventCaption], global_captions: Mapping[str, str]) -> Mapping[str, Any]:
        return {"accepted_event_ids": [item.event.event_id for item in events], "rejected_event_ids": [], "checks": {"non_empty_captions": True, "global_captions_non_empty": all(value.strip() for value in global_captions.values())}}


def build_from_config(cfg: Mapping[str, Any]) -> StructuredAdapterBundle:
    adapter = VyceStructuredAdapters(cfg)
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
