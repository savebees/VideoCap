"""Clients for the judge VLM and the claims/omission text LLM, plus robust
JSON parsing for Gemma3's pseudo-JSON output."""

import ast
import base64
import json
import re

from openai import OpenAI


def make_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="not-needed")


def image_content(jpeg_bytes_list: list[bytes]) -> list[dict]:
    """Multiple frames as standard OpenAI image_url items (NOT video_url)."""
    items = []
    for b in jpeg_bytes_list:
        b64 = base64.b64encode(b).decode("ascii")
        items.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return items


_TRAILING_COMMA = re.compile(r",(\s*[\]}])")


def _balanced(text: str, open_ch: str, close_ch: str):
    """Return the first top-level balanced {open..close} span (ignoring brackets
    inside strings), or None. Handles trailing prose after the JSON."""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None  # unbalanced (truncated)


def _salvage_array(text: str):
    """Best-effort recovery of a truncated top-level array: keep elements up to the
    last top-level comma (the last complete element), then close the array. Works
    for arrays of strings and of objects. Used only when strict parsing fails — it
    recovers the model's own intended items, not a different method."""
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    last_top_comma = None
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        elif c == "," and depth == 1:
            last_top_comma = i
    if last_top_comma is not None:
        return text[start:last_top_comma] + "]"
    return None


def parse_json_strict(text: str):
    """Parse a JSON array/object from model output. Robust to markdown fences,
    trailing prose, trailing commas, and a truncated final element. Raises if no
    JSON can be recovered (no silent substitution of a different result)."""
    if text is None:
        raise ValueError("model returned no content")
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", cleaned)  # strip markdown code fences

    span = _balanced(cleaned, "[", "]") or _balanced(cleaned, "{", "}")
    candidates = []
    if span is not None:
        candidates.append(span)
        candidates.append(_TRAILING_COMMA.sub(r"\1", span))
    salv = _salvage_array(cleaned)
    if salv is not None:
        candidates.append(_TRAILING_COMMA.sub(r"\1", salv))

    last_err = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = e
        # Gemma3 often emits single-quoted pseudo-JSON; literal_eval recovers it.
        try:
            pyish = re.sub(r"\btrue\b", "True", cand)
            pyish = re.sub(r"\bfalse\b", "False", pyish)
            pyish = re.sub(r"\bnull\b", "None", pyish)
            return ast.literal_eval(pyish)
        except (ValueError, SyntaxError):
            pass
    raise ValueError(f"no parseable JSON in model output (last error: {last_err}): {cleaned[:300]!r}")
