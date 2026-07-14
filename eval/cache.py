"""Atomic per-(system, split, video_id) result cache. "_complete": true marks a
fully written record; anything else is recomputed, never silently used."""

import json
import os


def _atomic_write(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_valid(path: str):
    """Return the cached dict if present, parseable, and marked complete; else None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            obj = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # corrupt/partial -> recompute
    if not isinstance(obj, dict) or not obj.get("_complete"):
        return None
    return obj


def result_path(cache_dir: str, system: str, split: str, video_id: str) -> str:
    return os.path.join(cache_dir, "results", system, f"{split}__{video_id}.json")


def save_result(cache_dir: str, system: str, split: str, video_id: str, record: dict):
    record = dict(record)
    record["_complete"] = True
    _atomic_write(result_path(cache_dir, system, split, video_id), record)


def shared_path(cache_dir: str, split: str, video_id: str) -> str:
    return os.path.join(cache_dir, "shared", f"{split}__{video_id}.json")


def load_shared(cache_dir: str, split: str, video_id: str):
    return load_valid(shared_path(cache_dir, split, video_id))


def save_shared(cache_dir: str, split: str, video_id: str, record: dict):
    record = dict(record)
    record["_complete"] = True
    _atomic_write(shared_path(cache_dir, split, video_id), record)
