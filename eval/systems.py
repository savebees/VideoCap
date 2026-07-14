"""Caption assembly and shared normalization. One (split, video_id) x system ->
one caption string. Missing output files raise: a hole must surface, not hide."""

import json
import os
import re

_BULLET = re.compile(r"^[ \t]*([*\-+]|\d+[.)])[ \t]+", re.MULTILINE)
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]*", re.MULTILINE)
_BOLD_ITALIC = re.compile(r"(\*\*|\*|__|_)")
_BACKTICKS = re.compile(r"`+")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{2,}")


def normalize_caption(text: str) -> str:
    """Strip markdown, keep the words, fold whitespace. All systems share this."""
    if text is None:
        raise ValueError("normalize_caption received None")
    t = _LINK.sub(r"\1", text)
    t = _HEADING.sub("", t)
    t = _BULLET.sub("", t)
    t = _BACKTICKS.sub("", t)
    t = _BOLD_ITALIC.sub("", t)
    t = t.replace(">", " ")
    t = _WS.sub(" ", t)
    t = _NL.sub("\n", t)
    return " ".join(line.strip() for line in t.splitlines() if line.strip()).strip()


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"caption file missing: {path}")
    with open(path) as f:
        return json.load(f)


def _norm_action(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().strip().rstrip(".")).strip()


def _action_sentence(a: dict) -> str:
    """One action sentence; prepend the subject only if not already in the description."""
    subj = (a.get("subject") or "").strip()
    desc = (a.get("description") or "").strip()
    if not desc:
        sent = subj
    elif subj and subj.lower() not in desc.lower():
        sent = f"{subj} {desc}"
    else:
        sent = desc
    if sent and sent[-1] not in ".!?":
        sent += "."
    return (sent[:1].upper() + sent[1:]) if sent else sent


def _assemble_pipeline(doc: dict) -> str:
    """Dense captions + action sentences in time order; action sentences globally
    deduplicated (surveillance re-detects the same actor across windows)."""
    segs = doc.get("segments")
    if not segs:
        raise ValueError(f"pipeline annotation has no segments: {doc.get('video_id')}")
    ordered = sorted(segs, key=lambda s: float(s["start"]))
    parts = []
    seen = set()
    for s in ordered:
        desc = s.get("description")
        if desc is None:
            raise ValueError(f"segment missing description in {doc.get('video_id')} scene {s.get('scene_id')}")
        if desc.strip():
            parts.append(desc.strip())
        for a in s.get("actions", []):
            sent = _action_sentence(a)
            key = _norm_action(sent)
            if key and key not in seen:
                seen.add(key)
                parts.append(sent)
    return " ".join(parts)


def load_caption(system: dict, split: str, video_id: str, config: dict) -> str:
    """Return the raw caption text for one system on one clip."""
    kind = system["kind"]
    name = system["name"]
    out = config["splits"][split]["out"]   # dataset-organized pipeline output dir
    if kind == "pipeline":
        path = os.path.join(out, video_id, "annotation.json")
        return _assemble_pipeline(_read_json(path))
    if kind == "a3b":
        path = os.path.join(out, video_id, "baseline.json")
        doc = _read_json(path)
    elif kind == "model":
        path = os.path.join(config["results_root"], name, split, f"{video_id}.json")
        doc = _read_json(path)
    else:
        raise ValueError(f"unknown system kind: {kind}")
    cap = doc.get("caption")
    if cap is None:
        raise ValueError(f"caption field missing in {path}")
    return cap


def list_clips(config: dict) -> list[tuple[str, str]]:
    """Enumerate every (split, video_id) clip from the source video dirs."""
    clips = []
    for split, spec in config["splits"].items():
        d, ext = spec["dir"], spec["ext"]
        if not os.path.isdir(d):
            raise FileNotFoundError(f"split dir missing: {d}")
        for fname in sorted(os.listdir(d)):
            if fname.endswith(ext):
                clips.append((split, os.path.splitext(fname)[0]))
    if not clips:
        raise RuntimeError("no clips found across splits")
    return clips


def video_path(config: dict, split: str, video_id: str) -> str:
    spec = config["splits"][split]
    p = os.path.join(spec["dir"], video_id + spec["ext"])
    if not os.path.exists(p):
        raise FileNotFoundError(f"video missing: {p}")
    return p
