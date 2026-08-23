"""Run orchestration and provenance artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from videocap.config import Config
from videocap.dataset import manifest_sha256
from videocap.io import atomic_write_json, atomic_write_jsonl
from videocap.pipeline import VideoCap
from videocap.structured import VideoSample


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    summary: Mapping[str, Any]


def _git_state() -> tuple[str | None, bool]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode:
        return None, False
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return commit.stdout.strip(), bool(status.stdout.strip())


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"videocap__{timestamp}__{uuid.uuid4().hex[:8]}"


def run(
    pipeline: VideoCap,
    samples: Sequence[VideoSample],
    config: Config,
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    run_id: str | None = None,
) -> RunResult:
    resolved_id = run_id or _run_id()
    if Path(resolved_id).name != resolved_id:
        raise ValueError("run_id must be a plain directory name")
    run_dir = Path(output_root).expanduser().resolve() / resolved_id
    run_dir.mkdir(parents=True, exist_ok=False)
    stages_dir = run_dir / "stages"
    stages_dir.mkdir()

    config_document = config.to_dict()
    config_json = json.dumps(config_document, sort_keys=True, separators=(",", ":"))
    git_commit, git_dirty = _git_state()
    manifest = {
        "schema_version": "videocap-run/v0.1",
        "run_id": resolved_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "videocap_version": pipeline.version,
        "dataset": {
            "manifest": Path(manifest_path).name,
            "sha256": manifest_sha256(manifest_path),
            "videos": len(samples),
        },
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "config.json", config_document)

    started = time.perf_counter()
    records: list[Mapping[str, Any]] = []
    failures: list[Mapping[str, Any]] = []
    for sample in samples:
        sample_dir = stages_dir / sample.video_id
        try:
            records.append(pipeline.process(sample, sample_dir))
        except Exception as exc:
            failure = {
                "video_id": sample.video_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            failures.append(failure)
            sample_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(sample_dir / "failure.json", failure)

    atomic_write_jsonl(run_dir / config.pipeline.output_name, records)
    atomic_write_jsonl(run_dir / "failures.jsonl", failures)
    summary = {
        "run_id": resolved_id,
        "videos": len(samples),
        "succeeded": len(records),
        "failed": len(failures),
        "duration_sec": round(time.perf_counter() - started, 3),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    return RunResult(run_dir, summary)


__all__ = ["RunResult", "run"]
