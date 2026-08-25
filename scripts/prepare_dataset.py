#!/usr/bin/env python3
"""Build a VideoCap JSONL manifest from a directory of videos."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from videocap.dataset import load_manifest
from videocap.io import atomic_write_jsonl

VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def _duration_ms(video_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = result.stdout.strip()
    if duration != "N/A":
        return round(float(duration) * 1_000)

    packets = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,duration_time",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not packets:
        raise ValueError(f"video stream has no packets: {video_path}")
    pts, packet_duration = map(float, packets[-1].split(","))
    return round((pts + packet_duration) * 1_000)


def prepare_dataset(video_dir: str | Path, output: str | Path) -> int:
    root = Path(video_dir).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    videos = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise ValueError(f"no supported videos found in {root}")

    records = []
    video_ids: set[str] = set()
    for video_path in videos:
        relative = video_path.relative_to(root)
        video_id = relative.with_suffix("").as_posix().replace("/", "__")
        if video_id in video_ids:
            raise ValueError(f"duplicate video_id after normalization: {video_id}")
        video_ids.add(video_id)
        records.append(
            {
                "video_id": video_id,
                "video_path": Path(os.path.relpath(video_path, output_path.parent)).as_posix(),
                "duration_ms": _duration_ms(video_path),
            }
        )

    atomic_write_jsonl(output_path, records)
    load_manifest(output_path)
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", help="Directory scanned recursively for videos")
    parser.add_argument("--output", default="videos.jsonl", help="Output JSONL manifest")
    args = parser.parse_args()
    count = prepare_dataset(args.video_dir, args.output)
    print(f"Wrote {count} videos to {Path(args.output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
