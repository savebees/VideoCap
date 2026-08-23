import json

import pytest

from videocap.dataset import load_manifest, manifest_sha256


def test_manifest_resolves_relative_video_paths(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "demo",
                "video_path": "video.mp4",
                "duration_ms": 2_000,
                "metadata": {"split": "demo"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sample = load_manifest(manifest)[0]

    assert sample.video_path == video.resolve()
    assert sample.metadata == {"split": "demo"}
    assert len(manifest_sha256(manifest)) == 64


def test_manifest_rejects_duplicate_video_ids(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    record = {"video_id": "same", "video_path": "video.mp4", "duration_ms": 1_000}
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate video_id"):
        load_manifest(manifest)


def test_manifest_rejects_video_ids_that_escape_the_stage_directory(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(
        json.dumps({"video_id": "../outside", "video_path": "video.mp4", "duration_ms": 1000})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="directory name"):
        load_manifest(manifest)
