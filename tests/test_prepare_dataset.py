from scripts import prepare_dataset
from videocap.dataset import load_manifest


def test_prepare_dataset_builds_a_valid_recursive_manifest(tmp_path, monkeypatch):
    video_dir = tmp_path / "videos"
    nested = video_dir / "nested"
    nested.mkdir(parents=True)
    (video_dir / "first.mp4").write_bytes(b"video")
    (nested / "second.mov").write_bytes(b"video")
    (video_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    output = tmp_path / "videos.jsonl"
    monkeypatch.setattr(prepare_dataset, "_duration_ms", lambda path: len(path.name) * 1_000)

    count = prepare_dataset.prepare_dataset(video_dir, output)
    samples = load_manifest(output)

    assert count == 2
    assert [sample.video_id for sample in samples] == ["first", "nested__second"]
    assert [sample.duration_ms for sample in samples] == [9_000, 10_000]
