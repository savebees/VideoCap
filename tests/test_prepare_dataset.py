from pathlib import Path
from types import SimpleNamespace

from scripts import prepare_dataset
from videocap.dataset import load_manifest


def test_duration_uses_the_video_stream(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="29.696363\n")

    monkeypatch.setattr(prepare_dataset.subprocess, "run", run)

    assert prepare_dataset._duration_ms(Path("clip.mp4")) == 29_696
    assert commands[0][commands[0].index("-select_streams") + 1] == "v:0"
    assert commands[0][commands[0].index("-show_entries") + 1] == "stream=duration"


def test_duration_uses_the_final_video_packet_when_stream_duration_is_missing(monkeypatch):
    responses = iter(("N/A\n", "0.000000,0.033000\n29.663000,0.033000\n"))

    def run(command, **kwargs):
        return SimpleNamespace(stdout=next(responses))

    monkeypatch.setattr(prepare_dataset.subprocess, "run", run)

    assert prepare_dataset._duration_ms(Path("clip.mp4")) == 29_696


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
