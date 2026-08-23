from videocap.config import PipelineConfig
from videocap.pipeline import VideoCap, processing_windows
from videocap.structured import (
    DIMENSIONS,
    EventCaption,
    EventProposal,
    EventWindow,
    VideoSample,
    WindowCaption,
)


class FakeVLM:
    def caption_window(self, sample, window):
        return WindowCaption(
            window.window_id,
            {name: f"{name} for {window.window_id}" for name in DIMENSIONS},
            window.evidence_frames_ms,
        )

    def review_event_boundary(self, sample, proposal, windows):
        return EventWindow(
            proposal.event_id,
            proposal.start_ms,
            proposal.end_ms - 1,
            (proposal.start_ms, proposal.end_ms - 1),
        )

    def caption_event(self, sample, event):
        return EventCaption(event, "A complete event caption.", (event.start_ms, event.end_ms))


class FakeLLM:
    def propose_events(self, windows, captions):
        return (
            EventProposal(
                "event_0000",
                tuple(window.window_id for window in windows),
                "One coherent activity.",
                windows[0].start_ms,
                windows[-1].end_ms,
            ),
        )

    def merge_global_caption(self, windows, window_captions, events):
        return {name: f"Global {name}." for name in DIMENSIONS}


def test_processing_windows_use_the_public_default_profile():
    windows = processing_windows(60_000, PipelineConfig())
    assert [(window.start_ms, window.end_ms) for window in windows] == [
        (0, 24_000),
        (22_000, 46_000),
        (44_000, 60_000),
    ]
    assert all(len(window.evidence_frames_ms) == 8 for window in windows)
    assert windows[-1].evidence_frames_ms[-1] == 59_999


def test_pipeline_writes_only_real_annotation_stages(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    sample = VideoSample("demo", video, 4_000)
    pipeline = VideoCap(
        FakeVLM(),
        FakeLLM(),
        PipelineConfig(window_ms=4_000, overlap_ms=0, evidence_frames=3),
    )

    record = pipeline.process(sample, tmp_path / "stages")

    assert record["schema_version"] == "videocap/v0.2"
    assert record["events"][0]["caption"] == "A complete event caption."
    assert {path.name for path in (tmp_path / "stages").iterdir()} == {
        "processing_windows.jsonl",
        "window_captions.jsonl",
        "event_proposals.jsonl",
        "event_boundaries.jsonl",
        "event_captions.jsonl",
        "global_caption.json",
    }
