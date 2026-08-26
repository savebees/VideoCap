from pathlib import Path
from types import SimpleNamespace

import pytest

from videocap.adapters.vlm import (
    VLM,
    coarse_boundary_frames,
    fine_boundary_frames,
    parse_boundary,
    uniform_timestamps,
)
from videocap.structured import EventProposal, ProcessingWindow, VideoSample


def test_uniform_sampling_includes_both_endpoints():
    assert uniform_timestamps(100, 500, 3) == (100, 300, 500)


def test_frame_extraction_stays_before_the_media_endpoint(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=b"jpeg")

    monkeypatch.setattr("videocap.adapters.vlm.subprocess.run", run)
    vlm = VLM.__new__(VLM)
    vlm.frame_height = 460
    vlm._frames = {}
    sample = VideoSample("clip", Path("clip.mp4"), 20_020)

    vlm._image(sample, 20_019)

    assert commands[0][commands[0].index("-ss") + 1] == "19.920"


def test_boundary_sampling_uses_one_fps_coarse_and_dynamic_fine_review():
    window = ProcessingWindow("W0001", 0, 24_000, (0, 23_999))
    event = EventProposal("event_0000", ("W0001",), "A person cooks.", 0, 24_000)
    coarse, starts, ends = coarse_boundary_frames(event, (window,))
    assert len(coarse) == 24
    assert coarse == starts == ends
    assert coarse == (*range(0, 23_000, 1_000), 23_999)

    fine, starts, ends = fine_boundary_frames(event, 5_000, 20_000, coarse)
    assert starts == (
        3_000,
        3_364,
        3_727,
        4_091,
        4_455,
        4_818,
        5_182,
        5_545,
        5_909,
        6_273,
        6_636,
        7_000,
    )
    assert ends == (
        18_000,
        18_364,
        18_727,
        19_091,
        19_455,
        19_818,
        20_182,
        20_545,
        20_909,
        21_273,
        21_636,
        22_000,
    )
    assert len(fine) == 24


def test_coarse_sampling_caps_long_events_and_fine_sampling_keeps_media_edges():
    windows = (
        ProcessingWindow("W0001", 0, 24_000, (0, 23_999)),
        ProcessingWindow("W0002", 22_000, 40_000, (22_000, 39_999)),
    )
    event = EventProposal(
        "event_0000", ("W0001", "W0002"), "A rabbit explores a meadow.", 0, 40_000
    )

    coarse, starts, ends = coarse_boundary_frames(event, windows)
    assert len(coarse) == 24
    assert coarse == starts == ends
    assert coarse[0] == 0
    assert coarse[-1] == 39_999

    fine, starts, ends = fine_boundary_frames(event, 0, 39_999, coarse)
    assert starts[0] == 0
    assert ends[-1] == 39_999
    assert len(fine) <= 24


def test_fine_sampling_expands_with_long_event_coarse_spacing():
    windows = tuple(
        ProcessingWindow(
            f"W{index + 1:04d}",
            index * 22_000,
            min(index * 22_000 + 24_000, 90_000),
            (index * 22_000, min(index * 22_000 + 24_000, 90_000) - 1),
        )
        for index in range(4)
    )
    event = EventProposal(
        "event_0000",
        tuple(window.window_id for window in windows),
        "A person completes one long activity.",
        0,
        90_000,
    )
    coarse, _, _ = coarse_boundary_frames(event, windows)
    coarse_start, coarse_end = coarse[3], coarse[-4]

    fine, starts, ends = fine_boundary_frames(event, coarse_start, coarse_end, coarse)

    local_start_gap = max(coarse_start - coarse[2], coarse[4] - coarse_start)
    local_end_gap = max(coarse_end - coarse[-5], coarse[-3] - coarse_end)
    assert starts[0] == coarse_start - local_start_gap
    assert starts[-1] == coarse_start + local_start_gap
    assert ends[0] == coarse_end - local_end_gap
    assert ends[-1] == coarse_end + local_end_gap
    assert len(starts) == len(ends) == 12
    assert len(fine) == 24


def test_boundary_parser_requires_supplied_frames():
    assert parse_boundary(
        "STATUS: OK\nSTART_MS: 1000\nEND_MS: 3000",
        start_frames=(0, 1_000),
        end_frames=(3_000, 4_000),
        coarse=True,
    ) == (1_000, 3_000)

    with pytest.raises(ValueError, match="supplied frames"):
        parse_boundary(
            "START_MS: 1500\nEND_MS: 3000",
            start_frames=(0, 1_000),
            end_frames=(3_000, 4_000),
            coarse=False,
        )
