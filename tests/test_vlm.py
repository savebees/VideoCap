import pytest

from videocap.adapters.vlm import (
    coarse_boundary_frames,
    fine_boundary_frames,
    parse_boundary,
    uniform_timestamps,
)
from videocap.structured import EventProposal, ProcessingWindow


def test_uniform_sampling_includes_both_endpoints():
    assert uniform_timestamps(100, 500, 3) == (100, 300, 500)


def test_boundary_sampling_is_bounded_and_uses_four_fps_for_fine_review():
    window = ProcessingWindow("W0001", 0, 24_000, (0, 23_999))
    event = EventProposal("event_0000", ("W0001",), "A person cooks.", 0, 24_000)
    coarse, starts, ends = coarse_boundary_frames(event, (window,))
    assert len(coarse) == 24
    assert coarse == starts == ends
    assert coarse[-1] == 23_999

    fine, starts, ends = fine_boundary_frames(event, 5_000, 20_000)
    assert starts == tuple(range(3_500, 6_500, 250))
    assert ends == tuple(range(18_500, 21_500, 250))
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
