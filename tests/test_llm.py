import pytest

from videocap.adapters.llm import parse_dimensions, parse_event_proposals
from videocap.structured import DIMENSIONS, ProcessingWindow


def test_dimension_parser_requires_all_fields_in_order():
    text = "\n".join(f"[{name}]\n{name} caption" for name in DIMENSIONS)
    assert tuple(parse_dimensions(text)) == DIMENSIONS

    with pytest.raises(ValueError, match="in order"):
        parse_dimensions("[short]\nOnly one field")


def test_event_parser_accepts_shared_windows_and_derives_intervals():
    windows = (
        ProcessingWindow("W0001", 0, 10_000, (0, 9_999)),
        ProcessingWindow("W0002", 9_000, 19_000, (9_000, 18_999)),
    )
    text = """EVENT
WINDOWS: W0001, W0002
CAPTION: A person prepares food.
END_EVENT
EVENT
WINDOWS: W0002
CAPTION: The person serves the food.
END_EVENT"""

    proposals = parse_event_proposals(text, windows)

    assert proposals[0].source_window_ids == ("W0001", "W0002")
    assert (proposals[0].start_ms, proposals[0].end_ms) == (0, 19_000)
    assert proposals[1].event_id == "event_0001"


def test_event_parser_rejects_nonconsecutive_windows():
    windows = tuple(
        ProcessingWindow(
            f"W{index:04d}",
            index * 1000,
            (index + 1) * 1000,
            (index * 1000, (index + 1) * 1000 - 1),
        )
        for index in range(1, 4)
    )
    with pytest.raises(ValueError, match="consecutive"):
        parse_event_proposals(
            "EVENT\nWINDOWS: W0001, W0003\nCAPTION: Jump.\nEND_EVENT",
            windows,
        )
