from videocap.qa import derive_qa


def test_derive_qa_uses_video_understanding_tasks():
    record = {
        "schema_version": "videocap/v0.2",
        "video_id": "demo",
        "duration_ms": 10_000,
        "captions": {
            "short": "A rabbit leaves its burrow.",
            "main_object": "A rabbit wakes, stretches, and walks outside.",
            "background": "The scene moves from a burrow to a meadow.",
            "camera": "Wide shots alternate with close-ups.",
            "detailed": "A rabbit wakes, exits its burrow, and enters a meadow.",
        },
        "events": [
            {
                "event_id": "event_0000",
                "start_ms": 1_000,
                "end_ms": 4_000,
                "evidence_frames_ms": [1_500, 3_500],
                "caption": "The rabbit leaves its burrow.",
            }
        ],
    }

    result = derive_qa(record, split="train")
    by_task = {example["task"]: example for example in result["examples"]}

    assert set(by_task) == {
        "video_summary",
        "action_recognition",
        "scene_transition",
        "camera_understanding",
        "temporal_reasoning",
        "event_understanding",
        "temporal_grounding",
    }
    assert by_task["temporal_grounding"]["answer"] == "From 1000 ms to 4000 ms."
    assert by_task["temporal_grounding"]["provenance"] == {
        "event_ids": ["event_0000"],
        "evidence_frames_ms": [1_500, 3_500],
    }
