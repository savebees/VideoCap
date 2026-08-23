import json

import pytest

from videocap.config import Config


def test_config_keeps_vlm_and_llm_endpoints_independent(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "pipeline": {"window_ms": 10_000, "overlap_ms": 1_000},
                "vlm": {
                    "base_url": "https://vision.example/v1/",
                    "api_key_env": "VISION_KEY",
                    "model": "vision-model",
                    "frame_height": 512,
                },
                "llm": {
                    "base_url": "https://text.example/v1",
                    "api_key_env": "TEXT_KEY",
                    "model": "text-model",
                },
            }
        ),
        encoding="utf-8",
    )

    config = Config.load(path)

    assert config.vlm.base_url == "https://vision.example/v1"
    assert config.vlm.frame_height == 512
    assert config.llm.model == "text-model"
    assert config.pipeline.evidence_frames == 8


def test_pipeline_config_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="overlap_ms"):
        Config.from_dict(
            {
                "pipeline": {"window_ms": 1000, "overlap_ms": 1000},
                "vlm": {"base_url": "x", "api_key_env": "X", "model": "x"},
                "llm": {"base_url": "x", "api_key_env": "X", "model": "x"},
            }
        )
