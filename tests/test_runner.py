import json

from videocap.config import Config
from videocap.runner import run
from videocap.structured import VideoSample


class StubPipeline:
    version = "test"

    def process(self, sample, output_dir):
        output_dir.mkdir(parents=True)
        return {"video_id": sample.video_id}


def test_runner_writes_compact_reproducibility_artifacts(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(
        json.dumps({"video_id": "demo", "video_path": "video.mp4", "duration_ms": 1000}) + "\n",
        encoding="utf-8",
    )
    config = Config.from_dict(
        {
            "vlm": {"base_url": "https://vlm.example/v1", "api_key_env": "VLM_KEY", "model": "v"},
            "llm": {"base_url": "https://llm.example/v1", "api_key_env": "LLM_KEY", "model": "l"},
        }
    )

    result = run(
        StubPipeline(),
        (VideoSample("demo", video, 1000),),
        config,
        manifest,
        tmp_path / "runs",
        run_id="test-run",
    )

    assert result.summary["succeeded"] == 1
    assert {path.name for path in result.run_dir.iterdir()} == {
        "annotations.jsonl",
        "config.json",
        "failures.jsonl",
        "manifest.json",
        "stages",
        "summary.json",
    }
    resolved = json.loads((result.run_dir / "config.json").read_text(encoding="utf-8"))
    assert resolved["vlm"]["api_key_env"] == "VLM_KEY"
    assert "api_key" not in resolved["vlm"]
