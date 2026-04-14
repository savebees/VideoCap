import json
import re
import subprocess


def get_video_metadata(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    video_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    fmt = probe.get("format", {})
    num, den = video_stream.get("avg_frame_rate", "0/1").split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0

    return {
        "duration": float(fmt.get("duration", video_stream.get("duration", 0))),
        "fps": round(fps, 2),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "codec": video_stream.get("codec_name", "unknown"),
        "file_size_mb": round(int(fmt.get("size", 0)) / (1024 * 1024), 2),
    }


def parse_json_array(raw_text: str) -> list:
    """Parse JSON array from VLM output via regex extraction."""
    if not raw_text or not raw_text.strip():
        raise json.JSONDecodeError("Empty response", raw_text or "", 0)
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if match is None:
        raise json.JSONDecodeError("No JSON array found", raw_text, 0)
    return json.loads(match.group(0))
