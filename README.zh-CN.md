# VideoCap

[English](README.md)

VideoCap 将本地视频转换为两层互补标注：完整视频的五维 caption，以及具有时间边界的语义事件 caption。项目采用一条固定、可检查的生产链路，使 processing window、模型证据、事件边界和最终结果都能被直接追溯。

## 流程

```text
视频 manifest
  -> 生成重叠 processing windows
  -> VLM：为每个窗口生成五维 caption
  -> LLM：根据 short + main_object 提议语义事件
  -> VLM：粗定位并精修事件边界
  -> VLM：为每个定位后的事件生成 caption
  -> LLM：合并事件与窗口证据，生成全局五维 caption
  -> 最终 JSONL + 分阶段产物
```

Processing window 是模型输入单元，不是标注边界。LLM 一次性把连续窗口组织成语义事件，VLM 再从视觉证据中选择起止时间，并且只描述最终时间窗内的内容。全局合并以有时间依据的事件为叙事主线，用窗口 caption 补充主体、环境、镜头和细节信息。

## 环境要求

- Python 3.10 或更高版本
- 系统 `PATH` 中可以调用 `ffmpeg`
- 支持 `image_url` data URI 的 OpenAI-compatible Chat Completions 接口

推荐使用 `uv` 安装：

```bash
git clone https://github.com/savebees/VideoCap.git
cd VideoCap
uv sync --locked
```

也可以使用标准 editable 安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 配置

全部运行参数集中在 [`configs/videocap.json`](configs/videocap.json)。VLM 和 LLM 使用同一套精简的 OpenAI-compatible 配置结构，也可以分别连接不同 provider 或模型。

```json
{
  "pipeline": {
    "window_ms": 24000,
    "overlap_ms": 2000,
    "evidence_frames": 8,
    "output_name": "annotations.jsonl"
  },
  "vlm": {
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key_env": "SILICONFLOW_API_KEY",
    "model": "Qwen/Qwen3.6-35B-A3B",
    "frame_height": 460,
    "timeout_sec": 120,
    "max_retries": 2,
    "extra_body": {"enable_thinking": false}
  },
  "llm": {
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key_env": "SILICONFLOW_API_KEY",
    "model": "Qwen/Qwen3.6-35B-A3B",
    "timeout_sec": 120,
    "max_retries": 2,
    "extra_body": {"enable_thinking": false}
  }
}
```

API key 只从配置指定的环境变量读取。Run 目录中的配置会保留 endpoint 和模型信息，但不会写入凭证。

## 视频 Manifest

输入文件采用 UTF-8 JSONL，每行对应一个视频：

```json
{"video_id":"video_001","video_path":"videos/video_001.mp4","duration_ms":68320}
{"video_id":"video_002","video_path":"videos/video_002.mp4","duration_ms":124500,"metadata":{"split":"demo"}}
```

相对路径以 manifest 所在目录为基准解析。`video_id` 必须唯一，视频文件必须存在，时长单位为毫秒。

## 运行

```bash
export SILICONFLOW_API_KEY="..."
uv run videocap run videos.jsonl \
  --config configs/videocap.json \
  --output-root runs
```

每次执行都会创建独立的 run 目录：

```text
runs/<run_id>/
├── annotations.jsonl
├── config.json
├── failures.jsonl
├── manifest.json
├── summary.json
└── stages/<video_id>/
    ├── processing_windows.jsonl
    ├── window_captions.jsonl
    ├── event_proposals.jsonl
    ├── event_boundaries.jsonl
    ├── event_captions.jsonl
    ├── global_caption.json
    └── failure.json              # 仅在该视频失败时生成
```

Run manifest 会记录数据集与配置哈希、VideoCap 版本、Git 状态和创建时间。单个视频失败时会明确留下错误记录，同时保留同一批次中已经成功的结果。

## 输出格式

```json
{
  "schema_version": "videocap/v0.2",
  "video_id": "video_001",
  "duration_ms": 68320,
  "captions": {
    "short": "一名男子准备并端出一道做好的菜。",
    "main_object": "一名男子准备食材、完成烹饪并端走装好的餐盘。",
    "background": "活动发生在厨房和相邻的用餐区域。",
    "camera": "镜头主要以中景记录，并从料理台跟随至餐桌。",
    "detailed": "一名男子准备食材、完成烹饪、装盘，并把食物端到桌上。"
  },
  "events": [
    {
      "event_id": "event_0000",
      "start_ms": 1750,
      "end_ms": 51250,
      "evidence_frames_ms": [1750, 51250],
      "caption": "一名男子准备食材、完成烹饪，并将做好的食物装盘。"
    }
  ]
}
```

公开 schema 位于 [`videocap/schemas/videocap.schema.json`](videocap/schemas/videocap.schema.json)。Prompt 模板集中在 [`videocap/prompts`](videocap/prompts)，与 provider 无关的 VLM 和 LLM 实现则分别位于 [`videocap/adapters`](videocap/adapters)。五维 caption 采用 [AuroraCap](https://github.com/wenhaochai/aurora) 公开的 VDC prompt 体系。

## 开发

```bash
uv sync --locked
uv run ruff check .
uv run pytest
uv build
```

当前版本明确不包含的工作统一记录在 [`TODO.md`](TODO.md)。

## License

VideoCap 使用 [MIT License](LICENSE) 发布。
