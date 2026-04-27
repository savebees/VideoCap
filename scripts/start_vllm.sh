#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3.6-35B-A3B}"
TP_SIZE="${TP_SIZE:-2}"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICES:-0,1}"
vllm serve "$MODEL_DIR" \
    --served-model-name "Qwen3.6-35B-A3B" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --override-generation-config '{"max_pixels": 156800}' \
    --media-io-kwargs '{"video": {"num_frames": -1}}'
