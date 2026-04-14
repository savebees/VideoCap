#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models/Qwen2.5-VL-32B-Instruct}"
TP_SIZE="${TP_SIZE:-2}"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICES:-0,1}"
vllm serve "$MODEL_DIR" \
    --served-model-name "Qwen2.5-VL-32B-Instruct" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --trust-remote-code \
    --override-generation-config '{"max_pixels": 156800}'
