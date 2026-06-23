#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_DIR="${MODEL_DIR:-/mnt/data/zxhe/models/Qwen3.6-35B-A3B}"
# Total GPUs = TP_SIZE * DP_SIZE
TP_SIZE="${TP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-1}"
# headroom for multi-frame prefills; lower on small cards
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES="${GPU_DEVICES:-0,1,2,3}"
vllm serve "$MODEL_DIR" \
    --served-model-name "Qwen3.6-35B-A3B" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "$TP_SIZE" \
    --data-parallel-size "$DP_SIZE" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --media-io-kwargs '{"video": {"num_frames": -1}}'
