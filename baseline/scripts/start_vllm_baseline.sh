#!/bin/bash
# Second vLLM replica for the A3B baseline (GPUs 4-7, :8001); same served model
# as the pipeline replica on 0-3.
set -e
BASE=/nfshdd/21039533r/zxhe
CACHE=$BASE/cache
export TMPDIR=$BASE/tmp
export XDG_CACHE_HOME=$CACHE
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
export TRITON_CACHE_DIR=$CACHE/triton
export HF_HOME=$CACHE/hf
export VLLM_CACHE_ROOT=$CACHE/vllm
export OUTLINES_CACHE_DIR=$CACHE/outlines
export PATH=/localnvme/zxhe/miniconda3/envs/physeval/bin:$PATH
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL_DIR=$BASE/models/Qwen3.6-35B-A3B

exec vllm serve "$MODEL_DIR" \
    --served-model-name "Qwen3.6-35B-A3B" \
    --host 0.0.0.0 --port 8001 \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --media-io-kwargs '{"video": {"num_frames": -1}}'
