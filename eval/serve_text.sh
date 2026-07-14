#!/bin/bash
# Text replica (claims + omission): Gemma3-27B-it, GPUs 4-7, :8011.
BASE=/nfshdd/21039533r/zxhe; CACHE=$BASE/cache
export TMPDIR=$BASE/tmp XDG_CACHE_HOME=$CACHE HF_HOME=$CACHE/hf VLLM_CACHE_ROOT=$CACHE/vllm OUTLINES_CACHE_DIR=$CACHE/outlines TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 XDG_CONFIG_HOME=$CACHE/config
export PATH=/localnvme/zxhe/miniconda3/envs/physeval/bin:$PATH PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4,5,6,7
exec vllm serve $BASE/models/gemma-3-27b-it --served-model-name gemma-3-27b-it \
  --host 0.0.0.0 --port 8011 --tensor-parallel-size 4 --max-model-len 32768 \
  --dtype bfloat16 --gpu-memory-utilization 0.85 --trust-remote-code
