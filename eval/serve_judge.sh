#!/bin/bash
# Visual judge: Gemma3-27B-it, GPUs 0-3, :8010. mm cache off (vLLM assertion bug);
# image limit raised for 48-frame requests.
BASE=/nfshdd/21039533r/zxhe; CACHE=$BASE/cache
export TMPDIR=$BASE/tmp XDG_CACHE_HOME=$CACHE HF_HOME=$CACHE/hf VLLM_CACHE_ROOT=$CACHE/vllm OUTLINES_CACHE_DIR=$CACHE/outlines TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 XDG_CONFIG_HOME=$CACHE/config
export PATH=/localnvme/zxhe/miniconda3/envs/physeval/bin:$PATH PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3
exec vllm serve $BASE/models/gemma-3-27b-it --served-model-name gemma-3-27b-it \
  --host 0.0.0.0 --port 8010 --tensor-parallel-size 4 --max-model-len 32768 \
  --dtype bfloat16 --gpu-memory-utilization 0.90 --trust-remote-code \
  --limit-mm-per-prompt '{"image": 60}' --mm-processor-cache-gb 0 \
  --media-io-kwargs '{"video": {"num_frames": -1}}'
