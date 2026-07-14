#!/bin/bash
# One model baseline, full lifecycle: download -> serve -> smoke -> splits -> stop.
# Usage: run_one_model.sh <TAG> <HF_ID> <SERVED_NAME> <INPUT_FORMAT> [EXTRA_SERVE_ARGS...]
set -u
TAG=$1; HF_ID=$2; SERVED=$3; INPUT_FORMAT=$4; shift 4
EXTRA_SERVE_ARGS=("$@")

BASE=/nfshdd/21039533r/zxhe
REPO=$BASE/dense-video-annotator
CACHE=$BASE/cache
MODELS=$BASE/models
export TMPDIR=$BASE/tmp
export XDG_CACHE_HOME=$CACHE
export HF_HOME=$CACHE/hf
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
export TRITON_CACHE_DIR=$CACHE/triton
export VLLM_CACHE_ROOT=$CACHE/vllm
export OUTLINES_CACHE_DIR=$CACHE/outlines
export PATH=/localnvme/zxhe/miniconda3/envs/physeval/bin:$PATH
export PYTORCH_ALLOC_CONF=expandable_segments:True
# telemetry writes to the full root disk; disable
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export XDG_CONFIG_HOME=$BASE/cache/config
# GPU set + port parameterizable so two tracks can run in parallel
GPUS="${GPUS:-4,5,6,7}"
PORT="${PORT:-8002}"
# memory knobs; InternVL needs smaller MAXLEN/MAXSEQS or KV cache won't fit
MAXLEN="${MAXLEN:-32768}"
GPUUTIL="${GPUUTIL:-0.85}"
MAXSEQS="${MAXSEQS:-256}"
VID=$BASE/datasets/nwpu_campus/NWPUCampusDataset/videos
YT=$BASE/datasets/youtube
PIQ=$BASE/datasets/physics_iq/30FPS
NPROC="${NPROC:-8}"
# override SPLITS to run a subset, e.g. SPLITS="physics_iq"
SPLITS="${SPLITS:-Test Train youtube physics_iq}"
cd $REPO

log(){ echo "[$(date +%H:%M:%S)] [$TAG] $*"; }

# 1. download
MODEL_DIR=$MODELS/$TAG
if [ -f "$MODEL_DIR/config.json" ]; then
  log "model already present at $MODEL_DIR"
else
  log "downloading $HF_ID -> $MODEL_DIR"
  hf download "$HF_ID" --local-dir "$MODEL_DIR" \
      --exclude "*.pth" --exclude "original/*" --exclude "*.gguf" >> logs/baseline_dl_$TAG.log 2>&1
  if [ ! -f "$MODEL_DIR/config.json" ]; then log "DOWNLOAD FAILED"; exit 2; fi
fi

# 2. serve
log "starting vLLM (TP=4, port $PORT)"
CUDA_VISIBLE_DEVICES=$GPUS nohup vllm serve "$MODEL_DIR" \
    --served-model-name "$SERVED" \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 4 --data-parallel-size 1 \
    --dtype bfloat16 --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$GPUUTIL" \
    --max-num-seqs "$MAXSEQS" \
    --trust-remote-code \
    "${EXTRA_SERVE_ARGS[@]}" \
    > logs/baseline_vllm_$TAG.log 2>&1 &
VLLM_PID=$!
log "vLLM PID $VLLM_PID; waiting for ready..."

ready=0
for i in $(seq 1 120); do   # up to ~30 min (download+compile cold)
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/v1/models 2>/dev/null)" = "200" ]; then ready=1; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then log "vLLM DIED during startup"; tail -20 logs/baseline_vllm_$TAG.log; exit 3; fi
  sleep 15
done
if [ "$ready" != "1" ]; then log "vLLM NOT READY (timeout)"; kill -9 $VLLM_PID 2>/dev/null; exit 4; fi
log "vLLM ready"

stop_vllm(){
  log "stopping vLLM"
  kill -9 $VLLM_PID 2>/dev/null
  sleep 2
  # TP workers rename themselves; kill whatever still holds this track's GPUs
  local p
  for p in $(nvidia-smi -i $GPUS --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
    kill -9 "$p" 2>/dev/null
  done
  sleep 5
}

# 3. per-split configs
mk_cfg(){ # <split> <outdir>
  local split=$1
  local outdir=$2
  local cfg=baseline/cfg_${TAG}_${split}.yaml
  cat > "$cfg" <<EOF
vlm_model: "$SERVED"
vllm_base_url: "http://localhost:$PORT/v1"
num_frames: 32
frame_quality: 95
frame_max_long_side: 672
vlm_temperature: 0.7
vlm_presence_penalty: 1.5
vlm_max_tokens: 2048
vlm_top_p: 0.8
vlm_top_k: 20
enable_thinking: false
input_format: "$INPUT_FORMAT"
output_dir: "$outdir"
EOF
  echo "$cfg"
}
CFG_TEST=$(mk_cfg Test       baseline/results/$TAG/Test)
CFG_TRAIN=$(mk_cfg Train      baseline/results/$TAG/Train)
CFG_YT=$(mk_cfg youtube    baseline/results/$TAG/youtube)
CFG_PIQ=$(mk_cfg physics_iq baseline/results/$TAG/physics_iq)

# 4. smoke test on the run's own first split (never re-forces another dataset)
case "${SPLITS%% *}" in
  Test)       SMOKE_VID="$VID/Test/D001_03.avi";  SMOKE_CFG=$CFG_TEST  ;;
  Train)      SMOKE_VID="$VID/Train/D001_03.avi"; SMOKE_CFG=$CFG_TRAIN ;;
  youtube)    SMOKE_VID=$(ls "$YT"/*.mp4 | head -1);  SMOKE_CFG=$CFG_YT  ;;
  physics_iq) SMOKE_VID=$(ls "$PIQ"/*.mp4 | head -1); SMOKE_CFG=$CFG_PIQ ;;
esac
SMOKE_ID=$(basename "$SMOKE_VID"); SMOKE_ID="${SMOKE_ID%.*}"
log "smoke test on ${SPLITS%% *}/$SMOKE_ID"
python -u baseline/run_model_baseline.py --video "$SMOKE_VID" --config "$SMOKE_CFG" --force > logs/baseline_smoke_$TAG.log 2>&1
sw=$(python -c "import json;print(json.load(open('baseline/results/$TAG/${SPLITS%% *}/$SMOKE_ID.json'))['word_count'])" 2>/dev/null || echo 0)
if [ "${sw:-0}" -lt 1 ]; then log "SMOKE FAILED (empty caption)"; tail -25 logs/baseline_smoke_$TAG.log; stop_vllm; exit 5; fi
log "smoke OK ($sw words)"

# 5. run splits (parallel per-video)
run_split(){ # <name> <dir> <cfg>
  local name=$1 dir=$2 cfg=$3
  log "RUN split $name"
  ls "$dir"/*.avi "$dir"/*.mp4 2>/dev/null | \
    xargs -P "$NPROC" -I{} bash -c 'python -u baseline/run_model_baseline.py --video "{}" --config "'"$cfg"'" >> logs/baseline_run_'"$TAG"'_'"$name"'.log 2>&1 && echo "OK $(basename "{}")" || echo "FAIL $(basename "{}")"' \
    >> logs/baseline_progress_$TAG.log 2>&1
  log "DONE split $name"
}
for s in $SPLITS; do
  case "$s" in
    Test)       run_split Test       "$VID/Test"  "$CFG_TEST"  ;;
    Train)      run_split Train      "$VID/Train" "$CFG_TRAIN" ;;
    youtube)    run_split youtube    "$YT"        "$CFG_YT"    ;;
    physics_iq) run_split physics_iq "$PIQ"       "$CFG_PIQ"   ;;
    *) log "UNKNOWN SPLIT '$s'"; stop_vllm; exit 6 ;;
  esac
done

# 6. stop
stop_vllm
for s in $SPLITS; do
  log "MODEL DONE: $s=$(ls baseline/results/$TAG/$s/*.json 2>/dev/null | wc -l)"
done
