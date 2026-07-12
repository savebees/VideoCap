#!/bin/bash
# Process ONE Physics-IQ video. Args: <video_path>
v=$1
BASE=/nfshdd/21039533r/zxhe
REPO=$BASE/dense-video-annotator
CACHE=$BASE/cache
export TMPDIR=$BASE/tmp XDG_CACHE_HOME=$CACHE HF_HOME=$CACHE/hf
export PATH=/localnvme/zxhe/miniconda3/envs/physeval/bin:$PATH
cd $REPO
id=$(basename "$v" .mp4)
if [ -f "results/physics_iq/$id/annotation.json" ]; then echo "SKIP $id"; exit 0; fi
echo "RUN $id (start $(date +%H:%M:%S))"
if python -u src/pipeline.py --video "$v" --config configs/physics_iq.yaml >> logs/par_physics_iq.log 2>&1; then
  echo "OK  $id (end $(date +%H:%M:%S))"
else
  echo "FAIL $id (exit $?)"
fi
