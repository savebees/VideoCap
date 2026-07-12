#!/bin/bash
# Parallel Physics-IQ annotation: N concurrent video workers against the TP=4 vLLM
# on :8000. Physics-IQ is one flat split of 198 take-1 clips, so unlike NWPU there is
# no per-split config or Train/Test filename collision to work around.
# Resumable: a clip whose annotation.json already exists is skipped.
BASE=/nfshdd/21039533r/zxhe
REPO=$BASE/dense-video-annotator
VID=$BASE/datasets/physics_iq/30FPS
export TMPDIR=$BASE/tmp
NPROC="${NPROC:-8}"
cd $REPO
echo "########## [$(date)] PHYSICS-IQ PARALLEL BATCH START (NPROC=$NPROC) ##########"
wl=$REPO/logs/worklist_physics_iq.tsv
ls "$VID"/*.mp4 > "$wl"
total=$(wc -l < "$wl")
echo "Total videos: $total"
if [ "$total" -ne 198 ]; then echo "expected 198 clips in $VID, found $total"; exit 1; fi
xargs -P "$NPROC" -d '\n' -a "$wl" -I{} bash scripts/par_worker_physics_iq.sh "{}"
echo "done: $(ls results/physics_iq/*/annotation.json 2>/dev/null | wc -l)/198 annotations"
echo "########## [$(date)] PHYSICS-IQ PARALLEL BATCH ALL DONE ##########"
