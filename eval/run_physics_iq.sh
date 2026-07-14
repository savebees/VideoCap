#!/bin/bash
# Physics-IQ eval driver: check completeness -> serve two Gemma3 replicas ->
# run_eval -> stop. Runs run_eval.py from evalenv (spacy pins numpy<2; the vLLM
# serving env needs numpy 2.x, so the two envs must stay separate).
set -u
BASE=/nfshdd/21039533r/zxhe
REPO=$BASE/dense-video-annotator
EVALPY=/localnvme/zxhe/miniconda3/envs/evalenv/bin/python
CFG=eval/config_physics_iq.yaml
cd $REPO

log(){ echo "[$(date +%H:%M:%S)] [eval] $*"; }

# 1. every system needs all 198 captions or run_eval raises mid-run
log "checking caption completeness"
missing=0
n=$(ls results/physics_iq/*/annotation.json 2>/dev/null | wc -l); log "  pipeline        $n/198"; [ "$n" -ne 198 ] && missing=1
n=$(ls results/physics_iq/*/baseline.json 2>/dev/null | wc -l);   log "  qwen3.6-a3b     $n/198"; [ "$n" -ne 198 ] && missing=1
for t in qwen2.5-vl-32b internvl3-38b internvl3.5-38b llama-vid-7b qwen2.5-vl-7b; do
  n=$(ls baseline/results/$t/physics_iq/*.json 2>/dev/null | wc -l)
  log "  $t $n/198"
  [ "$n" -ne 198 ] && missing=1
done
if [ "$missing" = "1" ]; then log "ABORT: some system is incomplete — eval would raise FileNotFoundError"; exit 1; fi

# 2. free GPUs held by annotation servers
log "stopping annotation vLLMs"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 "$p" 2>/dev/null; done
sleep 10

# 3. judge (:8010, GPUs 0-3) + text (:8011, GPUs 4-7)
log "starting Gemma3 judge (:8010) and text (:8011)"
nohup bash eval/serve_judge.sh > logs/eval_judge_piq.log 2>&1 &
nohup bash eval/serve_text.sh  > logs/eval_text_piq.log  2>&1 &
for i in $(seq 1 120); do
  j=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8010/v1/models 2>/dev/null)
  t=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8011/v1/models 2>/dev/null)
  [ "$j" = "200" ] && [ "$t" = "200" ] && break
  sleep 15
done
if [ "$j$t" != "200200" ]; then log "ABORT: judge=$j text=$t not ready"; tail -20 logs/eval_judge_piq.log; exit 2; fi
log "both Gemma3 replicas ready"

# 4. eval
log "running eval (198 clips x 7 systems)"
$EVALPY -u eval/run_eval.py --config $CFG
rc=$?

log "stopping Gemma3 replicas"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 "$p" 2>/dev/null; done

if [ "$rc" != "0" ]; then log "EVAL FAILED (rc=$rc)"; exit $rc; fi
log "EVAL DONE -> results/physics_iq/metrics/summary.csv"
cat results/physics_iq/metrics/summary.csv
