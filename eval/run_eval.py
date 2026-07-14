"""Eval orchestrator: full run, resumable via per-(system,clip) atomic cache.
Per clip (shared): judge frames + salient list. Per (clip, system): claims ->
judge verdicts -> omission coverage -> richness."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

# spaCy must import before decord (loading decord first breaks spaCy's import)
import spacy  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cache as cachemod
import judge as judgemod
import systems as sysmod
from claims import decompose
from frames import cached_frames
from llm_client import make_client
from systems import normalize_caption
from omission import coverage
from richness import richness


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_shared(config, judge_client, split, vid, force):
    """Per-clip shared artifacts: salient reference list (one judge call)."""
    if not force:
        c = cachemod.load_shared(config["cache_dir"], split, vid)
        if c is not None:
            return c
    vpath = sysmod.video_path(config, split, vid)
    frames = cached_frames(config["cache_dir"], split, vid, config["judge_frames"],
                           vpath, config["frame_max_long_side"], config["frame_quality"])
    salient = judgemod.list_salient(judge_client, config, frames)
    rec = {"split": split, "video_id": vid, "salient": salient}
    cachemod.save_shared(config["cache_dir"], split, vid, rec)
    return rec


def process_unit(config, system, split, vid, shared, judge_client, claim_client, force):
    if not force:
        c = cachemod.load_valid(cachemod.result_path(config["cache_dir"], system["name"], split, vid))
        if c is not None:
            return c

    raw = sysmod.load_caption(system, split, vid, config)
    norm = normalize_caption(raw)
    vpath = sysmod.video_path(config, split, vid)

    claims = decompose(norm, claim_client, config)
    pos = [c for c in claims if c["type"] != "negation"]
    neg = [c for c in claims if c["type"] == "negation"]

    frames = cached_frames(config["cache_dir"], split, vid, config["judge_frames"],
                           vpath, config["frame_max_long_side"], config["frame_quality"])
    verdicts = judgemod.verify_claims(judge_client, config, frames, pos)

    neg_verdicts = {}
    if neg:
        dense = cached_frames(config["cache_dir"], split, vid, config["negation_frames"],
                              vpath, config["frame_max_long_side"], config["frame_quality"])
        neg_verdicts = judgemod.verify_negations(judge_client, config, dense, neg)

    sup = sum(1 for v in verdicts.values() if v["verdict"] == "supported")
    con = sum(1 for v in verdicts.values() if v["verdict"] == "contradicted")
    nv = sum(1 for v in verdicts.values() if v["verdict"] == "not_verifiable")
    hr = (con / (sup + con)) if (sup + con) else None

    neg_sup = sum(1 for v in neg_verdicts.values() if v["verdict"] == "supported")
    neg_con = sum(1 for v in neg_verdicts.values() if v["verdict"] == "contradicted")
    neg_acc = (neg_sup / (neg_sup + neg_con)) if (neg_sup + neg_con) else None

    omis = coverage(norm, shared["salient"], claim_client, config)
    rich = richness(norm, config)

    claim_detail = []
    for c in claims:
        v = (neg_verdicts if c["type"] == "negation" else verdicts).get(c["claim_id"], {})
        claim_detail.append({**c, "verdict": v.get("verdict"), "evidence": v.get("evidence")})

    record = {
        "system": system["name"], "split": split, "video_id": vid,
        "raw_caption": raw, "norm_caption": norm,
        "faithfulness": {
            "n_pos_claims": len(pos), "supported": sup, "contradicted": con,
            "not_verifiable": nv, "hallucination_rate": hr,
        },
        "negation": {
            "n_neg_claims": len(neg), "supported": neg_sup, "contradicted": neg_con,
            "empty_scene_accuracy": neg_acc,
        },
        "omission": omis,
        "richness": rich,
        "alignment": {},   # clip_sim added later by clip_score.py --merge
        "claims": claim_detail,
    }
    cachemod.save_result(config["cache_dir"], system["name"], split, vid, record)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of clips")
    args = ap.parse_args()

    config = load_config(args.config)
    clips = sysmod.list_clips(config)
    if args.limit:
        clips = clips[:args.limit]
    systems = config["systems"]
    print(f"[eval] {len(clips)} clips x {len(systems)} systems", flush=True)

    judge_client = make_client(config["judge_base_url"])
    claim_client = make_client(config["claim_base_url"])

    # Phase A: shared per-clip salient lists
    with ThreadPoolExecutor(max_workers=config["judge_concurrency"]) as ex:
        futs = {ex.submit(ensure_shared, config, judge_client, s, v, args.force): (s, v) for s, v in clips}
        for i, f in enumerate(as_completed(futs), 1):
            f.result()  # raise on any failure
            if i % 50 == 0:
                print(f"[eval] shared {i}/{len(clips)}", flush=True)
    shared = {(s, v): cachemod.load_shared(config["cache_dir"], s, v) for s, v in clips}

    # Phase B: clip-major ordering lets the judge reuse each clip's image prefix
    # across systems via prefix caching.
    units = [(sy, s, v) for (s, v) in clips for sy in systems]
    done = 0
    with ThreadPoolExecutor(max_workers=config["judge_concurrency"]) as ex:
        futs = {ex.submit(process_unit, config, sy, s, v, shared[(s, v)],
                          judge_client, claim_client, args.force): (sy["name"], s, v)
                for sy, s, v in units}
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 50 == 0:
                print(f"[eval] units {done}/{len(units)}", flush=True)

    # Per-dataset assembly (cache -> {output_root}/{dataset}/metrics/...) + summaries
    from summarize import finalize
    paths = finalize(config)
    print(f"[eval] wrote per-dataset metrics ({len(paths)} datasets)", flush=True)
    print("[eval] summary.csv written", flush=True)


if __name__ == "__main__":
    main()
