"""CLIP ViT-L/14 caption-video alignment (coarse sanity check, not primary).
Captions are split into sentences (77-token text encoder) and mean-pooled so
long captions aren't truncated unfairly. Writes eval/cache/clip/; add the
clip_sim column afterwards with --merge."""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systems as sysmod  # noqa: E402

_SENT = re.compile(r"[^.!?]+[.!?]?")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _sample_frames(video_path, n):
    from decord import VideoReader, cpu  # lazy: only the no-frame-cache fallback needs it
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise RuntimeError(f"no frames in {video_path}")
    idx = [int(x) for x in np.linspace(0, total - 1, min(n, total))]
    arr = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(f) for f in arr]


def _cached_frames(cache_dir, split, vid, n_want):
    """Reuse the judge's already-extracted frame JPEGs (eval/cache/frames/
    {split}__{vid}__nN/) and uniformly subsample n_want — far less NFS IO than
    re-decoding the source video. Returns None if no usable cache exists."""
    import glob
    dirs = sorted(glob.glob(os.path.join(cache_dir, "frames", f"{split}__{vid}__n*")),
                  key=lambda d: -int(d.rsplit("__n", 1)[1]))
    for d in dirs:
        jpgs = sorted(glob.glob(os.path.join(d, "*.jpg")))
        if len(jpgs) >= n_want:
            idx = [int(x) for x in np.linspace(0, len(jpgs) - 1, n_want)]
            return [Image.open(jpgs[i]).convert("RGB") for i in idx]
    return None


def _sentences(text):
    sents = [s.strip() for s in _SENT.findall(text) if s.strip()]
    return sents or [text.strip() or " "]


class CLIPScorer:
    def __init__(self, model_path, device="cuda"):
        self.model = CLIPModel.from_pretrained(model_path).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_path)
        self.device = device

    def score(self, caption, video_path, n_frames):
        return self.score_frames(caption, _sample_frames(video_path, n_frames))

    @torch.no_grad()
    def score_frames(self, caption, frames):
        img = self.proc(images=frames, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**img)
        imf = (out.pooler_output if hasattr(out, "pooler_output") else out).float()
        imf = imf / imf.norm(dim=-1, keepdim=True)
        imf = imf.mean(dim=0, keepdim=True)
        imf = imf / imf.norm(dim=-1, keepdim=True)

        sents = _sentences(caption)
        txt = self.proc(text=sents, return_tensors="pt", padding=True,
                        truncation=True, max_length=77).to(self.device)
        out = self.model.get_text_features(**txt)
        tf = (out.pooler_output if hasattr(out, "pooler_output") else out).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
        tf = tf.mean(dim=0, keepdim=True)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        return float((tf @ imf.T).item())


def _clip_path(cache_dir, system, split, vid):
    return os.path.join(cache_dir, "clip", system, f"{split}__{vid}.json")


def run(config, model_path, force):
    scorer = CLIPScorer(model_path)
    n = config.get("clip_frames", 16)
    clips = sysmod.list_clips(config)
    systems = config["systems"]
    print(f"[clip] {len(clips)} clips x {len(systems)} systems, {n} frames", flush=True)
    done = 0
    for sy in systems:
        for split, vid in clips:
            out = _clip_path(config["cache_dir"], sy["name"], split, vid)
            if os.path.exists(out) and not force:
                done += 1
                continue
            cap = sysmod.load_caption(sy, split, vid, config)
            frames = _cached_frames(config["cache_dir"], split, vid, n)
            if frames is None:   # fallback: decode the source video
                frames = _sample_frames(sysmod.video_path(config, split, vid), n)
            sim = scorer.score_frames(cap, frames)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tmp = out + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"clip_score": sim}, f)
            os.replace(tmp, out)
            done += 1
            if done % 200 == 0:
                print(f"[clip] {done}/{len(clips) * len(systems)}", flush=True)
    print(f"[clip] done {done}", flush=True)


def merge(config):
    """Per dataset: add clip_sim to each per_video.jsonl record + a summary.csv column."""
    import csv
    root = config["output_root"]
    cache = config["cache_dir"]
    datasets = {config["splits"][s]["dataset"] for s in config["splits"]}
    for dataset in sorted(datasets):
        mdir = os.path.join(root, dataset, "metrics")
        means = {}
        for sy in config["systems"]:
            name = sy["name"]
            path = os.path.join(mdir, name, "per_video.jsonl")
            if not os.path.exists(path):
                continue
            recs = [json.loads(l) for l in open(path)]
            scores = []
            for r in recs:
                with open(_clip_path(cache, name, r["split"], r["video_id"])) as f:
                    s = json.load(f)["clip_score"]
                r.setdefault("alignment", {})["clip_sim"] = s
                scores.append(s)
            with open(path, "w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            means[name] = sum(scores) / len(scores) if scores else None
        spath = os.path.join(mdir, "summary.csv")
        if not os.path.exists(spath):
            continue
        rows = list(csv.DictReader(open(spath)))
        for row in rows:
            m = means.get(row["system"])
            row["clip_sim"] = round(m, 4) if m is not None else ""
        cols = list(rows[0].keys())
        with open(spath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"[clip] {dataset}: merged clip_sim ({len(means)} systems)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge", action="store_true", help="merge cached clip scores into outputs")
    args = ap.parse_args()
    config = load_config(args.config)
    if args.merge:
        merge(config)
    else:
        run(config, args.model_path, args.force)


if __name__ == "__main__":
    main()
