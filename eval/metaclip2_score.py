"""Meta CLIP 2 (ViT-H/14) alignment, same recipe as clip_score.py but
self-contained (captions.jsonl + pre-extracted frames) so it can run off-NFS."""

import argparse
import json
import os
import re

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

_SENT = re.compile(r"[^.!?]+[.!?]?")


def _sentences(text):
    s = [x.strip() for x in _SENT.findall(text) if x.strip()]
    return s or [text.strip() or " "]


def _load_frames(frames_dir, split, vid):
    d = os.path.join(frames_dir, f"{split}__{vid}")
    jpgs = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    return [Image.open(os.path.join(d, f)).convert("RGB") for f in jpgs]


class MetaClip2:
    def __init__(self, model_path, device="cuda"):
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_path)
        self.device = device

    @torch.no_grad()
    def image_feat(self, frames):
        px = self.proc(images=frames, return_tensors="pt").to(self.device)
        # Meta CLIP 2's get_image_features returns a model output; the joint-space
        # embedding is .pooler_output (== image_embeds), 1024-d.
        f = self.model.get_image_features(**px).pooler_output.float()
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.mean(dim=0, keepdim=True)
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text_feat(self, caption):
        t = self.proc(text=_sentences(caption), return_tensors="pt", padding=True,
                      truncation=True).to(self.device)
        f = self.model.get_text_features(**t).pooler_output.float()
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.mean(dim=0, keepdim=True)
        return f / f.norm(dim=-1, keepdim=True)

    def score(self, caption, imf):
        return float((self.text_feat(caption) @ imf.T).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--captions", required=True)   # jsonl: system/split/vid/caption
    ap.add_argument("--frames-dir", required=True)  # {split}__{vid}/*.jpg
    ap.add_argument("--out", required=True)         # jsonl: system/split/vid/metaclip2_sim
    args = ap.parse_args()

    # group captions by clip so each clip's frames are encoded once
    by_clip = {}
    for line in open(args.captions):
        r = json.loads(line)
        by_clip.setdefault((r["split"], r["vid"]), []).append((r["system"], r["caption"]))

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["system"], r["split"], r["vid"]))

    scorer = MetaClip2(args.model_path)
    total = sum(len(v) for v in by_clip.values())
    print(f"[mc2] {len(by_clip)} clips, {total} (system,clip) pairs", flush=True)
    n = 0
    with open(args.out, "a") as fh:
        for (split, vid), items in by_clip.items():
            if all((sy, split, vid) in done for sy, _ in items):
                n += len(items)
                continue
            imf = scorer.image_feat(_load_frames(args.frames_dir, split, vid))
            for sy, cap in items:
                if (sy, split, vid) in done:
                    continue
                sim = scorer.score(cap, imf)
                fh.write(json.dumps({"system": sy, "split": split, "vid": vid,
                                     "metaclip2_sim": sim}) + "\n")
                fh.flush()
                n += 1
            if n % 200 < len(items):
                print(f"[mc2] {n}/{total}", flush=True)
    print(f"[mc2] done {n}", flush=True)


if __name__ == "__main__":
    main()
