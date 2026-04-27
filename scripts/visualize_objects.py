"""Render object detections from objects.json onto each frame jpg.

Usage:
  python scripts/visualize_objects.py --video_id 0_0fNS2qODw
  python scripts/visualize_objects.py --all
"""

import argparse
import colorsys
import hashlib
import json
import os
import sys
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "output")


def color_for_label(label: str) -> tuple[int, int, int]:
    h = int(hashlib.md5(label.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_frame(img: Image.Image, dets: list[dict], font: ImageFont.ImageFont) -> Image.Image:
    canvas = img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        color = color_for_label(d["label"])
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)

        text = f"{d['label']} {d['confidence']:.2f}"
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), text, font=font)
        tw, th = tx1 - tx0, ty1 - ty0
        ly = max(0, y1 - th - 4)
        draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=color + (220,))
        draw.text((x1 + 3, ly + 2), text, fill=(255, 255, 255), font=font)
    return canvas


def visualize_video(video_id: str) -> None:
    video_dir = os.path.join(OUTPUT_DIR, video_id)
    objects_path = os.path.join(video_dir, "objects.json")
    frames_dir = os.path.join(video_dir, "frames")
    viz_dir = os.path.join(video_dir, "viz")

    if not os.path.isfile(objects_path):
        print(f"[skip] {video_id}: no objects.json")
        return
    if not os.path.isdir(frames_dir):
        print(f"[skip] {video_id}: no frames/")
        return
    os.makedirs(viz_dir, exist_ok=True)

    with open(objects_path) as f:
        doc = json.load(f)

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for d in doc["objects"]:
        by_frame[d["frame_index"]].append(d)

    font = load_font(16)
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    n_with, n_without = 0, 0
    for fname in frame_files:
        idx = int(os.path.splitext(fname)[0])
        with Image.open(os.path.join(frames_dir, fname)) as img:
            annotated = draw_frame(img, by_frame.get(idx, []), font)
            annotated.save(os.path.join(viz_dir, fname), quality=90)
        if by_frame.get(idx):
            n_with += 1
        else:
            n_without += 1
    print(f"[ok] {video_id}: {len(frame_files)} frames -> {viz_dir} "
          f"({n_with} with detections, {n_without} empty, {len(doc['objects'])} total objects)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", help="Single video id under data/output/")
    parser.add_argument("--all", action="store_true", help="Process every video in data/output/")
    args = parser.parse_args()

    if args.all:
        ids = sorted(d for d in os.listdir(OUTPUT_DIR)
                     if os.path.isdir(os.path.join(OUTPUT_DIR, d)))
    elif args.video_id:
        ids = [args.video_id]
    else:
        parser.error("Provide --video_id or --all")

    for vid in ids:
        visualize_video(vid)


if __name__ == "__main__":
    main()
