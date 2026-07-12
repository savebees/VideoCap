"""Download the Physics-IQ benchmark videos used by this pipeline.

Fetches ``full-videos/take-1/30FPS`` from the public GCS bucket
``physics-iq-benchmark`` (198 files, ~1.8 GB): the FULL 8 s clips, not the 3 s
conditioning / 5 s continuation splits that the generative-model benchmark uses.
The bucket is public, so this needs neither gsutil nor credentials.

Original filenames are preserved, since the pipeline derives ``video_id`` from
the basename.

``descriptions.csv`` is deliberately NOT downloaded here: it is eval-side
reference material and must stay out of any pipeline input path. Fetch it
separately if needed, e.g.

    curl -L -o eval/physics_iq_descriptions.csv \
      https://raw.githubusercontent.com/google-deepmind/physics-IQ-benchmark/main/descriptions/descriptions_original.csv

Usage:
    python scripts/download_physics_iq.py [--out data/physics_iq] [--workers 8]
"""

import argparse
import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BUCKET = "physics-iq-benchmark"
PREFIX = "full-videos/take-1/30FPS/"
EXPECTED_COUNT = 198


def list_objects() -> list[dict]:
    """List the take-1 30FPS full videos via the public GCS JSON API."""
    url = (
        f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
        f"?prefix={urllib.parse.quote(PREFIX)}&fields=items(name,size),nextPageToken"
    )
    items: list[dict] = []
    page_token = None
    while True:
        page_url = url if page_token is None else f"{url}&pageToken={page_token}"
        with urllib.request.urlopen(page_url) as resp:
            payload = json.load(resp)
        items.extend(payload["items"])
        page_token = payload.get("nextPageToken")
        if page_token is None:
            break
    if len(items) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNT} objects under gs://{BUCKET}/{PREFIX}, got {len(items)}"
        )
    return items


def download(obj: dict, out_dir: str) -> str:
    """Download one object, skipping it if a complete local copy already exists."""
    name = obj["name"]
    size = int(obj["size"])
    dest = os.path.join(out_dir, os.path.basename(name))
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest

    url = f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(name)}"
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    written = os.path.getsize(tmp)
    if written != size:
        os.remove(tmp)
        raise RuntimeError(f"{name}: expected {size} bytes, wrote {written}")
    os.rename(tmp, dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/physics_iq", help="Destination directory")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent downloads")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    objects = list_objects()
    total_gb = sum(int(o["size"]) for o in objects) / 1e9
    print(f"Downloading {len(objects)} videos ({total_gb:.2f} GB) to {args.out}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, dest in enumerate(pool.map(lambda o: download(o, args.out), objects), start=1):
            print(f"[{i}/{len(objects)}] {os.path.basename(dest)}")

    have = [f for f in os.listdir(args.out) if f.endswith(".mp4")]
    if len(have) != EXPECTED_COUNT:
        raise RuntimeError(f"{args.out} holds {len(have)} mp4 files, expected {EXPECTED_COUNT}")
    print(f"Done: {len(have)} videos in {args.out}")


if __name__ == "__main__":
    main()
