"""Invalidate baseline captions that were generated with fewer than num_frames frames.

extract_uniform_frames used to cap the sampling rate at 1 fps, so any clip shorter
than num_frames seconds silently got fewer frames than requested (an 8 s Physics-IQ
clip got 8 instead of 32; short NWPU clips got as few as 6). The cap is gone and every
clip now yields a full uniform sample, so those captions are stale: they were produced
from a different, poorer view of the video than a re-run would produce.

This deletes the stale caption files AND their eval cache entries, so a re-run of the
baseline scripts regenerates exactly those clips (both runners skip clips whose output
already exists) and the eval recomputes exactly those (system, clip) units. Clips that
already had the full frame count are left alone — no wasted GPU time.

A clip whose source genuinely holds fewer than num_frames frames (rare: a 1 s 10 fps
clip only has 10) is re-run to an identical result, so sweeping it in costs one wasted
inference and never a wrong number.

Usage:
    python scripts/invalidate_undersampled_baselines.py            # report only
    python scripts/invalidate_undersampled_baselines.py --apply    # delete
"""

import argparse
import glob
import json
import os

EXPECTED_FRAMES = 32
CACHE_RESULTS = "eval/cache/results"

# (glob, how to recover (system, split, video_id) from the path)
SOURCES = [
    # pipeline-output dirs: results/{dataset}/{split}/{id}/baseline.json (NWPU)
    #                       results/{dataset}/{id}/baseline.json          (youtube, physics_iq)
    ("results/*/*/*/baseline.json", "a3b_nested"),
    ("results/*/*/baseline.json", "a3b_flat"),
    # served-model baselines: baseline/results/{system}/{split}/{id}.json
    ("baseline/results/*/*/*.json", "model"),
]


def _identify(path: str, kind: str) -> tuple[str, str, str]:
    """Return (system, split, video_id) for a caption file."""
    parts = path.split(os.sep)
    if kind == "a3b_nested":       # results/nwpu_campus/Test/D001_01/baseline.json
        return "qwen3.6-a3b", parts[2], parts[3]
    if kind == "a3b_flat":         # results/physics_iq/0001_.../baseline.json
        return "qwen3.6-a3b", parts[1], parts[2]
    if kind == "model":            # baseline/results/internvl3-38b/Test/D001_01.json
        return parts[2], parts[3], os.path.splitext(parts[4])[0]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="actually delete (default: report only)")
    ap.add_argument("--frames", type=int, default=EXPECTED_FRAMES)
    args = ap.parse_args()

    stale = []
    for pattern, kind in SOURCES:
        for path in glob.glob(pattern):
            with open(path) as f:
                doc = json.load(f)
            n = doc.get("num_frames")
            if n is None or n >= args.frames:
                continue
            stale.append((path, *_identify(path, kind), n))

    print(f"caption files with < {args.frames} frames: {len(stale)}")
    by_system: dict[str, int] = {}
    for _, system, _, _, _ in stale:
        by_system[system] = by_system.get(system, 0) + 1
    for system, count in sorted(by_system.items(), key=lambda x: -x[1]):
        print(f"  {system:<18} {count}")

    cache_hits = []
    for _, system, split, vid, _ in stale:
        cpath = os.path.join(CACHE_RESULTS, system, f"{split}__{vid}.json")
        if os.path.exists(cpath):
            cache_hits.append(cpath)
    print(f"matching eval cache entries: {len(cache_hits)}")

    if not args.apply:
        print("\nreport only — re-run with --apply to delete these and free them for re-generation")
        return

    for path, *_ in stale:
        os.remove(path)
    for cpath in cache_hits:
        os.remove(cpath)
    print(f"\ndeleted {len(stale)} caption files + {len(cache_hits)} cache entries")
    print("re-run the baseline scripts (they skip clips whose output still exists), then re-run eval")


if __name__ == "__main__":
    main()
