"""LLaMA-VID naive caption baseline. Not vLLM-servable (pinned torch 2.0.1 /
transformers 4.31 stack) — runs in its own llamavid venv against its own codebase,
sharded across GPUs via --num_shards/--shard_idx.

Must run with cwd = the LLaMA-VID repo: the checkpoint config carries a relative
image_processor path. HF generate has no presence_penalty (known divergence).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from decord import VideoReader, cpu

from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

PROMPT = "Describe this video in detail."
N_FRAMES = 32
VID_EXT = (".mp4", ".avi", ".mkv", ".mov", ".webm")


def load_uniform_frames(video_path: str, n: int = N_FRAMES) -> np.ndarray:
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    n = min(n, total)
    idx = np.linspace(0, total - 1, n).astype(int)
    return vr.get_batch(idx).asnumpy()


def caption_one(model, tokenizer, image_processor, video_path: str) -> tuple[str, int]:
    frames = load_uniform_frames(video_path)
    num_frames = len(frames)
    image_tensor = image_processor.preprocess(frames, return_tensors='pt')['pixel_values'].half().cuda()
    image_tensor = [image_tensor]

    conv = conv_templates["llava_v1"].copy()
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + PROMPT
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + PROMPT
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    model.update_prompt([[PROMPT]])
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, images=image_tensor,
            do_sample=True, temperature=0.7, top_p=0.8,
            max_new_tokens=1024, use_cache=True,
            stopping_criteria=[stopping])
    caption = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    if caption.endswith(stop_str):
        caption = caption[:-len(stop_str)].strip()
    return caption, int(num_frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--splits", required=True,
                    help="comma list of name=dir, e.g. Test=/a/Test,Train=/a/Train")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    splits = []
    for spec in args.splits.split(","):
        name, d = spec.split("=", 1)
        splits.append((name, d))

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, model_name, False, False)
    print(f"[LLaMA-VID] model loaded (shard {args.shard_idx}/{args.num_shards})", flush=True)

    for split, vdir in splits:
        out_dir = os.path.join(args.output_root, split)
        os.makedirs(out_dir, exist_ok=True)
        vids = sorted(f for f in os.listdir(vdir) if f.endswith(VID_EXT))
        vids = vids[args.shard_idx::args.num_shards]
        print(f"[LLaMA-VID] split {split}: {len(vids)} videos (this shard)", flush=True)
        for vname in vids:
            vid = os.path.splitext(vname)[0]
            out_path = os.path.join(out_dir, f"{vid}.json")
            if os.path.exists(out_path) and not args.force:
                continue
            t0 = time.perf_counter()
            try:
                caption, num_frames = caption_one(model, tokenizer, image_processor,
                                                  os.path.join(vdir, vname))
                doc = {
                    "video_id": vid, "caption": caption,
                    "word_count": len(caption.split()), "num_frames": num_frames,
                    "model": "llama-vid-7b", "input_format": "video",
                    "timings_sec": {"total": round(time.perf_counter() - t0, 2)},
                }
                with open(out_path, "w") as f:
                    json.dump(doc, f, indent=2, ensure_ascii=False)
                print(f"[LLaMA-VID] OK {split}/{vid}: {doc['word_count']}w {num_frames}f", flush=True)
            except Exception as e:
                print(f"[LLaMA-VID] FAIL {split}/{vid}: {e}", flush=True)
    print(f"[LLaMA-VID] shard {args.shard_idx} DONE", flush=True)


if __name__ == "__main__":
    main()
