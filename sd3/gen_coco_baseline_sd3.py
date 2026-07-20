#!/usr/bin/env python3
"""Generate plain SD3 baseline images for the first N COCO prompts
(same prompts/indices/settings as results/sd3_tcsa/coco, for a fair CLIP-30K
comparison). No TCSA hooks — just the stock StableDiffusion3Pipeline.

Splits the prompt list across multiple GPUs via torch.multiprocessing
(round-robin chunking).

Run inside the `sd3_gen` conda env.

Usage:
    python gen_coco_baseline_sd3.py \
        --save_dir /storage/pk_mridul/TCSA/results/coco_baseline_sd3 \
        --gpus 0 1
"""
import argparse
import os

import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

SD3_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _worker(rank, gpu_id, indices, prompts, save_dir, model_id, steps, guidance, img_size, seed, dtype_str):
    import torch
    from diffusers import StableDiffusion3Pipeline

    device = f"cuda:{gpu_id}"
    dtype = _DTYPE_MAP[dtype_str]
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(leave=False)

    for i, prompt in zip(tqdm(indices, desc=f"[gpu{gpu_id}] coco_baseline_sd3"), prompts):
        out_path = os.path.join(save_dir, f"{i}_0.png")
        if os.path.exists(out_path):
            continue
        if not prompt.strip():
            continue
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=img_size,
            width=img_size,
            generator=generator,
        ).images[0]
        image.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=SD3_MODEL_ID)
    ap.add_argument("--coco_csv", default="/storage/pk_mridul/TCSA/datasets/coco_30k_10k.csv")
    ap.add_argument("--n_coco", type=int, default=1000)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    df = pd.read_csv(args.coco_csv)
    col = "prompt" if "prompt" in df.columns else "caption"
    df = df.head(args.n_coco).reset_index(drop=True)
    all_indices = list(df.index)
    all_prompts = df[col].astype(str).tolist()
    print(f"CoCo: {len(all_prompts)} prompts  |  GPUs: {args.gpus}")

    num_gpus = len(args.gpus)
    chunks_idx = [all_indices[i::num_gpus] for i in range(num_gpus)]
    chunks_pmt = [all_prompts[i::num_gpus] for i in range(num_gpus)]
    for rank, (ci, gpu_id) in enumerate(zip(chunks_idx, args.gpus)):
        print(f"  rank {rank} -> gpu {gpu_id}: {len(ci)} prompts (indices {ci[0]}..{ci[-1]})")

    if num_gpus == 1:
        _worker(0, args.gpus[0], chunks_idx[0], chunks_pmt[0], args.save_dir,
                args.model_id, args.steps, args.guidance, args.img_size, args.seed, args.dtype)
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank, gpu_id in enumerate(args.gpus):
            p = mp.Process(
                target=_worker,
                args=(rank, gpu_id, chunks_idx[rank], chunks_pmt[rank], args.save_dir,
                      args.model_id, args.steps, args.guidance, args.img_size, args.seed, args.dtype),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        failed = [p.exitcode for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"{len(failed)} worker(s) failed: {failed}")

    done = sum(1 for i in all_indices if os.path.exists(os.path.join(args.save_dir, f"{i}_0.png")))
    print(f"\nDone. {done}/{len(all_indices)} images saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
