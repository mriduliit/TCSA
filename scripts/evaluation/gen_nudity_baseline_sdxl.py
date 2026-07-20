#!/usr/bin/env python3
"""Generate plain SDXL baseline images for the MMA-Diffusion NSFW adversarial
prompts (mma-diffusion-nsfw-adv-prompts.csv), using the 'adv_prompt' column.
No TCSA/SAFREE hooks - just the stock StableDiffusionXLPipeline.

Splits the prompt list across multiple GPUs via torch.multiprocessing
(round-robin chunking).

Run inside the `casteer` conda env.

Usage:
    python gen_nudity_baseline_sdxl.py \
        --save_dir /storage/pk_mridul/TCSA/results/nudity_baseline_sdxl \
        --gpus 0 1 2 3
"""
import argparse
import os

import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm


def _worker(rank, gpu_id, indices, prompts, save_dir, model_id, steps, guidance, img_size, seed):
    import torch
    from diffusers import StableDiffusionXLPipeline

    device = f"cuda:{gpu_id}"
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(leave=False)

    for i, prompt in zip(tqdm(indices, desc=f"[gpu{gpu_id}] nudity_baseline_sdxl"), prompts):
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
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--csv", default="/storage/pk_mridul/TCSA/datasets/mma-diffusion-nsfw-adv-prompts.csv")
    ap.add_argument("--prompt_col", default="adv_prompt")
    ap.add_argument("--n_prompts", type=int, default=1000)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--img_size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df.head(args.n_prompts).reset_index(drop=True)
    all_indices = list(df.index)
    all_prompts = df[args.prompt_col].astype(str).tolist()
    print(f"MMA nudity prompts: {len(all_prompts)}  |  GPUs: {args.gpus}")

    num_gpus = len(args.gpus)
    chunks_idx = [all_indices[i::num_gpus] for i in range(num_gpus)]
    chunks_pmt = [all_prompts[i::num_gpus] for i in range(num_gpus)]
    for rank, (ci, gpu_id) in enumerate(zip(chunks_idx, args.gpus)):
        print(f"  rank {rank} -> gpu {gpu_id}: {len(ci)} prompts (indices {ci[0]}..{ci[-1]})")

    if num_gpus == 1:
        _worker(0, args.gpus[0], chunks_idx[0], chunks_pmt[0], args.save_dir,
                args.model_id, args.steps, args.guidance, args.img_size, args.seed)
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank, gpu_id in enumerate(args.gpus):
            p = mp.Process(
                target=_worker,
                args=(rank, gpu_id, chunks_idx[rank], chunks_pmt[rank], args.save_dir,
                      args.model_id, args.steps, args.guidance, args.img_size, args.seed),
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
