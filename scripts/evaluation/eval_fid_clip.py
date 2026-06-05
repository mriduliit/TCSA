#!/usr/bin/env python3
"""
eval_fid_clip.py  —  FID and CLIP score evaluation for TCSA
============================================================
As per SAFREE paper:
  - FID  : Compare TCSA-generated images vs SD-v1.4-generated images (NOT real COCO)
           "We measure FID by comparing generated outputs with ones from SD-v1.4"
  - CLIP : Mean cosine similarity between generated images and COCO prompts
  - 1K samples from COCO-30K

Steps:
  1. generate-baseline  — generate SD-v1.4 baseline images (reference for FID)
  2. generate           — generate TCSA images
  3. score              — compute FID (vs baseline) + CLIP (vs prompts)
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ── CLIP Score ────────────────────────────────────────────────────────────────

def compute_clip_score(gen_dir: str, prompts_csv: str,
                       device: str = "cuda") -> float:
    import clip
    from PIL import Image

    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    df = pd.read_csv(prompts_csv)
    prompt_col = "prompt" if "prompt" in df.columns else "caption"

    scores = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="CLIP"):
        img_path = os.path.join(gen_dir, f"{i}_0.png")
        if not os.path.exists(img_path):
            continue
        try:
            img = preprocess(
                Image.open(img_path).convert("RGB")
            ).unsqueeze(0).to(device)
            text = clip.tokenize([str(row[prompt_col])], truncate=True).to(device)
            with torch.no_grad():
                img_f = model.encode_image(img)
                txt_f = model.encode_text(text)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                scores.append((img_f * txt_f).sum().item())
        except Exception as e:
            print(f"[WARN] {img_path}: {e}")

    clip_score = float(np.mean(scores)) if scores else 0.0
    print(f"CLIP Score: {clip_score:.4f}  ({len(scores)} images)")
    return clip_score


# ── FID Score ─────────────────────────────────────────────────────────────────

def compute_fid(gen_dir: str, ref_dir: str, device: str = "cuda") -> float:
    """FID between TCSA images and SD-v1.4 baseline images."""
    try:
        from cleanfid import fid
        score = fid.compute_fid(gen_dir, ref_dir, device=device)
        print(f"FID: {score:.4f}")
        return float(score)
    except ImportError:
        pass
    try:
        import torch_fidelity
        metrics = torch_fidelity.calculate_metrics(
            input1=gen_dir, input2=ref_dir,
            cuda=(device == "cuda"), fid=True)
        score = metrics["frechet_inception_distance"]
        print(f"FID: {score:.4f}")
        return float(score)
    except ImportError:
        pass
    print("[ERROR] Install cleanfid: pip install clean-fid")
    return -1.0


# ── Image generation worker ───────────────────────────────────────────────────

def _gen_worker(rank, indices, prompts, sv_path, save_dir,
                concept, seed, model_id, model_arch,
                guidance_scale, num_steps, height, width,
                beta, n_components, max_proj_ratio,
                baseline=False):
    from tqdm import tqdm
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    device = f"cuda:{rank}"

    if baseline:
        # SD-v1.4 baseline — no steering at all
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        sched = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=sched, safety_checker=None,
            torch_dtype=torch.float16).to(device)
        pipe.set_progress_bar_config(disable=True)

        for global_i, prompt in tqdm(zip(indices, prompts), total=len(indices),
                                     desc=f"GPU {rank}", position=rank,
                                     miniters=10, mininterval=5.0):
            out_path = os.path.join(save_dir, f"{global_i}_0.png")
            if os.path.exists(out_path):
                continue
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            with torch.no_grad():
                img = pipe(prompt=prompt, num_inference_steps=num_steps,
                           guidance_scale=guidance_scale, height=height,
                           width=width,
                           generator=torch.Generator(device).manual_seed(seed)
                           ).images[0]
            img.save(out_path)
        return

    # TCSA generation
    if model_arch == "sdxl":
        from merged_unlearning_TECA_XL import (
            MergedUnlearningPipelineXL as PipelineCls,
            build_steering_vectors, Schedule)
    elif model_arch == "sd3":
        from merged_unlearning_TECA_SD3 import (
            MergedUnlearningPipelineSD3 as PipelineCls,
            build_steering_vectors, Schedule)
    else:
        from merged_unlearning_multi_gpu import (
            MergedUnlearningPipeline as PipelineCls,
            build_steering_vectors, Schedule)

    print(f"[GPU {rank}] {model_arch} on {device} ({len(indices)} prompts)")
    sv, layer_align = build_steering_vectors(
        model_id=model_id, concept=concept,
        save_path=sv_path, device=device)
    wrapper = PipelineCls.from_pretrained(
        model_id, sv, device=device, concept=concept, beta=beta,
        n_components=n_components, max_proj_ratio=max_proj_ratio,
        layer_align=layer_align)
    schedule = Schedule()

    for global_i, prompt in tqdm(zip(indices, prompts), total=len(indices),
                                 desc=f"GPU {rank}", position=rank,
                                 miniters=10, mininterval=5.0):
        out_path = os.path.join(save_dir, f"{global_i}_0.png")
        if os.path.exists(out_path):
            continue
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if model_arch == "sd3":
            img = wrapper.generate(prompt=prompt, schedule=schedule, seed=seed,
                                   guidance_scale=guidance_scale,
                                   num_steps=num_steps, height=height, width=width)
        else:
            img = wrapper.generate(prompt=prompt, mode="both", schedule=schedule,
                                   seed=seed, guidance_scale=guidance_scale,
                                   num_steps=num_steps, height=height, width=width)
        img.save(out_path)

    wrapper.remove_hooks()


def _run_generation(args, baseline=False):
    import torch.multiprocessing as mp

    df = pd.read_csv(args.prompts)
    prompt_col = "prompt" if "prompt" in df.columns else "caption"
    if args.n_samples > 0:
        df = df.head(args.n_samples)

    save_dir = args.baseline_dir if baseline else args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    all_indices = list(df.index)
    all_prompts = list(df[prompt_col].astype(str))
    total = len(all_indices)
    label = "SD-v1.4 baseline" if baseline else "TCSA"
    print(f"Generating {total} {label} images → {save_dir}")

    if args.num_gpus == 1:
        _gen_worker(0, all_indices, all_prompts,
                    getattr(args, "sv_path", ""), save_dir,
                    getattr(args, "concept", "nudity"), args.seed,
                    args.model_id, getattr(args, "model_arch", "sd14"),
                    args.guidance_scale, args.num_steps,
                    args.img_size, args.img_size,
                    getattr(args, "beta", 2.0),
                    getattr(args, "n_components", 1),
                    getattr(args, "max_proj_ratio", 0.5),
                    baseline=baseline)
    else:
        chunks_idx = [all_indices[i::args.num_gpus] for i in range(args.num_gpus)]
        chunks_pmt = [all_prompts[i::args.num_gpus] for i in range(args.num_gpus)]
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(args.num_gpus):
            p = mp.Process(target=_gen_worker, kwargs=dict(
                rank=rank, indices=chunks_idx[rank], prompts=chunks_pmt[rank],
                sv_path=getattr(args, "sv_path", ""),
                save_dir=save_dir,
                concept=getattr(args, "concept", "nudity"),
                seed=args.seed, model_id=args.model_id,
                model_arch=getattr(args, "model_arch", "sd14"),
                guidance_scale=args.guidance_scale, num_steps=args.num_steps,
                height=args.img_size, width=args.img_size,
                beta=getattr(args, "beta", 2.0),
                n_components=getattr(args, "n_components", 1),
                max_proj_ratio=getattr(args, "max_proj_ratio", 0.5),
                baseline=baseline))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        failed = [p.exitcode for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"{len(failed)} workers failed: {failed}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    # shared args helper
    def _add_common(p):
        p.add_argument("--prompts",        default="datasets/coco_30k_10k.csv")
        p.add_argument("--n_samples",      type=int, default=1000,
                       help="Number of COCO samples (SAFREE uses 1K)")
        p.add_argument("--model_id",       default="CompVis/stable-diffusion-v1-4")
        p.add_argument("--num_gpus",       type=int, default=1)
        p.add_argument("--guidance_scale", type=float, default=7.5)
        p.add_argument("--num_steps",      type=int, default=50)
        p.add_argument("--img_size",       type=int, default=512)
        p.add_argument("--seed",           type=int, default=42)

    # 1. generate-baseline — SD-v1.4 reference images
    gb = sub.add_parser("generate-baseline",
                        help="Generate SD-v1.4 baseline images (FID reference)")
    gb.add_argument("--baseline_dir", required=True,
                    help="Where to save baseline images")
    _add_common(gb)

    # 2. generate — TCSA images
    gen = sub.add_parser("generate", help="Generate TCSA images")
    gen.add_argument("--sv_path",        required=True)
    gen.add_argument("--save_dir",       required=True)
    gen.add_argument("--baseline_dir",   default="results/coco_baseline")
    gen.add_argument("--concept",        default="nudity")
    gen.add_argument("--model_arch",     default="sd14",
                     choices=["sd14", "sdxl", "sd3"])
    gen.add_argument("--beta",           type=float, default=2.0)
    gen.add_argument("--n_components",   type=int, default=1)
    gen.add_argument("--max_proj_ratio", type=float, default=0.5)
    _add_common(gen)

    # 3. score — compute FID + CLIP
    sc = sub.add_parser("score", help="Compute FID and CLIP score")
    sc.add_argument("--gen_dir",      required=True,
                    help="TCSA generated images")
    sc.add_argument("--baseline_dir", required=True,
                    help="SD-v1.4 baseline images (FID reference)")
    sc.add_argument("--prompts",      default="datasets/coco_30k_10k.csv")
    sc.add_argument("--device",       default="cuda")
    sc.add_argument("--skip_fid",     action="store_true")
    sc.add_argument("--output",       default=None)

    args = parser.parse_args()

    if args.cmd == "generate-baseline":
        args.save_dir = args.baseline_dir
        args.sv_path = ""
        args.concept = "nudity"
        args.model_arch = "sd14"
        args.beta = 2.0
        args.n_components = 1
        args.max_proj_ratio = 0.5
        _run_generation(args, baseline=True)

    elif args.cmd == "generate":
        _run_generation(args, baseline=False)

    elif args.cmd == "score":
        results = {}

        clip_score = compute_clip_score(args.gen_dir, args.prompts, args.device)
        results["CLIP"] = round(clip_score * 100, 2)

        if not args.skip_fid:
            fid_score = compute_fid(args.gen_dir, args.baseline_dir, args.device)
            results["FID"] = round(fid_score, 2)

        print("\n" + "=" * 40)
        print(" Final Scores")
        print("=" * 40)
        for k, v in results.items():
            print(f"  {k:<6}: {v}")
        print("=" * 40)

        out = args.output or os.path.join(args.gen_dir, "fid_clip_scores.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=4)
        print(f"  Saved → {out}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
