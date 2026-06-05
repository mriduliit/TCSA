"""
evaluate_coco.py  —  COCO-30k image quality evaluation for TECA
================================================================

Measures whether TECA damages normal (safe) image generation by
generating images for COCO prompts under different modes and computing:

  FID   — Frechet Inception Distance vs real COCO images
           Lower is better. Measures distribution-level quality.

  CLIP  — Mean CLIP similarity between generated image and its prompt
           Higher is better. Measures prompt faithfulness.

The key comparison is:
  mode=none  (unsafe baseline)  vs  mode=both  (TECA)

If FID and CLIP are close between the two, TECA preserves image quality.
If they diverge, TECA is damaging normal generation.

Usage
-----
    # generate images for all 4 modes (takes ~2-4 hours for 1000 prompts)
    python evaluate_coco.py generate \\
        --sv_path  sv_nudity_weighted.pt \\
        --save_dir coco_eval \\
        --n        1000 \\
        --device   cuda

    # compute FID and CLIP scores from generated images
    python evaluate_coco.py score \\
        --save_dir coco_eval \\
        --device   cuda

    # quick sanity check with 50 prompts
    python evaluate_coco.py generate \\
        --sv_path  sv_nudity_weighted.pt \\
        --save_dir coco_eval_quick \\
        --n        50 \\
        --device   cuda

    python evaluate_coco.py score \\
        --save_dir coco_eval_quick \\
        --device   cuda

Requirements
------------
    pip install torch torchvision diffusers transformers
    pip install clean-fid clip  (for scoring)
    pip install pandas Pillow tqdm

Output structure
----------------
    coco_eval/
        none/          <- unsafe baseline images
        text/          <- Space 1 only
        ca/            <- Space 2 only
        both/          <- TECA (text + CA + TCL)
        scores.json    <- FID and CLIP scores for all modes

Notes
-----
- FID requires at least 2048 images for reliable estimates.
  Use --n 2048 or more for paper results. 50 is only for debugging.
- This file imports MergedUnlearningPipeline from merged_unlearning_TECA.py
  which must be in the same directory or on PYTHONPATH.
- The COCO CSV is at examples/prompts/coco_30k_10k.csv in your repo.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

log = logging.getLogger("evaluate_coco")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODES = ["none", "text", "ca", "both"]
COCO_CSV = os.path.join(
    os.path.dirname(__file__),
    "examples/prompts/coco_30k_10k.csv"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_prompts(csv_path: str, n: Optional[int]) -> List[str]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    prompts = df["prompt"].dropna().astype(str).tolist()
    if n is not None:
        prompts = prompts[:n]
    log.info("Loaded %d COCO prompts from %s", len(prompts), csv_path)
    return prompts


def _img_path(save_dir: str, mode: str, idx: int) -> str:
    return os.path.join(save_dir, mode, f"{idx:05d}.png")


# ---------------------------------------------------------------------------
# generate subcommand
# ---------------------------------------------------------------------------

def cmd_generate(args) -> None:
    """Generate COCO images under all 4 modes."""
    # import TECA pipeline
    sys.path.insert(0, os.path.dirname(__file__))
    from merged_unlearning_TECA import (
        MergedUnlearningPipeline, build_steering_vectors, make_schedule
    )

    prompts = _load_prompts(args.coco_csv, args.n)

    log.info("Loading steering vectors from %s", args.sv_path)
    sv = build_steering_vectors(
        model_id   = args.model_id,
        concept    = args.concept,
        num_pairs  = args.num_pairs,
        save_path  = args.sv_path,
        device     = args.device,
    )

    log.info("Loading pipeline …")
    wrapper = MergedUnlearningPipeline.from_pretrained(
        model_id = args.model_id,
        sv       = sv,
        device   = args.device,
        concept  = args.concept,
        beta     = args.beta,
    )

    schedule = make_schedule(
        mode      = args.schedule,
        tau_relax = args.tau_relax,
        w_min     = args.w_min,
    )

    # create output dirs
    for mode in MODES:
        os.makedirs(os.path.join(args.save_dir, mode), exist_ok=True)

    total = len(prompts)
    for idx, prompt in enumerate(prompts):
        log.info("[%d/%d] %s", idx + 1, total, prompt[:70])

        for mode in MODES:
            out_path = _img_path(args.save_dir, mode, idx)
            if os.path.exists(out_path):
                continue   # resume support

            img = wrapper.generate(
                prompt         = prompt,
                mode           = mode,
                schedule       = schedule if mode != "none" else None,
                seed           = args.seed,
                guidance_scale = args.guidance,
                num_steps      = args.steps,
                height         = args.img_size,
                width          = args.img_size,
            )
            img.save(out_path)

    wrapper.remove_hooks()
    log.info("Generation done → %s", args.save_dir)


# ---------------------------------------------------------------------------
# score subcommand
# ---------------------------------------------------------------------------

def cmd_score(args) -> None:
    """Compute FID and CLIP scores for all modes."""

    # ── CLIP score ────────────────────────────────────────────────────────
    def compute_clip_score(mode: str) -> float:
        try:
            import clip as openai_clip
        except ImportError:
            log.warning("clip not installed — skipping CLIP score. "
                        "pip install clip")
            return -1.0

        model, preprocess = openai_clip.load("ViT-B/32", device=args.device)
        model.eval()

        img_dir = os.path.join(args.save_dir, mode)
        img_files = sorted(Path(img_dir).glob("*.png"))

        if not img_files:
            log.warning("No images found in %s", img_dir)
            return -1.0

        # reload prompts to align with image indices
        prompts = _load_prompts(args.coco_csv, len(img_files))

        scores = []
        batch = args.clip_batch

        for start in range(0, len(img_files), batch):
            batch_files   = img_files[start:start + batch]
            batch_prompts = prompts[start:start + len(batch_files)]

            images = torch.stack([
                preprocess(Image.open(f).convert("RGB"))
                for f in batch_files
            ]).to(args.device)

            texts = openai_clip.tokenize(
                [p[:77] for p in batch_prompts],
                truncate=True
            ).to(args.device)

            with torch.no_grad():
                img_feat  = model.encode_image(images)
                txt_feat  = model.encode_text(texts)
                img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                sims      = (img_feat * txt_feat).sum(dim=-1)
                scores.extend(sims.cpu().tolist())

            if (start // batch) % 10 == 0:
                log.info("  CLIP [%s] %d/%d", mode, start + len(batch_files), len(img_files))

        return float(sum(scores) / len(scores)) if scores else -1.0

    # ── FID ───────────────────────────────────────────────────────────────
    def compute_fid(mode: str, ref_mode: str = "none") -> float:
        """
        Compute FID between `mode` images and `ref_mode` images.

        We use the generated unsafe baseline (mode='none') as reference
        rather than real COCO images, because:
          1. Real COCO images may not be on the server
          2. This directly measures how much TECA changes the output
             distribution relative to unfiltered generation

        If you have real COCO images, pass --real_coco_dir and we compare
        against those instead.
        """
        try:
            from cleanfid import fid
        except ImportError:
            log.warning("clean-fid not installed — skipping FID. "
                        "pip install clean-fid")
            return -1.0

        gen_dir = os.path.join(args.save_dir, mode)

        if args.real_coco_dir and os.path.isdir(args.real_coco_dir):
            ref_dir = args.real_coco_dir
            log.info("FID [%s] vs real COCO images", mode)
        else:
            ref_dir = os.path.join(args.save_dir, ref_mode)
            log.info("FID [%s] vs baseline [%s]", mode, ref_mode)

        if not os.path.exists(gen_dir) or not os.listdir(gen_dir):
            log.warning("No images in %s — skipping FID", gen_dir)
            return -1.0

        score = fid.compute_fid(
            gen_dir,
            ref_dir,
            device    = torch.device(args.device),
            num_workers = 4,
        )
        return float(score)

    # ── run all ───────────────────────────────────────────────────────────
    results = {}

    log.info("Computing CLIP scores …")
    for mode in MODES:
        clip_score = compute_clip_score(mode)
        results[mode] = {"clip": round(clip_score, 4)}
        log.info("  CLIP [%s] = %.4f", mode, clip_score)

    log.info("Computing FID scores …")
    for mode in MODES:
        fid_score = compute_fid(mode, ref_mode="none")
        results[mode]["fid"] = round(fid_score, 2)
        log.info("  FID  [%s] = %.2f", mode, fid_score)

    # ── print table ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Mode':<12}  {'CLIP ↑':>10}  {'FID ↓':>10}")
    print("-" * 55)
    for mode in MODES:
        r = results[mode]
        label = {
            "none": "Unsafe (base)",
            "text": "Text-ortho",
            "ca":   "CA-steer",
            "both": "TECA (ours)",
        }[mode]
        print(f"{label:<20}  {r['clip']:>10.4f}  {r['fid']:>10.2f}")
    print("=" * 55)
    print()
    print("FID is relative to the unsafe baseline (mode=none).")
    print("Closer to 0 = less quality damage from the method.")
    if args.real_coco_dir:
        print(f"FID computed against real COCO images: {args.real_coco_dir}")

    # save results
    out_path = os.path.join(args.save_dir, "scores.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Scores saved → %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="COCO image quality evaluation for TECA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── generate ──────────────────────────────────────────────────────────
    g = sub.add_parser("generate",
                       help="Generate COCO images under all 4 modes.")
    g.add_argument("--sv_path",    required=True,
                   help="Path to steering vectors .pt file")
    g.add_argument("--save_dir",   required=True,
                   help="Root output directory")
    g.add_argument("--coco_csv",   default=COCO_CSV,
                   help="Path to coco_30k_10k.csv")
    g.add_argument("--n",          type=int, default=1000,
                   help="Number of prompts (2048+ for reliable FID)")
    g.add_argument("--model_id",   default="CompVis/stable-diffusion-v1-4")
    g.add_argument("--concept",    default="nudity")
    g.add_argument("--num_pairs",  type=int, default=50)
    g.add_argument("--device",     default="cuda")
    g.add_argument("--seed",       type=int, default=42)
    g.add_argument("--steps",      type=int, default=50)
    g.add_argument("--guidance",   type=float, default=7.5)
    g.add_argument("--img_size",   type=int, default=512)
    g.add_argument("--beta",       type=float, default=2.0)
    g.add_argument("--schedule",   default="front",
                   choices=["front", "always", "band"])
    g.add_argument("--tau_relax",  type=float, default=0.2)
    g.add_argument("--w_min",      type=float, default=0.5)

    # ── score ─────────────────────────────────────────────────────────────
    s = sub.add_parser("score",
                       help="Compute FID and CLIP from generated images.")
    s.add_argument("--save_dir",      required=True)
    s.add_argument("--coco_csv",      default=COCO_CSV)
    s.add_argument("--device",        default="cuda")
    s.add_argument("--clip_batch",    type=int, default=64)
    s.add_argument("--real_coco_dir", default=None,
                   help="Optional: path to real COCO val2017 images for "
                        "absolute FID (instead of baseline-relative FID)")

    args = p.parse_args()

    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "score":
        cmd_score(args)