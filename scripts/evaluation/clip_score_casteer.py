#!/usr/bin/env python3
"""
clip_score_casteer.py — CLIPScore computed the same way CASteer does
(scripts/diffusion/produce_scores.py -> core.eval.clip.clip_score), as an
alternative to eval_fid_clip.py's plain-cosine CLIP metric.

Differences from eval_fid_clip.compute_clip_score:
  - Official CLIPScore formula (Hessel et al. 2021):
        CLIPScore(I, T) = w * max(cos(E_I, E_T), 0),   w = 2.5
    (eval_fid_clip.py instead reports raw cosine similarity * 100 — no
    re-weighting, no clamping of negative similarities to zero.)
  - CLIP backbone: ViT-B/32 (CASteer default) instead of ViT-L/14.
  - Custom preprocessing pipeline matching the official CLIPScore repo
    (Resize -> CenterCrop -> ToTensor -> Normalize) instead of CLIP's
    bundled `preprocess`, which additionally does its own resize/crop with
    slightly different interpolation/cropping behaviour.

Any of these three differences alone can shift the absolute numbers and,
more importantly, how much the score varies across ablation modes — useful
for sanity-checking the "CLIP barely moves across modes" observation.

Usage
-----
python clip_score_casteer.py \\
    --ablation_dir results/sd14_ablation \\
    --coco_csv datasets/coco_30k_10k.csv \\
    --modes none text ca both
"""

from __future__ import annotations

import argparse
import json
import os

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm


# ── CASteer's exact CLIP preprocessing (core/eval/clip.py::get_clip_preprocess) ──

def get_clip_preprocess(n_px: int = 224):
    def _convert(image):
        return image.convert("RGB")

    image_preprocess = Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        _convert,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    def text_preprocess(text):
        return clip.tokenize(text, truncate=True)

    return image_preprocess, text_preprocess


# ── CASteer's exact CLIPScore formula (core/eval/clip.py::clip_score) ─────────

@torch.no_grad()
def clip_score_paired(
    image_paths: list,
    texts: list,
    metric: str = "cosine100",
    w: float = 2.5,
    clip_model: str = "ViT-B/32",
    n_px: int = 224,
    device: str = "cuda",
    batch_size: int = 50,
) -> np.ndarray:
    """
    Per-pair score, in either of two conventions:

      "cosine100"  ->  100 * cos(E_I, E_T)
                       This is the "CLIP-30K" convention CASteer actually
                       reports in its locality table (Table 3): values land
                       in the ~28-31 range for COCO-30K, matching e.g.
                       "SD v1.4: 31.34", "Ours (clip): 31.09".

      "clipscore"  ->  w * max(cos(E_I, E_T), 0)   (w = 2.5)
                       The official CLIPScore (Hessel et al. 2021) used
                       inside CASteer's core/eval/clip.py::clip_score for
                       *concept*-presence scoring (produce_scores.py),
                       not for the COCO-30K locality numbers. Lands in
                       ~0.0-1.0 — divide Table-3-style numbers by ~100*w
                       to cross-check: 31.09/100/2.5 ≈ 0.124... no — actually
                       0.7793 (clipscore) / 2.5 * 100 = 31.17, i.e. the two
                       conventions are the SAME cosine similarity, just
                       reported on different scales (×100 vs. ×2.5+clamp).
    """
    model, _ = clip.load(clip_model, device=device)
    model.eval()
    image_preprocess, text_preprocess = get_clip_preprocess(n_px)

    scores = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="CASteer CLIP-30K"):
        batch_imgs = image_paths[i:i + batch_size]
        batch_txts = texts[i:i + batch_size]

        imgs = torch.stack([image_preprocess(Image.open(p)) for p in batch_imgs]).to(device)
        txts = text_preprocess(batch_txts).to(device)

        img_f = model.encode_image(imgs)
        txt_f = model.encode_text(txts)
        img_f = img_f / img_f.norm(dim=1, p=2, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=1, p=2, keepdim=True)

        sim = (img_f * txt_f).sum(dim=1)
        if metric == "cosine100":
            scores.append((100.0 * sim).cpu())
        elif metric == "clipscore":
            scores.append((w * sim).clamp(min=0).cpu())
        else:
            raise ValueError(f"Unknown metric '{metric}'")

    return torch.cat(scores).numpy()


def score_mode_dir(coco_dir: str, prompts_csv: str, device: str,
                   clip_model: str, metric: str, w: float) -> tuple:
    df = pd.read_csv(prompts_csv)
    col = "prompt" if "prompt" in df.columns else "caption"

    image_paths, texts = [], []
    for i, row in df.iterrows():
        p = os.path.join(coco_dir, f"{i}_0.png")
        if os.path.exists(p):
            image_paths.append(p)
            texts.append(str(row[col]))

    if not image_paths:
        return None, 0

    scores = clip_score_paired(image_paths, texts, metric=metric, w=w,
                               clip_model=clip_model, device=device)
    return float(np.mean(scores)), len(scores)


def main():
    ap = argparse.ArgumentParser(
        description="CASteer-style CLIPScore (w=2.5, ViT-B/32, official preprocess) "
                    "for the TCSA ablation's CoCo images — drop-in alternative "
                    "metric for the CLIP↑ column of the comparison table.")
    ap.add_argument("--ablation_dir", default=None,
                    help="Dir with {mode}/coco/{i}_0.png subfolders, e.g. results/sd14_ablation")
    ap.add_argument("--coco_csv", default="/storage/pk_mridul/TCSA/datasets/coco_30k_10k.csv")
    ap.add_argument("--modes", nargs="*", default=["none", "text", "ca", "both"])
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="Flat label=path pairs (no {mode}/coco nesting), "
                         "e.g. --dirs TCSA=results/coco_eval_tecaa_sd1.4 "
                         "baseline=results/coco_baseline_sd14")
    ap.add_argument("--clip_model", default="ViT-B/32")
    ap.add_argument("--metric", choices=["cosine100", "clipscore"], default="cosine100",
                    help="cosine100 = 100*cos(image,text)  -> matches CASteer's reported "
                         "CLIP-30K table values (~28-31). "
                         "clipscore = official w*max(cos,0) -> CASteer's concept-CLIPScore "
                         "convention (~0-1), used in produce_scores.py for concept presence.")
    ap.add_argument("--w", type=float, default=2.5, help="Re-weighting factor for --metric clipscore (official=2.5)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    key = "CLIP30K_casteer" if args.metric == "cosine100" else "CLIPScore_casteer"
    results = {}

    if args.dirs:
        for pair in args.dirs:
            label, path = pair.split("=", 1)
            if not os.path.isdir(path):
                print(f"[skip] {path} not found")
                continue
            score, n = score_mode_dir(path, args.coco_csv, args.device, args.clip_model, args.metric, args.w)
            results[label] = {key: round(score, 2) if score is not None else None, "n_images": n}
            print(f"{label:10s}  {key}={results[label][key]}  (n={n})")
    else:
        if not args.ablation_dir:
            ap.error("Provide either --ablation_dir or --dirs")
        for mode in args.modes:
            coco_dir = os.path.join(args.ablation_dir, mode, "coco")
            if not os.path.isdir(coco_dir):
                print(f"[skip] {coco_dir} not found")
                continue
            score, n = score_mode_dir(coco_dir, args.coco_csv, args.device, args.clip_model, args.metric, args.w)
            results[mode] = {key: round(score, 2) if score is not None else None, "n_images": n}
            print(f"{mode:6s}  {key}={results[mode][key]}  (n={n})")

    out = args.output or (os.path.join(args.ablation_dir, "clip_scores_casteer.json")
                          if args.ablation_dir else "clip_scores_casteer.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
