#!/usr/bin/env python3
"""
ablation/tcl_diagnostic.py — mechanistic TCL (text concept load) diagnostic
================================================================================

Addresses the reviewer comment: "The paper's thesis is unconditional
suppression hurts clean inputs and TCL gating fixes this. You never show a
head-to-head of gated vs ungated (lambda=1) on clean, non-target prompts."

The earlier image-level ablation (run_sd14_component_ablation.py, tcl_off
variant) showed gated vs ungated produce statistically IDENTICAL COCO
CLIP/FID (Delta=0.01 on both) -- a null result. This script explains WHY,
without generating a single image: it computes the actual `tcl` value
(`_compute_text_concept_load`'s sigmoid(3 * max_token_dot)`) for every
prompt across several full datasets, using ONLY the CLIP text encoder (no
UNet/VAE loaded, no diffusion sampling) -- a few minutes of pure text
encoding instead of hours of image generation.

If COCO prompts already sit at tcl~0, the null FID/CLIP result is
EXPECTED (the gate has nothing to gate -- the dots/clamp mechanism alone
already suppresses nothing for genuinely unrelated content). If instead
some prompt class has tcl well above 0 while still being safe, THAT class
is where gated vs ungated should actually diverge, and is worth a full
image-generation ablation.

Datasets (all from examples/examples/prompts/, full unfiltered size):
    i2p.csv                              (4703 rows, all categories;
                                           931 tagged "sexual")
    coco_30k_10k.csv                     (9966 rows, clean captions)
    nudity-ring-a-bell.csv                (79 sensitive/normal PAIRS --
                                           same scene, unsafe vs safe)
    mma-diffusion-nsfw-adv-prompts.csv    (1000 rows, adversarial NSFW,
                                           adv_prompt column)
    p4dn_16_prompt.csv                   (151 rows, adversarial P4D-N)

New file. No edits to merged_unlearning_multi_gpu.py -- reimplements the
text-direction + TCL formula standalone (CLIPTokenizer/CLIPTextModel only,
no diffusers pipeline) for speed.
Run inside the `casteer` conda env.
"""
import argparse
import json
import math
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

PROMPTS_DIR = "/storage/pk_mridul/TCSA/examples/examples/prompts"

# Exact CONCEPT_VOCAB["nudity"] / SAFE_CONCEPT_VOCAB["nudity"] from
# merged_unlearning_multi_gpu.py (verified identical, copied not imported
# to avoid pulling in the diffusers UNet/VAE dependency chain).
UNSAFE_VOCAB = [
    "nudity", "nude", "naked", "pornography", "explicit content",
    "erotic", "nsfw", "topless", "genitals", "sex", "sexual",
    "pornographic", "adult content", "bare skin", "undressed",
]
SAFE_VOCAB = [
    "clothed person", "fully dressed", "wearing clothes", "outfit",
    "fashion", "attire", "dressed woman", "dressed man",
    "person in clothes", "portrait", "clothed", "wearing outfit",
    "dressed people", "fashion model", "person wearing shirt",
]


def main(args):
    device = args.device
    tok = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    enc = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").eval().to(device)

    @torch.no_grad()
    def encode_mean(text):
        t = tok(text, padding="max_length", max_length=tok.model_max_length,
                 truncation=True, return_tensors="pt")
        emb = enc(t.input_ids.to(device))[0]
        return emb.float().mean(dim=1).squeeze(0)

    @torch.no_grad()
    def encode_seq_with_mask(text):
        t = tok(text, padding="max_length", max_length=tok.model_max_length,
                 truncation=True, return_tensors="pt")
        cond = enc(t.input_ids.to(device))[0].float()  # [1, seq, dim]
        return cond, t.attention_mask[0]  # attention_mask: 1 for BOS+content+EOS, 0 for trailing pad

    print("Building text direction d from CONCEPT_VOCAB['nudity'] / SAFE_CONCEPT_VOCAB['nudity'] ...")
    mean_unsafe = torch.stack([encode_mean(w) for w in UNSAFE_VOCAB]).mean(dim=0)
    mean_safe = torch.stack([encode_mean(w) for w in SAFE_VOCAB]).mean(dim=0)
    d = mean_unsafe - mean_safe
    d = d / d.norm().clamp(min=1e-8)
    print(f"  d built, norm={d.norm().item():.4f}")

    @torch.no_grad()
    def tcl_variants(prompt: str):
        """Returns (unmasked, mask_padding_only, mask_padding_and_bos) tcl values."""
        cond, attn_mask = encode_seq_with_mask(prompt)
        token_dots = cond[0] @ d.to(cond.dtype)  # [77]

        # (a) as actually deployed: max over ALL 77 positions (BOS + content + EOS + padding)
        max_unmasked = token_dots.max().item()

        # (b) exclude trailing padding only (attention_mask==1: BOS + content + EOS)
        real_dots = token_dots[attn_mask.bool()]
        max_mask_pad = real_dots.max().item()

        # (c) also exclude BOS (position 0) -- max over content tokens + EOS only
        real_no_bos = token_dots[attn_mask.bool()][1:]
        max_mask_pad_bos = real_no_bos.max().item() if len(real_no_bos) > 0 else max_mask_pad

        sig = lambda x: 1.0 / (1.0 + math.exp(-3.0 * x))
        return sig(max_unmasked), sig(max_mask_pad), sig(max_mask_pad_bos)

    def tcl_batch(prompts, desc):
        unmasked, mask_pad, mask_pad_bos = [], [], []
        for p in tqdm(prompts, desc=desc):
            if not str(p).strip():
                continue
            a, b, c = tcl_variants(str(p))
            unmasked.append(a); mask_pad.append(b); mask_pad_bos.append(c)
        return unmasked, mask_pad, mask_pad_bos

    def summarize(vals, label):
        if not vals:
            return {"label": label, "n": 0}
        t = torch.tensor(vals)
        return {
            "label": label, "n": len(vals),
            "mean": round(t.mean().item(), 4),
            "median": round(t.median().item(), 4),
            "p90": round(t.quantile(0.9).item(), 4),
            "p10": round(t.quantile(0.1).item(), 4),
            "frac_above_0.5": round((t > 0.5).float().mean().item(), 4),
        }

    def run_dataset(prompts, key, label):
        unmasked, mask_pad, mask_pad_bos = tcl_batch(prompts, label)
        results[key] = {
            "unmasked (as-deployed)": summarize(unmasked, label),
            "mask_padding_only": summarize(mask_pad, label),
            "mask_padding_and_bos": summarize(mask_pad_bos, label),
        }

    results = {}

    # ── COCO: clean captions, full 9966 ──
    coco = pd.read_csv(os.path.join(PROMPTS_DIR, "coco_30k_10k.csv"))
    run_dataset(coco["prompt"].tolist(), "coco_clean", "COCO captions (clean, full)")

    # ── I2P: full 4703, split by sexual vs non-sexual category ──
    i2p = pd.read_csv(os.path.join(PROMPTS_DIR, "i2p.csv"))
    is_sexual = i2p["categories"].astype(str).str.lower().str.contains("sexual")
    run_dataset(i2p.loc[is_sexual, "prompt"].tolist(), "i2p_sexual", "I2P prompts tagged 'sexual' (full)")
    run_dataset(i2p.loc[~is_sexual, "prompt"].tolist(), "i2p_nonsexual",
                "I2P prompts NOT tagged 'sexual' (full, e.g. violence/hate/shocking)")

    # ── Ring-A-Bell: paired sensitive vs normal (same scene) ──
    ring = pd.read_csv(os.path.join(PROMPTS_DIR, "nudity-ring-a-bell.csv"))
    run_dataset(ring["sensitive prompt"].tolist(), "ring_sensitive", "Ring-A-Bell 'sensitive prompt' (unsafe, paired)")
    run_dataset(ring["normal prompt"].tolist(), "ring_normal", "Ring-A-Bell 'normal prompt' (safe counterpart, same scene)")

    # ── MMA-Diffusion: adversarial NSFW (obfuscated) ──
    mma = pd.read_csv(os.path.join(PROMPTS_DIR, "mma-diffusion-nsfw-adv-prompts.csv"))
    run_dataset(mma["adv_prompt"].dropna().tolist(), "mma_adversarial", "MMA-Diffusion adv_prompt (full, obfuscated NSFW)")

    # ── P4D-N: adversarial prompts ──
    p4dn = pd.read_csv(os.path.join(PROMPTS_DIR, "p4dn_16_prompt.csv"))
    run_dataset(p4dn["prompt"].dropna().tolist(), "p4dn_adversarial", "P4D-N prompt (full, adversarial)")

    out_path = os.path.join(args.save_dir, "tcl_diagnostic_results.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    for variant in ["unmasked (as-deployed)", "mask_padding_only", "mask_padding_and_bos"]:
        print("\n" + "=" * 100)
        print(f"VARIANT: {variant}")
        print(f"{'Dataset':<55} {'n':>6} {'mean':>8} {'median':>8} {'p10':>8} {'p90':>8} {'%>0.5':>8}")
        print("-" * 100)
        for key, r in results.items():
            row = r[variant]
            if row["n"] == 0:
                continue
            print(f"{row['label']:<55} {row['n']:>6} {row['mean']:>8} {row['median']:>8} {row['p10']:>8} {row['p90']:>8} {row['frac_above_0.5']:>8}")
    print("=" * 100)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_dir", default="/storage/pk_mridul/TCSA/ablation/results/tcl_diagnostic")
    args = ap.parse_args()
    main(args)
