#!/usr/bin/env python3
"""
ablation/tcl_distribution_plot.py
==================================
Plots the distribution of TCL (Text-Concept Load) values across:
  - COCO prompts (1000 safe/benign prompts)
  - I2P-Sexual prompts (931 unsafe prompts)

Two variants compared:
  - Buggy TCL (includes BOS token): everything saturates to ~0.999
  - Fixed TCL (BOS excluded):       safe prompts cluster near 0,
                                     unsafe prompts cluster near 1

This directly demonstrates that the TCL gate *discriminates* between
safe and unsafe prompts — the key claim for its inclusion in TCSA.

Run inside the casteer conda env. CPU-only (no image generation).
"""
import argparse
import math
import os
import sys

import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, "/storage/pk_mridul/TCSA/scripts/evaluation")
import merged_unlearning_multi_gpu as mu

ap = argparse.ArgumentParser()
ap.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
ap.add_argument("--sv_path", default="/storage/pk_mridul/TCSA/steering_vectors/sv_nudity_beta_1.pt")
ap.add_argument("--i2p_small_csv", default="/storage/pk_mridul/TCSA/examples/examples/prompts/1774333861566_unsafe-prompts4703copy.csv")
ap.add_argument("--i2p_main_csv", default="/storage/pk_mridul/TCSA/datasets/i2p.csv")
ap.add_argument("--coco_csv", default="/storage/pk_mridul/TCSA/datasets/coco_30k_10k.csv")
ap.add_argument("--n_coco", type=int, default=1000)
ap.add_argument("--n_i2p_sexual", type=int, default=931)
ap.add_argument("--device", default="cuda")
ap.add_argument("--save_dir", default="/storage/pk_mridul/TCSA/ablation/results/tcl_distribution")
args = ap.parse_args()

os.makedirs(args.save_dir, exist_ok=True)


# ── load prompts ──────────────────────────────────────────────────────────────

def load_i2p_sexual(small_csv, main_csv, n):
    small = pd.read_csv(small_csv)
    sex = small[small["categories"].astype(str).str.contains("sexual", case=False, na=False)]
    prompts = sex["prompt"].astype(str).tolist()
    if len(prompts) < n:
        main = pd.read_csv(main_csv)
        sex2 = main[main["categories"].astype(str).str.contains("sexual", case=False, na=False)]
        extra = [p for p in sex2["prompt"].astype(str).tolist() if p not in prompts]
        prompts += extra[:n - len(prompts)]
    return prompts[:n]

def load_i2p_by_category(main_csv):
    """Return dict: category -> list of prompts."""
    df = pd.read_csv(main_csv)
    cats = {}
    for _, row in df.iterrows():
        for cat in str(row.get("categories", "")).split(","):
            cat = cat.strip().lower()
            if cat:
                cats.setdefault(cat, []).append(str(row["prompt"]))
    return cats

coco_df = pd.read_csv(args.coco_csv)
ccol = "prompt" if "prompt" in coco_df.columns else "caption"
coco_prompts = coco_df[ccol].astype(str).head(args.n_coco).tolist()
print(f"COCO: {len(coco_prompts)} prompts")

i2p_sexual = load_i2p_sexual(args.i2p_small_csv, args.i2p_main_csv, args.n_i2p_sexual)
print(f"I2P-Sexual: {len(i2p_sexual)} prompts")

i2p_by_cat = load_i2p_by_category(args.i2p_main_csv)
print(f"I2P categories: {list(i2p_by_cat.keys())}")


# ── load text encoder + concept direction ─────────────────────────────────────

print("Loading text encoder...")
tokenizer = CLIPTokenizer.from_pretrained(
    args.model_id, subfolder="tokenizer", cache_dir="/storage/pk_mridul/TCSA/cache")
text_encoder = CLIPTextModel.from_pretrained(
    args.model_id, subfolder="text_encoder", cache_dir="/storage/pk_mridul/TCSA/cache").to(args.device)
text_encoder.eval()

print("Building text-space concept direction (same formula as pipeline)...")

def encode_mean(words, tokenizer, text_encoder, device):
    """Mean-pool the token embeddings for a list of words (one word at a time)."""
    embs = []
    for word in words:
        tok = tokenizer(word, padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb = text_encoder(tok.input_ids.to(device))[0][0].float()  # [seq, dim]
        embs.append(emb.mean(dim=0))
    return torch.stack(embs)  # [n_words, dim]

unsafe_embs = encode_mean(mu.CONCEPT_VOCAB["nudity"], tokenizer, text_encoder, args.device)
safe_embs   = encode_mean(mu.SAFE_CONCEPT_VOCAB["nudity"], tokenizer, text_encoder, args.device)
direction = unsafe_embs.mean(0) - safe_embs.mean(0)
d = (direction / direction.norm().clamp(min=1e-8)).to(args.device).float()  # [dim]
print(f"Concept direction shape: {d.shape}")


def encode_prompt(prompt):
    """Returns (cond_emb [seq, dim], attention_mask [seq])."""
    tok = tokenizer(prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        emb = text_encoder(tok.input_ids.to(args.device))[0][0].float()  # [seq, dim]
    return emb, tok.attention_mask[0]


def tcl_buggy(emb):
    """Original (buggy): max over ALL tokens including BOS."""
    token_dots = emb @ d          # [seq]
    max_dot = token_dots.max().item()
    return 1.0 / (1.0 + math.exp(-3.0 * max_dot))


def tcl_fixed(emb, mask):
    """Fixed: exclude BOS (index 0) and padding from max."""
    token_dots = emb @ d          # [seq]
    real = token_dots[mask.bool()]  # drop padding
    real_no_bos = real[1:] if real.shape[0] > 1 else real  # drop BOS
    max_dot = real_no_bos.max().item()
    return 1.0 / (1.0 + math.exp(-3.0 * max_dot))


# ── compute TCL for all prompt sets ──────────────────────────────────────────

def compute_tcl_list(prompts, label):
    buggy_vals, fixed_vals = [], []
    for p in tqdm(prompts, desc=label):
        emb, mask = encode_prompt(p)
        buggy_vals.append(tcl_buggy(emb))
        fixed_vals.append(tcl_fixed(emb, mask))
    return buggy_vals, fixed_vals


coco_buggy, coco_fixed   = compute_tcl_list(coco_prompts, "COCO")
i2p_buggy,  i2p_fixed    = compute_tcl_list(i2p_sexual,   "I2P-Sexual")

# also per-category for Option 2 support
cat_fixed = {}
for cat, prompts in i2p_by_cat.items():
    _, fvals = compute_tcl_list(prompts[:200], f"I2P-{cat}")
    cat_fixed[cat] = fvals

import json
with open(os.path.join(args.save_dir, "tcl_values.json"), "w") as f:
    json.dump({
        "coco_buggy": coco_buggy,
        "coco_fixed": coco_fixed,
        "i2p_sexual_buggy": i2p_buggy,
        "i2p_sexual_fixed": i2p_fixed,
        "i2p_by_category_fixed": {k: v for k, v in cat_fixed.items()},
    }, f, indent=2)
print("Saved TCL values.")

# ── statistics ────────────────────────────────────────────────────────────────

print("\n=== TCL Statistics ===")
for name, vals in [("COCO (buggy)", coco_buggy), ("COCO (fixed)", coco_fixed),
                   ("I2P-Sexual (buggy)", i2p_buggy), ("I2P-Sexual (fixed)", i2p_fixed)]:
    arr = np.array(vals)
    print(f"{name:25s}  mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"std={arr.std():.4f}  <0.5: {(arr<0.5).mean()*100:.1f}%  >0.9: {(arr>0.9).mean()*100:.1f}%")

# ── PLOT 1: Buggy vs Fixed side-by-side KDE ───────────────────────────────────

from scipy.stats import gaussian_kde

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

for ax, buggy_c, fixed_c, buggy_i, fixed_i, title in [
    (axes[0], coco_buggy, coco_fixed, i2p_buggy, i2p_fixed, "TCL Gate: Buggy vs Fixed")]:

    bins = np.linspace(0, 1, 60)

    # fixed
    ax.hist(coco_fixed,  bins=bins, alpha=0.55, color="#2196F3", label="COCO / safe (fixed)",    density=True)
    ax.hist(i2p_fixed,   bins=bins, alpha=0.55, color="#F44336", label="I2P-Sexual / unsafe (fixed)", density=True)
    ax.set_xlabel("TCL value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("TCL Distribution — Fixed (BOS excluded)", fontsize=12)
    ax.legend(fontsize=10)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

# buggy plot
ax2 = axes[1]
ax2.hist(coco_buggy,  bins=bins, alpha=0.55, color="#2196F3", label="COCO / safe (buggy)",    density=True)
ax2.hist(i2p_buggy,   bins=bins, alpha=0.55, color="#F44336", label="I2P-Sexual / unsafe (buggy)", density=True)
ax2.set_xlabel("TCL value", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.set_title("TCL Distribution — Buggy (BOS included)", fontsize=12)
ax2.legend(fontsize=10)
ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

plt.suptitle("Text-Concept Load Gate: Safe vs Unsafe Prompt Discrimination", fontsize=13, fontweight="bold")
plt.tight_layout()
out1 = os.path.join(args.save_dir, "tcl_distribution_buggy_vs_fixed.pdf")
plt.savefig(out1, bbox_inches="tight", dpi=150)
plt.savefig(out1.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"Saved -> {out1}")
plt.close()

# ── PLOT 2: Fixed only — all I2P categories ───────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
colors = plt.cm.tab10(np.linspace(0, 1, len(cat_fixed)))

# COCO baseline
ax.hist(coco_fixed, bins=bins, alpha=0.4, color="gray", label="COCO (safe)", density=True)

for (cat, vals), col in zip(sorted(cat_fixed.items()), colors):
    arr = np.array(vals)
    ax.hist(arr, bins=bins, alpha=0.4, color=col, label=f"I2P-{cat} (n={len(vals)})", density=True)

ax.set_xlabel("TCL value (fixed)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("TCL Distribution per I2P Category vs COCO (Fixed Gate)", fontsize=12)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
out2 = os.path.join(args.save_dir, "tcl_distribution_by_category.pdf")
plt.savefig(out2, bbox_inches="tight", dpi=150)
plt.savefig(out2.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"Saved -> {out2}")
plt.close()

print("\nDone. Plots saved to:", args.save_dir)
