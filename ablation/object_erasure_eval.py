#!/usr/bin/env python3
"""
ablation/object_erasure_eval.py — NLCE-style evaluation for TCSA's
object-erasure generations (separate from object_erasure_tcsa.py, which
only handles generation).

Reads the baseline/ (mode="none") and steered/ (mode="both", TCSA)
directories produced by `object_erasure_tcsa.py generate` and computes the
exact same five metrics as NLCE's Table 1 (Oxford Flowers / Stanford Dogs):

    Acc_t (down) — target-species classification accuracy (lower = better erasure)
    Acc_r (up)   — neighbor/other-species classification accuracy (higher = better retention)
    H_o   (up)   — harmonic mean of erasure-success (1-Acc_t) and retention (Acc_r)
    CS    (up)   — CLIP score (image-text alignment / fidelity)
    KID   (down) — Kernel Inception Distance vs the unsteered baseline

Methodology, copied exactly from the NLCE repo (alirezafarashah/NLCE):
  - Acc_t / Acc_r: NLCE's own classifiers (SR-GNN for dogs, a private HF
    checkpoint for flowers) are not available to us, so we substitute CLIP
    zero-shot classification over the full species label set — the
    standard substitute used elsewhere in the concept-erasure literature
    (ESD/MACE/Receler). The accuracy DEFINITION matches NLCE's
    `Stanford_Dogs_Classification.ipynb` / `Oxford_Flowers_Classification.ipynb`
    `analyze_retention_erasure()` exactly: Acc_t = % of target-species
    images correctly predicted as the target; Acc_r = % of other-species
    images correctly predicted as their own species.
  - H_o: identical formula to NLCE's `Hcc_dec`:
        H = 2 * ((1-Acc_t/100)*(Acc_r/100)) / ((1-Acc_t/100)+(Acc_r/100)) * 100
  - CS: identical method to NLCE's `evaluation/nudity/clip_score.py` —
    HF `CLIPModel`/`CLIPProcessor` ("openai/clip-vit-base-patch32"),
    `logits_per_image`, averaged over every image against the prompt that
    generated it.
  - KID: identical method to NLCE's `evaluation/nudity/fid_score.py` —
    `cleanfid.fid.compute_kid(dir1, dir2)`. NLCE computes KID of each
    method's generations against SD v1.4 (the unsteered model); we use our
    own "baseline" dir (mode="none") as that exact reference, so KID
    measures how far TCSA's "steered" outputs drift from unsteered SD1.4.

New file, does not modify object_erasure_tcsa.py.
Run inside the `casteer` conda env.
"""
import argparse
import json
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def main(args):
    import clip as openai_clip

    df = pd.read_csv(args.csv)
    target_rows = df[df["type"] == "erased"]
    target = target_rows["species"].iloc[0]
    all_species = sorted(df["species"].unique().tolist())
    print(f"Target: {target}  |  {len(all_species)} species total  |  {len(df)} rows")

    device = args.device
    baseline_dir = os.path.join(args.save_dir, "baseline")
    steered_dir = os.path.join(args.save_dir, "steered")

    # ── Acc_t / Acc_r via CLIP zero-shot classification ──────────────────────
    zs_model, zs_preprocess = openai_clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        text_tokens = openai_clip.tokenize([f"a photo of a {s}" for s in all_species]).to(device)
        text_feats = zs_model.encode_text(text_tokens).float()
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

    def classify(img_path):
        img = zs_preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = zs_model.encode_image(img).float()
            feat /= feat.norm(dim=-1, keepdim=True)
            sims = (feat @ text_feats.T).squeeze(0)
        return all_species[sims.argmax().item()]

    target_correct = {"baseline": 0, "steered": 0, "n": 0}
    neighbor_correct = {"baseline": 0, "steered": 0, "n": 0}
    per_row = []

    for row in tqdm(list(df.itertuples()), desc="classifying (Acc_t/Acc_r)"):
        idx = row.Index
        species_slug = slug(row.species)
        out_base = os.path.join(baseline_dir, f"{idx}_{species_slug}.png")
        out_steer = os.path.join(steered_dir, f"{idx}_{species_slug}.png")
        if not (os.path.exists(out_base) and os.path.exists(out_steer)):
            continue

        pred_base = classify(out_base)
        pred_steer = classify(out_steer)
        is_target = (row.species == target)

        bucket = target_correct if is_target else neighbor_correct
        bucket["n"] += 1
        bucket["baseline"] += int(pred_base == row.species)
        bucket["steered"] += int(pred_steer == row.species)
        per_row.append({"idx": idx, "species": row.species, "is_target": is_target,
                         "pred_baseline": pred_base, "pred_steered": pred_steer})

    del zs_model
    torch.cuda.empty_cache()

    def pct(c, n):
        return round(100 * c / n, 2) if n else None

    acc_t_base = pct(target_correct["baseline"], target_correct["n"])
    acc_t_steer = pct(target_correct["steered"], target_correct["n"])
    acc_r_base = pct(neighbor_correct["baseline"], neighbor_correct["n"])
    acc_r_steer = pct(neighbor_correct["steered"], neighbor_correct["n"])

    # ── H_o: NLCE's exact harmonic-mean formula ──────────────────────────────
    def harmonic(acc_t, acc_r):
        if acc_t is None or acc_r is None:
            return None
        e, r = 1 - acc_t / 100, acc_r / 100
        return round(2 * e * r / (e + r) * 100, 2) if (e + r) > 0 else 0.0

    h_o_base = harmonic(acc_t_base, acc_r_base)
    h_o_steer = harmonic(acc_t_steer, acc_r_steer)

    # ── CS: NLCE's exact CLIP-score method ───────────────────────────────────
    from transformers import CLIPProcessor, CLIPModel
    cs_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    cs_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def mean_clip_score(img_dir):
        scores = []
        for row in tqdm(list(df.itertuples()), desc=f"CS {os.path.basename(img_dir)}"):
            idx = row.Index
            species_slug = slug(row.species)
            path = os.path.join(img_dir, f"{idx}_{species_slug}.png")
            if not os.path.exists(path):
                continue
            image = Image.open(path).convert("RGB")
            inputs = cs_processor(text=str(row.prompt), images=image,
                                   return_tensors="pt", padding=True)
            with torch.no_grad():
                out = cs_model(**{k: v.to(device) for k, v in inputs.items()})
            scores.append(out.logits_per_image[0][0].item())
        return round(float(sum(scores) / len(scores)), 2) if scores else None

    cs_base = mean_clip_score(baseline_dir)
    cs_steer = mean_clip_score(steered_dir)
    del cs_model
    torch.cuda.empty_cache()

    # ── KID: cleanfid, baseline (unsteered) vs steered (TCSA) ───────────────
    from cleanfid import fid
    try:
        kid = round(float(fid.compute_kid(baseline_dir, steered_dir, device=device,
                                           use_dataparallel=False)), 4)
    except Exception as e:
        kid = None
        print(f"KID computation failed: {e}")

    results = {
        "target": target, "n_species": len(all_species),
        "target_erasure": {"n": target_correct["n"], "baseline_acc_pct": acc_t_base, "steered_acc_pct": acc_t_steer},
        "neighbor_retention": {"n": neighbor_correct["n"], "baseline_acc_pct": acc_r_base, "steered_acc_pct": acc_r_steer},
        "H_o": {"baseline": h_o_base, "steered": h_o_steer},
        "CS": {"baseline": cs_base, "steered": cs_steer},
        "KID": kid,
        "per_row": per_row,
    }

    out_path = os.path.join(args.save_dir, "nlce_style_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Target: {target}  (NLCE-style metrics, TCSA = our method)")
    print(f"  Acc_t  : {acc_t_steer}%   (baseline {acc_t_base}%)   [lower = better erasure]")
    print(f"  Acc_r  : {acc_r_steer}%   (baseline {acc_r_base}%)   [higher = better retention]")
    print(f"  H_o    : {h_o_steer}%     (baseline {h_o_base}%)     [higher = better balance]")
    print(f"  CS     : {cs_steer}       (baseline {cs_base})       [higher = better fidelity]")
    print(f"  KID    : {kid}            [lower = closer to unsteered SD v1.4]")
    print("=" * 70)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(args)
