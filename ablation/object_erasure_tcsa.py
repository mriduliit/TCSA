#!/usr/bin/env python3
"""
ablation/object_erasure_tcsa.py — TCSA on NLCE's object-erasure benchmark
================================================================================

NLCE (CVPR 2026, github.com/alirezafarashah/NLCE) introduces a fine-grained
object-erasure benchmark: erase ONE species (e.g. the dog breed "Bluetick",
or the flower type "camellia") while preserving every OTHER species in the
same fine-grained category (94 Stanford Dogs breeds / 34 Oxford Flowers
types) -- this "neighbor preservation" is NLCE's central claim.

This script applies TCSA to that exact benchmark (same prompt CSVs, same
species, same seeds: `ablation/nlce_data/config_dogs_bluetick.csv` and
`config_flowers_camellia.csv`, copied verbatim from the NLCE repo) -- in
EXACTLY the way TCSA was already extended for style erasure in
`merged_unlearning_TECA_style.py`: add a new concept to CONCEPT_VOCAB /
SAFE_CONCEPT_VOCAB and a dynamic paired-prompt generator, reusing the
unmodified `build_steering_vectors` / `MergedUnlearningPipeline` machinery
from merged_unlearning_multi_gpu.py via monkey-patch (no edits to that
file, same pattern as every other ablation/*.py in this repo).

Concept definition for an object (target species X, siblings = all other
species in the same CSV):
    CONCEPT_VOCAB[X]      = [X, "a X", "a photo of a X", "an image of a X"]
    SAFE_CONCEPT_VOCAB[X] = a fixed sample of ~10 sibling species names
    paired prompts        = ("a photo of a X", "a photo of a {sibling}")
                             cycling through siblings -- mirrors the style
                             concept's dynamic ImageNet-subject pairing.

Evaluation metric (NLCE's own classifiers -- a private SR-GNN checkpoint
for dogs, an unpublished HF checkpoint for flowers -- are not available to
us). We substitute CLIP zero-shot classification over the full species
label set, the standard metric family used elsewhere in the concept-
erasure literature (ESD/MACE/Receler):
    - target erasure accuracy  = % of target-species images classified as
                                  the target species (want LOW after TCSA)
    - neighbor retention accuracy = % of other-species images classified
                                  correctly as their own species (want HIGH
                                  and unchanged after TCSA -- this is
                                  exactly NLCE's neighbor-preservation axis)
both computed for "none" (unsteered baseline) and "both" (full TCSA).

Usage:
    # phase 1: generate (resumable, shardable across GPUs)
    CUDA_VISIBLE_DEVICES=1 python ablation/object_erasure_tcsa.py generate \\
        --csv ablation/nlce_data/config_flowers_camellia.csv \\
        --target camellia --save_dir ablation/results/object_erasure/flowers_camellia \\
        --shard_idx 0 --num_shards 1

    # phase 2: evaluate (CLIP zero-shot, after all shards finish generating)
    python ablation/object_erasure_tcsa.py evaluate \\
        --csv ablation/nlce_data/config_flowers_camellia.csv \\
        --target camellia --save_dir ablation/results/object_erasure/flowers_camellia

Run inside the `casteer` conda env.
"""
import argparse
import json
import os
import random
import sys

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, "/storage/pk_mridul/TCSA/scripts/evaluation")
import merged_unlearning_multi_gpu as mu  # noqa: E402
from merged_unlearning_multi_gpu import (  # noqa: E402
    MergedUnlearningPipeline, Schedule, build_steering_vectors,
)


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def setup_object_concept(target: str, siblings: list, num_neg_vocab: int = 10, seed: int = 0):
    """Monkey-patch CONCEPT_VOCAB / SAFE_CONCEPT_VOCAB / _paired_prompts for `target`.

    Mirrors merged_unlearning_TECA_style.py's pattern for artist/style concepts,
    but for a fine-grained object species with explicit sibling species as the
    contrastive "safe" set (instead of "photorealistic, no style").
    """
    rng = random.Random(seed)
    neg_sample = rng.sample(siblings, min(num_neg_vocab, len(siblings)))

    mu.CONCEPT_VOCAB[target] = [
        target, f"a {target}", f"a photo of a {target}", f"an image of a {target}",
    ]
    mu.SAFE_CONCEPT_VOCAB[target] = neg_sample

    _orig_paired_prompts = mu._paired_prompts

    def _patched_paired_prompts(concept, num_pairs):
        if concept == target:
            pairs = []
            i = 0
            while len(pairs) < num_pairs:
                sib = siblings[i % len(siblings)]
                pairs.append((f"a photo of a {target}", f"a photo of a {sib}"))
                i += 1
            return pairs[:num_pairs]
        return _orig_paired_prompts(concept, num_pairs)

    mu._paired_prompts = _patched_paired_prompts


def cmd_generate(args):
    df = pd.read_csv(args.csv)
    target_rows = df[df["type"] == "erased"]
    assert target_rows["species"].nunique() == 1, "Expected exactly one 'erased' species"
    target = target_rows["species"].iloc[0]
    siblings = sorted(df[df["species"] != target]["species"].unique().tolist())
    print(f"Target species: {target}  |  {len(siblings)} sibling species  |  {len(df)} total rows")

    setup_object_concept(target, siblings)

    os.makedirs(args.save_dir, exist_ok=True)
    sv_dir = os.path.join(args.save_dir, "..", "sv_cache")
    os.makedirs(sv_dir, exist_ok=True)
    sv_path = os.path.join(sv_dir, f"sv_{slug(target)}.pt")

    # NOTE: use_text_weighting=False and layer_align=None below — this mirrors
    # style_erasure_TECA_final.py exactly (Van Gogh/Picasso/Monet), NOT the
    # adaptive-per-layer-beta path in merged_unlearning_multi_gpu.py's default
    # nudity pipeline. For a fine-grained single-entity concept (one dog breed,
    # one flower species — same category as an artist "style"), the adaptive
    # beta's softmax-over-alignment can concentrate almost all suppression
    # budget onto a single CA layer (observed beta=13.4 on a single "up" layer
    # for "Bluetick", vs <5 for broad concepts like nudity), which blows that
    # layer's feature map out of distribution and collapses the image into a
    # repeating-tile hallucination. Flat global beta avoids this entirely.
    sv, _ = build_steering_vectors(
        model_id=args.model_id, concept=target, num_pairs=args.num_pairs,
        save_path=sv_path, device=args.device, num_steps=30,
        use_text_weighting=False)

    wrapper = MergedUnlearningPipeline.from_pretrained(
        args.model_id, sv, device=args.device, concept=target,
        beta=args.beta, n_components=1, max_proj_ratio=args.max_proj_ratio,
        layer_align=None)

    baseline_dir = os.path.join(args.save_dir, "baseline")
    steered_dir = os.path.join(args.save_dir, "steered")
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(steered_dir, exist_ok=True)

    rows = list(df.itertuples())
    shard_rows = rows[args.shard_idx::args.num_shards]
    print(f"Shard {args.shard_idx}/{args.num_shards}: {len(shard_rows)} rows")

    for row in tqdm(shard_rows, desc=f"shard{args.shard_idx}"):
        idx = row.Index
        species_slug = slug(row.species)
        seed = int(row.seed)
        prompt = str(row.prompt)

        out_base = os.path.join(baseline_dir, f"{idx}_{species_slug}.png")
        out_steer = os.path.join(steered_dir, f"{idx}_{species_slug}.png")

        if not os.path.exists(out_base):
            img = wrapper.generate(prompt=prompt, mode="none", schedule=None,
                                    seed=seed, guidance_scale=args.guidance,
                                    num_steps=args.steps, height=args.img_size,
                                    width=args.img_size)
            img.save(out_base)

        if not os.path.exists(out_steer):
            img = wrapper.generate(prompt=prompt, mode="both", schedule=Schedule(),
                                    seed=seed, guidance_scale=args.guidance,
                                    num_steps=args.steps, height=args.img_size,
                                    width=args.img_size)
            img.save(out_steer)

    wrapper.remove_hooks()
    print("Shard done.")


def cmd_evaluate(args):
    import clip
    from PIL import Image

    df = pd.read_csv(args.csv)
    target_rows = df[df["type"] == "erased"]
    target = target_rows["species"].iloc[0]
    all_species = sorted(df["species"].unique().tolist())
    print(f"Target: {target}  |  {len(all_species)} species total")

    device = args.device
    model, preprocess = clip.load("ViT-B/32", device=device)

    with torch.no_grad():
        text_tokens = clip.tokenize([f"a photo of a {s}" for s in all_species]).to(device)
        text_feats = model.encode_text(text_tokens).float()
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

    def classify(img_path):
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img).float()
            feat /= feat.norm(dim=-1, keepdim=True)
            sims = (feat @ text_feats.T).squeeze(0)
        return all_species[sims.argmax().item()]

    baseline_dir = os.path.join(args.save_dir, "baseline")
    steered_dir = os.path.join(args.save_dir, "steered")

    results = {"target": target, "n_species": len(all_species), "per_row": []}
    target_correct = {"baseline": 0, "steered": 0, "n": 0}
    neighbor_correct = {"baseline": 0, "steered": 0, "n": 0}

    for row in tqdm(list(df.itertuples()), desc="classifying"):
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

        results["per_row"].append({
            "idx": idx, "species": row.species, "is_target": is_target,
            "pred_baseline": pred_base, "pred_steered": pred_steer,
        })

    def pct(c, n):
        return round(100 * c / n, 2) if n else None

    results["target_erasure"] = {
        "n": target_correct["n"],
        "baseline_acc_pct": pct(target_correct["baseline"], target_correct["n"]),
        "steered_acc_pct": pct(target_correct["steered"], target_correct["n"]),
        "note": "lower steered_acc_pct = better erasure (target no longer recognized)",
    }
    results["neighbor_retention"] = {
        "n": neighbor_correct["n"],
        "baseline_acc_pct": pct(neighbor_correct["baseline"], neighbor_correct["n"]),
        "steered_acc_pct": pct(neighbor_correct["steered"], neighbor_correct["n"]),
        "note": "steered_acc_pct close to baseline_acc_pct = good neighbor preservation",
    }

    out_path = os.path.join(args.save_dir, "clip_zeroshot_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"TARGET ERASURE   ({target}, n={target_correct['n']}):")
    print(f"  baseline accuracy : {results['target_erasure']['baseline_acc_pct']}%")
    print(f"  TCSA accuracy     : {results['target_erasure']['steered_acc_pct']}%  (lower = better erasure)")
    print(f"\nNEIGHBOR RETENTION (all other species, n={neighbor_correct['n']}):")
    print(f"  baseline accuracy : {results['neighbor_retention']['baseline_acc_pct']}%")
    print(f"  TCSA accuracy     : {results['neighbor_retention']['steered_acc_pct']}%  (close to baseline = good)")
    print("=" * 70)
    print(f"Saved -> {out_path}")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)

g = sub.add_parser("generate")
g.add_argument("--csv", required=True)
g.add_argument("--save_dir", required=True)
g.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
g.add_argument("--device", default="cuda")
g.add_argument("--steps", type=int, default=30)
g.add_argument("--guidance", type=float, default=7.5)
g.add_argument("--img_size", type=int, default=512)
g.add_argument("--beta", type=float, default=2.0)
g.add_argument("--max_proj_ratio", type=float, default=0.5)
g.add_argument("--num_pairs", type=int, default=30)
g.add_argument("--shard_idx", type=int, default=0)
g.add_argument("--num_shards", type=int, default=1)

e = sub.add_parser("evaluate")
e.add_argument("--csv", required=True)
e.add_argument("--save_dir", required=True)
e.add_argument("--device", default="cuda")

args = ap.parse_args()
if args.cmd == "generate":
    cmd_generate(args)
elif args.cmd == "evaluate":
    cmd_evaluate(args)
