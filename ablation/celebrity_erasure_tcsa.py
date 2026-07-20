#!/usr/bin/env python3
"""
ablation/celebrity_erasure_tcsa.py — TCSA on NLCE's celebrity-erasure benchmark
================================================================================

Mirrors `ablation/object_erasure_tcsa.py` exactly (same flat-global-beta fix,
same Acc_t/Acc_r/H_o/CS/KID metric family), applied to NLCE's celebrity
erasure setup (`celebrity_erasure.py`): erase ONE celebrity's identity while
preserving ~9 "neighbor" celebrities (visually/contextually similar public
figures) the same way object erasure preserves sibling species.

Campaigns (target + neighbors, exactly as in NLCE's `celebrity_erasure.py`,
minus 3 individuals with no freely-licensed Wikipedia photo available --
Mark Bezos, Miguel Bezos, Emma Kenney -- and de-duplicating "Gerald Trump"/
"Harvey Trump", both clearly typos for Donald Trump in the original script,
to a single "Donald Trump" reference):
    Elon Musk     vs Jeff Bezos, Mark Zuckerberg, Tim Cook, Bill Gates,
                     Steve Jobs, Satya Nadella, Richard Branson, Warren Buffett
    Anna Kendrick vs Katheryn Winnick, Jenny Lewis, Nicki Clyne, Nicole Parker,
                     Danielle Lloyd, Emily Kinney, Hannah Marks, Kyla Kenedy,
                     Vicky McClure
    Bill Clinton  vs Hillary Clinton, Joe Biden, Barack Obama, Bernie Sanders,
                     John Kerry, Ronald Reagan, Donald Trump, Jeb Bush, Al Gore

Concept setup (identical pattern to object_erasure_tcsa.py's
`setup_object_concept`, just "species" -> "celebrity name"):
    CONCEPT_VOCAB[target]      = [target, "a photo of {target}", ...]
    SAFE_CONCEPT_VOCAB[target] = sample of neighbor names
    paired prompts             = ("a portrait of {target}", "a portrait of {neighbor}")
                                  cycling through neighbors

Same fix as object erasure: `layer_align=None` (flat global beta=2.0,
matching style_erasure_TECA_final.py), `use_text_weighting=False` -- the
adaptive per-layer beta caused catastrophic image collapse on fine-grained
single-entity concepts (confirmed on "Bluetick"); a specific person's face
is exactly the same kind of narrow, texture/identity-localized concept, so
we use the same safe configuration from the start rather than re-discovering
the same failure.

Prompts: NLCE's own joint-portrait fidelity CSVs, used verbatim (same
prompts and seeds as their actual Table 2 / Figure 4 / Figure 5 benchmark)
-- "A portrait of {target} and {other}", 150 rows per target, ~50 unique
"other" celebrities each repeated across a few seeds:
    ablation/nlce_data/celebrity50_fidelity_elon_musk_150.csv
    ablation/nlce_data/celebrity50_fidelity_anna_150.csv
    ablation/nlce_data/celebrity50_fidelity_bill_clinton_150.csv

The steering-vector CONCEPT definition still uses the separate "neighbor"
list (target vs. semantically-close public figures) from
`celebrity_erasure.py`'s `erasure_celeb_neighbors` -- this defines what
direction is erased. The generation prompts (this benchmark) test it on a
DIFFERENT, broader set of "other" celebrities co-occurring in the same
joint portrait, exactly matching NLCE's actual fidelity-test methodology.

New file. No edits to merged_unlearning_multi_gpu.py or object_erasure_tcsa.py.
Run inside the `casteer` conda env (generation only -- face-recognition
evaluation runs separately in the `nlce_face` env via celebrity_erasure_eval.py).
"""
import argparse
import json
import os
import re
import sys

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, "/storage/pk_mridul/TCSA/scripts/evaluation")
import merged_unlearning_multi_gpu as mu  # noqa: E402
from merged_unlearning_multi_gpu import (  # noqa: E402
    MergedUnlearningPipeline, Schedule, build_steering_vectors,
)

# Neighbor lists used ONLY to build the steering vector's concept direction
# (target vs. semantically-close public figures) -- exactly NLCE's
# `erasure_celeb_neighbors`, minus 3 individuals with no freely-licensed
# Wikipedia photo (Mark Bezos, Miguel Bezos, Emma Kenney) and de-duplicating
# "Gerald Trump"/"Harvey Trump" (clear typos for Donald Trump) to one name.
CONCEPT_NEIGHBORS = {
    "Elon Musk": [
        "Jeff Bezos", "Mark Zuckerberg", "Tim Cook", "Bill Gates", "Steve Jobs",
        "Satya Nadella", "Richard Branson", "Warren Buffett",
    ],
    "Anna Kendrick": [
        "Katheryn Winnick", "Jenny Lewis", "Nicki Clyne", "Nicole Parker",
        "Danielle Lloyd", "Emily Kinney", "Hannah Marks", "Kyla Kenedy", "Vicky McClure",
    ],
    "Bill Clinton": [
        "Hillary Clinton", "Joe Biden", "Barack Obama", "Bernie Sanders",
        "John Kerry", "Ronald Reagan", "Donald Trump", "Jeb Bush", "Al Gore",
    ],
}

# NLCE's actual fidelity-test CSVs (joint portrait: target + a different
# "other" celebrity), used verbatim for the generation benchmark.
FIDELITY_CSV = {
    "Elon Musk": "celebrity50_fidelity_elon_musk_150.csv",
    "Anna Kendrick": "celebrity50_fidelity_anna_150.csv",
    "Bill Clinton": "celebrity50_fidelity_bill_clinton_150.csv",
}


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


_KNOWN_SUFFIXES = [" at a public event", " in an official photo", " on the red carpet"]


def _parse_other(prompt: str, target: str) -> str | None:
    """Prompts use several templates ('A portrait of {t} and {o}',
    'An image capturing {t} and {o} at a public event',
    '{t} and {o} in an official photo', ...) -- split on the known
    target name rather than assuming one fixed template."""
    marker = f"{target} and "
    if marker not in prompt:
        return None
    other = prompt.split(marker, 1)[1]
    for suf in _KNOWN_SUFFIXES:
        if other.endswith(suf):
            other = other[: -len(suf)]
            break
    return other.strip()


def build_rows(target: str, csv_dir: str):
    """Load NLCE's fidelity CSV verbatim: (idx, prompt, seed, target, other)."""
    df = pd.read_csv(os.path.join(csv_dir, FIDELITY_CSV[target]))
    rows = []
    for idx, row in df.iterrows():
        other = _parse_other(row["prompt"], target)
        rows.append({
            "idx": idx, "prompt": row["prompt"], "seed": int(row["evaluation_seed"]),
            "target": target, "other": other,
        })
    return rows


def setup_celebrity_concept(target: str, neighbors: list, num_neg_vocab: int = 8):
    """Same pattern as object_erasure_tcsa.py's setup_object_concept."""
    import random
    rng = random.Random(0)
    neg_sample = neighbors if len(neighbors) <= num_neg_vocab else rng.sample(neighbors, num_neg_vocab)

    mu.CONCEPT_VOCAB[target] = [
        target, f"a photo of {target}", f"a portrait of {target}", f"an image of {target}",
    ]
    mu.SAFE_CONCEPT_VOCAB[target] = neg_sample

    _orig_paired_prompts = mu._paired_prompts

    def _patched_paired_prompts(concept, num_pairs):
        if concept == target:
            pairs = []
            i = 0
            while len(pairs) < num_pairs:
                nb = neighbors[i % len(neighbors)]
                pairs.append((f"a portrait of {target}", f"a portrait of {nb}"))
                i += 1
            return pairs[:num_pairs]
        return _orig_paired_prompts(concept, num_pairs)

    mu._paired_prompts = _patched_paired_prompts


def cmd_generate(args):
    target = args.target
    if target not in CONCEPT_NEIGHBORS:
        raise ValueError(f"Unknown target '{target}'. Choose from {list(CONCEPT_NEIGHBORS)}")
    neighbors = CONCEPT_NEIGHBORS[target]
    rows = build_rows(target, args.csv_dir)
    print(f"Target: {target}  |  {len(neighbors)} concept-neighbors  |  {len(rows)} fidelity rows")

    setup_celebrity_concept(target, neighbors)

    os.makedirs(args.save_dir, exist_ok=True)
    sv_dir = os.path.join(args.save_dir, "..", "sv_cache")
    os.makedirs(sv_dir, exist_ok=True)
    sv_path = os.path.join(sv_dir, f"sv_{slug(target)}.pt")

    # Same fix as object erasure: flat global beta, no text-weighted SV build.
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

    with open(os.path.join(args.save_dir, "rows.json"), "w") as f:
        json.dump(rows, f, indent=2)

    shard_rows = rows[args.shard_idx::args.num_shards]
    print(f"Shard {args.shard_idx}/{args.num_shards}: {len(shard_rows)} rows")

    for row in tqdm(shard_rows, desc=f"shard{args.shard_idx}"):
        idx = row["idx"]
        other_slug = slug(row["other"]) if row["other"] else "unknown"
        seed = row["seed"]
        prompt = row["prompt"]

        out_base = os.path.join(baseline_dir, f"{idx}_{other_slug}.png")
        out_steer = os.path.join(steered_dir, f"{idx}_{other_slug}.png")

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


ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True, choices=list(CONCEPT_NEIGHBORS))
ap.add_argument("--csv_dir", default="/storage/pk_mridul/TCSA/ablation/nlce_data")
ap.add_argument("--save_dir", required=True)
ap.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
ap.add_argument("--device", default="cuda")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--guidance", type=float, default=7.5)
ap.add_argument("--img_size", type=int, default=512)
ap.add_argument("--beta", type=float, default=2.0)
ap.add_argument("--max_proj_ratio", type=float, default=0.5)
ap.add_argument("--num_pairs", type=int, default=30)
ap.add_argument("--shard_idx", type=int, default=0)
ap.add_argument("--num_shards", type=int, default=1)
args = ap.parse_args()
cmd_generate(args)
