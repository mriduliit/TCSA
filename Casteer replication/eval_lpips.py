"""
Compute LPIPS_e (erased-artist prompts) and LPIPS_u (other-artist prompts)
between SD-v1.4 baseline images and CASteer artist-erased images.

Run inside the `casteer` conda env:

    python eval_lpips.py \
        --baseline_dir /storage/pk_mridul/TCSA/results/artist_baseline_vangogh \
        --erased_dir   "/storage/pk_mridul/TCSA/Casteer replication/results/vangogh" \
        --csv          /storage/pk_mridul/TCSA/datasets/big_artist_prompts.csv \
        --artist       "Van Gogh" \
        --output       "/storage/pk_mridul/TCSA/Casteer replication/results/vangogh/lpips_scores.json"
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import lpips
from PIL import Image
from tqdm import tqdm


def _load(path, device):
    img = Image.open(path).convert("RGB").resize((512, 512))
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--erased_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--artist", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    loss_fn = lpips.LPIPS(net="alex").to(args.device)
    df = pd.read_csv(args.csv)

    scores_e, scores_u = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        idx = row["case_number"]
        base_path = os.path.join(args.baseline_dir, f"{idx}.png")
        erased_path = os.path.join(args.erased_dir, f"{idx}.png")
        if not os.path.exists(base_path) or not os.path.exists(erased_path):
            continue

        with torch.no_grad():
            score = loss_fn(_load(base_path, args.device), _load(erased_path, args.device)).item()

        row_artist = str(row.get("artist", ""))
        if args.artist.lower() in row_artist.lower():
            scores_e.append(score)
        else:
            scores_u.append(score)

    results = {
        "LPIPS_e": round(float(np.mean(scores_e)), 4) if scores_e else 0.0,
        "LPIPS_u": round(float(np.mean(scores_u)), 4) if scores_u else 0.0,
        "n_erased": len(scores_e),
        "n_unerased": len(scores_u),
    }
    print(f"LPIPS_e: {results['LPIPS_e']}  ({results['n_erased']} images)")
    print(f"LPIPS_u: {results['LPIPS_u']}  ({results['n_unerased']} images)")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
