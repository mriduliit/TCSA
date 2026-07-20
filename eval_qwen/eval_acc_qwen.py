#!/usr/bin/env python3
"""
eval_acc_qwen.py — Compute Acc_e and Acc_u for artist concept removal using Qwen2.5-VL
========================================================================================
Replicates the GPT-4o-based Acc_e / Acc_u evaluation from eval_artist.py but uses
Qwen2.5-VL (local, no API cost) for Yes/No style identification.

Metrics:
  Acc_e  — fraction of erased-artist-prompted images identified as that artist's style
           (lower = better erasure)
  Acc_u  — fraction of unerased-artist-prompted images correctly identified as that
           artist's style  (higher = style preserved)

Run inside the `pci` conda env (has transformers>=4.51, qwen_vl_utils, torch 2.6):

    python eval_acc_qwen.py \
        --artist "Van Gogh" \
        --tcsa_dir /storage/pk_mridul/TCSA/results/artist_final_vangogh_beta1 \
        --output   /storage/pk_mridul/TCSA/eval_qwen/acc_vangogh.json \
        --device cuda:0

    python eval_acc_qwen.py \
        --artist "Kelly McKernan" \
        --tcsa_dir /storage/pk_mridul/TCSA/results/artist_kelly_final_beta1 \
        --output   /storage/pk_mridul/TCSA/eval_qwen/acc_kelly.json \
        --device cuda:0
"""

import argparse
import json
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── Dataset config ──────────────────────────────────────────────────────────

DATASET_ROOT = "/storage/pk_mridul/TCSA/datasets"

ARTIST_CONFIG = {
    "Van Gogh": {
        "csv": os.path.join(DATASET_ROOT, "big_artist_prompts.csv"),
        "erased": "Van Gogh",
        "all_artists": ["Van Gogh", "Pablo Picasso", "Rembrandt", "Warhol", "Caravaggio"],
    },
    "Kelly McKernan": {
        "csv": os.path.join(DATASET_ROOT, "short_niche_art_prompts.csv"),
        "erased": "Kelly McKernan",
        "all_artists": ["Kelly McKernan", "Kinkade", "Tyler Edlin", "Kilian Eng", "Ajin"],
    },
}

QUESTION_TEMPLATE = (
    "Does this image appear to be in the style of {artist}? "
    "Answer with only 'Yes' or 'No'."
)


def load_model(model_id: str, device: str):
    print(f"Loading Qwen2.5-VL from {model_id} on {device} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


@torch.no_grad()
def classify_image(model, processor, img_path: str, artist: str, device: str) -> bool | None:
    try:
        question = QUESTION_TEMPLATE.format(artist=artist)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text",  "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        output_ids = model.generate(**inputs, max_new_tokens=8)
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"[WARN] {img_path}: {e}")
        return None


def compute_acc(tcsa_dir, artist, model, processor, device, n_runs=3):
    cfg = ARTIST_CONFIG[artist]
    df = pd.read_csv(cfg["csv"])
    erased = cfg["erased"].lower()

    acc_e_runs, acc_u_runs = [], []

    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        yes_e = total_e = 0
        yes_u = total_u = 0
        log = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Qwen2.5-VL run {run+1}"):
            idx = row["case_number"]
            img_path = os.path.join(tcsa_dir, f"{idx}.png")
            if not os.path.exists(img_path):
                continue

            row_artist = str(row.get("artist", ""))
            is_erased_prompt = erased in row_artist.lower()
            style_to_ask = artist if is_erased_prompt else row_artist

            answer = classify_image(model, processor, img_path, style_to_ask, device)
            if answer is None:
                continue

            log.append({"idx": idx, "row_artist": row_artist,
                        "style_asked": style_to_ask, "qwen_yes": answer,
                        "is_erased_prompt": is_erased_prompt})

            if is_erased_prompt:
                total_e += 1
                if answer: yes_e += 1
            else:
                total_u += 1
                if answer: yes_u += 1

        run_e = yes_e / max(total_e, 1)
        run_u = yes_u / max(total_u, 1)
        acc_e_runs.append(run_e)
        acc_u_runs.append(run_u)
        print(f"  Run {run+1}: Acc_e={run_e:.4f}  Acc_u={run_u:.4f}")

    import numpy as np
    acc_e = float(np.mean(acc_e_runs))
    acc_u = float(np.mean(acc_u_runs))

    results = {
        "artist": artist,
        "tcsa_dir": tcsa_dir,
        "Acc_e": round(acc_e, 4),
        "Acc_u": round(acc_u, 4),
        "Acc_e_runs": [round(x, 4) for x in acc_e_runs],
        "Acc_u_runs": [round(x, 4) for x in acc_u_runs],
    }
    print(f"\nAcc_e = {acc_e:.4f}  (lower=better, n_runs={n_runs})")
    print(f"Acc_u = {acc_u:.4f}  (higher=better, n_runs={n_runs})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", required=True, choices=list(ARTIST_CONFIG.keys()))
    ap.add_argument("--tcsa_dir", required=True,
                    help="Dir with {case_number}.png images (TCSA output)")
    ap.add_argument("--output", required=True, help="Path to save JSON results")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_runs", type=int, default=3)
    args = ap.parse_args()

    model, processor = load_model(args.model_id, args.device)
    results = compute_acc(args.tcsa_dir, args.artist, model, processor, args.device, args.n_runs)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
