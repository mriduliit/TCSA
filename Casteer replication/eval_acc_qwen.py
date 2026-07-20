"""
Compute Acc_e and Acc_u for CASteer artist-erased images using Qwen2.5-VL.

Acc_e — fraction of erased-artist-prompted images identified as that artist's
        style (lower = better erasure)
Acc_u — fraction of unerased-artist-prompted images correctly identified as
        their (other) artist's style (higher = style preserved)

Run inside the `pci` conda env:

    python eval_acc_qwen.py \
        --artist "Van Gogh" \
        --erased_dir "/storage/pk_mridul/TCSA/Casteer replication/results/vangogh" \
        --csv /storage/pk_mridul/TCSA/datasets/big_artist_prompts.csv \
        --output "/storage/pk_mridul/TCSA/Casteer replication/results/vangogh/acc_scores.json" \
        --device cuda:0
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

QUESTION_TEMPLATE = (
    "Does this image appear to be in the style of {artist}? "
    "Answer with only 'Yes' or 'No'."
)


def load_model(model_id, device):
    print(f"Loading {model_id} on {device} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


@torch.no_grad()
def classify_image(model, processor, img_path, artist, device):
    try:
        question = QUESTION_TEMPLATE.format(artist=artist)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": question},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)
        output_ids = model.generate(**inputs, max_new_tokens=8)
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"[WARN] {img_path}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", required=True)
    ap.add_argument("--erased_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_runs", type=int, default=3)
    args = ap.parse_args()

    model, processor = load_model(args.model_id, args.device)
    df = pd.read_csv(args.csv)
    erased = args.artist.lower()

    acc_e_runs, acc_u_runs = [], []
    for run in range(args.n_runs):
        yes_e = total_e = 0
        yes_u = total_u = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"run {run+1}/{args.n_runs}"):
            idx = row["case_number"]
            img_path = os.path.join(args.erased_dir, f"{idx}.png")
            if not os.path.exists(img_path):
                continue

            row_artist = str(row.get("artist", ""))
            is_erased_prompt = erased in row_artist.lower()
            style_to_ask = args.artist if is_erased_prompt else row_artist

            answer = classify_image(model, processor, img_path, style_to_ask, args.device)
            if answer is None:
                continue

            if is_erased_prompt:
                total_e += 1
                if answer:
                    yes_e += 1
            else:
                total_u += 1
                if answer:
                    yes_u += 1

        acc_e_runs.append(yes_e / max(total_e, 1))
        acc_u_runs.append(yes_u / max(total_u, 1))
        print(f"  Run {run+1}: Acc_e={acc_e_runs[-1]:.4f}  Acc_u={acc_u_runs[-1]:.4f}")

    results = {
        "artist": args.artist,
        "erased_dir": args.erased_dir,
        "Acc_e": round(float(np.mean(acc_e_runs)), 4),
        "Acc_u": round(float(np.mean(acc_u_runs)), 4),
        "Acc_e_runs": [round(x, 4) for x in acc_e_runs],
        "Acc_u_runs": [round(x, 4) for x in acc_u_runs],
    }
    print(f"\nAcc_e = {results['Acc_e']}  Acc_u = {results['Acc_u']}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
