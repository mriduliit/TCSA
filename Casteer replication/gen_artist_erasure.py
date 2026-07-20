"""
Generate artist-erased images using CASteer cross-attention output steering
on SD-v1.4, for a CSV of (case_number, prompt, evaluation_seed, artist) rows.

Run inside the `casteer` conda env, e.g.:

    python gen_artist_erasure.py \
        --csv /storage/pk_mridul/TCSA/datasets/big_artist_prompts.csv \
        --sv_path "/storage/pk_mridul/TCSA/Casteer replication/steering_vectors/Van Gogh.pt" \
        --save_dir "/storage/pk_mridul/TCSA/Casteer replication/results/vangogh" \
        --strength 2.0 --intermediate_clipping
"""
import argparse
import os
import sys

sys.path.insert(0, "/storage/pk_mridul/CASteer")

import pandas as pd
import torch
from tqdm import tqdm

from core.controller import CrossAttentionOutputSteering
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from core.pickle import unpickle
from core.utils import init_pipeline_for_image_model, run_image_model, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--sv_path", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--strength", type=float, default=2.0)
    ap.add_argument("--intermediate_clipping", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    os.chdir("/storage/pk_mridul/CASteer")

    pipeline = init_pipeline_for_image_model("sd14")
    pipeline.set_progress_bar_config(disable=True)
    device = get_device()

    source_concept = unpickle(args.sv_path)
    vector_control = CrossAttentionOutputSteering(
        target_concepts=[None],
        source_concepts=[source_concept],
        strength=args.strength,
        device=device,
        intermediate_clipping=args.intermediate_clipping,
        use_first_diffusion_step=False,
    )

    model_component = getattr(pipeline, "transformer", None) or pipeline.unet
    diffusion_register_vector_controls_with_hooks(
        model_component, vector_control, model_type=DiffusionModelType.from_model("sd14"))

    for _, row in tqdm(df.iterrows(), total=len(df)):
        idx = row["case_number"]
        out_path = os.path.join(args.save_dir, f"{idx}.png")
        if os.path.exists(out_path):
            continue
        prompt = row["prompt"]
        seed = int(row["evaluation_seed"])
        images = run_image_model(model_type="sd14", pipe=pipeline, prompt=prompt, seed=seed, device=device)
        vector_control.reset()
        images[0].save(out_path)


if __name__ == "__main__":
    main()
