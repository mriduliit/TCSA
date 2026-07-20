#!/usr/bin/env python3
"""
eval_artist.py  —  Artist Concept Removal Evaluation for TCSA
==============================================================
Replicates Table 3 from the paper (Van Gogh / Kelly McKernan removal).

Metrics:
  LPIPS_e  — perceptual distance on ERASED artist prompts (higher = more erased)
  LPIPS_u  — perceptual distance on UNERASED artist prompts (higher = style preserved)
  Acc_e    — GPT-4o accuracy identifying erased artist (lower = better erasure)
  Acc_u    — GPT-4o accuracy identifying unerased artists (higher = style preserved)

Pipeline:
  1. Generate images with SD-v1.4 baseline (reference for LPIPS)
  2. Generate images with TCSA (erased artist)
  3. Compute LPIPS between baseline and TCSA images
  4. Use GPT-4o to classify which artist style is in each image

Usage
-----
    # Step 1: Generate baseline SD-v1.4 images
    python eval_artist.py generate-baseline \
        --artist    "Van Gogh" \
        --save_dir  results/artist_baseline_vangogh

    # Step 2: Generate TCSA images (with artist erased)
    python eval_artist.py generate \
        --artist    "Van Gogh" \
        --sv_path   steering_vectors/sv_vangogh_beta_1.pt \
        --save_dir  results/artist_tcsa_vangogh

    # Step 3: Compute LPIPS
    python eval_artist.py lpips \
        --baseline_dir  results/artist_baseline_vangogh \
        --tcsa_dir      results/artist_tcsa_vangogh \
        --artist        "Van Gogh"

    # Step 4: Compute Acc with GPT-4o
    python eval_artist.py acc \
        --tcsa_dir      results/artist_tcsa_vangogh \
        --artist        "Van Gogh" \
        --openai_key    sk-...
"""

import argparse
import base64
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# ── Dataset paths ─────────────────────────────────────────────────────────────
ARTIST_DATASETS = {
    "Van Gogh":       "datasets/big_artist_prompts.csv",
    "Kelly McKernan": "datasets/short_niche_art_prompts.csv",
}

# All artists in each dataset (for multi-choice QA)
FAMOUS_ARTISTS = [
    "Pablo Picasso", "Vincent Van Gogh", "Salvador Dali",
    "Claude Monet", "Frida Kahlo", "Leonardo da Vinci",
]
NICHE_ARTISTS = [
    "Tyler Edlin", "Kelly McKernan", "Kilian Eng",
    "Ajin", "Thomas Kinkade",
]


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_artist_dataset(artist: str) -> pd.DataFrame:
    path = ARTIST_DATASETS.get(artist)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset for {artist} not found at {path}.\n"
            f"Copy from SAFREE: datasets/big_artist_prompts.csv or "
            f"datasets/short_niche_art_prompts.csv")
    return pd.read_csv(path)


# ── Image generation ──────────────────────────────────────────────────────────
def _worker(rank: int, indices: list, prompts: list, seeds: list,
            save_dir: str, sv_path: str, concept: str, model_id: str,
            model_arch: str, guidance_scale: float, num_steps: int,
            height: int, width: int, beta: float, n_components: int,
            max_proj_ratio: float, baseline: bool,
            mode: str = "both") -> None:
    """Single-GPU worker for both baseline and TCSA generation."""
    from tqdm import tqdm
    device = f"cuda:{rank}"
    os.makedirs(save_dir, exist_ok=True)

    if baseline:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        sched = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=sched, safety_checker=None,
            torch_dtype=torch.float16).to(device)
        pipe.set_progress_bar_config(disable=True)

        for idx, prompt, seed in tqdm(zip(indices, prompts, seeds),
                                      total=len(indices),
                                      desc=f"Baseline GPU{rank}",
                                      position=rank, miniters=5,
                                      mininterval=5.0):
            out = os.path.join(save_dir, f"{idx}.png")
            if os.path.exists(out):
                continue
            if not isinstance(prompt, str):
                continue
            with torch.no_grad():
                img = pipe(
                    prompt=prompt, num_inference_steps=num_steps,
                    guidance_scale=guidance_scale, height=height, width=width,
                    generator=torch.Generator(device).manual_seed(int(seed))
                ).images[0]
            img.save(out)
        return

    # TCSA generation — use style_erasure_TECA_allstyle (stable flat beta)
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    from style_erasure_TECA_final import (
        MergedUnlearningPipeline as StylePipeline,
        build_steering_vectors, Schedule)

    result = build_steering_vectors(
        concept=concept, model_id=model_id,
        save_path=sv_path, device=device)
    sv, layer_align = result if isinstance(result, tuple) else (result, None)
    wrapper = StylePipeline.from_pretrained(
        model_id=model_id, sv=sv, concept=concept,
        device=device, beta=beta,
        n_components=n_components,
        max_proj_ratio=max_proj_ratio)
    schedule = Schedule()

    for idx, prompt, seed in tqdm(zip(indices, prompts, seeds),
                                  total=len(indices),
                                  desc=f"TCSA GPU{rank}",
                                  position=rank, miniters=5,
                                  mininterval=5.0):
        out = os.path.join(save_dir, f"{idx}.png")
        if os.path.exists(out):
            continue
        if not isinstance(prompt, str):
            continue
        img = wrapper.generate(
            prompt=prompt, mode=mode, schedule=schedule,
            seed=int(seed), guidance_scale=guidance_scale,
            num_steps=num_steps, height=height, width=width)
        img.save(out)

    wrapper.remove_hooks()
    print(f"[GPU {rank}] done.")


def _generate_images(indices, prompts, seeds, save_dir,
                     sv_path, concept, model_id, model_arch,
                     guidance_scale, num_steps, height, width,
                     beta, n_components, max_proj_ratio,
                     baseline=False, device="cuda", num_gpus=1,
                     mode="both"):
    """Dispatch generation across num_gpus GPUs."""
    import torch.multiprocessing as mp
    os.makedirs(save_dir, exist_ok=True)

    if num_gpus == 1:
        _worker(rank=0, indices=indices, prompts=prompts, seeds=seeds,
                save_dir=save_dir, sv_path=sv_path, concept=concept,
                model_id=model_id, model_arch=model_arch,
                guidance_scale=guidance_scale, num_steps=num_steps,
                height=height, width=width, beta=beta,
                n_components=n_components, max_proj_ratio=max_proj_ratio,
                baseline=baseline, mode=mode)
    else:
        chunks_idx = [indices[i::num_gpus] for i in range(num_gpus)]
        chunks_pmt = [prompts[i::num_gpus]  for i in range(num_gpus)]
        chunks_sed = [seeds[i::num_gpus]    for i in range(num_gpus)]

        for rank, ci in enumerate(chunks_idx):
            print(f"  GPU {rank} -> {len(ci)} prompts")

        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(target=_worker, kwargs=dict(
                rank=rank, indices=chunks_idx[rank],
                prompts=chunks_pmt[rank], seeds=chunks_sed[rank],
                save_dir=save_dir, sv_path=sv_path, concept=concept,
                model_id=model_id, model_arch=model_arch,
                guidance_scale=guidance_scale, num_steps=num_steps,
                height=height, width=width, beta=beta,
                n_components=n_components, max_proj_ratio=max_proj_ratio,
                baseline=baseline, mode=mode))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        failed = [p.exitcode for p in processes if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"{len(failed)} workers failed: {failed}")


# ── LPIPS computation ─────────────────────────────────────────────────────────
def compute_lpips(baseline_dir: str, tcsa_dir: str,
                  artist: str, device: str = "cuda") -> dict:
    """Compute LPIPS_e (erased artist) and LPIPS_u (other artists)."""
    try:
        import lpips
    except ImportError:
        print("Install lpips: pip install lpips")
        return {}

    loss_fn = lpips.LPIPS(net="alex").to(device)
    df = load_artist_dataset(artist)

    scores_e, scores_u = [], []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="LPIPS"):
        idx = row["case_number"]
        base_path = os.path.join(baseline_dir, f"{idx}.png")
        tcsa_path = os.path.join(tcsa_dir, f"{idx}.png")

        if not os.path.exists(base_path) or not os.path.exists(tcsa_path):
            continue

        def _load(p):
            img = Image.open(p).convert("RGB").resize((512, 512))
            arr = np.array(img).astype(np.float32) / 127.5 - 1.0
            return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            score = loss_fn(_load(base_path), _load(tcsa_path)).item()

        # erased artist prompts vs other artists
        row_artist = str(row.get("artist", ""))
        if artist.lower() in row_artist.lower():
            scores_e.append(score)
        else:
            scores_u.append(score)

    results = {
        "LPIPS_e": round(float(np.mean(scores_e)), 4) if scores_e else 0.0,
        "LPIPS_u": round(float(np.mean(scores_u)), 4) if scores_u else 0.0,
        "n_erased": len(scores_e),
        "n_unerased": len(scores_u),
    }
    print(f"\nLPIPS_e: {results['LPIPS_e']}  ({results['n_erased']} images)")
    print(f"LPIPS_u: {results['LPIPS_u']}  ({results['n_unerased']} images)")
    return results


# ── GPT-4o Acc computation ────────────────────────────────────────────────────
def _encode_image(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _gpt4o_classify(client, img_path: str, style: str) -> bool:
    """Ask GPT-4o if image is in the given artist style. Returns True if Yes."""
    question = f"Is this picture in {style} style? Just tell me Yes or No."
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url",
                     "image_url": {
                         "url": f"data:image/png;base64,{_encode_image(img_path)}",
                         "detail": "low"
                     }},
                ],
            }],
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
    except Exception as e:
        print(f"[WARN] GPT-4o error for {img_path}: {e}")
        return None


def compute_acc_llava(tcsa_dir: str, artist: str, device: str = "cuda") -> dict:
    """Compute Acc_e and Acc_u using LLaVA (free, no API key needed).

    Same methodology as GPT-4o version:
    "Is this picture in {style} style? Just tell me Yes or No."
    Averaged over 3 runs.
    """
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    import torch
    from PIL import Image

    print("\nLoading LLaVA model (free alternative to GPT-4o)...")
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    df = load_artist_dataset(artist)

    if artist == "Van Gogh":
        all_artists = ["Van Gogh", "Picasso", "Rembrandt", "Warhol", "Caravaggio"]
    else:
        all_artists = ["Kelly McKernan", "Kinkade", "Tyler Edlin", "Kilian Eng", "Ajin"]

    n_runs = 3
    acc_e_runs, acc_u_runs = [], []

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        yes_e, total_e = 0, 0
        yes_u, total_u = 0, 0

        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"LLaVA Run {run+1}"):
            idx = row["case_number"]
            img_path = os.path.join(tcsa_dir, f"{idx}.png")
            if not os.path.exists(img_path):
                continue

            row_artist = str(row.get("artist", ""))
            is_erased = artist.lower() in row_artist.lower()
            style = artist if is_erased else row_artist

            # Build prompt
            question = f"Is this picture in {style} style? Just tell me Yes or No."
            prompt = f"[INST] <image>\n{question} [/INST]"

            img = Image.open(img_path).convert("RGB")
            inputs = processor(prompt, img, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=5)
            answer = processor.decode(output[0], skip_special_tokens=True).lower()
            is_yes = "yes" in answer.split("[/inst]")[-1]

            if is_erased:
                total_e += 1
                if is_yes:
                    yes_e += 1
            else:
                total_u += 1
                if is_yes:
                    yes_u += 1

        acc_e_runs.append(yes_e / max(total_e, 1))
        acc_u_runs.append(yes_u / max(total_u, 1))

    acc_e = float(np.mean(acc_e_runs))
    acc_u = float(np.mean(acc_u_runs))

    results = {
        "Acc_e": round(acc_e, 4),
        "Acc_u": round(acc_u, 4),
        "model": "LLaVA-1.6-Mistral-7B",
        "Acc_e_runs": [round(x, 4) for x in acc_e_runs],
        "Acc_u_runs": [round(x, 4) for x in acc_u_runs],
    }
    print(f"\nAcc_e: {results['Acc_e']}  (lower is better)")
    print(f"Acc_u: {results['Acc_u']}  (higher is better)")
    return results, []


def compute_acc(tcsa_dir: str, artist: str, openai_key: str,
                device: str = "cuda") -> dict:
    """Compute Acc_e and Acc_u using GPT-4o Yes/No style identification.

    As per paper appendix:
    - Acc_e: ask GPT-4o if erased artist style present in erased-artist prompted images
             → lower is better (style successfully removed)
    - Acc_u: ask GPT-4o if each other artist style present in their prompted images
             → higher is better (other styles preserved)
    Averaged over 3 runs as per paper.
    """
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)

    df = load_artist_dataset(artist)

    # Artists in the dataset
    if artist == "Van Gogh":
        all_artists = ["Van Gogh", "Picasso", "Rembrandt", "Warhol", "Caravaggio"]
    else:  # Kelly McKernan
        all_artists = ["Kelly McKernan", "Kinkade", "Tyler Edlin", "Kilian Eng", "Ajin"]

    n_runs = 3
    acc_e_runs, acc_u_runs = [], []

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")
        yes_e, total_e = 0, 0
        yes_u, total_u = 0, 0
        results_log = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"GPT-4o Run {run+1}"):
            idx = row["case_number"]
            img_path = os.path.join(tcsa_dir, f"{idx}.png")
            if not os.path.exists(img_path):
                continue

            row_artist = str(row.get("artist", ""))
            is_erased_prompt = artist.lower() in row_artist.lower()

            # Ask about the artist in the prompt
            style = artist if is_erased_prompt else row_artist
            answer = _gpt4o_classify(client, img_path, style)
            if answer is None:
                continue

            results_log.append({
                "idx": idx, "row_artist": row_artist,
                "style_asked": style, "gpt_yes": answer,
                "is_erased_prompt": is_erased_prompt,
            })

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

    acc_e = float(np.mean(acc_e_runs))
    acc_u = float(np.mean(acc_u_runs))

    results = {
        "Acc_e": round(acc_e, 4),   # lower is better
        "Acc_u": round(acc_u, 4),   # higher is better
        "Acc_e_runs": [round(x, 4) for x in acc_e_runs],
        "Acc_u_runs": [round(x, 4) for x in acc_u_runs],
    }
    print(f"\nAcc_e: {results['Acc_e']}  (avg over {n_runs} runs) — lower is better")
    print(f"Acc_u: {results['Acc_u']}  (avg over {n_runs} runs) — higher is better")
    return results, results_log


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Artist concept removal evaluation (LPIPS + GPT-4o Acc)")
    sub = parser.add_subparsers(dest="cmd")

    def _add_gen_common(p):
        p.add_argument("--artist",   required=True,
                       choices=["Van Gogh", "Kelly McKernan"])
        p.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
        p.add_argument("--num_gpus", type=int, default=1)
        p.add_argument("--device",   default="cuda")
        p.add_argument("--guidance_scale", type=float, default=7.5)
        p.add_argument("--num_steps",      type=int,   default=50)
        p.add_argument("--img_size",       type=int,   default=512)

    # generate-baseline
    gb = sub.add_parser("generate-baseline",
                        help="Generate SD-v1.4 baseline images")
    gb.add_argument("--save_dir", required=True)
    _add_gen_common(gb)

    # generate (TCSA)
    gen = sub.add_parser("generate", help="Generate TCSA images")
    gen.add_argument("--save_dir",        required=True)
    gen.add_argument("--sv_path",         required=True)
    gen.add_argument("--concept",         default=None,
                     help="Concept to erase (default: artist name)")
    gen.add_argument("--model_arch",      default="sd14",
                     choices=["sd14", "sdxl"])
    gen.add_argument("--beta",            type=float, default=2.0)
    gen.add_argument("--n_components",    type=int,   default=1)
    gen.add_argument("--max_proj_ratio",  type=float, default=0.5)
    gen.add_argument("--mode",            default="both",
                     choices=["none", "text", "ca", "both"],
                     help="Erasure mode: both=TECA, ca=CA-only, text=text-only, none=baseline")
    _add_gen_common(gen)

    # lpips
    lp = sub.add_parser("lpips", help="Compute LPIPS_e and LPIPS_u")
    lp.add_argument("--baseline_dir", required=True)
    lp.add_argument("--tcsa_dir",     required=True)
    lp.add_argument("--artist",       required=True,
                    choices=["Van Gogh", "Kelly McKernan"])
    lp.add_argument("--device",       default="cuda")
    lp.add_argument("--output",       default=None)

    # acc
    ac = sub.add_parser("acc", help="Compute Acc_e and Acc_u using GPT-4o")
    ac.add_argument("--tcsa_dir",   required=True)
    ac.add_argument("--artist",     required=True,
                    choices=["Van Gogh", "Kelly McKernan"])
    ac.add_argument("--openai_key", required=True,
                    help="OpenAI API key")
    ac.add_argument("--device",     default="cuda")
    ac.add_argument("--output",     default=None)

    al = sub.add_parser("acc-llava",
                        help="Compute Acc_e and Acc_u using LLaVA (no API key needed)")
    al.add_argument("--tcsa_dir", required=True)
    al.add_argument("--artist",   required=True,
                    choices=["Van Gogh", "Kelly McKernan"])
    al.add_argument("--device",   default="cuda")
    al.add_argument("--output",   default=None)

    args = parser.parse_args()

    if args.cmd == "generate-baseline":
        df = load_artist_dataset(args.artist)
        _generate_images(
            indices=list(df["case_number"]),
            prompts=list(df["prompt"]),
            seeds=list(df["evaluation_seed"]),
            save_dir=args.save_dir,
            sv_path="", concept="", model_id=args.model_id,
            model_arch="sd14", guidance_scale=args.guidance_scale,
            num_steps=args.num_steps, height=args.img_size, width=args.img_size,
            beta=2.0, n_components=1, max_proj_ratio=0.5,
            baseline=True, device=args.device, num_gpus=args.num_gpus)

    elif args.cmd == "generate":
        df = load_artist_dataset(args.artist)
        concept = args.concept or args.artist
        _generate_images(
            indices=list(df["case_number"]),
            prompts=list(df["prompt"]),
            seeds=list(df["evaluation_seed"]),
            save_dir=args.save_dir,
            sv_path=args.sv_path, concept=concept,
            model_id=args.model_id, model_arch=args.model_arch,
            guidance_scale=args.guidance_scale, num_steps=args.num_steps,
            height=args.img_size, width=args.img_size,
            beta=args.beta, n_components=args.n_components,
            max_proj_ratio=args.max_proj_ratio,
            baseline=False, device=args.device, num_gpus=args.num_gpus,
            mode=args.mode)

    elif args.cmd == "lpips":
        results = compute_lpips(
            args.baseline_dir, args.tcsa_dir, args.artist, args.device)
        out = args.output or os.path.join(args.tcsa_dir, "lpips_scores.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved → {out}")

    elif args.cmd == "acc":
        results, log = compute_acc(
            args.tcsa_dir, args.artist, args.openai_key, args.device)
        out = args.output or os.path.join(args.tcsa_dir, "acc_scores.json")
        log_out = out.replace(".json", "_log.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=4)
        with open(log_out, "w") as f:
            json.dump(log, f, indent=4)
        print(f"\nSaved → {out}")

    elif args.cmd == "acc-llava":
        results, _ = compute_acc_llava(
            args.tcsa_dir, args.artist, args.device)
        out = args.output or os.path.join(args.tcsa_dir, "acc_llava_scores.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved → {out}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
