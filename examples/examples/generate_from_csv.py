"""
Generate comparison images for every prompt in a file.

For each prompt produces three images (same seed):
  col1 — Unsafe      : unfiltered generation
  col2 — SAFREE      : token-subspace projection (prior art)
  col3 — Vocab Ortho : our vocabulary-based orthogonal PCI (Method 6)

and saves them folderwise under --save_dir:

  <save_dir>/
    prompt_000_<slug>/
        col1_unsafe.png
        col2_safree.png
        col3_vocab_ortho.png
        comparison.png          ← 3-column side-by-side grid
    prompt_001_<slug>/
        ...
    summary.png                 ← all prompts × 3 methods in one grid

Supported input formats
-----------------------
  CSV  — must have a text column (default: "target_prompt").
          Pass --col <column_name> to use a different column.
  TXT  — one prompt per line; blank lines and # comments are skipped.

Usage
-----
    python examples/generate_from_csv.py \\
        --prompts  SAFREE/dataset/mma-diffusion-nsfw-adv-prompts.csv \\
        --save_dir results/run_01 \\
        --n        10            \\
        --col      target_prompt \\
        --seed     42            \\
        --device   cuda:3

Optional flags
--------------
    --n          N        Number of prompts to process (default: all)
    --col        NAME     CSV column name for prompts (default: target_prompt)
    --seed       INT      RNG seed (default: 42)
    --device     STR      CUDA device (default: cuda:3)
    --steps      INT      Inference steps (default: 50)
    --guidance   FLOAT    CFG scale (default: 7.5)
    --img_size   INT      Image size in px (default: 512)
    --skip_safree         Skip SAFREE (faster, one model only)
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import sys
import textwrap
from typing import List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFREE_DIR = os.path.join(REPO_ROOT, "SAFREE")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SAFREE_DIR)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("generate_from_csv")

# ── SAFREE nudity concept vocabulary ────────────────────────────────────────
SAFREE_NEG_SPACE = [
    "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
    "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
    "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
    "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification",
]
SAFREE_NEG_PROMPT = ", ".join(SAFREE_NEG_SPACE)

LABEL_H  = 64
FONT_SZ  = 15
PAD      = 6
COL_LABELS = [
    "1. Unsafe (no filter)",
    "2. SAFREE (token subspace)",
    "3. Vocab Ortho PCI (ours)",
]


# ── Grid helpers ─────────────────────────────────────────────────────────────

def _label_img(text: str, w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SZ)
    except Exception:
        font = ImageFont.load_default()
    lines = textwrap.wrap(text, width=max(1, w // (FONT_SZ // 2)))
    y = 5
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font)
        draw.text(((w - (bb[2] - bb[0])) // 2, y), line, font=font,
                  fill=(30, 30, 30))
        y += FONT_SZ + 3
    return img


def make_row(images: List[Image.Image], labels: List[str],
             row_tag: str = "") -> Image.Image:
    """One horizontal row: optional left tag, images + labels below."""
    w, h = images[0].size
    tag_w = 70 if row_tag else 0
    n = len(images)
    total_w = tag_w + n * w + (n - 1) * PAD
    row = Image.new("RGB", (total_w, h + LABEL_H), (220, 220, 220))
    if row_tag:
        row.paste(_label_img(row_tag, tag_w, h + LABEL_H), (0, 0))
    for i, (img, lbl) in enumerate(zip(images, labels)):
        x = tag_w + i * (w + PAD)
        row.paste(img, (x, 0))
        row.paste(_label_img(lbl, w, LABEL_H), (x, h))
    return row


def make_summary(rows: List[Image.Image]) -> Image.Image:
    total_h = sum(r.height for r in rows) + (len(rows) - 1) * PAD
    canvas = Image.new("RGB", (rows[0].width, total_h), (180, 180, 180))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + PAD
    return canvas


def slug(text: str, max_len: int = 30) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len]


# ── Pipeline loaders ─────────────────────────────────────────────────────────

def load_standard(model_id: str, device: str):
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    sched = DPMSolverMultistepScheduler.from_pretrained(model_id,
                                                        subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=sched,
        torch_dtype=torch.float32, safety_checker=None,
    ).to(device)
    pipe.unet.eval()
    return pipe, sched


def load_safree(model_id: str, scheduler, device: str):
    spec = importlib.util.spec_from_file_location(
        "modified_stable_diffusion_pipeline",
        os.path.join(SAFREE_DIR, "models",
                     "modified_stable_diffusion_pipeline.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pipe = mod.ModifiedStableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler,
        torch_dtype=torch.float32, safety_checker=None,
    ).to(device)
    pipe.unet.eval()
    return pipe


# ── SAFREE logger shim ───────────────────────────────────────────────────────

class _SLog:
    def log(self, msg):
        log.debug("[SAFREE] %s", msg)


# ── Prompt loader ─────────────────────────────────────────────────────────────

def load_prompts(path: str, col: str, n: Optional[int]) -> List[str]:
    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available: {df.columns.tolist()}")
        prompts = df[col].dropna().astype(str).tolist()
    else:
        with open(path) as f:
            prompts = [l.strip() for l in f
                       if l.strip() and not l.startswith("#")]
    if n is not None:
        prompts = prompts[:n]
    return prompts


# ── Per-prompt generation ────────────────────────────────────────────────────

def gen_unsafe(pci, prompt: str, seed: int) -> Image.Image:
    return pci.run_full(prompt=prompt, seed=seed)


def gen_safree(pipe, prompt: str, seed: int, device: str,
               num_steps: int, guidance: float, img_size: int) -> Image.Image:
    gen = torch.Generator(device=device).manual_seed(seed)
    imgs = pipe(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        height=img_size, width=img_size,
        negative_prompt=SAFREE_NEG_PROMPT,
        negative_prompt_space=SAFREE_NEG_SPACE,
        generator=gen,
        safree_dict={
            "safree":    True,
            "svf":       False,
            "lra":       False,
            "re_attn_t": [-1, 1001],
            "alpha":     0.01,
            "logger":    _SLog(),
            "up_t":      10,
            "category":  "nudity",
        },
    )
    return imgs[0]


def gen_vocab_ortho(pci, prompt: str, seed: int, schedule,
                    max_proj_ratio: Optional[float] = None,
                    censor_threshold: Optional[float] = None,
                    token_threshold: Optional[float] = 0.1) -> Image.Image:
    return pci.run_orthogonal_vocab(
        prompt=prompt,
        schedule=schedule,
        seed=seed,
        concept="nudity",
        n_components=3,
        renormalize=True,
        max_proj_ratio=max_proj_ratio,
        censor_threshold=censor_threshold,
        token_threshold=token_threshold,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)

    log.info("Prompt file : %s", args.prompts)
    log.info("Save dir    : %s", args.save_dir)
    log.info("Device      : %s", args.device)

    # Load prompts
    prompts = load_prompts(args.prompts, args.col, args.n)
    log.info("Prompts loaded: %d", len(prompts))

    # Load standard pipeline
    log.info("Loading standard pipeline …")
    pipe_std, scheduler = load_standard(args.model_id, args.device)

    # Load SAFREE pipeline
    safree_pipe = None
    if not args.skip_safree:
        try:
            log.info("Loading SAFREE pipeline …")
            safree_pipe = load_safree(args.model_id, scheduler, args.device)
            log.info("SAFREE ready.")
        except Exception as exc:
            log.warning("SAFREE failed to load (%s) — col 2 will be blank.",
                        type(exc).__name__)
            import gc; gc.collect(); torch.cuda.empty_cache()

    # Build PCI adapter + schedule
    from pci_unlearning_v2 import (auto_adapter, PromptConditionedIntervention,
                                   make_uniform_schedule)
    adapter  = auto_adapter(pipe_std, num_inference_steps=args.steps,
                            device=args.device)
    pci      = PromptConditionedIntervention(adapter, guidance_scale=args.guidance,
                                             height=args.img_size,
                                             width=args.img_size)
    schedule = make_uniform_schedule(tau_lo=0.0, tau_hi=1.0)

    # ── Process each prompt ──────────────────────────────────────────────────
    summary_rows: List[Image.Image] = []
    blank = Image.new("RGB", (args.img_size, args.img_size), (200, 200, 200))

    for idx, prompt in enumerate(prompts):
        folder_name = f"prompt_{idx:03d}_{slug(prompt)}"
        folder = os.path.join(args.save_dir, folder_name)
        os.makedirs(folder, exist_ok=True)

        log.info("─" * 60)
        log.info("[%d/%d] %s", idx + 1, len(prompts), prompt[:80])

        images: List[Image.Image] = []

        # Col 1 — Unsafe
        log.info("  col1: unsafe …")
        img = gen_unsafe(pci, prompt, args.seed)
        img.save(os.path.join(folder, "col1_unsafe.png"))
        images.append(img)

        # Col 2 — SAFREE
        if safree_pipe is not None:
            log.info("  col2: SAFREE …")
            img = gen_safree(safree_pipe, prompt, args.seed, args.device,
                             args.steps, args.guidance, args.img_size)
        else:
            img = blank.copy()
        img.save(os.path.join(folder, "col2_safree.png"))
        images.append(img)

        # Col 3 — Vocab Ortho PCI
        log.info("  col3: vocab ortho PCI …")
        img = gen_vocab_ortho(pci, prompt, args.seed, schedule,
                              max_proj_ratio=args.max_proj_ratio,
                              censor_threshold=args.censor_threshold,
                              token_threshold=args.token_threshold)
        img.save(os.path.join(folder, "col3_vocab_ortho.png"))
        images.append(img)

        # Per-prompt comparison grid
        row = make_row(images, COL_LABELS, row_tag="")
        grid_path = os.path.join(folder, "comparison.png")
        row.save(grid_path)
        log.info("  Saved → %s", grid_path)

        summary_rows.append(make_row(images, COL_LABELS,
                                     row_tag=f"{idx}"))

    # ── Summary grid ─────────────────────────────────────────────────────────
    if summary_rows:
        log.info("Building summary grid …")
        summary = make_summary(summary_rows)
        summary_path = os.path.join(args.save_dir, "summary.png")
        summary.save(summary_path)
        log.info("Summary → %s", summary_path)

    print("\n" + "=" * 60)
    print(f"Done.  {len(prompts)} prompts processed.")
    print(f"Results : {args.save_dir}")
    print(f"Summary : {os.path.join(args.save_dir, 'summary.png')}")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate Unsafe / SAFREE / Vocab-Ortho comparisons "
                    "for every prompt in a file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prompts",  required=True,
                   help="Path to CSV or TXT prompt file.")
    p.add_argument("--save_dir", required=True,
                   help="Root output directory. Created if it doesn't exist.")
    p.add_argument("--col",      default="target_prompt",
                   help="CSV column name containing the prompts.")
    p.add_argument("--n",        type=int, default=None,
                   help="Max number of prompts to process (default: all).")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   default="cuda:3")
    p.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--steps",    type=int,   default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--img_size", type=int,   default=512)
    p.add_argument("--skip_safree", action="store_true",
                   help="Skip SAFREE (faster, runs one model only).")
    p.add_argument("--max_proj_ratio", type=float, default=None,
                   help="Max fraction of token norm removed per direction. "
                        "None (default) disables clamping.")
    p.add_argument("--censor_threshold", type=float, default=None,
                   help="Mean per-token cosine to unsafe direction above which "
                        "a placeholder is returned instead of generating. "
                        "None (default) disables blocking.")
    p.add_argument("--token_threshold", type=float, default=0.1,
                   help="Per-token cosine threshold below which a token is skipped "
                        "entirely during orthogonalization. Default 0.1. "
                        "None disables selective masking.")
    main(p.parse_args())
