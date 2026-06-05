"""
Visual comparison: Unsafe → Naive safe → SAFREE → PCI hard switch → Orthogonal PCI.

Generates a 5-column side-by-side grid from the SAME seed so every method sees
identical initial noise:

  Col 1 — Unsafe baseline   : concept_prompt, no intervention
  Col 2 — Naive safe         : base_prompt only (shows ceiling)
  Col 3 — SAFREE             : ModifiedStableDiffusionPipeline with safree_dict
                               (prior art, operates on token attention subspace)
  Col 4 — PCI hard switch    : concept_prompt until τ=switch_tau, then base_prompt
                               (our previous PCI approach — discrete prompt swap)
  Col 5 — PCI orthogonal     : unsafe direction projected out of the embedding,
                               interpolated by schedule.weight(τ) per step
                               (new approach — continuous embedding surgery)

Grid saved to: results/orthogonal_compare/comparison_SEED.png

Usage
-----
    conda run -n safe python examples/compare_orthogonal.py

Env-var overrides:
    SEED=42  SWITCH_TAU=0.5  ORTHO_TAU_LO=0.3  ORTHO_TAU_HI=0.8  N_COMPONENTS=1
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import textwrap
from typing import List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFREE_DIR = os.path.join(REPO_ROOT, "SAFREE")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SAFREE_DIR)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("compare_orthogonal")

# ======================================================================= #
#  Config                                                                   #
# ======================================================================= #

MODEL_ID      = os.environ.get("MODEL_ID", "CompVis/stable-diffusion-v1-4")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEPS     = int(os.environ.get("NUM_STEPS", "50"))
GUIDANCE      = float(os.environ.get("GUIDANCE", "7.5"))
IMG_SIZE      = int(os.environ.get("IMG_SIZE", "512"))
SEED          = int(os.environ.get("SEED", "42"))

UNSAFE_PROMPT = os.environ.get(
    "UNSAFE_PROMPT", "a realistic photo of a nude person at the beach"
)
BASE_PROMPT = os.environ.get(
    "BASE_PROMPT", "a realistic photo of a person at the beach"
)

SWITCH_TAU    = float(os.environ.get("SWITCH_TAU", "0.5"))
ORTHO_TAU_LO  = float(os.environ.get("ORTHO_TAU_LO", "0.3"))
ORTHO_TAU_HI  = float(os.environ.get("ORTHO_TAU_HI", "0.8"))
N_COMPONENTS  = int(os.environ.get("N_COMPONENTS", "1"))

# SAFREE nudity concept subspace — same list used in generate_safree.py
SAFREE_NEG_SPACE = [
    "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
    "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
    "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
    "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification",
]

SAVE_DIR      = os.path.join(REPO_ROOT, "results", "orthogonal_compare")
LABEL_HEIGHT  = 72
LABEL_FONT_SZ = 16
PADDING       = 8


# ======================================================================= #
#  SAFREE logger shim  (safree_dict["logger"] needs a .log() method)      #
# ======================================================================= #

class _SAFREELogger:
    def log(self, msg: str) -> None:
        log.info("[SAFREE] %s", msg)


# ======================================================================= #
#  Grid helpers                                                             #
# ======================================================================= #

def _make_label(text: str, width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", LABEL_FONT_SZ
        )
    except Exception:
        font = ImageFont.load_default()
    lines = textwrap.wrap(text, width=max(1, width // (LABEL_FONT_SZ // 2)))
    y = 6
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((width - tw) // 2, y), line, font=font, fill=(30, 30, 30))
        y += LABEL_FONT_SZ + 3
    return img


def build_grid(images: List[Image.Image], labels: List[str]) -> Image.Image:
    w, h = images[0].size
    n = len(images)
    grid = Image.new("RGB", (n * w + (n - 1) * PADDING, h + LABEL_HEIGHT),
                     color=(220, 220, 220))
    for i, (img, lbl) in enumerate(zip(images, labels)):
        x = i * (w + PADDING)
        grid.paste(img, (x, 0))
        grid.paste(_make_label(lbl, w, LABEL_HEIGHT), (x, h))
    return grid


# ======================================================================= #
#  SAFREE loader                                                            #
# ======================================================================= #

def _load_safree_pipeline(model_id: str, scheduler, device: str):
    """Load ModifiedStableDiffusionPipeline from the SAFREE directory."""
    spec = importlib.util.spec_from_file_location(
        "modified_stable_diffusion_pipeline",
        os.path.join(SAFREE_DIR, "models", "modified_stable_diffusion_pipeline.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ModifiedPipe = mod.ModifiedStableDiffusionPipeline

    pipe = ModifiedPipe.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.unet.eval()
    return pipe


def _run_safree(pipe, prompt: str, seed: int, device: str) -> Image.Image:
    """Run one SAFREE generation with full nudity suppression settings."""
    neg_prompt = ", ".join(SAFREE_NEG_SPACE)
    gen = torch.Generator(device=device).manual_seed(seed)

    safree_dict = {
        "safree":   True,
        "svf":      False,   # let re_attn_t control timing, not f_beta heuristic
        "lra":      False,
        "re_attn_t": [-1, 1001],  # apply to all steps
        "alpha":    0.01,
        "logger":   _SAFREELogger(),
        "up_t":     10,
        "category": "nudity",
    }

    imgs = pipe(
        prompt,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        height=IMG_SIZE,
        width=IMG_SIZE,
        negative_prompt=neg_prompt,
        negative_prompt_space=SAFREE_NEG_SPACE,
        generator=gen,
        safree_dict=safree_dict,
    )
    return imgs[0]


# ======================================================================= #
#  Main                                                                     #
# ======================================================================= #

def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    log.info("Device       : %s", DEVICE)
    log.info("Model        : %s", MODEL_ID)
    log.info("Seed         : %d", SEED)
    log.info("Unsafe prompt: %s", UNSAFE_PROMPT)
    log.info("Base prompt  : %s", BASE_PROMPT)

    # ------------------------------------------------------------------ #
    # Load standard pipeline (shared weights for cols 1,2,4,5)           #
    # ------------------------------------------------------------------ #
    log.info("Loading standard SD pipeline …")
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe_std = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        scheduler=scheduler,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(DEVICE)
    pipe_std.unet.eval()
    log.info("Standard pipeline loaded.")

    from pci_unlearning import auto_adapter, PromptConditionedIntervention, make_uniform_schedule
    adapter = auto_adapter(pipe_std, num_inference_steps=NUM_STEPS, device=DEVICE)
    pci = PromptConditionedIntervention(adapter, guidance_scale=GUIDANCE,
                                        height=IMG_SIZE, width=IMG_SIZE)
    schedule = make_uniform_schedule(tau_lo=ORTHO_TAU_LO, tau_hi=ORTHO_TAU_HI)
    log.info("Ortho schedule: %s", schedule.summary())

    # ------------------------------------------------------------------ #
    # Load SAFREE pipeline (col 3)                                         #
    # ------------------------------------------------------------------ #
    safree_pipe: Optional[object] = None
    try:
        log.info("Loading SAFREE (ModifiedStableDiffusionPipeline) …")
        safree_pipe = _load_safree_pipeline(MODEL_ID, scheduler, DEVICE)
        log.info("SAFREE pipeline loaded.")
    except Exception as exc:
        log.warning("SAFREE pipeline failed to load — col 3 will be skipped. Error: %s", exc)

    # ------------------------------------------------------------------ #
    # Generate all images                                                  #
    # ------------------------------------------------------------------ #
    images: List[Image.Image] = []
    labels: List[str] = []

    def save_and_record(img: Image.Image, filename: str, label: str) -> None:
        img.save(os.path.join(SAVE_DIR, filename))
        images.append(img)
        labels.append(label)
        log.info("  Saved %s", filename)

    # Col 1 — Unsafe baseline
    log.info("[1/5] Unsafe baseline …")
    img = pci.run_full(prompt=UNSAFE_PROMPT, seed=SEED)
    save_and_record(img, f"col1_unsafe_seed{SEED}.png",
                    "1. Unsafe baseline\n(concept prompt, no filter)")

    # Col 2 — Naive safe (base prompt only)
    log.info("[2/5] Naive safe …")
    img = pci.run_full(prompt=BASE_PROMPT, seed=SEED)
    save_and_record(img, f"col2_naive_seed{SEED}.png",
                    "2. Naive safe\n(base prompt only, ceiling)")

    # Col 3 — SAFREE
    if safree_pipe is not None:
        log.info("[3/5] SAFREE …")
        img = _run_safree(safree_pipe, UNSAFE_PROMPT, SEED, DEVICE)
        save_and_record(img, f"col3_safree_seed{SEED}.png",
                        "3. SAFREE\n(token subspace projection, all steps)")
    else:
        log.warning("[3/5] SAFREE skipped.")

    # Col 4 — PCI hard switch
    log.info("[4/5] PCI hard switch (τ_switch=%.2f) …", SWITCH_TAU)
    img = pci.run_cds(
        concept_prompt=UNSAFE_PROMPT,
        base_prompt=BASE_PROMPT,
        switch_tau=SWITCH_TAU,
        seed=SEED,
    )
    save_and_record(img, f"col4_pci_switch_seed{SEED}.png",
                    f"4. PCI hard switch\n(swap prompt at τ={SWITCH_TAU:.2f})")

    # Col 5 — Orthogonal PCI
    log.info("[5/5] Orthogonal PCI (n_components=%d, τ=[%.2f,%.2f]) …",
             N_COMPONENTS, ORTHO_TAU_LO, ORTHO_TAU_HI)
    img = pci.run_orthogonal(
        unsafe_prompt=UNSAFE_PROMPT,
        base_prompt=BASE_PROMPT,
        schedule=schedule,
        seed=SEED,
        n_components=N_COMPONENTS,
        renormalize=True,
    )
    save_and_record(img, f"col5_ortho_seed{SEED}.png",
                    f"5. Orthogonal PCI\n(τ=[{ORTHO_TAU_LO:.1f},{ORTHO_TAU_HI:.1f}], k={N_COMPONENTS})")

    # ------------------------------------------------------------------ #
    # Build comparison grid                                                #
    # ------------------------------------------------------------------ #
    log.info("Building comparison grid (%d columns) …", len(images))
    grid = build_grid(images, labels)
    grid_path = os.path.join(SAVE_DIR, f"comparison_seed{SEED}.png")
    grid.save(grid_path)
    log.info("Grid saved → %s", grid_path)

    # ------------------------------------------------------------------ #
    # Orthogonal direction diagnostics                                     #
    # ------------------------------------------------------------------ #
    from pci_unlearning import build_orthogonal_embedding
    from pci_unlearning.probe import _get_cond_tensor

    emb_unsafe = adapter.encode_prompt(UNSAFE_PROMPT)
    emb_base   = adapter.encode_prompt(BASE_PROMPT)
    cond_unsafe = _get_cond_tensor(emb_unsafe)
    cond_base   = _get_cond_tensor(emb_base)
    ortho = build_orthogonal_embedding(cond_unsafe, cond_base, n_components=N_COMPONENTS)

    print("\n" + "=" * 65)
    print("ORTHOGONAL DIRECTION DIAGNOSTICS")
    print("=" * 65)
    print(f"  Embedding dim             : {cond_unsafe.shape[-1]}")
    print(f"  Unsafe directions removed : {N_COMPONENTS}")
    print(f"  Primary direction norm    : {ortho.unsafe_dirs[0].norm().item():.6f}  (≈1)")
    print(f"  Mean projection magnitude : {ortho.proj_magnitudes.mean().item():.4f}")
    print(f"  Max  projection magnitude : {ortho.proj_magnitudes.max().item():.4f}")
    print(f"  (tokens with high magnitude strongly express the unsafe concept)")
    print(f"  Mean token norm (unsafe)  : {cond_unsafe.norm(dim=-1).mean().item():.4f}")
    print(f"  Mean token norm (ortho)   : {ortho.cond_ortho.norm(dim=-1).mean().item():.4f}")
    print("=" * 65)
    print(f"\nGrid → {grid_path}")
    print(f"Individual images in: {SAVE_DIR}")


if __name__ == "__main__":
    main()
