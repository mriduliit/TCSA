"""
style_erasure_TECA.py  —  Two-Space Training-Free Style Erasure
================================================================

Same TECA approach as nudity removal but for artistic styles.

The idea is identical — dual-space intervention:

  SPACE 1 — TEXT EMBEDDING
      Build a style direction d from the style name and related vocab.
      At each denoising step: project out d from the prompt embedding.
      Removes the style from the instruction before the UNet sees it.

  SPACE 2 — CROSS-ATTENTION ACTIVATIONS
      Build per-layer / per-step steering vectors from paired generations:
          positive: "{ImageNet class}, {style} style"
          negative: "{ImageNet class}"
      At inference: subtract the style direction from every CA patch.
      Catches whatever style the UNet rebuilds from its learned priors.

TCL (Text Concept Load) gates Space 2 by how strongly the prompt
carries the style in text space — same cross-space connection as nudity.

Supported styles (out of the box):
    van_gogh, picasso, monet, anime, ukiyo_e, watercolor,
    oil_painting, sketch, impressionism, cubism

You can add any style by adding entries to STYLE_VOCAB and STYLE_SAFE_VOCAB.

Usage
-----
    # build steering vectors for Van Gogh style (one time)
    python style_erasure_TECA.py build-vectors \\
        --style     van_gogh \\
        --save_path steering_vectors/style_van_gogh.pt \\
        --device    cuda

    # single prompt comparison (4 columns)
    python style_erasure_TECA.py single \\
        --prompt    "a cat in the style of Van Gogh" \\
        --style     van_gogh \\
        --sv_path   steering_vectors/style_van_gogh.pt \\
        --save_path results/style_test.png \\
        --device    cuda

    # batch on a CSV
    python style_erasure_TECA.py batch \\
        --prompts   dataset/style_prompts.csv \\
        --col       prompt \\
        --style     van_gogh \\
        --sv_path   steering_vectors/style_van_gogh.pt \\
        --save_dir  results/style_van_gogh \\
        --n         20 \\
        --device    cuda

    # erase anime style with SDXL
    python style_erasure_TECA.py build-vectors \\
        --style      anime \\
        --model_id   stabilityai/stable-diffusion-xl-base-1.0 \\
        --model_arch sdxl \\
        --save_path  steering_vectors/style_anime_sdxl.pt \\
        --device     cuda

    python style_erasure_TECA.py batch \\
        --prompts    dataset/style_prompts.csv \\
        --col        prompt \\
        --style      anime \\
        --model_id   stabilityai/stable-diffusion-xl-base-1.0 \\
        --model_arch sdxl \\
        --sv_path    steering_vectors/style_anime_sdxl.pt \\
        --save_dir   results/style_anime_sdxl \\
        --img_size   1024 \\
        --device     cuda

Output (4-column grid per prompt):
    col1  Unsafe                  (original style intact)
    col2  Text-ortho only         (style removed from text embedding)
    col3  CA-steer only           (style removed from activations)
    col4  TECA merged             (both spaces — the full method)

Requirements:  diffusers transformers torch Pillow pandas tqdm
Self-contained — does NOT import pci_unlearning or merged_unlearning_TECA.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import logging
import math
import os
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("style_erasure_teca")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SteeringVectors = Dict[int, Dict[str, List[torch.Tensor]]]

# =========================================================================
#  Style definitions
# =========================================================================

# Text-space vocab — defines the direction d in CLIP embedding space.
# These words anchor the style direction in text space.
STYLE_VOCAB: Dict[str, List[str]] = {
    "van_gogh": [
        "Van Gogh", "Van Gogh style", "post-impressionist", "swirling brushstrokes",
        "thick impasto", "Starry Night", "Sunflowers painting", "expressionist brushwork",
        "vivid colors Van Gogh", "painterly swirls",
    ],
    "picasso": [
        "Picasso", "Picasso style", "cubist", "cubism", "Picasso painting",
        "fragmented faces", "abstract cubist", "geometric abstraction",
        "multiple perspectives painting", "analytic cubism",
    ],
    "monet": [
        "Monet", "Claude Monet", "impressionist", "Monet style", "water lilies painting",
        "soft brushstrokes", "Monet garden", "impressionism", "hazy light painting",
        "plein air impressionist",
    ],
    "anime": [
        "anime", "anime style", "manga", "Japanese animation", "anime art",
        "anime illustration", "cel shading", "anime character", "kawaii anime",
        "anime aesthetic",
    ],
    "ukiyo_e": [
        "ukiyo-e", "ukiyo-e style", "Japanese woodblock", "woodblock print",
        "Hokusai", "Great Wave", "edo period art", "Japanese print art",
        "flat woodblock style", "ukiyo art",
    ],
    "watercolor": [
        "watercolor", "watercolor painting", "watercolor art", "watercolor style",
        "soft watercolor", "watercolor wash", "transparent watercolor",
        "loose watercolor", "watercolor illustration", "aquarelle",
    ],
    "oil_painting": [
        "oil painting", "oil on canvas", "classical oil painting", "old master painting",
        "oil paint texture", "fine art oil", "portrait oil painting",
        "renaissance oil painting", "baroque oil painting", "oil medium painting",
    ],
    "sketch": [
        "pencil sketch", "charcoal drawing", "sketch art", "hand-drawn sketch",
        "graphite drawing", "pencil drawing style", "line art sketch",
        "rough sketch", "artist sketch", "gestural drawing",
    ],
    "impressionism": [
        "impressionism", "impressionist painting", "impressionist style",
        "loose brushstrokes", "plein air painting", "light and color impressionist",
        "impressionist art movement", "Renoir style", "Sisley style",
        "post-impressionist",
    ],
    "cubism": [
        "cubism", "cubist art", "cubist painting", "geometric cubism",
        "analytical cubism", "synthetic cubism", "fractured perspective",
        "cubist abstraction", "multiple viewpoints painting", "Braque style",
    ],
}

# Safe vocab — opposite of the style, used for contrastive direction.
STYLE_SAFE_VOCAB: Dict[str, List[str]] = {
    "van_gogh": [
        "photorealistic", "photograph", "realistic photo", "neutral style",
        "plain image", "no artistic style", "documentary photo",
    ],
    "picasso": [
        "photorealistic", "realistic portrait", "photograph", "natural perspective",
        "realistic painting", "no abstraction", "anatomically correct",
    ],
    "monet": [
        "sharp photograph", "crisp photo", "photorealistic", "high definition photo",
        "clear image", "no painterly effect", "digital photo",
    ],
    "anime": [
        "photorealistic", "live action", "real photograph", "realistic human",
        "documentary photo", "no animation style", "hyper realistic",
    ],
    "ukiyo_e": [
        "photorealistic", "modern photograph", "digital art", "3D render",
        "realistic painting", "no traditional style", "contemporary image",
    ],
    "watercolor": [
        "oil painting", "acrylic painting", "digital art", "photorealistic",
        "sharp edges", "no watercolor", "solid color painting",
    ],
    "oil_painting": [
        "watercolor", "sketch", "digital art", "photograph", "pencil drawing",
        "no oil paint", "flat illustration",
    ],
    "sketch": [
        "oil painting", "photograph", "colorful painting", "digital render",
        "no sketch", "full color illustration", "photorealistic",
    ],
    "impressionism": [
        "photorealistic", "sharp photograph", "hyperrealism", "precise painting",
        "no brushstrokes", "digital art", "clean render",
    ],
    "cubism": [
        "photorealistic", "realistic painting", "natural proportion",
        "perspective accurate", "no abstraction", "classical realism",
    ],
}

# Human-readable display name for each style key
STYLE_DISPLAY: Dict[str, str] = {
    "van_gogh":      "Van Gogh",
    "picasso":       "Picasso",
    "monet":         "Monet",
    "anime":         "Anime",
    "ukiyo_e":       "Ukiyo-e",
    "watercolor":    "Watercolor",
    "oil_painting":  "Oil Painting",
    "sketch":        "Sketch",
    "impressionism": "Impressionism",
    "cubism":        "Cubism",
}

# ImageNet classes used for paired prompt generation (same as CASteer)
IMAGENET_CLASSES_50 = [
    "a dog", "a cat", "a bird", "a fish", "a horse", "a lion", "a tiger",
    "an elephant", "a bear", "a zebra", "a giraffe", "a monkey", "a rabbit",
    "a deer", "a wolf", "a fox", "a penguin", "a dolphin", "a whale",
    "a shark", "a butterfly", "a bee", "a spider", "a crab", "a frog",
    "a turtle", "a snake", "a rose", "a sunflower", "a tree", "a mountain",
    "a beach", "a river", "a forest", "a castle", "a house", "a car",
    "a bicycle", "a train", "an airplane", "a boat", "a bridge", "a clock",
    "a book", "a chair", "a table", "a lamp", "a bottle", "a cup", "a hat",
]


# =========================================================================
#  Paired prompt builder for styles (CASteer approach)
# =========================================================================

STYLE_CONTRASTIVE: Dict[str, List[str]] = {
    "van_gogh":      ["Picasso", "Rembrandt", "Andy Warhol", "Caravaggio", "Monet"],
    "picasso":       ["Van Gogh", "Rembrandt", "Andy Warhol", "Caravaggio", "Monet"],
    "monet":         ["Van Gogh", "Picasso", "Rembrandt", "Andy Warhol", "Caravaggio"],
    "anime":         ["photorealistic", "oil painting", "watercolor", "sketch", "3D render"],
    "ukiyo_e":       ["photorealistic", "oil painting", "watercolor", "sketch", "3D render"],
    "watercolor":    ["oil painting", "acrylic", "photorealistic", "sketch", "3D render"],
    "oil_painting":  ["watercolor", "sketch", "photorealistic", "3D render", "anime"],
    "sketch":        ["oil painting", "watercolor", "photorealistic", "3D render", "anime"],
    "impressionism": ["photorealistic", "cubism", "sketch", "3D render", "watercolor"],
    "cubism":        ["photorealistic", "impressionism", "sketch", "3D render", "watercolor"],
    "kelly_mckernan":["Tyler Edlin", "Kilian Eng", "Thomas Kinkade", "Ajin", "Monet"],
}


def build_style_pairs(
    style: str,
    num_pairs: int = 50,
) -> List[Tuple[str, str]]:
    """
    Build (positive, negative) prompt pairs for style steering vectors.

    Uses style-vs-other-style pairs (not style-vs-no-style) so the
    steering vector captures Van Gogh specifically, not artistic-vs-photo:
        positive: "{subject}, {target_style} style"
        negative: "{subject}, {other_style} style"

    Other styles are cycled from STYLE_CONTRASTIVE.
    """
    display = STYLE_DISPLAY.get(style, style)
    other_styles = STYLE_CONTRASTIVE.get(style, ["photorealistic"])

    subjects = IMAGENET_CLASSES_50 * ((num_pairs // len(IMAGENET_CLASSES_50)) + 1)
    subjects = subjects[:num_pairs]

    pairs = []
    for i, subj in enumerate(subjects):
        neg_style = other_styles[i % len(other_styles)]
        pos = f"{subj}, {display} style"
        neg = f"{subj}, {neg_style} style"
        pairs.append((pos, neg))
    return pairs


# =========================================================================
#  Schedule
# =========================================================================

@dataclass
class Schedule:
    mode:      str   = "front"
    tau_lo:    float = 0.0
    tau_hi:    float = 1.0
    tau_relax: float = 0.2
    w_min:     float = 0.5

    def weight(self, tau: float) -> float:
        if self.mode == "always":
            return 1.0
        if self.mode == "band":
            return 1.0 if self.tau_lo <= tau <= self.tau_hi else 0.0
        if self.mode == "front":
            if tau >= self.tau_relax:
                return 1.0
            return self.w_min + (1.0 - self.w_min) * (
                tau / max(self.tau_relax, 1e-8))
        return 1.0

    def summary(self) -> str:
        return (f"Schedule(mode={self.mode}, tau_lo={self.tau_lo}, "
                f"tau_hi={self.tau_hi}, tau_relax={self.tau_relax}, "
                f"w_min={self.w_min})")


def make_schedule(mode="front", tau_lo=0.0, tau_hi=1.0,
                  tau_relax=0.2, w_min=0.5) -> Schedule:
    return Schedule(mode=mode, tau_lo=tau_lo, tau_hi=tau_hi,
                    tau_relax=tau_relax, w_min=w_min)


# =========================================================================
#  Text-space orthogonalization (same as TECA)
# =========================================================================

def orthogonalize_embedding(
    cond: torch.Tensor,
    dirs: torch.Tensor,
    renormalize: bool = True,
    max_proj_ratio: Optional[float] = 0.5,
) -> torch.Tensor:
    """
    Project style direction out of every token in the prompt embedding.

    Args:
        cond:            [B, seq, dim]  prompt token embeddings
        dirs:            [k, dim]       style directions (k=1 default)
        renormalize:     restore original token norms after projection
        max_proj_ratio:  cap on how much of a token's norm can be removed
    """
    if dirs.dim() == 1:
        dirs = dirs.unsqueeze(0)
    dirs = dirs.to(dtype=cond.dtype, device=cond.device)
    original_norms = cond.norm(dim=-1, keepdim=True)
    out = cond
    for d in dirs:
        proj = (out * d).sum(dim=-1, keepdim=True)
        if max_proj_ratio is not None:
            token_norms = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            ceiling     = max_proj_ratio * token_norms
            proj        = proj.sign() * proj.abs().clamp(max=ceiling)
        out = out - proj * d
    if renormalize:
        new_norms = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        out = out * (original_norms / new_norms)
    return out


# =========================================================================
#  CA hook (Space 2) — identical to TECA, just different concept
# =========================================================================

class _CASteerHook:
    """
    Four-way gated CA hook (upgraded from 3-way):
        alpha_k = max(beta_l * w * tcl * <v, patch_k>, 0)
        patch_k = patch_k - alpha_k * v

    beta_l = adaptive per-layer beta from concept-alignment scores
    w      = temporal schedule weight
    tcl    = text concept load (style strength from text space)
    v      = CA steering vector for this layer/step
    """

    def __init__(self, sv: SteeringVectors, beta: float = 2.0,
                 layer_align=None,
                 clip: bool = True, align_temp: float = 1.0) -> None:
        self.sv   = sv
        self.beta = beta
        self.clip = clip
        self._hooks: List = []
        self._step: int   = 0
        self._w:    float = 0.0
        self._tcl:  float = 1.0
        self._lidx: Dict[str, int] = defaultdict(int)
        self._sorted_steps = sorted(sv.keys())

        # Build adaptive beta per layer from alignment scores
        self.beta_layer: Dict[str, Dict[int, float]] = {}
        if layer_align:
            # Auto-calibrate: scale global beta relative to mean alignment
            # so that beta is consistent regardless of concept magnitude
            all_scores = [v for place in layer_align
                         for v in layer_align[place].values()]
            mean_align = sum(all_scores) / max(len(all_scores), 1)
            # target mean alignment contribution = 1.0
            # so effective beta = beta / mean_align (self-calibrating)
            calibrated_beta = beta / max(mean_align, 1e-8)
            log.info("Auto-calibrated beta: %.4f (global=%.2f / mean_align=%.4f)",
                     calibrated_beta, beta, mean_align)

            for place in layer_align:
                scores = layer_align[place]
                if not scores:
                    continue
                n = len(scores)
                vals = [scores[li] for li in sorted(scores)]
                max_v = max(vals)
                exps = [math.exp((v - max_v) / max(align_temp, 1e-8))
                        for v in vals]
                s = sum(exps)
                weights = [(e / s) * n for e in exps]
                self.beta_layer[place] = {
                    li: calibrated_beta * weights[i]
                    for i, li in enumerate(sorted(scores))
                }
            log.info("Adaptive beta per layer:")
            for place in sorted(self.beta_layer):
                betas = [f"{self.beta_layer[place][li]:.3f}"
                         for li in sorted(self.beta_layer[place])]
                log.info("  %s: %s", place, betas)
        else:
            log.info("No layer_align — using global beta=%.2f", beta)

    def set_step(self, step: int, w: float,
                 text_concept_load: float = 1.0) -> None:
        self._step = step
        self._w    = w
        self._tcl  = text_concept_load
        self._lidx = defaultdict(int)

    def _lookup(self, place: str, li: int):
        try:
            return self.sv[self._step][place][li]
        except (KeyError, IndexError):
            if not self._sorted_steps:
                return None
            nearest = min(self._sorted_steps, key=lambda s: abs(s - self._step))
            try:
                return self.sv[nearest][place][li]
            except (KeyError, IndexError):
                return None

    def _hook_fn(self, place: str):
        def fn(module, inp, out):
            if self._w <= 0.0:
                return out
            li = self._lidx[place]
            self._lidx[place] += 1
            v = self._lookup(place, li)
            if v is None:
                return out
            v = v.to(out.device).to(out.dtype)
            beta_l = self.beta_layer.get(place, {}).get(li, self.beta)
            x = out
            orig = x.shape
            if x.dim() == 4:
                x = x.reshape(x.shape[0], x.shape[1], -1)
            if x.shape[0] > 1:
                cond = x[x.shape[0] // 2:].clone()
            else:
                cond = x.clone()
            dots  = cond[0] @ v
            alpha = beta_l * self._w * self._tcl * dots
            if self.clip:
                alpha = torch.clamp(alpha, min=0.0)
            cond[0] = cond[0] - alpha.unsqueeze(-1) * v.unsqueeze(0)
            if x.shape[0] > 1:
                x = x.clone()
                x[x.shape[0] // 2:] = cond
            else:
                x = cond
            if len(orig) == 4:
                x = x.reshape(orig)
            return x.to(out.dtype)
        return fn

    def register(self, unet) -> None:
        for name, module in unet.named_modules():
            if module.__class__.__name__ == "BasicTransformerBlock":
                if getattr(module, "attn2", None) is not None:
                    place = ("down" if "down" in name
                             else "up" if "up" in name else "mid")
                    self._hooks.append(
                        module.attn2.register_forward_hook(
                            self._hook_fn(place)))
        log.info("CA hooks registered on %d layers", len(self._hooks))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

class _CACollector:
    def __init__(self):
        self._step  = 0
        self.data   = defaultdict(lambda: defaultdict(list))
        self._hooks = []

    def reset(self):
        self._step = 0
        self.data  = defaultdict(lambda: defaultdict(list))

    def advance(self):
        self._step += 1

    def _fn(self, place):
        def fn(module, inp, out):
            x = out.detach().float()
            if x.dim() == 4:
                x = x.reshape(x.shape[0], x.shape[1], -1)
            if x.shape[0] > 1:
                x = x[x.shape[0] // 2:]
            self.data[self._step][place].append(x.mean(dim=(0, 1)).cpu())
        return fn

    def register(self, unet):
        for name, module in unet.named_modules():
            if module.__class__.__name__ == "BasicTransformerBlock":
                if getattr(module, "attn2", None) is not None:
                    place = ("down" if "down" in name
                             else "up" if "up" in name else "mid")
                    self._hooks.append(
                        module.attn2.register_forward_hook(self._fn(place)))
        log.info("Collector hooks: %d layers", len(self._hooks))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# =========================================================================
#  Build steering vectors
# =========================================================================

def build_steering_vectors(
    style:      str,
    model_id:   str   = "CompVis/stable-diffusion-v1-4",
    model_arch: str   = "sd14",
    num_pairs:  int   = 50,
    save_path:  Optional[str] = None,
    device:     str   = "cuda",
    num_steps:  int   = 50,
    seed:       int   = 0,
) -> SteeringVectors:
    """
    Build per-layer / per-step CA steering vectors for style erasure.

    Paired prompts follow CASteer's style format:
        positive: "{ImageNet class}, {Style} style"
        negative: "{ImageNet class}"

    Each pair uses a unique seed (seed + i) to avoid correlated activations.
    """
    if save_path and os.path.exists(save_path):
        log.info("Loading cached steering vectors from %s", save_path)
        data = torch.load(save_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "sv" in data:
            return data["sv"], data.get("layer_align", {})
        return data, {}

    if style not in STYLE_VOCAB:
        raise ValueError(f"Unknown style '{style}'. "
                         f"Supported: {list(STYLE_VOCAB)}")

    log.info("Building style steering vectors: style=%s num_pairs=%d arch=%s",
             style, num_pairs, model_arch)

    # load pipeline
    pipe = _load_pipeline(model_id, model_arch, device, torch.float32)
    pipe.set_progress_bar_config(disable=True)

    pairs = build_style_pairs(style, num_pairs)
    log.info("Using %d prompt pairs for style='%s'", len(pairs), style)

    def _collect_weighted(prompts, weights=None):
        if weights is None:
            weights = [1.0] * len(prompts)
        coll  = _CACollector()
        coll.register(pipe.unet)
        wsum  = defaultdict(lambda: defaultdict(list))
        wacc  = defaultdict(lambda: defaultdict(list))

        for i, (prompt, w) in enumerate(zip(prompts, weights)):
            pseed = seed + i
            log.info("  [%d/%d] w=%.3f seed=%d  %s",
                     i + 1, len(prompts), w, pseed, prompt[:70])
            coll.reset()

            def _cb(p, step, ts, kwargs):
                coll.advance()
                return kwargs

            pipe(prompt=prompt, num_inference_steps=num_steps,
                 generator=torch.Generator(device=device).manual_seed(pseed),
                 callback_on_step_end=_cb, output_type="latent")

            for step in coll.data:
                for place in coll.data[step]:
                    for li, vec in enumerate(coll.data[step][place]):
                        if len(wsum[step][place]) <= li:
                            wsum[step][place].append(vec.clone() * w)
                            wacc[step][place].append(w)
                        else:
                            wsum[step][place][li] += vec * w
                            wacc[step][place][li] += w

        coll.remove()
        means = {}
        for step in wsum:
            means[step] = {}
            for place in wsum[step]:
                means[step][place] = [
                    wsum[step][place][li] / max(wacc[step][place][li], 1e-8)
                    for li in range(len(wsum[step][place]))
                ]
        return means

    log.info("Collecting POSITIVE (style-present) activations ...")
    pos = _collect_weighted([p for p, _ in pairs])

    log.info("Collecting NEGATIVE (style-absent) activations ...")
    neg = _collect_weighted([n for _, n in pairs])

    sv: SteeringVectors = {}
    for step in pos:
        if step not in neg:
            continue
        sv[step] = {}
        for place in pos[step]:
            if place not in neg[step]:
                continue
            sv[step][place] = []
            for li in range(min(len(pos[step][place]), len(neg[step][place]))):
                diff = pos[step][place][li] - neg[step][place][li]
                sv[step][place].append(
                    (diff / diff.norm().clamp(min=1e-8)).cpu())

    # Compute layer_align: mean concept-alignment score per layer
    from collections import defaultdict as _dd
    la_sum: dict = _dd(lambda: _dd(float))
    la_cnt: dict = _dd(lambda: _dd(int))
    for step in pos:
        if step not in neg:
            continue
        for place in pos[step]:
            if place not in neg[step]:
                continue
            for li in range(min(len(pos[step][place]), len(neg[step][place]))):
                diff = pos[step][place][li] - neg[step][place][li]
                la_sum[place][li] += diff.norm().item()
                la_cnt[place][li] += 1
    layer_align = {
        place: {li: la_sum[place][li] / max(la_cnt[place][li], 1)
                for li in la_sum[place]}
        for place in la_sum
    }
    for place in sorted(layer_align):
        scores = [f"{layer_align[place][li]:.3f}"
                  for li in sorted(layer_align[place])]
        log.info("  layer_align[%s]: %s", place, scores)

    log.info("Steering vectors built: %d steps", len(sv))

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save({"sv": sv, "layer_align": layer_align}, save_path)
        log.info("Saved → %s  (sv + layer_align)", save_path)

    return sv, layer_align


# =========================================================================
#  Pipeline loader helper
# =========================================================================

def _load_pipeline(model_id: str, arch: str, device: str, dtype):
    if arch == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        sched = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, scheduler=sched, torch_dtype=torch.float16,
            variant="fp16", use_safetensors=True).to(device)
        pipe.vae = pipe.vae.float()
    else:
        sched = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=sched, torch_dtype=dtype,
            safety_checker=None).to(device)
    return pipe


# =========================================================================
#  Style Erasure Pipeline
# =========================================================================

class StyleErasurePipeline:
    """
    TECA style erasure pipeline.

    Works for both SD 1.4 (arch='sd14') and SDXL (arch='sdxl').
    Same dual-space logic as nudity removal — only the concept vocab changes.
    """

    def __init__(self, pipe, sv: SteeringVectors, style: str,
                 arch: str = "sd14", beta: float = 2.0,
                 n_components: int = 1,
                 max_proj_ratio: Optional[float] = 0.5,
                 layer_align=None) -> None:
        self.pipe           = pipe
        self.arch           = arch
        self.beta           = beta
        self.max_proj_ratio = max_proj_ratio
        self._hook          = _CASteerHook(sv, beta=beta,
                                           layer_align=layer_align, clip=True)
        self._hook.register(pipe.unet)
        self.text_dirs      = self._build_text_direction(style, n_components)
        display             = STYLE_DISPLAY.get(style, style)
        log.info("Style direction for '%s': shape=%s",
                 display, tuple(self.text_dirs.shape))

    @classmethod
    def from_pretrained(
        cls,
        model_id:       str,
        sv:             SteeringVectors,
        style:          str,
        arch:           str   = "sd14",
        device:         str   = "cuda",
        beta:           float = 2.0,
        n_components:   int   = 1,
        max_proj_ratio: Optional[float] = 0.5,
        layer_align=None,
    ) -> "StyleErasurePipeline":
        pipe = _load_pipeline(model_id, arch, device, torch.float32)
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe, sv, style=style, arch=arch, beta=beta,
                   n_components=n_components, max_proj_ratio=max_proj_ratio,
                   layer_align=layer_align)

    # ── text direction ────────────────────────────────────────────────────

    def _encode_word(self, text: str) -> torch.Tensor:
        """Encode a single word/phrase → [768] via CLIP-L."""
        tok = self.pipe.tokenizer
        enc = self.pipe.text_encoder
        t   = tok(text, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb = enc(t.input_ids.to(self.pipe.device))[0]
        return emb.float().mean(dim=1).squeeze(0)   # [768]

    def _build_text_direction(self, style: str,
                              n_components: int) -> torch.Tensor:
        """Build contrastive style direction from STYLE_VOCAB."""
        if style not in STYLE_VOCAB:
            raise ValueError(f"Unknown style '{style}'. "
                             f"Supported: {list(STYLE_VOCAB)}")
        vocab      = STYLE_VOCAB[style]
        safe_vocab = STYLE_SAFE_VOCAB.get(style, [])
        embs       = torch.stack([self._encode_word(w) for w in vocab]).float()

        if n_components == 1:
            mu = embs.mean(dim=0)
            if safe_vocab:
                safe = torch.stack([self._encode_word(w)
                                    for w in safe_vocab]).float().mean(dim=0)
                mu   = mu - safe
            return (mu / mu.norm().clamp(min=1e-8)).unsqueeze(0)   # [1, 768]

        if safe_vocab:
            safe = torch.stack([self._encode_word(w)
                                for w in safe_vocab]).float().mean(dim=0)
            embs = embs - safe.unsqueeze(0)
        _, _, Vh = torch.linalg.svd(embs, full_matrices=False)
        return Vh[:n_components]   # [k, 768]

    # ── TCL ───────────────────────────────────────────────────────────────

    def _compute_tcl(self, cond_clipl: torch.Tensor) -> float:
        """Text Concept Load: sigmoid(3 * max token dot product with style dir)."""
        d          = self.text_dirs[0].to(cond_clipl.device).to(cond_clipl.dtype)
        token_dots = cond_clipl[0] @ d
        return float(1.0 / (1.0 + math.exp(-3.0 * token_dots.max().item())))

    # ── encoding ─────────────────────────────────────────────────────────

    def _encode_sd14(self, prompt: str, neg: str = ""):
        tok, enc = self.pipe.tokenizer, self.pipe.text_encoder
        def _e(t):
            ids = tok(t, padding="max_length",
                      max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids
            with torch.no_grad():
                return enc(ids.to(self.pipe.device))[0].float()
        return _e(neg), _e(prompt)

    def _encode_sdxl(self, prompt: str, neg: str = ""):
        with torch.no_grad():
            pe, npe, pooled_pe, pooled_npe = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt,
                negative_prompt=neg, negative_prompt_2=neg,
                device=self.pipe.device,
                do_classifier_free_guidance=True)
        return npe.float(), pe.float(), pooled_npe.float(), pooled_pe.float()

    # ── generate ─────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:          str,
        mode:            str   = "both",
        schedule:        Optional[Schedule] = None,
        seed:            int   = 42,
        negative_prompt: str   = "",
        guidance_scale:  float = 7.5,
        num_steps:       int   = 50,
        height:          int   = 512,
        width:           int   = 512,
    ) -> Image.Image:
        """Generate an image with style erasure.

        mode: "none" | "text" | "ca" | "both"
            none  — no erasure (baseline)
            text  — text-space orthogonalization only (Space 1)
            ca    — CA activation steering only (Space 2)
            both  — dual-space TECA (full method)
        """
        if mode not in ("none", "text", "ca", "both"):
            raise ValueError(f"Bad mode '{mode}'")
        use_text = mode in ("text", "both")
        use_ca   = mode in ("ca",   "both")

        pipe       = self.pipe
        unet_dtype = next(pipe.unet.parameters()).dtype

        pipe.scheduler.set_timesteps(num_steps, device=pipe.device)
        timesteps = pipe.scheduler.timesteps
        taus      = [float(t.item()) / 1000.0 for t in timesteps]

        if self.arch == "sdxl":
            uncond, cond, pooled_uncond, pooled_cond = self._encode_sdxl(
                prompt, negative_prompt)
            # for SDXL: CLIP-L is first 768 dims of the 2048-d embedding
            cond_clipl = cond[:, :, :768]
        else:
            uncond, cond = self._encode_sd14(prompt, negative_prompt)
            cond_clipl   = cond   # SD 1.4 is 768-d throughout

        # SPACE 1: orthogonalize style direction from prompt embedding
        cond_ortho = None
        if use_text:
            if self.arch == "sdxl":
                cl_ortho   = orthogonalize_embedding(
                    cond_clipl, self.text_dirs,
                    renormalize=True, max_proj_ratio=self.max_proj_ratio)
                cond_ortho = torch.cat([cl_ortho, cond[:, :, 768:]], dim=-1)
            else:
                cond_ortho = orthogonalize_embedding(
                    cond, self.text_dirs,
                    renormalize=True, max_proj_ratio=self.max_proj_ratio)

        # TCL: how strongly does the prompt carry the style in text space?
        tcl = self._compute_tcl(cond_clipl) if use_ca else 1.0
        log.debug("tcl=%.3f  prompt='%s'", tcl, prompt[:60])

        # init latents
        shape  = (1, pipe.unet.config.in_channels,
                  height // pipe.vae_scale_factor,
                  width  // pipe.vae_scale_factor)
        gen    = torch.Generator(device=pipe.device).manual_seed(seed)
        latent = torch.randn(shape, generator=gen,
                             device=pipe.device, dtype=unet_dtype)
        latent = (latent * pipe.scheduler.init_noise_sigma).to(unet_dtype)

        if self.arch == "sdxl":
            time_ids = torch.tensor(
                [height, width, 0, 0, height, width],
                dtype=unet_dtype, device=pipe.device).unsqueeze(0)

        for step_idx, (tau, native_t) in enumerate(zip(taus, timesteps)):
            w = schedule.weight(tau) if schedule is not None else 1.0

            # SPACE 1: interpolate between original and orthogonalized embedding
            if use_text:
                if w <= 0.0:   cond_used = cond
                elif w >= 1.0: cond_used = cond_ortho
                else:          cond_used = (1.0 - w) * cond + w * cond_ortho
            else:
                cond_used = cond

            # SPACE 2: CA hook with temporal + TCL gate
            self._hook.set_step(step_idx, w if use_ca else 0.0,
                                text_concept_load=tcl)

            if self.arch == "sdxl":
                combined = torch.cat([uncond, cond_used]).to(unet_dtype)
                lat2     = pipe.scheduler.scale_model_input(
                    torch.cat([latent] * 2), native_t).to(unet_dtype)
                added    = {
                    "text_embeds": torch.cat(
                        [pooled_uncond, pooled_cond]).to(unet_dtype),
                    "time_ids": torch.cat([time_ids, time_ids]),
                }
                with torch.no_grad():
                    noise = pipe.unet(lat2, native_t,
                                      encoder_hidden_states=combined,
                                      added_cond_kwargs=added).sample
            else:
                combined = torch.cat([uncond, cond_used]).to(unet_dtype)
                lat2     = pipe.scheduler.scale_model_input(
                    torch.cat([latent] * 2), native_t).to(unet_dtype)
                with torch.no_grad():
                    noise = pipe.unet(lat2, native_t,
                                      encoder_hidden_states=combined).sample

            unc, cnd = noise.chunk(2)
            latent   = pipe.scheduler.step(
                unc + guidance_scale * (cnd - unc), native_t, latent).prev_sample

        # decode (float32 for VAE stability)
        lat = latent.float() / pipe.vae.config.scaling_factor
        with torch.no_grad():
            img = pipe.vae.decode(lat).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().float().permute(0, 2, 3, 1).numpy()
        return Image.fromarray((img[0] * 255).astype("uint8"))

    def remove_hooks(self) -> None:
        self._hook.remove()


# =========================================================================
#  Grid / comparison helpers
# =========================================================================

LABEL_H   = 60
FONT_SZ   = 14
PAD       = 6
PROMPT_W  = 240
COL_LABELS = [
    "1. Unsafe (style intact)",
    "2. Text-ortho only",
    "3. CA-steer only",
    "4. TECA merged",
]


def _label_img(text: str, w: int, h: int) -> Image.Image:
    img  = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SZ)
    except Exception:
        font = ImageFont.load_default()
    for i, line in enumerate(textwrap.wrap(text, width=w // 8)):
        draw.text((4, 4 + i * (FONT_SZ + 2)), line, font=font, fill=(30, 30, 30))
    return img


def make_row(images: List[Image.Image], prompt_text: str = "") -> Image.Image:
    w, h = images[0].size
    images = [img.resize((w, h), Image.LANCZOS) for img in images]
    n      = len(images)
    total_w = PROMPT_W + n * w + n * PAD
    canvas  = Image.new("RGB", (total_w, h + LABEL_H), (220, 220, 220))

    # prompt panel on the left
    p_img = Image.new("RGB", (PROMPT_W, h + LABEL_H), (245, 245, 245))
    draw  = ImageDraw.Draw(p_img)
    try:
        pfont = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        pfont = ImageFont.load_default()
    for i, line in enumerate(textwrap.wrap(prompt_text, width=28)):
        draw.text((6, 6 + i * 14), line, font=pfont, fill=(50, 50, 50))
    canvas.paste(p_img, (0, 0))

    # image columns
    for i, (img, lbl) in enumerate(zip(images, COL_LABELS)):
        x = PROMPT_W + i * (w + PAD)
        canvas.paste(img, (x, 0))
        canvas.paste(_label_img(lbl, w, LABEL_H), (x, h))
    return canvas


def slug(text: str, n: int = 35) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:n]


def compare_prompt(
    wrapper:        StyleErasurePipeline,
    prompt:         str,
    schedule:       Schedule,
    seed:           int   = 42,
    guidance_scale: float = 7.5,
    num_steps:      int   = 50,
    height:         int   = 512,
    width:          int   = 512,
) -> List[Image.Image]:
    images = []
    for mode in ("none", "text", "ca", "both"):
        log.info("  %s ...", mode)
        try:
            img = wrapper.generate(
                prompt=prompt, mode=mode, schedule=schedule,
                seed=seed, guidance_scale=guidance_scale,
                num_steps=num_steps, height=height, width=width)
        except Exception as e:
            log.error("  [%s] failed: %s", mode, e)
            img = Image.new("RGB", (width, height), (80, 0, 0))
        images.append(img)
    return images


# =========================================================================
#  Prompt loader
# =========================================================================

def _load_prompts(path: str, col: str, n: Optional[int]) -> List[str]:
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
    return prompts[:n] if n is not None else prompts


# =========================================================================
#  Batch runner
# =========================================================================

def run_batch(
    prompts_path:   str,
    sv:             SteeringVectors,
    schedule:       Schedule,
    save_dir:       str,
    style:          str,
    col:            str   = "prompt",
    n:              Optional[int] = None,
    model_id:       str   = "CompVis/stable-diffusion-v1-4",
    arch:           str   = "sd14",
    device:         str   = "cuda",
    seed:           int   = 42,
    guidance_scale: float = 7.5,
    num_steps:      int   = 50,
    height:         int   = 512,
    width:          int   = 512,
    beta:           float = 2.0,
    n_components:   int   = 1,
    max_proj_ratio: Optional[float] = 0.5,
    layer_align=None,
) -> None:
    prompts = _load_prompts(prompts_path, col, n)
    os.makedirs(save_dir, exist_ok=True)
    log.info("Schedule : %s", schedule.summary())
    log.info("Style    : %s  arch=%s", STYLE_DISPLAY.get(style, style), arch)
    log.info("Processing %d prompts → %s", len(prompts), save_dir)

    wrapper = StyleErasurePipeline.from_pretrained(
        model_id=model_id, sv=sv, style=style, arch=arch,
        device=device, beta=beta, n_components=n_components,
        max_proj_ratio=max_proj_ratio, layer_align=layer_align)

    col_files   = ["col1_unsafe.png", "col2_text.png",
                   "col3_ca.png", "col4_merged.png"]
    summary_rows: List[Image.Image] = []

    for idx, prompt in enumerate(prompts):
        folder   = os.path.join(save_dir, f"prompt_{idx:03d}_{slug(prompt)}")
        out_path = os.path.join(folder, "comparison.png")

        if os.path.exists(out_path):
            log.info("[%d/%d] skip (exists)", idx + 1, len(prompts))
            summary_rows.append(Image.open(out_path))
            continue

        log.info("-" * 60)
        log.info("[%d/%d] %s", idx + 1, len(prompts), prompt[:80])
        os.makedirs(folder, exist_ok=True)

        images = compare_prompt(wrapper, prompt, schedule, seed=seed,
                                guidance_scale=guidance_scale,
                                num_steps=num_steps, height=height, width=width)
        for img, fname in zip(images, col_files):
            img.save(os.path.join(folder, fname))

        row = make_row(images, prompt_text=prompt)
        row.save(out_path)
        summary_rows.append(row)
        log.info("  saved → %s", out_path)

    wrapper.remove_hooks()

    if summary_rows:
        total_h = sum(r.height for r in summary_rows) + (len(summary_rows) - 1) * PAD
        canvas  = Image.new("RGB", (summary_rows[0].width, total_h), (180, 180, 180))
        y = 0
        for r in summary_rows:
            canvas.paste(r, (0, y))
            y += r.height + PAD
        canvas.save(os.path.join(save_dir, "summary.png"))
        log.info("Summary → %s/summary.png", save_dir)

    print("\n" + "=" * 60)
    print(f"Done. {len(prompts)} prompts processed.")
    print(f"Style  : {STYLE_DISPLAY.get(style, style)}")
    print(f"Results: {save_dir}")
    print("=" * 60)


# =========================================================================
#  CLI
# =========================================================================

def _add_common(sp):
    sp.add_argument("--model_id",   default="CompVis/stable-diffusion-v1-4")
    sp.add_argument("--model_arch", default="sd14", choices=["sd14", "sdxl"],
                    help="Model architecture")
    sp.add_argument("--device",     default="cuda")
    sp.add_argument("--seed",       type=int,   default=42)
    sp.add_argument("--steps",      type=int,   default=50)
    sp.add_argument("--guidance",   type=float, default=7.5)
    sp.add_argument("--img_size",   type=int,   default=512)
    sp.add_argument("--beta",       type=float, default=2.0)
    sp.add_argument("--n_components",   type=int,   default=1)
    sp.add_argument("--max_proj_ratio", type=float, default=0.5)
    sp.add_argument("--schedule",   default="front",
                    choices=["always", "front", "band"])
    sp.add_argument("--tau_lo",     type=float, default=0.0)
    sp.add_argument("--tau_hi",     type=float, default=1.0)
    sp.add_argument("--tau_relax",  type=float, default=0.2)
    sp.add_argument("--w_min",      type=float, default=0.5)


def _sched(a) -> Schedule:
    return make_schedule(mode=a.schedule, tau_lo=a.tau_lo, tau_hi=a.tau_hi,
                         tau_relax=a.tau_relax, w_min=a.w_min)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="TECA Style Erasure — training-free artistic style removal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── build-vectors ──────────────────────────────────────────────────────
    bv = sub.add_parser("build-vectors",
                        help="Build CA steering vectors for a style.")
    bv.add_argument("--style",      required=True,
                    choices=list(STYLE_VOCAB),
                    help="Style to erase")
    bv.add_argument("--model_id",   default="CompVis/stable-diffusion-v1-4")
    bv.add_argument("--model_arch", default="sd14", choices=["sd14", "sdxl"])
    bv.add_argument("--num_pairs",  type=int, default=50)
    bv.add_argument("--save_path",  required=True)
    bv.add_argument("--device",     default="cuda")
    bv.add_argument("--steps",      type=int, default=50)
    bv.add_argument("--seed",       type=int, default=0)

    # ── single ─────────────────────────────────────────────────────────────
    sg = sub.add_parser("single",
                        help="4-column comparison for one prompt.")
    sg.add_argument("--prompt",    required=True)
    sg.add_argument("--style",     required=True, choices=list(STYLE_VOCAB))
    sg.add_argument("--sv_path",   required=True)
    sg.add_argument("--save_path", default="style_comparison.png")
    sg.add_argument("--num_pairs", type=int, default=50)
    _add_common(sg)

    # ── batch ──────────────────────────────────────────────────────────────
    bt = sub.add_parser("batch",
                        help="4-column comparison over a CSV/TXT file.")
    bt.add_argument("--prompts",   required=True)
    bt.add_argument("--style",     required=True, choices=list(STYLE_VOCAB))
    bt.add_argument("--sv_path",   required=True)
    bt.add_argument("--save_dir",  default="results/style_erasure")
    bt.add_argument("--col",       default="prompt")
    bt.add_argument("--n",         type=int, default=20)
    bt.add_argument("--num_pairs", type=int, default=50)
    _add_common(bt)

    args = p.parse_args()

    if args.cmd == "build-vectors":
        build_steering_vectors(
            style      = args.style,
            model_id   = args.model_id,
            model_arch = args.model_arch,
            num_pairs  = args.num_pairs,
            save_path  = args.save_path,
            device     = args.device,
            num_steps  = args.steps,
            seed       = args.seed,
        )

    elif args.cmd == "single":
        mpr = None if args.max_proj_ratio < 0 else args.max_proj_ratio
        sv, layer_align = build_steering_vectors(
            style=args.style, model_id=args.model_id,
            model_arch=args.model_arch, num_pairs=args.num_pairs,
            save_path=args.sv_path, device=args.device,
            num_steps=args.steps, seed=0)
        wrapper  = StyleErasurePipeline.from_pretrained(
            model_id=args.model_id, sv=sv, style=args.style,
            arch=args.model_arch, device=args.device, beta=args.beta,
            n_components=args.n_components, max_proj_ratio=mpr,
            layer_align=layer_align)
        schedule = _sched(args)
        log.info("Schedule: %s", schedule.summary())
        images   = compare_prompt(
            wrapper, args.prompt, schedule, seed=args.seed,
            guidance_scale=args.guidance, num_steps=args.steps,
            height=args.img_size, width=args.img_size)
        row = make_row(images, prompt_text=args.prompt)
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or ".",
                    exist_ok=True)
        row.save(args.save_path)
        wrapper.remove_hooks()
        print(f"Saved → {args.save_path}")

    elif args.cmd == "batch":
        mpr = None if args.max_proj_ratio < 0 else args.max_proj_ratio
        sv, layer_align = build_steering_vectors(
            style=args.style, model_id=args.model_id,
            model_arch=args.model_arch, num_pairs=args.num_pairs,
            save_path=args.sv_path, device=args.device,
            num_steps=args.steps, seed=0)
        run_batch(
            prompts_path   = args.prompts,
            sv             = sv,
            schedule       = _sched(args),
            save_dir       = args.save_dir,
            style          = args.style,
            col            = args.col,
            n              = args.n,
            model_id       = args.model_id,
            arch           = args.model_arch,
            device         = args.device,
            seed           = args.seed,
            guidance_scale = args.guidance,
            num_steps      = args.steps,
            height         = args.img_size,
            width          = args.img_size,
            beta           = args.beta,
            n_components   = args.n_components,
            max_proj_ratio = mpr,
            layer_align    = layer_align,
        )