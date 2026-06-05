"""
style_erasure_TECA.py  —  Two-Space Training-Free Style Erasure
================================================================

The idea (merging your method + your partner's)
-----------------------------------------------
Concept erasure has two independent places you can intervene during a
Stable-Diffusion generation:

  SPACE 1 — TEXT EMBEDDING  (your method)
      The CLIP text embedding `cond` [1, 77, 768] is the *instruction*
      the UNet attends to.  We remove the unsafe concept from it by
      projecting out a concept direction built from a word vocabulary:

          d          = normalize( mean(unsafe_vocab) - mean(safe_vocab) )
          e_ortho[i] = e[i] - (e[i] . d) * d        # per token

  SPACE 2 — CROSS-ATTENTION ACTIVATIONS  (your partner's method, CASteer)
      Even with a cleaned instruction the UNet's learned priors can
      rebuild the concept inside its feature maps.  So we *also* steer
      the output of every cross-attention (`attn2`) layer, patch by patch:

          ca_new[patch] = ca[patch] - beta * w * max(<v, ca[patch]>, 0) * v

      where `v` is a per-layer / per-step steering vector built from 50
      paired (concept-present vs concept-absent) image generations.

These are DIFFERENT spaces (text = 768-d, activations = 320/640/1280-d),
so they need DIFFERENT operators — you cannot reuse one vector for both.
They share only the *concept definition*.  Applied together they are
"defense in depth": the text edit cleans the instruction, the activation
steer cleans whatever the network rebuilds.

What we keep from PCI / what we drop
------------------------------------
We DROP the expensive PCI step-detection (finding the exact tau where the
concept commits via CIS curves + a VQA scorer).  We KEEP PCI's *mechanism*
— varying the conditioning over the denoising trajectory — as a cheap
FIXED schedule.  For erasure the intervention should be strongest EARLY
(high tau decides global composition), so the default schedule is
front-loaded: full strength early, gently relaxed near the end to recover
fine detail.  No per-image computation, no curve fitting.

Comparison output (4 columns, same seed)
----------------------------------------
  col1  Unsafe                      (no intervention)
  col2  Text-ortho only             (SPACE 1)
  col3  CA-steer only               (SPACE 2)
  col4  Merged: text + CA           (SPACE 1 + SPACE 2)  <- the new method

Columns 2/3/4 are exactly the ablation that tells you whether the merge is
worth it: the merge is justified only if col4 beats max(col2, col3).

Usage
-----
    # build CA steering vectors once (cached to a .pt file), then run a batch
    python style_erasure_TECA.py batch \\
        --prompts dataset/style.csv --col prompt --n 10 \\
        --concept van_gogh --sv_path sv_van_gogh.pt \\
        --save_dir results/style --device cuda

    # one prompt
    python style_erasure_TECA.py single \\
        --prompt "a cat in the style of Van Gogh" \\
        --concept van_gogh --sv_path sv_van_gogh.pt \\
        --save_path out.png --device cuda

    # build vectors explicitly
    python style_erasure_TECA.py build-vectors \\
        --concept van_gogh --num_pairs 50 \\
        --save_path sv_van_gogh.pt --device cuda

Requirements:  diffusers transformers torch Pillow pandas tqdm
This file is self-contained — it does NOT import pci_unlearning.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import logging
import os
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("style_erasure_teca")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =========================================================================
#  Concept definition — vocab (for text space) + paired prompts (for CA space)
# =========================================================================

# Word vocabulary -> defines the TEXT-SPACE direction (your method).
# Cleaner than full-sentence pairs because single words carry little
# contextual noise.
# Supported styles — add new entries here to support more styles.
# Keys are used as --concept argument values.
CONCEPT_VOCAB: Dict[str, List[str]] = {
    "van_gogh": [
        "Van Gogh", "Van Gogh style", "post-impressionist", "swirling brushstrokes",
        "thick impasto", "Starry Night", "Sunflowers painting", "painterly swirls",
        "vivid colors Van Gogh", "expressionist brushwork",
    ],
    "picasso": [
        "Picasso", "Picasso style", "cubist", "cubism", "Picasso painting",
        "fragmented faces", "abstract cubist", "geometric abstraction",
        "multiple perspectives painting", "analytic cubism",
    ],
    "monet": [
        "Monet", "Claude Monet", "impressionist", "Monet style",
        "water lilies painting", "soft brushstrokes", "Monet garden",
        "impressionism", "hazy light painting", "plein air impressionist",
    ],
    "anime": [
        "anime", "anime style", "manga", "Japanese animation", "anime art",
        "anime illustration", "cel shading", "anime character",
        "kawaii anime", "anime aesthetic",
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
        "oil painting", "oil on canvas", "classical oil painting",
        "old master painting", "oil paint texture", "fine art oil",
        "portrait oil painting", "renaissance oil painting", "baroque oil painting",
        "oil medium painting",
    ],
    "sketch": [
        "pencil sketch", "charcoal drawing", "sketch art", "hand-drawn sketch",
        "graphite drawing", "pencil drawing style", "line art sketch",
        "rough sketch", "artist sketch", "gestural drawing",
    ],
    "impressionism": [
        "impressionism", "impressionist painting", "impressionist style",
        "loose brushstrokes", "plein air painting", "impressionist art",
        "light and color impressionist", "Renoir style", "Sisley style",
        "post-impressionist",
    ],
    "cubism": [
        "cubism", "cubist art", "cubist painting", "geometric cubism",
        "analytical cubism", "synthetic cubism", "fractured perspective",
        "cubist abstraction", "multiple viewpoints painting", "Braque style",
    ],
}

# Safe counterpart vocab -> contrastive direction cancels the shared
# "photorealistic human" component so we erase nudity-vs-clothed rather
# than cartoon-vs-photo.
SAFE_CONCEPT_VOCAB: Dict[str, List[str]] = {
    "van_gogh":     ["photorealistic", "photograph", "realistic photo",
                     "digital photo", "no artistic style", "plain image"],
    "picasso":      ["photorealistic", "realistic portrait", "photograph",
                     "natural perspective", "realistic painting", "anatomically correct"],
    "monet":        ["sharp photograph", "crisp photo", "photorealistic",
                     "high definition photo", "clear image", "digital photo"],
    "anime":        ["photorealistic", "live action", "real photograph",
                     "realistic human", "documentary photo", "hyper realistic"],
    "ukiyo_e":      ["photorealistic", "modern photograph", "digital art",
                     "3D render", "realistic painting", "contemporary image"],
    "watercolor":   ["oil painting", "acrylic painting", "digital art",
                     "photorealistic", "sharp edges", "solid color painting"],
    "oil_painting": ["watercolor", "sketch", "digital art", "photograph",
                     "pencil drawing", "flat illustration"],
    "sketch":       ["oil painting", "photograph", "colorful painting",
                     "digital render", "full color illustration", "photorealistic"],
    "impressionism":["photorealistic", "sharp photograph", "hyperrealism",
                     "precise painting", "digital art", "clean render"],
    "cubism":       ["photorealistic", "realistic painting", "natural proportion",
                     "perspective accurate", "classical realism"],
}

# Paired prompts -> define the ACTIVATION-SPACE steering vectors (CASteer).
# Full sentences are fine here: averaging 50 real generations cancels the
# contextual noise, and the difference is taken per layer / per step.
# ImageNet classes used as subjects for style paired prompts.
# Following CASteer get_prompts_style():
#   positive: "{subject}, {Style} style"
#   negative: "{subject}"
_IMAGENET_SUBJECTS = [
    "a dog", "a cat", "a bird", "a fish", "a horse", "a lion", "a tiger",
    "an elephant", "a bear", "a zebra", "a giraffe", "a monkey", "a rabbit",
    "a deer", "a wolf", "a fox", "a penguin", "a dolphin", "a whale",
    "a shark", "a butterfly", "a bee", "a spider", "a crab", "a frog",
    "a turtle", "a snake", "a rose", "a sunflower", "a tree", "a mountain",
    "a beach", "a river", "a forest", "a castle", "a house", "a car",
    "a bicycle", "a train", "an airplane", "a boat", "a bridge", "a clock",
    "a book", "a chair", "a table", "a lamp", "a bottle", "a cup", "a hat",
]

# Human-readable style names used in the positive prompt
_STYLE_DISPLAY = {
    "van_gogh":     "Van Gogh",
    "picasso":      "Picasso",
    "monet":        "Monet",
    "anime":        "anime",
    "ukiyo_e":      "ukiyo-e",
    "watercolor":   "watercolor",
    "oil_painting": "oil painting",
    "sketch":       "pencil sketch",
    "impressionism":"impressionist",
    "cubism":       "cubist",
}

# PAIRED_PROMPTS is generated dynamically from ImageNet subjects.
# Kept as a dict for compatibility with _paired_prompts().
PAIRED_PROMPTS: Dict[str, List[Tuple[str, str]]] = {}

def _paired_prompts(concept: str, num_pairs: int) -> List[Tuple[str, str]]:
    """Return `num_pairs` (style-present, style-absent) prompt pairs.

    Follows CASteer get_prompts_style():
        positive: "{ImageNet subject}, {Style} style"
        negative: "{ImageNet subject}"

    Cycles through ImageNet subjects until num_pairs is reached.
    """
    if concept not in CONCEPT_VOCAB:
        raise ValueError(f"Unknown concept/style '{concept}'. "
                         f"Supported: {list(CONCEPT_VOCAB)}")
    display = _STYLE_DISPLAY.get(concept, concept)
    subjects = (_IMAGENET_SUBJECTS * ((num_pairs // len(_IMAGENET_SUBJECTS)) + 1))
    pairs = []
    for subj in subjects[:num_pairs]:
        pairs.append((f"{subj}, {display} style", subj))
    return pairs


# =========================================================================
#  Fixed temporal schedule  (PCI mechanism, no PCI analysis)
# =========================================================================

@dataclass
class Schedule:
    """Per-tau intervention weight in [0, 1].

    tau is the normalised denoising step: tau=1 is pure noise (start,
    global structure decided here), tau=0 is the clean image (end, only
    fine detail).

    Modes:
      always  weight = 1 everywhere (strongest erasure).
      front   weight = 1 while tau >= tau_relax, then linearly decays to
              w_min at tau=0.  Erasure stays strong through the steps that
              decide composition, then relaxes so late detail recovers.
              This is the recommended default.
      band    weight = 1 inside [tau_lo, tau_hi], 0 outside (PCI-style
              window — kept only for comparison / ablation).
    """
    mode: str = "front"
    tau_lo: float = 0.0
    tau_hi: float = 1.0
    tau_relax: float = 0.2
    w_min: float = 0.5

    def weight(self, tau: float) -> float:
        if self.mode == "always":
            return 1.0
        if self.mode == "band":
            return 1.0 if self.tau_lo <= tau <= self.tau_hi else 0.0
        if self.mode == "front":
            if tau >= self.tau_relax:
                return 1.0
            # linear ramp from w_min (at tau=0) up to 1.0 (at tau_relax)
            return self.w_min + (1.0 - self.w_min) * (tau / max(self.tau_relax, 1e-6))
        raise ValueError(f"Unknown schedule mode '{self.mode}'")

    def summary(self) -> str:
        return (f"Schedule(mode={self.mode}, tau_lo={self.tau_lo}, "
                f"tau_hi={self.tau_hi}, tau_relax={self.tau_relax}, "
                f"w_min={self.w_min})")


def make_schedule(mode: str = "front", tau_lo: float = 0.0, tau_hi: float = 1.0,
                  tau_relax: float = 0.2, w_min: float = 0.5) -> Schedule:
    return Schedule(mode=mode, tau_lo=tau_lo, tau_hi=tau_hi,
                    tau_relax=tau_relax, w_min=w_min)


# =========================================================================
#  SPACE 1 — text-embedding orthogonalization
# =========================================================================

def orthogonalize_embedding(
    cond: torch.Tensor,
    dirs: torch.Tensor,
    renormalize: bool = True,
    max_proj_ratio: Optional[float] = 0.5,
) -> torch.Tensor:
    """Project the unsafe subspace out of every token of a text embedding.

    For each direction d in `dirs` (applied sequentially):

        cond[:, i, :] -= proj * d

    `max_proj_ratio` caps how much of a token's norm can be removed per
    direction (0.5 = keep >=50% of every token) — this prevents semantic
    collapse on heavily-explicit prompts, which CFG would otherwise
    amplify into text-like garbage.  `renormalize` restores each token's
    original norm afterwards.

    Args:
        cond:           [B, seq, dim] text embedding to clean.
        dirs:           [k, dim] orthonormal unsafe directions.
        renormalize:    Rescale tokens to their original norm after removal.
        max_proj_ratio: Cap on fractional removal per direction (None=off).
    Returns:
        Cleaned embedding, same shape as `cond`.
    """
    if dirs.dim() == 1:
        dirs = dirs.unsqueeze(0)
    dirs = dirs.to(dtype=cond.dtype, device=cond.device)
    original_norms = cond.norm(dim=-1, keepdim=True)
    out = cond
    for d in dirs:
        proj = (out * d).sum(dim=-1, keepdim=True)            # [B, seq, 1]
        if max_proj_ratio is not None:
            token_norms = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            ceiling = max_proj_ratio * token_norms
            proj = proj.sign() * proj.abs().clamp(max=ceiling)
        out = out - proj * d
    if renormalize:
        new_norms = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        out = out * (original_norms / new_norms)
    return out


# =========================================================================
#  SPACE 2 (build) — collect cross-attention activations
# =========================================================================

# SteeringVectors: dict[step -> dict[place -> list[Tensor[dim]]]]
SteeringVectors = Dict[int, Dict[str, List[torch.Tensor]]]


class _CACollector:
    """Forward hook that records mean-pooled `attn2` outputs per layer/step."""

    def __init__(self) -> None:
        self._step: int = 0
        self._layer_counts: Dict[str, int] = defaultdict(int)
        self.data: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list))
        self._hooks: List = []

    def reset_step(self) -> None:
        self._step += 1
        self._layer_counts = defaultdict(int)

    def _hook_fn(self, place: str):
        def fn(module, inp, out):
            x = out.detach().float()
            if x.dim() == 4:                       # [B, seq, heads, hd]
                x = x.reshape(x.shape[0], x.shape[1], -1)
            if x.shape[0] > 1:                     # keep CFG-conditioned half
                x = x[x.shape[0] // 2:]
            vec = x.mean(dim=(0, 1))               # -> [dim]
            self._layer_counts[place] += 1
            self.data[self._step][place].append(vec.cpu())
        return fn

    def register(self, unet) -> None:
        for name, module in unet.named_modules():
            if module.__class__.__name__ == "BasicTransformerBlock":
                if getattr(module, "attn2", None) is not None:
                    place = ("down" if "down" in name
                             else "up" if "up" in name else "mid")
                    self._hooks.append(
                        module.attn2.register_forward_hook(self._hook_fn(place)))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def build_steering_vectors(
    model_id: str,
    concept: str = "nudity",
    num_pairs: int = 50,
    save_path: Optional[str] = None,
    device: str = "cuda",
    num_steps: int = 50,
    seed: int = 0,
) -> SteeringVectors:
    """Build per-layer / per-step CA steering vectors from paired generations.

        v[step][place][layer] = normalize( mean(CA_pos) - mean(CA_neg) )

    Cached to `save_path` (.pt); reloaded directly if it already exists.
    """
    if save_path and os.path.exists(save_path):
        log.info("Loading cached steering vectors from %s", save_path)
        return torch.load(save_path, map_location="cpu", weights_only=False)

    log.info("Building CA steering vectors: concept=%s num_pairs=%d",
             concept, num_pairs)
    sched = DPMSolverMultistepScheduler.from_pretrained(model_id,
                                                        subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=sched, torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    pairs = _paired_prompts(concept, num_pairs)
    log.info("Using %d prompt pairs", len(pairs))

    def _collect_mean(prompts: List[str]) -> SteeringVectors:
        collector = _CACollector()
        collector.register(pipe.unet)
        sum_data: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list))
        count: Dict[int, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list))

        for i, prompt in enumerate(prompts):
            prompt_seed = seed + i   # unique seed per prompt — prevents correlated activations
            log.info("  [%d/%d] seed=%d  %s", i + 1, len(prompts), prompt_seed, prompt[:60])
            collector._step = 0
            collector._layer_counts = defaultdict(int)
            collector.data = defaultdict(lambda: defaultdict(list))

            def _cb(pipe_, step, timestep, kwargs):
                collector.reset_step()
                return kwargs

            pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                generator=torch.Generator(device=device).manual_seed(prompt_seed),
                callback_on_step_end=_cb,
                output_type="latent",
            )
            for step in collector.data:
                for place in collector.data[step]:
                    for li, vec in enumerate(collector.data[step][place]):
                        if len(sum_data[step][place]) <= li:
                            sum_data[step][place].append(vec.clone())
                            count[step][place].append(1)
                        else:
                            sum_data[step][place][li] += vec
                            count[step][place][li] += 1

        collector.remove()
        means: SteeringVectors = {}
        for step in sum_data:
            means[step] = {}
            for place in sum_data[step]:
                means[step][place] = [
                    sum_data[step][place][li] / count[step][place][li]
                    for li in range(len(sum_data[step][place]))
                ]
        return means

    log.info("Collecting POSITIVE (concept-present) activations ...")
    pos = _collect_mean([p for p, _ in pairs])
    log.info("Collecting NEGATIVE (concept-absent) activations ...")
    neg = _collect_mean([n for _, n in pairs])

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
                sv[step][place].append((diff / diff.norm().clamp(min=1e-8)).cpu())

    log.info("Steering vectors built: %d steps", len(sv))
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(sv, save_path)
        log.info("Saved -> %s", save_path)
    return sv


# =========================================================================
#  SPACE 2 (apply) — cross-attention steering hook
# =========================================================================

class _CASteerHook:
    """Forward hook applying cross-space gated CA steering at inference.

    The steering gate now has THREE multiplicative components:

        alpha_k = max(β · w · tcl · <v_ca, ca_k>, 0)
        patch_k_new = patch_k - alpha_k · v_ca

    Where:
        w          — temporal schedule weight (when to intervene)
        tcl        — text concept load: how strongly the PROMPT carries the
                     concept in text-embedding space (cross-space signal)
        <v_ca,ca_k>— per-patch dot product with CA steering vector
                     (which patches are producing the concept in visual space)

    All three gates must fire for a patch to be suppressed.
    This is the cross-space information flow:
      - text-space tells us WHETHER the prompt intends the concept
      - activation-space tells us WHERE in the image it is forming
      - temporal schedule tells us WHEN to intervene

    text_concept_load is computed once per prompt in generate() from the
    prompt's token embeddings dotted with the text-space concept direction,
    then passed here via set_step().  It is a prompt-level scalar in [0, 1].
    """

    def __init__(self, sv: Optional[SteeringVectors], beta: float = 2.0,
                 clip: bool = True) -> None:
        self.sv = sv or {}
        self.beta = beta
        self.clip = clip
        self._hooks: List = []
        self._step: int = 0
        self._w: float = 0.0
        self._tcl: float = 1.0    # text concept load — default 1.0 (no text info)
        self._layer_idx: Dict[str, int] = defaultdict(int)

    def set_step(self, step: int, w: float, text_concept_load: float = 1.0) -> None:
        """Call before each denoising step.

        Args:
            step:               Current step index (0-indexed).
            w:                  Temporal schedule weight in [0, 1].
            text_concept_load:  Prompt-level concept strength from text space,
                                in [0, 1].  Pass 1.0 to disable (legacy behaviour).
        """
        self._step = step
        self._w = w
        self._tcl = text_concept_load
        self._layer_idx = defaultdict(int)

    def _lookup(self, place: str, li: int):
        try:
            return self.sv[self._step][place][li]
        except (KeyError, IndexError):
            if not self.sv:
                return None
            closest = min(self.sv.keys(), key=lambda s: abs(s - self._step))
            try:
                return self.sv[closest][place][li]
            except (KeyError, IndexError):
                return None

    def _hook_fn(self, place: str):
        def fn(module, inp, out):
            w = self._w
            if w <= 0.0:
                return out                       # disabled this step (schedule gate)
            tcl = self._tcl                      # text concept load gate
            li = self._layer_idx[place]
            self._layer_idx[place] += 1
            v = self._lookup(place, li)
            if v is None:
                return out
            v = v.to(out.device).to(out.dtype)   # [dim]

            x = out
            orig_shape = x.shape
            if x.dim() == 4:
                B, S, H, D = x.shape
                x = x.reshape(B, S, H * D)

            if x.shape[0] > 1:                   # steer CFG-conditioned half only
                cond = x[x.shape[0] // 2:].clone()
            else:
                cond = x.clone()

            # per-patch activation-space dot product: [seq]
            dots = cond[0] @ v

            # three-way gate: temporal × text-load × activation-alignment
            alpha = self.beta * w * tcl * dots
            if self.clip:
                alpha = torch.clamp(alpha, min=0.0)

            cond[0] = cond[0] - alpha.unsqueeze(-1) * v.unsqueeze(0)

            if x.shape[0] > 1:
                x = x.clone()
                x[x.shape[0] // 2:] = cond
            else:
                x = cond
            if len(orig_shape) == 4:
                x = x.reshape(orig_shape)
            return x.to(out.dtype)
        return fn

    def register(self, unet) -> None:
        for name, module in unet.named_modules():
            if module.__class__.__name__ == "BasicTransformerBlock":
                if getattr(module, "attn2", None) is not None:
                    place = ("down" if "down" in name
                             else "up" if "up" in name else "mid")
                    self._hooks.append(
                        module.attn2.register_forward_hook(self._hook_fn(place)))
        log.info("CA steering hooks registered on %d layers", len(self._hooks))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# =========================================================================
#  Merged pipeline
# =========================================================================

class MergedUnlearningPipeline:
    """Stable-Diffusion wrapper that can apply text-ortho, CA-steer, or both.

    A single `generate(..., mode=...)` call drives every comparison column:
        mode="none"  -> unsafe baseline
        mode="text"  -> SPACE 1 only
        mode="ca"    -> SPACE 2 only
        mode="both"  -> merged method
    """

    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        sv: Optional[SteeringVectors],
        concept: str = "nudity",
        beta: float = 2.0,
        n_components: int = 1,
        max_proj_ratio: Optional[float] = 0.5,
    ) -> None:
        self.pipe = pipe
        self.beta = beta
        self.max_proj_ratio = max_proj_ratio
        self._hook = _CASteerHook(sv, beta=beta, clip=True)
        self._hook.register(pipe.unet)
        # SPACE 1 direction — built once from the word vocabulary.
        self.text_dirs = self._build_text_direction(concept, n_components)
        log.info("Text-space direction: %s", tuple(self.text_dirs.shape))

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        sv: Optional[SteeringVectors],
        device: str = "cuda",
        concept: str = "nudity",
        beta: float = 2.0,
        n_components: int = 1,
        max_proj_ratio: Optional[float] = 0.5,
    ) -> "MergedUnlearningPipeline":
        sched = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=sched, torch_dtype=torch.float32,
            safety_checker=None,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe, sv, concept=concept, beta=beta,
                   n_components=n_components, max_proj_ratio=max_proj_ratio)

    # ---- encoding helpers ----

    def _encode_mean(self, text: str) -> torch.Tensor:
        """Encode `text` and mean-pool over the sequence -> [dim]."""
        tok, enc = self.pipe.tokenizer, self.pipe.text_encoder
        t = tok(text, padding="max_length", max_length=tok.model_max_length,
                truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb = enc(t.input_ids.to(self.pipe.device))[0]
        return emb.float().mean(dim=1).squeeze(0)

    def _encode_prompt(self, prompt: str, negative_prompt: str = ""):
        tok, enc = self.pipe.tokenizer, self.pipe.text_encoder

        def _enc(text):
            t = tok(text, padding="max_length", max_length=tok.model_max_length,
                    truncation=True, return_tensors="pt")
            with torch.no_grad():
                return enc(t.input_ids.to(self.pipe.device))[0]

        return _enc(negative_prompt), _enc(prompt)      # (uncond, cond)

    def _build_text_direction(self, concept: str, n_components: int) -> torch.Tensor:
        """Contrastive vocab direction: normalize(mean(unsafe) - mean(safe))."""
        if concept not in CONCEPT_VOCAB:
            raise ValueError(f"Unknown concept '{concept}'. "
                             f"Available: {list(CONCEPT_VOCAB)}")
        vocab = CONCEPT_VOCAB[concept]
        safe_vocab = SAFE_CONCEPT_VOCAB.get(concept)
        embs = torch.stack([self._encode_mean(w) for w in vocab]).float()
        if n_components == 1:
            mean_unsafe = embs.mean(dim=0)
            if safe_vocab:
                safe = torch.stack([self._encode_mean(w)
                                    for w in safe_vocab]).float().mean(dim=0)
                direction = mean_unsafe - safe
            else:
                direction = mean_unsafe
            direction = direction / direction.norm().clamp(min=1e-8)
            return direction.unsqueeze(0)               # [1, dim]
        if safe_vocab:
            safe = torch.stack([self._encode_mean(w)
                                for w in safe_vocab]).float().mean(dim=0)
            embs = embs - safe.unsqueeze(0)
        _, _, Vh = torch.linalg.svd(embs, full_matrices=False)
        return Vh[:n_components]                         # [k, dim]

    def _compute_text_concept_load(self, cond: torch.Tensor) -> float:
        """Compute how strongly the prompt embedding carries the concept direction.

        For each token in the prompt embedding we compute its dot product with
        the text-space concept direction.  We take the max over tokens (not mean)
        because a single explicit concept token — e.g. "nude" — should trigger
        strong suppression even if surrounded by benign context tokens.

        The result is then sigmoid-scaled to [0, 1] so it can multiply the
        existing gate without changing the scale of beta.

        Args:
            cond: [1, seq, dim]  — conditioned text embedding for the prompt.

        Returns:
            Scalar float in [0, 1] — 0 means no concept load, 1 means maximum.
        """
        import math
        # text_dirs: [k, dim]  (k=1 in default config)
        # take the primary direction (first component)
        d = self.text_dirs[0].to(cond.device).to(cond.dtype)   # [dim]

        # dot product with every token: [seq]
        token_dots = (cond[0] @ d)   # cond[0] is [seq, dim]

        # max over tokens — one explicit concept token is enough
        max_dot = token_dots.max().item()

        # sigmoid centres at 0: positive dot = concept present, negative = absent
        # we use a steeper sigmoid (scale=3) so it transitions sharply around 0
        load = 1.0 / (1.0 + math.exp(-3.0 * max_dot))
        return float(load)

    # ---- denoising ----

    def _init_latents(self, seed: int, height: int, width: int) -> torch.Tensor:
        pipe = self.pipe
        shape = (1, pipe.unet.config.in_channels,
                 height // pipe.vae_scale_factor,
                 width // pipe.vae_scale_factor)
        gen = torch.Generator(device=pipe.device).manual_seed(seed)
        latents = torch.randn(shape, generator=gen, device=pipe.device,
                              dtype=torch.float32)
        return latents * pipe.scheduler.init_noise_sigma

    def _denoise_step(self, latent, uncond, cond, native_t, guidance_scale):
        combined = torch.cat([uncond, cond])
        lat2 = torch.cat([latent] * 2)
        lat2 = self.pipe.scheduler.scale_model_input(lat2, native_t)
        with torch.no_grad():
            noise_pred = self.pipe.unet(
                lat2, native_t, encoder_hidden_states=combined).sample
        unc, cnd = noise_pred.chunk(2)
        noise = unc + guidance_scale * (cnd - unc)
        return self.pipe.scheduler.step(noise, native_t, latent).prev_sample

    def _decode(self, latent: torch.Tensor) -> Image.Image:
        lat = latent / self.pipe.vae.config.scaling_factor
        with torch.no_grad():
            img = self.pipe.vae.decode(lat).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((img[0] * 255).astype("uint8"))

    # ---- public ----

    def generate(
        self,
        prompt: str,
        mode: str = "both",
        schedule: Optional[Schedule] = None,
        seed: int = 42,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_steps: int = 50,
        height: int = 512,
        width: int = 512,
    ) -> Image.Image:
        """Generate one image.

        mode:  "none" | "text" | "ca" | "both"
        schedule:  Schedule controlling both intervention weights over tau.
                   None -> full strength every step.
        """
        if mode not in ("none", "text", "ca", "both"):
            raise ValueError(f"Bad mode '{mode}'")
        use_text = mode in ("text", "both")
        use_ca = mode in ("ca", "both")

        pipe = self.pipe
        pipe.scheduler.set_timesteps(num_steps, device=pipe.device)
        timesteps = pipe.scheduler.timesteps
        taus = [float(t.item()) / 1000.0 for t in timesteps]

        uncond, cond = self._encode_prompt(prompt, negative_prompt)

        # SPACE 1: build the cleaned embedding once (interpolated per step).
        cond_ortho = None
        if use_text:
            cond_ortho = orthogonalize_embedding(
                cond, self.text_dirs, renormalize=True,
                max_proj_ratio=self.max_proj_ratio)

        latent = self._init_latents(seed, height, width)

        # Cross-space signal: compute text concept load once per prompt.
        # This tells the CA hook how strongly the text intends the concept,
        # so activation-space suppression scales with text-space evidence.
        # tcl=1.0 when use_ca is False (no cross-space gating needed).
        tcl = self._compute_text_concept_load(cond) if use_ca else 1.0
        log.debug("text_concept_load=%.3f  prompt='%s'", tcl, prompt[:60])

        for step_idx, (tau, native_t) in enumerate(zip(taus, timesteps)):
            w = schedule.weight(tau) if schedule is not None else 1.0

            # SPACE 1 — interpolate text embedding by the schedule weight
            if use_text:
                if w <= 0.0:
                    cond_used = cond
                elif w >= 1.0:
                    cond_used = cond_ortho
                else:
                    cond_used = (1.0 - w) * cond + w * cond_ortho
            else:
                cond_used = cond

            # SPACE 2 — gate the CA hook with both temporal weight AND
            # text concept load (cross-space gate).
            # When w=0 the hook is a no-op regardless of tcl.
            self._hook.set_step(
                step_idx,
                w if use_ca else 0.0,
                text_concept_load=tcl,
            )

            latent = self._denoise_step(latent, uncond, cond_used,
                                        native_t, guidance_scale)

        return self._decode(latent)

    def remove_hooks(self) -> None:
        self._hook.remove()


# =========================================================================
#  Grid helpers
# =========================================================================

LABEL_H, FONT_SZ, PAD = 60, 15, 6
PROMPT_W, PROMPT_FSZ = 240, 12
COL_LABELS = [
    "1. Unsafe",
    "2. Text-ortho (embedding)",
    "3. CA-steer (activations)",
    "4. Merged: text + CA",
]


def _font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _label_img(text: str, w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _font(FONT_SZ)
    y = 5
    for line in textwrap.wrap(text, width=max(1, w // (FONT_SZ // 2))):
        bb = draw.textbbox((0, 0), line, font=font)
        draw.text(((w - (bb[2] - bb[0])) // 2, y), line, font=font,
                  fill=(30, 30, 30))
        y += FONT_SZ + 3
    return img


def _prompt_panel(text: str, h: int) -> Image.Image:
    img = Image.new("RGB", (PROMPT_W, h), (240, 243, 255))
    draw = ImageDraw.Draw(img)
    font = _font(PROMPT_FSZ)
    chars = max(1, (PROMPT_W - 12) * 2 // PROMPT_FSZ)
    y = 8
    for line in textwrap.wrap(text, width=chars):
        draw.text((6, y), line, font=font, fill=(15, 15, 60))
        y += PROMPT_FSZ + 3
        if y > h - PROMPT_FSZ - 4:
            draw.text((6, y), "...", font=font, fill=(80, 80, 120))
            break
    return img


def make_row(images: List[Image.Image], prompt_text: str = "") -> Image.Image:
    w, h = images[0].size
    left_w = PROMPT_W if prompt_text else 0
    n = len(images)
    row = Image.new("RGB", (left_w + n * w + (n - 1) * PAD, h + LABEL_H),
                    (220, 220, 220))
    if prompt_text:
        row.paste(_prompt_panel(prompt_text, h + LABEL_H), (0, 0))
    for i, (img, lbl) in enumerate(zip(images, COL_LABELS)):
        x = left_w + i * (w + PAD)
        row.paste(img, (x, 0))
        row.paste(_label_img(lbl, w, LABEL_H), (x, h))
    return row


def slug(text: str, max_len: int = 30) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:max_len]


# =========================================================================
#  Comparison + batch
# =========================================================================

def compare_prompt(
    wrapper: MergedUnlearningPipeline,
    prompt: str,
    schedule: Schedule,
    seed: int = 42,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    height: int = 512,
    width: int = 512,
) -> List[Image.Image]:
    """Return [unsafe, text-only, ca-only, merged] for one prompt (same seed)."""
    out = []
    for tag, mode in [("unsafe", "none"), ("text", "text"),
                      ("ca", "ca"), ("merged", "both")]:
        log.info("  %s ...", tag)
        out.append(wrapper.generate(
            prompt, mode=mode, schedule=schedule, seed=seed,
            guidance_scale=guidance_scale, num_steps=num_steps,
            height=height, width=width))
    return out


def _load_prompts(path: str, col: str, n: Optional[int]) -> List[str]:
    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. "
                             f"Available: {df.columns.tolist()}")
        prompts = df[col].dropna().astype(str).tolist()
    else:
        with open(path) as f:
            prompts = [l.strip() for l in f
                       if l.strip() and not l.startswith("#")]
    return prompts[:n] if n is not None else prompts


def run_batch(
    prompts_path: str,
    sv: Optional[SteeringVectors],
    schedule: Schedule,
    save_dir: str,
    col: str = "prompt",
    n: Optional[int] = None,
    concept: str = "nudity",
    seed: int = 42,
    model_id: str = "CompVis/stable-diffusion-v1-4",
    device: str = "cuda",
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    height: int = 512,
    width: int = 512,
    beta: float = 2.0,
    n_components: int = 1,
    max_proj_ratio: Optional[float] = 0.5,
) -> None:
    """Run the 4-column comparison for every prompt in a CSV/TXT file."""
    prompts = _load_prompts(prompts_path, col, n)
    os.makedirs(save_dir, exist_ok=True)
    log.info("Schedule    : %s", schedule.summary())
    log.info("Processing %d prompts -> %s", len(prompts), save_dir)

    wrapper = MergedUnlearningPipeline.from_pretrained(
        model_id, sv, device=device, concept=concept, beta=beta,
        n_components=n_components, max_proj_ratio=max_proj_ratio)

    col_files = ["col1_unsafe.png", "col2_text.png",
                 "col3_ca.png", "col4_merged.png"]
    summary_rows: List[Image.Image] = []

    for idx, prompt in enumerate(prompts):
        folder = os.path.join(save_dir, f"prompt_{idx:03d}_{slug(prompt)}")
        out_path = os.path.join(folder, "comparison.png")
        if os.path.exists(out_path):
            log.info("[%d/%d] skip (exists): %s", idx + 1, len(prompts),
                     prompt[:60])
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
        log.info("  saved -> %s", out_path)

    wrapper.remove_hooks()

    if summary_rows:
        total_h = sum(r.height for r in summary_rows) + (len(summary_rows) - 1) * PAD
        canvas = Image.new("RGB", (summary_rows[0].width, total_h), (180, 180, 180))
        y = 0
        for r in summary_rows:
            canvas.paste(r, (0, y))
            y += r.height + PAD
        summary_path = os.path.join(save_dir, "summary.png")
        canvas.save(summary_path)
        log.info("Summary -> %s", summary_path)

    print("\n" + "=" * 60)
    print(f"Done. {len(prompts)} prompts processed.")
    print(f"Results : {save_dir}")
    print("=" * 60)


# =========================================================================
#  CLI
# =========================================================================

def _add_common(sp):
    sp.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    sp.add_argument("--device",   default="cuda")
    sp.add_argument("--concept",  default="van_gogh",
                    choices=list(CONCEPT_VOCAB),
                    help="Style to erase")
    sp.add_argument("--seed",     type=int,   default=42)
    sp.add_argument("--steps",    type=int,   default=50)
    sp.add_argument("--guidance", type=float, default=7.5)
    sp.add_argument("--img_size", type=int,   default=512)
    sp.add_argument("--beta",     type=float, default=2.0,
                    help="CA suppression strength (2.0 = Householder).")
    sp.add_argument("--n_components", type=int, default=1,
                    help="Number of text-space directions to remove.")
    sp.add_argument("--max_proj_ratio", type=float, default=0.5,
                    help="Cap on per-token norm removed in text space "
                         "(use -1 to disable).")
    # schedule
    sp.add_argument("--schedule", default="front",
                    choices=["always", "front", "band"])
    sp.add_argument("--tau_lo",    type=float, default=0.0)
    sp.add_argument("--tau_hi",    type=float, default=1.0)
    sp.add_argument("--tau_relax", type=float, default=0.2)
    sp.add_argument("--w_min",     type=float, default=0.5)


def _schedule_from_args(a) -> Schedule:
    return make_schedule(mode=a.schedule, tau_lo=a.tau_lo, tau_hi=a.tau_hi,
                         tau_relax=a.tau_relax, w_min=a.w_min)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="TECA Style Erasure: training-free artistic style removal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    bv = sub.add_parser("build-vectors", help="Precompute CA steering vectors.")
    bv.add_argument("--model_id",  default="CompVis/stable-diffusion-v1-4")
    bv.add_argument("--concept",   default="van_gogh",
                    choices=list(CONCEPT_VOCAB),
                    help="Style to erase")
    bv.add_argument("--num_pairs", type=int, default=50)
    bv.add_argument("--save_path", required=True)
    bv.add_argument("--device",    default="cuda")
    bv.add_argument("--steps",     type=int, default=50)
    bv.add_argument("--seed",      type=int, default=0)

    sg = sub.add_parser("single", help="4-column comparison for one prompt.")
    sg.add_argument("--prompt",    required=True)
    sg.add_argument("--sv_path",   required=True)
    sg.add_argument("--save_path", default="merged_comparison.png")
    sg.add_argument("--num_pairs", type=int, default=50)
    _add_common(sg)

    bt = sub.add_parser("batch", help="4-column comparison over a CSV/TXT file.")
    bt.add_argument("--prompts",   default="dataset/nudity.csv")
    bt.add_argument("--sv_path",   required=True)
    bt.add_argument("--save_dir",  default="results/merged")
    bt.add_argument("--col",       default="prompt")
    bt.add_argument("--n",         type=int, default=10)
    bt.add_argument("--num_pairs", type=int, default=50)
    _add_common(bt)

    args = p.parse_args()

    if args.cmd == "build-vectors":
        build_steering_vectors(
            model_id=args.model_id, concept=args.concept,
            num_pairs=args.num_pairs, save_path=args.save_path,
            device=args.device, num_steps=args.steps, seed=args.seed)

    elif args.cmd == "single":
        mpr = None if args.max_proj_ratio < 0 else args.max_proj_ratio
        sv = build_steering_vectors(
            args.model_id, args.concept, args.num_pairs, args.sv_path,
            args.device, args.steps, seed=0)
        wrapper = MergedUnlearningPipeline.from_pretrained(
            args.model_id, sv, device=args.device, concept=args.concept,
            beta=args.beta, n_components=args.n_components, max_proj_ratio=mpr)
        schedule = _schedule_from_args(args)
        log.info("Schedule: %s", schedule.summary())
        images = compare_prompt(
            wrapper, args.prompt, schedule, seed=args.seed,
            guidance_scale=args.guidance, num_steps=args.steps,
            height=args.img_size, width=args.img_size)
        row = make_row(images, prompt_text=args.prompt)
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or ".",
                    exist_ok=True)
        row.save(args.save_path)
        wrapper.remove_hooks()
        print(f"Saved -> {args.save_path}")

    elif args.cmd == "batch":
        mpr = None if args.max_proj_ratio < 0 else args.max_proj_ratio
        sv = build_steering_vectors(
            args.model_id, args.concept, args.num_pairs, args.sv_path,
            args.device, args.steps, seed=0)
        run_batch(
            prompts_path=args.prompts, sv=sv, schedule=_schedule_from_args(args),
            save_dir=args.save_dir, col=args.col, n=args.n,
            concept=args.concept, seed=args.seed, model_id=args.model_id,
            device=args.device, guidance_scale=args.guidance,
            num_steps=args.steps, height=args.img_size, width=args.img_size,
            beta=args.beta, n_components=args.n_components, max_proj_ratio=mpr)