"""
casteer_pci.py  —  Temporally-Gated CASteer + PCI
==================================================

What this does
--------------
Combines two training-free concept-erasure ideas into one method:

  1. CASteer  (Gaintseva et al., 2025)
     Builds steering vectors from cross-attention *activations* collected
     during paired image generations (concept present vs absent).
     At inference it subtracts those vectors from every CA-layer output,
     patch by patch, using a dot-product gate so only patches that
     actually contain the concept are affected.

  2. Temporally-precise PCI  (your ICLR 2026 paper)
     The TemporalSchedule tells you *when* during denoising the concept is
     being committed (the τ window with the steepest CIS slope).
     Outside that window the concept either hasn't formed yet (τ high,
     pure noise) or is already baked in (τ low, refinement only).

The hybrid equation applied at every CA-layer output, per patch k:

    alpha_k = max(β · w(τ) · <v, ca_k>, 0)
    ca_k_new = ca_k - alpha_k · v

Where:
  v     = steering vector for this CA layer (from paired generations)
  τ     = normalised denoising timestep (1 = pure noise, 0 = clean image)
  w(τ)  = TemporalSchedule weight ∈ [0, 1]  ← PCI contribution
  β     = base suppression strength (2.0 = Householder reflection)

Quickstart
----------
    # Step 1: build steering vectors once
    python casteer_pci.py build-vectors \\
        --model_id CompVis/stable-diffusion-v1-4 \\
        --concept  nudity \\
        --num_pairs 50 \\
        --save_path steering_vectors/nudity_sd14.pt \\
        --device    cuda

    # Step 2 (recommended): compute a real CIS curve for the concept,
    # then derive a data-driven temporal schedule from it
    python casteer_pci.py build-schedule \\
        --model_id  CompVis/stable-diffusion-v1-4 \\
        --concept   nudity \\
        --save_path schedules/nudity_band.pkl \\
        --device    cuda

    # Step 3a: single prompt comparison (4-column grid)
    python casteer_pci.py single \\
        --prompt    "a beautiful woman on the beach" \\
        --sv_path   steering_vectors/nudity_sd14.pt \\
        --cis_path  schedules/nudity_band.pkl \\
        --save_path results/test.png \\
        --device    cuda

    # Step 3b: batch over a CSV
    python casteer_pci.py batch \\
        --prompts   examples/prompts/unsafe-prompts.csv \\
        --col       prompt \\
        --sv_path   steering_vectors/nudity_sd14.pt \\
        --cis_path  schedules/nudity_band.pkl \\
        --save_dir  results/run1 \\
        --device    cuda

    # Without a CIS curve: fall back to a uniform window
    python casteer_pci.py single \\
        --prompt    "a beautiful woman on the beach" \\
        --sv_path   steering_vectors/nudity_sd14.pt \\
        --tau_lo    0.3 --tau_hi 0.8 \\
        --save_path results/test.png \\
        --device    cuda

Requirements
------------
    pip install diffusers transformers torch Pillow tqdm pandas
    # pci_unlearning must be on PYTHONPATH:
    # export PYTHONPATH=/path/to/CASteer/examples:$PYTHONPATH

Layout
------
    CASteer/
    ├── casteer_pci.py          ← this file
    ├── examples/
    │   └── pci_unlearning/     ← your package (on PYTHONPATH)
    └── steering_vectors/       ← .pt files go here
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import pickle
import re
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("casteer_pci")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Type alias
# SteeringVectors: step_idx -> place -> list of Tensor[dim]
# ---------------------------------------------------------------------------

SteeringVectors = Dict[int, Dict[str, List[torch.Tensor]]]

# ---------------------------------------------------------------------------
# Concept prompt pairs for steering vector collection
# ---------------------------------------------------------------------------

PAIRED_PROMPTS: Dict[str, List[Tuple[str, str]]] = {
    "nudity": [
        ("a nude woman lying on a bed",          "a woman lying on a bed"),
        ("a naked man in the shower",             "a man in the shower"),
        ("an explicit erotic photo",              "a portrait photo"),
        ("a topless person on the beach",         "a person on the beach"),
        ("naked body in artistic photography",    "a person in artistic photography"),
        ("nude figure in a painting",             "a figure in a painting"),
        ("erotic sculpture with bare skin",       "a sculpture"),
        ("a nude model posing",                   "a model posing"),
        ("bare skin and nudity",                  "a photo of a person"),
        ("a naked couple embracing",              "a couple embracing"),
    ],
    "violence": [
        ("a brutal murder scene with blood",      "a scene outdoors"),
        ("graphic violence and gore",             "a dramatic photo"),
        ("a man being stabbed with a knife",      "a man standing outside"),
        ("soldiers in a bloody battle",           "soldiers in a field"),
        ("a violent fight with injuries",         "two people standing"),
    ],
}

_NUDITY_POS = [
    "a nude {subj} in {setting}",
    "naked {subj} in {setting}",
    "explicit nudity, {subj} in {setting}",
    "a topless {subj} in {setting}",
    "bare skin, {subj} in {setting}",
]
_NUDITY_NEG = [
    "a {subj} in {setting}",
    "a clothed {subj} in {setting}",
    "a fully dressed {subj} in {setting}",
    "a {subj} wearing clothes in {setting}",
    "a {subj} standing in {setting}",
]
_SUBJECTS = ["woman", "man", "person", "model", "figure", "couple"]
_SETTINGS = [
    "a bedroom", "a beach", "a studio", "a garden",
    "an artistic photo", "a painting", "a swimming pool",
    "a forest", "a bathroom", "a living room",
]


def _generate_nudity_pairs(num_pairs: int) -> List[Tuple[str, str]]:
    import itertools
    pairs: List[Tuple[str, str]] = list(PAIRED_PROMPTS["nudity"])
    combos = list(itertools.product(_SUBJECTS, _SETTINGS))
    tpl_idx = 0
    for subj, setting in itertools.cycle(combos):
        if len(pairs) >= num_pairs:
            break
        pos = _NUDITY_POS[tpl_idx % len(_NUDITY_POS)].format(subj=subj, setting=setting)
        neg = _NUDITY_NEG[tpl_idx % len(_NUDITY_NEG)].format(subj=subj, setting=setting)
        tpl_idx += 1
        if (pos, neg) not in pairs:
            pairs.append((pos, neg))
    return pairs[:num_pairs]


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def _load_pt(path: str) -> Any:
    """torch.load with weights_only=False (silences PyTorch 2.6 warning)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_schedule(cis_path: Optional[str], tau_lo: float, tau_hi: float,
                   schedule_mode: str = "early_lock"):
    """
    Return a TemporalSchedule.

    If cis_path is provided, load the pickled CISResult and derive a
    data-driven band schedule from it.  Otherwise fall back to a uniform
    window [tau_lo, tau_hi].
    """
    from pci_unlearning.schedule import derive_schedule_from_cis, make_uniform_schedule

    if cis_path and os.path.exists(cis_path):
        log.info("Loading CIS result from %s", cis_path)
        with open(cis_path, "rb") as f:
            cis_result = pickle.load(f)
        schedule = derive_schedule_from_cis(cis_result, mode=schedule_mode)
        log.info("Schedule: %s", schedule.summary())
        return schedule

    if cis_path:
        log.warning("CIS path '%s' not found — falling back to uniform [%.2f, %.2f]",
                    cis_path, tau_lo, tau_hi)

    schedule = make_uniform_schedule(tau_lo=tau_lo, tau_hi=tau_hi)
    log.info("Using uniform schedule [%.2f, %.2f]", tau_lo, tau_hi)
    return schedule


# ---------------------------------------------------------------------------
# Hook: collect CA outputs during steering-vector construction
# ---------------------------------------------------------------------------

class _CACollector:
    """
    Forward hook that records mean-pooled CA outputs per layer per step.

    Step counting fix: the hook fires DURING the forward pass of step N.
    We initialise _current_step=0 before generation starts and call
    advance_step() in callback_on_step_end (which fires AFTER step N
    completes).  So during step N's forward pass, _current_step == N.
    """

    def __init__(self) -> None:
        self._current_step: int = 0
        self._layer_counts: Dict[str, int] = defaultdict(int)
        # step -> place -> list[mean-pooled Tensor]
        self.data: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list))
        self._hooks: List = []

    def reset_for_prompt(self) -> None:
        """Call before running each new prompt."""
        self._current_step = 0
        self._layer_counts = defaultdict(int)
        self.data = defaultdict(lambda: defaultdict(list))

    def advance_step(self) -> None:
        """Call in callback_on_step_end — moves to next step index."""
        self._current_step += 1
        self._layer_counts = defaultdict(int)

    def _hook_fn(self, place: str):
        def fn(module, inp, out):
            x = out.detach().float()
            if x.dim() == 4:
                x = x.reshape(x.shape[0], x.shape[1], -1)
            # CFG doubles the batch: take only conditioned half
            if x.shape[0] > 1:
                x = x[x.shape[0] // 2:]
            vec = x.mean(dim=(0, 1))   # mean over batch and sequence → [dim]
            self.data[self._current_step][place].append(vec.cpu())
            self._layer_counts[place] += 1
        return fn

    def register(self, unet) -> None:
        for name, module in unet.named_modules():
            if module.__class__.__name__ == "BasicTransformerBlock":
                if hasattr(module, "attn2") and module.attn2 is not None:
                    place = ("down" if "down" in name
                             else "up" if "up" in name else "mid")
                    h = module.attn2.register_forward_hook(self._hook_fn(place))
                    self._hooks.append(h)
        log.info("Collector hooks registered on %d CA layers", len(self._hooks))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Build steering vectors
# ---------------------------------------------------------------------------

def build_steering_vectors(
    model_id: str,
    concept: str = "nudity",
    num_pairs: int = 50,
    save_path: Optional[str] = None,
    device: str = "cuda",
    num_steps: int = 50,
    base_seed: int = 0,
) -> SteeringVectors:
    """
    Run `num_pairs` paired generations and compute per-layer, per-step
    steering vectors:

        v[step][place][layer_idx] = normalise(mean(CA_pos) - mean(CA_neg))

    Each prompt pair uses a unique seed (base_seed + pair_index) so
    the activations are not correlated through shared noise.

    If save_path already exists the file is loaded directly.
    """
    if save_path and os.path.exists(save_path):
        log.info("Loading cached steering vectors from %s", save_path)
        sv = _load_pt(save_path)
        log.info("Loaded: %d steps", len(sv))
        return sv

    log.info("Building steering vectors — concept='%s', num_pairs=%d", concept, num_pairs)

    sched = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=sched,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    if concept == "nudity":
        pairs = _generate_nudity_pairs(num_pairs)
    elif concept in PAIRED_PROMPTS:
        base = PAIRED_PROMPTS[concept]
        pairs = (base * ((num_pairs // len(base)) + 1))[:num_pairs]
    else:
        raise ValueError(f"Unknown concept '{concept}'. Supported: {list(PAIRED_PROMPTS)}")

    log.info("Using %d prompt pairs", len(pairs))

    def _collect_means(prompts: List[str]) -> SteeringVectors:
        """Run pipeline on each prompt, accumulate mean CA activations."""
        collector = _CACollector()
        collector.register(pipe.unet)

        # Running sum and count: step -> place -> list[Tensor / int]
        sum_data: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list))
        counts: Dict[int, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list))

        for i, prompt in enumerate(prompts):
            prompt_seed = base_seed + i   # unique seed per prompt
            log.info("  [%d/%d] seed=%d  %s", i + 1, len(prompts), prompt_seed, prompt[:70])
            collector.reset_for_prompt()

            def _cb(pipe_, step, timestep, kwargs):
                collector.advance_step()
                return kwargs

            pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                generator=torch.Generator(device=device).manual_seed(prompt_seed),
                callback_on_step_end=_cb,
                output_type="latent",
            )

            for step, places in collector.data.items():
                for place, layer_vecs in places.items():
                    for li, vec in enumerate(layer_vecs):
                        if len(sum_data[step][place]) <= li:
                            sum_data[step][place].append(vec.clone())
                            counts[step][place].append(1)
                        else:
                            sum_data[step][place][li] = sum_data[step][place][li] + vec
                            counts[step][place][li] += 1

        collector.remove()

        means: SteeringVectors = {}
        for step in sum_data:
            means[step] = {}
            for place in sum_data[step]:
                means[step][place] = [
                    sum_data[step][place][li] / counts[step][place][li]
                    for li in range(len(sum_data[step][place]))
                ]
        return means

    pos_prompts = [p for p, _ in pairs]
    neg_prompts = [n for _, n in pairs]

    log.info("Collecting POSITIVE activations …")
    pos_means = _collect_means(pos_prompts)

    log.info("Collecting NEGATIVE activations …")
    neg_means = _collect_means(neg_prompts)

    # steering vector = normalise(pos_mean - neg_mean) per layer per step
    sv: SteeringVectors = {}
    for step in pos_means:
        if step not in neg_means:
            continue
        sv[step] = {}
        for place in pos_means[step]:
            if place not in neg_means[step]:
                continue
            sv[step][place] = []
            for li in range(len(pos_means[step][place])):
                if li >= len(neg_means[step][place]):
                    break
                diff = pos_means[step][place][li] - neg_means[step][place][li]
                norm = diff.norm().clamp(min=1e-8)
                sv[step][place].append((diff / norm).cpu())

    log.info("Steering vectors built: %d steps × %d places",
             len(sv), len(next(iter(sv.values()), {})))

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(sv, save_path)
        log.info("Saved → %s", save_path)

    return sv


# ---------------------------------------------------------------------------
# Build CIS schedule  (data-driven temporal window)
# ---------------------------------------------------------------------------

def build_cis_schedule(
    model_id: str,
    concept: str = "nudity",
    save_path: Optional[str] = None,
    device: str = "cuda",
    n_seeds: int = 30,
    num_steps: int = 50,
    schedule_mode: str = "band",
) -> Any:
    """
    Compute a CIS curve for `concept` and derive a TemporalSchedule from it.

    This runs n_seeds PCI switch experiments using pci_unlearning.compute_cis_curve,
    fits the curve, finds tau_50 / tau_70, and returns a TemporalSchedule.

    The schedule (and the underlying CISResult) are pickled to save_path so
    you can load them later with --cis_path.

    Args:
        model_id:      SD model.
        concept:       Concept key ("nudity", "violence").
        save_path:     Where to pickle the CISResult (loaded by --cis_path).
        device:        CUDA device.
        n_seeds:       Number of seeds for CIS curve estimation.
        num_steps:     Denoising steps.
        schedule_mode: "band", "slope_weighted", or "early_lock".

    Returns:
        TemporalSchedule
    """
    from pci_unlearning import (
        auto_adapter,
        compute_cis_curve, derive_schedule_from_cis,
        get_scorer,
    )

    log.info("Building CIS curve for concept='%s'", concept)

    sched = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=sched,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)

    adapter = auto_adapter(pipe, num_inference_steps=num_steps, device=device)

    concept_pairs = {
        "nudity":   ("a photo of a person", "a photo of a nude person"),
        "violence": ("a photo of a scene",  "a photo of a violent scene with blood"),
    }
    if concept not in concept_pairs:
        raise ValueError(f"No CIS prompt pair defined for concept '{concept}'. "
                         f"Available: {list(concept_pairs)}")

    base_prompt, concept_prompt = concept_pairs[concept]
    scorer = get_scorer("qwen")

    log.info("Running CIS curve: base='%s' -> concept='%s'", base_prompt, concept_prompt)
    cis_result = compute_cis_curve(
        adapter=adapter,
        base_prompt=base_prompt,
        concept_prompt=concept_prompt,
        concept=concept,
        scorer=scorer,
        num_seeds=n_seeds,
    )

    log.info("CIS result: tau_50=%.3f  tau_70=%.3f  bandwidth=%.3f",
             cis_result.tau_50 or -1,
             cis_result.tau_70 or -1,
             cis_result.bandwidth or -1)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(cis_result, f)
        log.info("CIS result saved → %s", save_path)

    schedule = derive_schedule_from_cis(cis_result, mode=schedule_mode)
    log.info("Schedule: %s", schedule.summary())

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return schedule



# ---------------------------------------------------------------------------
# Build CDS schedule  (deletion-direction temporal window)
# ---------------------------------------------------------------------------

def build_cds_schedule(
    model_id: str,
    concept: str = "nudity",
    save_path: Optional[str] = None,
    device: str = "cuda",
    n_seeds: int = 30,
    num_steps: int = 50,
    schedule_mode: str = "early_lock",
) -> Any:
    """
    Compute a CDS (Concept Deletion Success) curve and derive a TemporalSchedule.

    CDS measures: at what τ is it still possible to remove a concept by switching
    to the safe prompt?  High CDS at τ = concept not yet locked in.
    CDS drops to zero at the lock-in point — that is the latest step where
    suppression is still tractable.

    For unlearning, the correct intervention window is everything ABOVE the
    lock-in τ (i.e. early_lock mode), which is why that is the default here.

    The CISResult dataclass is reused to store the CDS curve — the semantics
    are inverted (high = successful deletion = concept absent) but the
    structure is identical.  Pass the saved .pkl to --cis_path in batch/single.

    Args:
        model_id:      SD model.
        concept:       Concept key ("nudity", "violence").
        save_path:     Where to pickle the CISResult for later use.
        device:        CUDA device.
        n_seeds:       Seeds to average over.
        num_steps:     Denoising steps.
        schedule_mode: How to derive schedule from curve.
                       "early_lock" recommended for deletion tasks —
                       activates for all τ ≥ tau_70 (before lock-in).
                       "band" also valid if you want a narrow window.

    Returns:
        TemporalSchedule
    """
    from pci_unlearning import (
        auto_adapter,
        compute_cds_curve, derive_schedule_from_cis,
        get_scorer,
    )

    log.info("Building CDS curve for concept='%s'", concept)

    sched = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=sched,
        torch_dtype=torch.float32,   # float32 required by UNetAdapter
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    adapter = auto_adapter(pipe, num_inference_steps=num_steps, device=device)

    # note: concept_prompt and base_prompt are SWAPPED vs CIS —
    # we start with the concept and measure how well switching to base removes it
    concept_pairs = {
        "nudity":   ("a photo of a nude person",             "a photo of a person"),
        "violence": ("a photo of a violent scene with blood","a photo of a scene"),
    }
    if concept not in concept_pairs:
        raise ValueError(f"No CDS prompt pair for concept '{concept}'. "
                         f"Available: {list(concept_pairs)}")

    concept_prompt, base_prompt = concept_pairs[concept]
    scorer = get_scorer("qwen")

    log.info("Running CDS curve: concept='%s' -> base='%s'", concept_prompt, base_prompt)
    cds_result = compute_cds_curve(
        adapter        = adapter,
        concept_prompt = concept_prompt,
        base_prompt    = base_prompt,
        concept        = concept,
        scorer         = scorer,
        num_seeds      = n_seeds,
    )

    log.info("CDS result: tau_50=%.3f  tau_70=%.3f  bandwidth=%.3f",
             cds_result.tau_50 or -1,
             cds_result.tau_70 or -1,
             cds_result.bandwidth or -1)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(cds_result, f)
        log.info("CDS result saved → %s", save_path)

    schedule = derive_schedule_from_cis(cds_result, mode=schedule_mode)
    log.info("Schedule: %s", schedule.summary())

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return schedule


# ---------------------------------------------------------------------------
# Hook: apply steering during inference
# ---------------------------------------------------------------------------

class _CASteerHook:
    """
    Forward hook that applies temporally-gated CASteer steering.

    Per CA-layer output, per patch k:
        alpha_k = max(β · w(τ) · <v, ca_k>, 0)
        ca_k_new = ca_k - alpha_k · v

    Call set_step(step_idx, weight) before each denoising step.
    """

    def __init__(
        self,
        sv: SteeringVectors,
        beta: float = 2.0,
        intermediate_clipping: bool = True,
    ) -> None:
        self.sv = sv
        self.beta = beta
        self.intermediate_clipping = intermediate_clipping
        self._hooks: List = []
        self._current_step: int = 0
        self._schedule_weight: float = 1.0
        self._layer_idx: Dict[str, int] = defaultdict(int)
        # pre-sort available step keys for fast nearest-step lookup
        self._sorted_steps: List[int] = sorted(sv.keys())

    def set_step(self, step_idx: int, weight: float) -> None:
        self._current_step = step_idx
        self._schedule_weight = weight
        self._layer_idx = defaultdict(int)

    def _nearest_step(self, step: int) -> int:
        """Return the closest available step key."""
        if not self._sorted_steps:
            return step
        return min(self._sorted_steps, key=lambda s: abs(s - step))

    def _hook_fn(self, place: str):
        def fn(module, inp, out):
            w = self._schedule_weight
            if w <= 0.0:
                return out

            step = self._current_step
            li = self._layer_idx[place]
            self._layer_idx[place] += 1

            # look up steering vector; fall back to nearest step if needed
            try:
                v = self.sv[step][place][li]
            except (KeyError, IndexError):
                nearest = self._nearest_step(step)
                try:
                    v = self.sv[nearest][place][li]
                except (KeyError, IndexError):
                    return out

            v = v.to(device=out.device, dtype=out.dtype)   # [dim]

            x = out
            orig_shape = x.shape
            if x.dim() == 4:
                B, S, H, D = x.shape
                x = x.reshape(B, S, H * D)

            # operate only on conditioned half (CFG)
            if x.shape[0] > 1:
                cond = x[x.shape[0] // 2:].clone()
            else:
                cond = x.clone()

            # per-patch dot product with steering vector: [seq]
            dots = (cond[0] @ v)

            if self.intermediate_clipping:
                alpha = torch.clamp(self.beta * w * dots, min=0.0)
            else:
                alpha = self.beta * w * dots

            cond[0] = cond[0] - alpha.unsqueeze(-1) * v

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
                if hasattr(module, "attn2") and module.attn2 is not None:
                    place = ("down" if "down" in name
                             else "up" if "up" in name else "mid")
                    h = module.attn2.register_forward_hook(self._hook_fn(place))
                    self._hooks.append(h)
        log.info("Steering hooks registered on %d CA layers", len(self._hooks))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

class TemporalCASteerPipeline:
    """
    Wraps a single loaded StableDiffusionPipeline with temporally-gated
    CASteer steering.  Load the pipeline once and reuse across prompts.
    """

    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        sv: SteeringVectors,
        beta: float = 2.0,
        intermediate_clipping: bool = True,
    ) -> None:
        self.pipe = pipe
        self.sv = sv
        self._hook = _CASteerHook(sv, beta, intermediate_clipping)
        self._hook.register(pipe.unet)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        sv: SteeringVectors,
        device: str = "cuda",
        beta: float = 2.0,
    ) -> "TemporalCASteerPipeline":
        sched = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=sched,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe, sv, beta=beta)

    # ---- internal helpers ----

    def _encode_prompt(self, prompt: str, negative_prompt: str = ""):
        tok = self.pipe.tokenizer
        enc = self.pipe.text_encoder
        device = self.pipe.device

        def _enc(text):
            t = tok(text, padding="max_length",
                    max_length=tok.model_max_length,
                    truncation=True, return_tensors="pt")
            with torch.no_grad():
                return enc(t.input_ids.to(device))[0]

        return _enc(negative_prompt), _enc(prompt)

    def _init_latents(self, seed: int, height: int, width: int,
                      num_steps: int) -> torch.Tensor:
        pipe = self.pipe
        pipe.scheduler.set_timesteps(num_steps, device=pipe.device)
        shape = (
            1,
            pipe.unet.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
        gen = torch.Generator(device=pipe.device).manual_seed(seed)
        latents = torch.randn(shape, generator=gen, device=pipe.device,
                              dtype=torch.float16)
        return latents * pipe.scheduler.init_noise_sigma

    def _denoise_step(self, latent, emb_pair, native_t,
                      guidance_scale: float = 7.5):
        uncond, cond = emb_pair
        combined = torch.cat([uncond, cond])
        lat2 = torch.cat([latent] * 2)
        lat2 = self.pipe.scheduler.scale_model_input(lat2, native_t)
        with torch.no_grad():
            noise_pred = self.pipe.unet(
                lat2, native_t, encoder_hidden_states=combined
            ).sample
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

    def _tau_from_timestep(self, t) -> float:
        """Convert a native scheduler timestep to normalised τ ∈ [0, 1]."""
        return float(t.item()) / 1000.0

    # ---- public API ----

    def generate(
        self,
        prompt: str,
        schedule,
        seed: int = 42,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_steps: int = 50,
        height: int = 512,
        width: int = 512,
    ) -> Image.Image:
        """Generate with temporally-gated CASteer suppression."""
        pipe = self.pipe
        pipe.scheduler.set_timesteps(num_steps, device=pipe.device)

        emb = self._encode_prompt(prompt, negative_prompt)
        latent = self._init_latents(seed, height, width, num_steps)

        for step_idx, native_t in enumerate(pipe.scheduler.timesteps):
            tau = self._tau_from_timestep(native_t)
            w = schedule.weight(tau)
            self._hook.set_step(step_idx, w)
            latent = self._denoise_step(latent, emb, native_t, guidance_scale)

        return self._decode(latent)

    def generate_unsteered(
        self,
        prompt: str,
        seed: int = 42,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_steps: int = 50,
        height: int = 512,
        width: int = 512,
    ) -> Image.Image:
        """Generate without any steering (unsafe baseline)."""
        pipe = self.pipe
        pipe.scheduler.set_timesteps(num_steps, device=pipe.device)

        emb = self._encode_prompt(prompt, negative_prompt)
        latent = self._init_latents(seed, height, width, num_steps)

        for step_idx, native_t in enumerate(pipe.scheduler.timesteps):
            # weight=0 at every step — hook does nothing
            self._hook.set_step(step_idx, 0.0)
            latent = self._denoise_step(latent, emb, native_t, guidance_scale)

        return self._decode(latent)

    def remove_hooks(self) -> None:
        self._hook.remove()


# ---------------------------------------------------------------------------
# Vocab Ortho PCI column (col3) — reuses the already-loaded pipeline
# ---------------------------------------------------------------------------

def _load_fp32_pipe(model_id: str, device: str) -> StableDiffusionPipeline:
    """Load a float32 pipeline for pci_unlearning's UNetAdapter (which does not cast embeddings)."""
    sched = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=sched,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    log.info("Loaded fp32 pipeline for Vocab Ortho PCI")
    return pipe


def _gen_vocab_ortho_from_pipe(
    prompt: str,
    pipe: StableDiffusionPipeline,   # must be float32
    schedule,
    seed: int = 42,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    height: int = 512,
    width: int = 512,
) -> Image.Image:
    """Run pci_unlearning.run_orthogonal_vocab using a dedicated float32 pipeline."""
    from pci_unlearning import auto_adapter, PromptConditionedIntervention

    adapter = auto_adapter(pipe, num_inference_steps=num_steps,
                           device=str(pipe.device))
    pci = PromptConditionedIntervention(
        adapter, guidance_scale=guidance_scale,
        height=height, width=width,
    )
    return pci.run_orthogonal_vocab(
        prompt=prompt,
        schedule=schedule,
        seed=seed,
        concept="nudity",
        n_components=3,
        renormalize=True,
    )


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

LABEL_H = 56
FONT_SZ  = 14
PAD      = 6
COL_LABELS = [
    "1. Unsafe",
    "2. CASteer (all steps)",
    "3. Vocab Ortho PCI",
    "4. CASteer + Temporal PCI",
]


def _label_img(text: str, w: int, h: int) -> Image.Image:
    img  = Image.new("RGB", (w, h), (255, 255, 255))
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
        draw.text(((w - (bb[2] - bb[0])) // 2, y), line,
                  font=font, fill=(30, 30, 30))
        y += FONT_SZ + 3
    return img


def _make_grid(images: List[Image.Image], labels: List[str]) -> Image.Image:
    # normalise all images to the same size (first image's size)
    w, h = images[0].size
    images = [img.resize((w, h), Image.LANCZOS) for img in images]
    n = len(images)
    total_w = n * w + (n - 1) * PAD
    canvas = Image.new("RGB", (total_w, h + LABEL_H), (220, 220, 220))
    for i, (img, lbl) in enumerate(zip(images, labels)):
        x = i * (w + PAD)
        canvas.paste(img, (x, 0))
        canvas.paste(_label_img(lbl, w, LABEL_H), (x, h))
    return canvas


# ---------------------------------------------------------------------------
# compare_methods  — loads the model ONCE, runs all 4 columns
# ---------------------------------------------------------------------------

def compare_methods(
    prompt: str,
    sv: SteeringVectors,
    schedule,
    save_path: str = "casteer_pci_comparison.png",
    seed: int = 42,
    model_id: str = "CompVis/stable-diffusion-v1-4",
    device: str = "cuda",
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    height: int = 512,
    width: int = 512,
    beta: float = 2.0,
    include_vocab_ortho: bool = True,
    pipe_wrapper: Optional[TemporalCASteerPipeline] = None,
    fp32_pipe=None,
) -> Image.Image:
    """
    Generate a 4-column comparison image for one prompt.

    If pipe_wrapper is provided (batch mode) the model is NOT reloaded —
    the wrapper is reused across prompts for efficiency.

    Columns:
        1. Unsafe baseline
        2. CASteer all steps (no temporal gating)
        3. Vocab Ortho PCI  (text-embedding level)
        4. CASteer + Temporal PCI  (the hybrid)
    """
    from pci_unlearning.schedule import make_uniform_schedule

    own_wrapper = pipe_wrapper is None
    if own_wrapper:
        pipe_wrapper = TemporalCASteerPipeline.from_pretrained(
            model_id, sv, device=device, beta=beta)

    try:
        # col 1: unsafe
        log.info("[1/4] Unsafe …")
        img_unsafe = pipe_wrapper.generate_unsteered(
            prompt, seed=seed, guidance_scale=guidance_scale,
            num_steps=num_steps, height=height, width=width,
        )

        # col 2: CASteer full (all steps)
        log.info("[2/4] CASteer (all steps) …")
        full_schedule = make_uniform_schedule(tau_lo=0.0, tau_hi=1.0)
        img_casteer = pipe_wrapper.generate(
            prompt, full_schedule, seed=seed,
            guidance_scale=guidance_scale,
            num_steps=num_steps, height=height, width=width,
        )

        # col 3: Vocab Ortho PCI — reuse the loaded pipeline
        if include_vocab_ortho:
            log.info("[3/4] Vocab Ortho PCI …")
            img_vocab_ortho = _gen_vocab_ortho_from_pipe(
                prompt, fp32_pipe or pipe_wrapper.pipe, schedule,
                seed=seed, guidance_scale=guidance_scale,
                num_steps=num_steps, height=height, width=width,
            )
        else:
            log.info("[3/4] Skipping Vocab Ortho PCI")
            img_vocab_ortho = Image.new("RGB", (width, height), (200, 200, 200))

        # col 4: CASteer + Temporal PCI
        log.info("[4/4] CASteer + Temporal PCI …")
        img_hybrid = pipe_wrapper.generate(
            prompt, schedule, seed=seed,
            guidance_scale=guidance_scale,
            num_steps=num_steps, height=height, width=width,
        )

    finally:
        if own_wrapper:
            pipe_wrapper.remove_hooks()

    active_labels = COL_LABELS[:]
    if not include_vocab_ortho:
        active_labels[2] = "3. (skipped)"

    images = [img_unsafe, img_casteer, img_vocab_ortho, img_hybrid]
    grid = _make_grid(images, active_labels)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    grid.save(save_path)
    log.info("Saved → %s", save_path)
    return grid


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _slug(text: str, max_len: int = 30) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len]


def run_batch(
    prompts_path: str,
    sv: SteeringVectors,
    schedule,
    save_dir: str,
    col: str = "prompt",
    n: Optional[int] = None,
    seed: int = 42,
    model_id: str = "CompVis/stable-diffusion-v1-4",
    device: str = "cuda",
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    height: int = 512,
    width: int = 512,
    beta: float = 2.0,
    include_vocab_ortho: bool = True,
) -> None:
    """
    Run compare_methods for every prompt in a CSV or TXT file.

    The pipeline is loaded ONCE before the loop and reused across all
    prompts — no repeated model loading.

    Output structure:
        <save_dir>/
            prompt_000_<slug>/comparison.png
            prompt_001_<slug>/comparison.png
            ...
            summary.png
    """
    # load prompts
    if prompts_path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(prompts_path)
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in CSV. "
                f"Available columns: {df.columns.tolist()}")
        prompts = df[col].dropna().astype(str).tolist()
    else:
        with open(prompts_path) as f:
            prompts = [l.strip() for l in f
                       if l.strip() and not l.startswith("#")]

    if n is not None:
        prompts = prompts[:n]

    log.info("Processing %d prompts → %s", len(prompts), save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # load pipelines once for the whole batch
    log.info("Loading float16 pipeline for CASteer cols 1/2/4 …")
    pipe_wrapper = TemporalCASteerPipeline.from_pretrained(
        model_id, sv, device=device, beta=beta)

    log.info("Loading float32 pipeline for Vocab Ortho PCI col3 …")
    fp32_pipe = _load_fp32_pipe(model_id, device) if include_vocab_ortho else None

    summary_rows: List[Image.Image] = []

    try:
        for idx, prompt in enumerate(prompts):
            folder   = os.path.join(save_dir, f"prompt_{idx:03d}_{_slug(prompt)}")
            out_path = os.path.join(folder, "comparison.png")

            if os.path.exists(out_path):
                log.info("[%d/%d] skip (exists): %s",
                         idx + 1, len(prompts), prompt[:60])
                summary_rows.append(Image.open(out_path))
                continue

            log.info("─" * 60)
            log.info("[%d/%d] %s", idx + 1, len(prompts), prompt[:80])
            os.makedirs(folder, exist_ok=True)

            try:
                grid = compare_methods(
                    prompt=prompt,
                    sv=sv,
                    schedule=schedule,
                    save_path=out_path,
                    seed=seed,
                    model_id=model_id,
                    device=device,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    height=height,
                    width=width,
                    beta=beta,
                    include_vocab_ortho=include_vocab_ortho,
                    pipe_wrapper=pipe_wrapper,   # reuse — no reload
                    fp32_pipe=fp32_pipe,
                )
                summary_rows.append(grid)
            except Exception as e:
                log.error("Failed on prompt %d: %s — %s", idx, prompt[:60], e)
                # save an error placeholder so the batch can continue
                err_img = Image.new("RGB", (width * 4, height), (80, 0, 0))
                summary_rows.append(err_img)

    finally:
        pipe_wrapper.remove_hooks()

    # build summary grid
    if summary_rows:
        total_h = sum(r.height for r in summary_rows) + (len(summary_rows) - 1) * PAD
        canvas  = Image.new("RGB", (summary_rows[0].width, total_h), (180, 180, 180))
        y = 0
        for r in summary_rows:
            canvas.paste(r, (0, y))
            y += r.height + PAD
        summary_path = os.path.join(save_dir, "summary.png")
        canvas.save(summary_path)
        log.info("Summary → %s", summary_path)

    print("\n" + "=" * 60)
    print(f"Done.  {len(prompts)} prompts processed.")
    print(f"Results: {save_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CASteer + Temporal PCI — temporally-gated CA steering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- build-vectors ----
    bv = sub.add_parser("build-vectors",
                        help="Precompute steering vectors from paired generations.")
    bv.add_argument("--model_id",   default="CompVis/stable-diffusion-v1-4")
    bv.add_argument("--concept",    default="nudity")
    bv.add_argument("--num_pairs",  type=int, default=50)
    bv.add_argument("--save_path",  required=True)
    bv.add_argument("--device",     default="cuda")
    bv.add_argument("--steps",      type=int, default=50)
    bv.add_argument("--seed",       type=int, default=0,
                    help="Base seed; each prompt pair uses seed+i")

    # ---- build-schedule ----
    bs = sub.add_parser("build-schedule",
                        help="Compute a CIS curve and derive a data-driven schedule.")
    bs.add_argument("--model_id",   default="CompVis/stable-diffusion-v1-4")
    bs.add_argument("--concept",    default="nudity")
    bs.add_argument("--save_path",  required=True,
                    help="Output .pkl path for the CISResult")
    bs.add_argument("--device",     default="cuda")
    bs.add_argument("--n_seeds",    type=int, default=30)
    bs.add_argument("--steps",      type=int, default=50)
    bs.add_argument("--mode",       default="band",
                    choices=["band", "slope_weighted", "early_lock"])


    # ---- build-cds ----
    bc = sub.add_parser("build-cds",
                        help="Compute a CDS curve — the correct schedule for deletion/unlearning.")
    bc.add_argument("--model_id",   default="CompVis/stable-diffusion-v1-4")
    bc.add_argument("--concept",    default="nudity")
    bc.add_argument("--save_path",  required=True,
                    help="Output .pkl path for the CDS result")
    bc.add_argument("--device",     default="cuda")
    bc.add_argument("--n_seeds",    type=int, default=30)
    bc.add_argument("--steps",      type=int, default=50)
    bc.add_argument("--mode",       default="early_lock",
                    choices=["band", "slope_weighted", "early_lock"],
                    help="Schedule mode (early_lock recommended for deletion)")

    # ---- single ----
    sg = sub.add_parser("single",
                        help="4-column comparison for one prompt.")
    sg.add_argument("--prompt",     required=True)
    sg.add_argument("--sv_path",    required=True)
    sg.add_argument("--save_path",  default="casteer_pci_comparison.png")
    sg.add_argument("--cis_path",   default=None,
                    help="Pickled CISResult from build-schedule (recommended). "
                         "Falls back to --tau_lo/--tau_hi if not set.")
    sg.add_argument("--tau_lo",     type=float, default=0.3,
                    help="Fallback window start (used only if --cis_path not set)")
    sg.add_argument("--tau_hi",     type=float, default=0.8,
                    help="Fallback window end")
    sg.add_argument("--mode",       default="early_lock",
                    choices=["band", "slope_weighted", "early_lock"],
                    help="How to derive schedule from CIS/CDS pkl")
    sg.add_argument("--model_id",   default="CompVis/stable-diffusion-v1-4")
    sg.add_argument("--device",     default="cuda")
    sg.add_argument("--seed",       type=int,   default=42)
    sg.add_argument("--steps",      type=int,   default=50)
    sg.add_argument("--guidance",   type=float, default=7.5)
    sg.add_argument("--img_size",   type=int,   default=512)
    sg.add_argument("--beta",       type=float, default=2.0)
    sg.add_argument("--no_vocab_ortho", action="store_true")

    # ---- batch ----
    bt = sub.add_parser("batch",
                        help="Run compare_methods over a CSV/TXT prompt file.")
    bt.add_argument("--prompts",    required=True)
    bt.add_argument("--sv_path",    required=True)
    bt.add_argument("--save_dir",   required=True)
    bt.add_argument("--col",        default="prompt")
    bt.add_argument("--n",          type=int, default=None)
    bt.add_argument("--cis_path",   default=None)
    bt.add_argument("--tau_lo",     type=float, default=0.3)
    bt.add_argument("--tau_hi",     type=float, default=0.8)
    bt.add_argument("--mode",       default="early_lock",
                    choices=["band", "slope_weighted", "early_lock"],
                    help="How to derive schedule from CIS/CDS pkl")
    bt.add_argument("--model_id",   default="CompVis/stable-diffusion-v1-4")
    bt.add_argument("--device",     default="cuda")
    bt.add_argument("--seed",       type=int,   default=42)
    bt.add_argument("--steps",      type=int,   default=50)
    bt.add_argument("--guidance",   type=float, default=7.5)
    bt.add_argument("--img_size",   type=int,   default=512)
    bt.add_argument("--beta",       type=float, default=2.0)
    bt.add_argument("--no_vocab_ortho", action="store_true")

    args = p.parse_args()

    # ------------------------------------------------------------------ #

    if args.cmd == "build-vectors":
        build_steering_vectors(
            model_id  = args.model_id,
            concept   = args.concept,
            num_pairs = args.num_pairs,
            save_path = args.save_path,
            device    = args.device,
            num_steps = args.steps,
            base_seed = args.seed,
        )

    elif args.cmd == "build-cds":
        build_cds_schedule(
            model_id      = args.model_id,
            concept       = args.concept,
            save_path     = args.save_path,
            device        = args.device,
            n_seeds       = args.n_seeds,
            num_steps     = args.steps,
            schedule_mode = args.mode,
        )

    elif args.cmd == "build-schedule":
        build_cis_schedule(
            model_id      = args.model_id,
            concept       = args.concept,
            save_path     = args.save_path,
            device        = args.device,
            n_seeds       = args.n_seeds,
            num_steps     = args.steps,
            schedule_mode = args.mode,
        )

    elif args.cmd == "single":
        sv       = _load_pt(args.sv_path)
        schedule = _load_schedule(args.cis_path, args.tau_lo, args.tau_hi,
                                  schedule_mode=args.mode)
        compare_methods(
            prompt              = args.prompt,
            sv                  = sv,
            schedule            = schedule,
            save_path           = args.save_path,
            seed                = args.seed,
            model_id            = args.model_id,
            device              = args.device,
            guidance_scale      = args.guidance,
            num_steps           = args.steps,
            height              = args.img_size,
            width               = args.img_size,
            beta                = args.beta,
            include_vocab_ortho = not args.no_vocab_ortho,
        )

    elif args.cmd == "batch":
        sv       = _load_pt(args.sv_path)
        schedule = _load_schedule(args.cis_path, args.tau_lo, args.tau_hi,
                                  schedule_mode=args.mode)
        run_batch(
            prompts_path        = args.prompts,
            sv                  = sv,
            schedule            = schedule,
            save_dir            = args.save_dir,
            col                 = args.col,
            n                   = args.n,
            seed                = args.seed,
            model_id            = args.model_id,
            device              = args.device,
            guidance_scale      = args.guidance,
            num_steps           = args.steps,
            height              = args.img_size,
            width               = args.img_size,
            beta                = args.beta,
            include_vocab_ortho = not args.no_vocab_ortho,
        )