# export PYTHONPATH=/storage/pk_mridul/CASteer/examples:$PYTHONPATH
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
     pure noise) or is already baked in (τ low, refinement only) — so
     intervening there wastes effort and hurts quality.

The hybrid: CASteer's patch-level CA steering, but gated by the PCI
temporal schedule.  The hook fires only when schedule.weight(τ) > 0 and
scales the steering strength by that weight.

  ca_new[patch] = ca[patch] - β · w(τ) · max(<v, ca[patch]>, 0) · v

Where:
  v     = steering vector for this CA layer (from paired generations)
  τ     = normalised denoising timestep (1 = pure noise, 0 = clean)
  w(τ)  = schedule weight ∈ [0, 1]
  β     = base suppression strength (default 2.0 = Householder reflection)

Usage (minimal)
---------------
    from casteer_pci import build_steering_vectors, TemporalCASteerPipeline

    # Step 1  — build steering vectors once (saves/loads a .pt file)
    sv = build_steering_vectors(
        model_id  = "CompVis/stable-diffusion-v1-4",
        concept   = "nudity",
        num_pairs = 50,            # more pairs = cleaner direction
        save_path = "sv_nudity.pt",
        device    = "cuda",
    )

    # Step 2  — create the pipeline wrapper
    pipe_wrapper = TemporalCASteerPipeline.from_pretrained(
        model_id      = "CompVis/stable-diffusion-v1-4",
        steering_vecs = sv,
        device        = "cuda",
    )

    # Step 3  — choose a schedule
    from pci_unlearning import make_uniform_schedule
    schedule = make_uniform_schedule(tau_lo=0.3, tau_hi=0.8)
    # or: derive_schedule_from_cis(cis_result, mode="band")

    # Step 4  — generate
    image = pipe_wrapper.generate(
        prompt   = "a beautiful woman on the beach",
        schedule = schedule,
        seed     = 42,
    )
    image.save("out.png")

Comparison helper
-----------------
    from casteer_pci import compare_methods

    compare_methods(
        prompt    = "a beautiful woman on the beach",
        sv        = sv,
        schedule  = schedule,
        save_path = "comparison.png",
        seed      = 42,
        device    = "cuda",
    )

Produces a 4-column PNG:
  col1  Unsafe baseline (no intervention)
  col2  CASteer only (full 50 steps, no temporal gating)
  col3  Vocab Ortho PCI (your existing method, text-embedding level)
  col4  CASteer + Temporal PCI gating  ← the new hybrid

Requirements
------------
    pip install diffusers transformers torch Pillow tqdm

Both pci_unlearning (from your repo) and this file must be importable.

Notes
-----
- This file is self-contained: it re-implements the minimal CASteer hook
  logic inline rather than importing from the CASteer repo, so you don't
  need to clone CASteer separately.
- Tested against SD 1.4 (CompVis/stable-diffusion-v1-4).  The hook
  targets BasicTransformerBlock.attn2 which exists in SD 1.4/1.5/2.1.
  For SDXL / DiT / FLUX you'd need to extend register_hooks() following
  the pattern in CASteer's diffusion_steering.py.
- Steering vectors are saved as plain torch dicts and are fast to reload.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import re
import textwrap
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("casteer_pci")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Concept vocabulary (reused from pci_unlearning.orthogonal)
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
        ("pornographic image of a couple",        "an image of a couple"),
        ("a nude model posing",                   "a model posing"),
        ("bare skin and nudity",                  "a photo of a person"),
    ],
    "violence": [
        ("a brutal murder scene with blood",      "a scene"),
        ("graphic violence and gore",             "a photo"),
        ("a man being stabbed",                   "a man standing"),
        ("soldiers in a bloody battle",           "soldiers in a field"),
        ("a violent fight with injuries",         "two people"),
    ],
}

# Fallback: generate `num_pairs` prompts from a small set of templates
_NUDITY_TEMPLATES_POS = [
    "a nude {subj} in {setting}",
    "naked {subj} in {setting}",
    "explicit nudity, {subj} in {setting}",
    "a topless {subj} in {setting}",
    "bare skin, {subj} in {setting}",
]
_NUDITY_TEMPLATES_NEG = [
    "a {subj} in {setting}",
    "a clothed {subj} in {setting}",
    "a fully dressed {subj} in {setting}",
    "a {subj} wearing clothes in {setting}",
    "a {subj} in {setting}",
]
_SUBJECTS = ["woman", "man", "person", "model", "figure", "couple"]
_SETTINGS = [
    "a bedroom", "a beach", "a studio", "a garden",
    "an artistic photo", "a painting", "a sculpture",
    "a swimming pool", "a forest", "a bathroom",
]


def _generate_nudity_pairs(num_pairs: int) -> List[Tuple[str, str]]:
    """Generate `num_pairs` (positive, negative) prompt pairs for nudity."""
    pairs: List[Tuple[str, str]] = list(PAIRED_PROMPTS["nudity"])
    import itertools
    combos = list(itertools.product(_SUBJECTS, _SETTINGS))
    tpl_idx = 0
    for subj, setting in itertools.cycle(combos):
        if len(pairs) >= num_pairs:
            break
        pos = _NUDITY_TEMPLATES_POS[tpl_idx % len(_NUDITY_TEMPLATES_POS)].format(
            subj=subj, setting=setting)
        neg = _NUDITY_TEMPLATES_NEG[tpl_idx % len(_NUDITY_TEMPLATES_NEG)].format(
            subj=subj, setting=setting)
        tpl_idx += 1
        if (pos, neg) not in pairs:
            pairs.append((pos, neg))
    return pairs[:num_pairs]


# ---------------------------------------------------------------------------
# Type alias
# SteeringVectors: dict[step_idx -> dict[place -> list[Tensor[heads, dim]]]]
# ---------------------------------------------------------------------------

SteeringVectors = Dict[int, Dict[str, List[torch.Tensor]]]


# ---------------------------------------------------------------------------
# Hook: collect CA outputs
# ---------------------------------------------------------------------------

class _CACollector:
    """Forward hook that collects mean-pooled CA outputs per layer/step."""

    def __init__(self) -> None:
        self._step: int = 0
        self._layer_counts: Dict[str, int] = defaultdict(int)
        # step -> place -> list[Tensor]
        self.data: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list))
        self._hooks: List = []

    def reset_step(self) -> None:
        self._step += 1
        self._layer_counts = defaultdict(int)

    def _hook_fn(self, place: str):
        def fn(module, inp, out):
            # out: [B, seq, dim]  (SD UNet attn2 output)
            x = out.detach().float()
            if x.dim() == 4:
                # [B, seq, heads, head_dim] — flatten heads
                x = x.reshape(x.shape[0], x.shape[1], -1)
            # Take only the conditioned half (CFG doubles batch)
            if x.shape[0] > 1:
                x = x[x.shape[0] // 2:]
            # Mean pool over sequence → [dim]
            vec = x.mean(dim=(0, 1))
            idx = self._layer_counts[place]
            self._layer_counts[place] += 1
            self.data[self._step][place].append(vec.cpu())
        return fn

    def register(self, unet) -> None:
        for name, module in unet.named_modules():
            if module.__class__.__name__ == "BasicTransformerBlock":
                if hasattr(module, "attn2") and module.attn2 is not None:
                    place = ("down" if "down" in name
                             else "up" if "up" in name else "mid")
                    h = module.attn2.register_forward_hook(self._hook_fn(place))
                    self._hooks.append(h)

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
    seed: int = 0,
) -> SteeringVectors:
    """
    Generate `num_pairs` paired images and compute per-layer, per-step
    steering vectors:

        v[step][place][layer_idx] = normalize(mean(CA_pos) - mean(CA_neg))

    If save_path already exists the file is loaded directly (fast reload).

    Args:
        model_id:   HuggingFace model ID (e.g. "CompVis/stable-diffusion-v1-4").
        concept:    Concept key ("nudity" or "violence").
        num_pairs:  Number of prompt pairs to average over.
        save_path:  Optional path to cache the vectors as a .pt file.
        device:     Torch device string.
        num_steps:  Denoising steps used when collecting activations.
        seed:       Fixed generation seed for reproducibility.

    Returns:
        SteeringVectors dict.
    """
    if save_path and os.path.exists(save_path):
        log.info("Loading cached steering vectors from %s", save_path)
        sv = torch.load(save_path, map_location="cpu")
        log.info("Loaded: %d steps × places", len(sv))
        return sv

    log.info("Building steering vectors for concept='%s', num_pairs=%d", concept, num_pairs)

    # ---- load pipeline ----
    sched = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=sched,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # ---- get prompt pairs ----
    if concept == "nudity":
        pairs = _generate_nudity_pairs(num_pairs)
    elif concept in PAIRED_PROMPTS:
        base = PAIRED_PROMPTS[concept]
        pairs = (base * ((num_pairs // len(base)) + 1))[:num_pairs]
    else:
        raise ValueError(f"Unknown concept '{concept}'. Supported: {list(PAIRED_PROMPTS)}")

    log.info("Using %d prompt pairs", len(pairs))

    def _collect_mean(prompts: List[str]) -> SteeringVectors:
        """Run pipeline on a list of prompts, return mean CA activations."""
        collector = _CACollector()
        collector.register(pipe.unet)

        sum_data: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list))
        count: Dict[int, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list))

        gen = torch.Generator(device=device).manual_seed(seed)

        for i, prompt in enumerate(prompts):
            log.info("  [%d/%d] %s", i + 1, len(prompts), prompt[:60])
            collector._step = -1
            collector._layer_counts = defaultdict(int)
            collector.data = defaultdict(lambda: defaultdict(list))

            # We need per-step data, so we run the loop manually
            # by wrapping the scheduler callback
            step_idx = [0]

            def _step_callback(pipe_, step, timestep, kwargs):
                collector.reset_step()   # increments step counter after each denoise step
                return kwargs

            pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                generator=torch.Generator(device=device).manual_seed(seed),
                callback_on_step_end=_step_callback,
                output_type="latent",
            )

            # accumulate
            for step in collector.data:
                for place in collector.data[step]:
                    layers = collector.data[step][place]
                    for li, vec in enumerate(layers):
                        if len(sum_data[step][place]) <= li:
                            sum_data[step][place].append(vec.clone())
                            count[step][place].append(1)
                        else:
                            sum_data[step][place][li] += vec
                            count[step][place][li] += 1

        collector.remove()

        # divide to get means
        means: SteeringVectors = {}
        for step in sum_data:
            means[step] = {}
            for place in sum_data[step]:
                means[step][place] = [
                    sum_data[step][place][li] / count[step][place][li]
                    for li in range(len(sum_data[step][place]))
                ]
        return means

    pos_prompts = [p for p, _ in pairs]
    neg_prompts = [n for _, n in pairs]

    log.info("Collecting activations for POSITIVE prompts …")
    pos_means = _collect_mean(pos_prompts)

    log.info("Collecting activations for NEGATIVE prompts …")
    neg_means = _collect_mean(neg_prompts)

    # ---- compute steering vectors ----
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

    log.info("Steering vectors built: %d steps", len(sv))

    # ---- clean up ----
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(sv, save_path)
        log.info("Saved to %s", save_path)

    return sv


# ---------------------------------------------------------------------------
# Hook: apply steering during inference
# ---------------------------------------------------------------------------

class _CASteerHook:
    """
    Forward hook that applies temporally-gated CASteer steering.

    At each CA layer output:
        alpha = max(beta * w(tau) * dot(v, patch), 0)
        patch_new = patch - alpha * v

    where w(tau) comes from the TemporalSchedule.
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

        # mutable state updated per-step by the pipeline wrapper
        self._current_step: int = 0
        self._schedule_weight: float = 1.0
        self._layer_idx: Dict[str, int] = defaultdict(int)

    # ---- called by pipeline wrapper before each denoising step ----
    def set_step(self, step_idx: int, weight: float) -> None:
        self._current_step = step_idx
        self._schedule_weight = weight
        self._layer_idx = defaultdict(int)

    def _hook_fn(self, place: str):
        def fn(module, inp, out):
            w = self._schedule_weight
            if w <= 0.0:
                return out   # outside temporal window — do nothing

            step = self._current_step
            li = self._layer_idx[place]
            self._layer_idx[place] += 1

            # look up steering vector for this step/place/layer
            try:
                v = self.sv[step][place][li]
            except (KeyError, IndexError):
                # step index may differ between collection and inference
                # fall back to closest available step
                available = sorted(self.sv.keys())
                if not available:
                    return out
                closest = min(available, key=lambda s: abs(s - step))
                try:
                    v = self.sv[closest][place][li]
                except (KeyError, IndexError):
                    return out

            v = v.to(out.device).to(out.dtype)  # [dim]

            x = out  # [B, seq, dim]  or  [B, seq, heads, head_dim]
            orig_shape = x.shape
            if x.dim() == 4:
                B, S, H, D = x.shape
                x = x.reshape(B, S, H * D)

            # steer only the conditioned half (CFG doubles batch)
            if x.shape[0] > 1:
                cond = x[x.shape[0] // 2:].clone()   # [1, seq, dim]
            else:
                cond = x.clone()

            # dot product per patch: [seq]
            dots = (cond[0] @ v)           # [seq]

            if self.intermediate_clipping:
                alpha = torch.clamp(self.beta * w * dots, min=0.0)   # [seq]
            else:
                alpha = self.beta * w * dots                           # [seq]

            delta = alpha.unsqueeze(-1) * v.unsqueeze(0)   # [seq, dim]
            cond[0] = cond[0] - delta

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
        log.info("CASteer hooks registered on %d CA layers", len(self._hooks))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

class TemporalCASteerPipeline:
    """
    Wraps a StableDiffusionPipeline and applies temporally-gated CASteer
    steering during generation.

    The denoising loop is taken over manually so we can call
    hook.set_step(step_idx, weight) before each step.
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
        self.beta = beta
        self.intermediate_clipping = intermediate_clipping
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
            torch_dtype=torch.float32,
            safety_checker=None,
        ).to(device)
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

        return _enc(negative_prompt), _enc(prompt)   # (uncond, cond)

    def _init_latents(self, seed: int, height: int, width: int) -> torch.Tensor:
        pipe = self.pipe
        pipe.scheduler.set_timesteps(50, device=pipe.device)
        shape = (
            1,
            pipe.unet.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
        gen = torch.Generator(device=pipe.device).manual_seed(seed)
        latents = torch.randn(shape, generator=gen, device=pipe.device,
                              dtype=torch.float32)
        return latents * pipe.scheduler.init_noise_sigma

    def _denoise_step(self, latent, emb_pair, native_t, guidance_scale=7.5):
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

    # ---- public API ----

    def generate(
        self,
        prompt: str,
        schedule,                          # TemporalSchedule from pci_unlearning
        seed: int = 42,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_steps: int = 50,
        height: int = 512,
        width: int = 512,
    ) -> Image.Image:
        """
        Generate an image with temporally-gated CASteer suppression.

        Args:
            prompt:          User prompt (safe or adversarial).
            schedule:        pci_unlearning.TemporalSchedule — controls
                             which timesteps are steered and how strongly.
            seed:            RNG seed.
            negative_prompt: CFG negative conditioning.
            guidance_scale:  CFG scale.
            num_steps:       Denoising steps.
            height, width:   Output image dimensions.

        Returns:
            PIL.Image.Image
        """
        pipe = self.pipe
        pipe.scheduler.set_timesteps(num_steps, device=pipe.device)
        tau_schedule = [float(t.item()) / 1000.0
                        for t in pipe.scheduler.timesteps]

        emb = self._encode_prompt(prompt, negative_prompt)
        latent = self._init_latents(seed, height, width)

        for step_idx, (tau, native_t) in enumerate(
                zip(tau_schedule, pipe.scheduler.timesteps)):
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
        """Generate without any steering (baseline)."""
        from pci_unlearning.schedule import make_uniform_schedule
        # weight=0 everywhere
        zero_schedule = make_uniform_schedule(tau_lo=2.0, tau_hi=3.0)
        return self.generate(
            prompt, zero_schedule, seed, negative_prompt,
            guidance_scale, num_steps, height, width,
        )

    def remove_hooks(self) -> None:
        self._hook.remove()


# ---------------------------------------------------------------------------
# Vocab Ortho PCI  (col3 — your existing method, for comparison)
# ---------------------------------------------------------------------------

def _gen_vocab_ortho(
    prompt: str,
    model_id: str,
    schedule,
    seed: int = 42,
    device: str = "cuda",
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    height: int = 512,
    width: int = 512,
) -> Image.Image:
    """Run your existing Vocab Ortho PCI method as a comparison column."""
    from pci_unlearning import auto_adapter, PromptConditionedIntervention

    sched = DPMSolverMultistepScheduler.from_pretrained(
        model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=sched,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)

    adapter = auto_adapter(pipe, num_inference_steps=num_steps, device=device)
    pci = PromptConditionedIntervention(
        adapter, guidance_scale=guidance_scale,
        height=height, width=width,
    )
    img = pci.run_orthogonal_vocab(
        prompt=prompt,
        schedule=schedule,
        seed=seed,
        concept="nudity",
        n_components=3,
        renormalize=True,
    )

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return img


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

LABEL_H = 56
FONT_SZ = 14
PAD = 6
COL_LABELS = [
    "1. Unsafe",
    "2. CASteer only",
    "3. Vocab Ortho PCI",
    "4. CASteer + Temporal PCI",
]


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
        draw.text(((w - (bb[2] - bb[0])) // 2, y), line,
                  font=font, fill=(30, 30, 30))
        y += FONT_SZ + 3
    return img


def _make_grid(images: List[Image.Image]) -> Image.Image:
    w, h = images[0].size
    n = len(images)
    total_w = n * w + (n - 1) * PAD
    canvas = Image.new("RGB", (total_w, h + LABEL_H), (220, 220, 220))
    for i, (img, lbl) in enumerate(zip(images, COL_LABELS)):
        x = i * (w + PAD)
        canvas.paste(img, (x, 0))
        canvas.paste(_label_img(lbl, w, LABEL_H), (x, h))
    return canvas


# ---------------------------------------------------------------------------
# compare_methods  — the main entry point for experiments
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
) -> Image.Image:
    """
    Generate a 4-column comparison image:

        col1  Unsafe baseline
        col2  CASteer (all steps active, no temporal gating)
        col3  Vocab Ortho PCI  (text-embedding level, your existing method)
        col4  CASteer + Temporal PCI  (the hybrid)

    Args:
        prompt:              Input prompt.
        sv:                  Precomputed steering vectors.
        schedule:            TemporalSchedule for cols 3 and 4.
        save_path:           Where to write the comparison PNG.
        seed:                RNG seed.
        model_id:            SD model.
        device:              CUDA device string.
        guidance_scale:      CFG scale.
        num_steps:           Denoising steps.
        height, width:       Image size in pixels.
        beta:                CASteer suppression strength (2.0 = Householder).
        include_vocab_ortho: If False, col3 is replaced with a blank (saves time).

    Returns:
        The assembled PIL.Image.Image grid.
    """
    from pci_unlearning.schedule import make_uniform_schedule

    pipe_wrapper = TemporalCASteerPipeline.from_pretrained(
        model_id, sv, device=device, beta=beta)

    # ---- col 1: unsafe ----
    log.info("[1/4] Unsafe …")
    img_unsafe = pipe_wrapper.generate_unsteered(
        prompt, seed=seed, guidance_scale=guidance_scale,
        num_steps=num_steps, height=height, width=width,
    )

    # ---- col 2: CASteer only (all steps) ----
    log.info("[2/4] CASteer only (all steps) …")
    full_schedule = make_uniform_schedule(tau_lo=0.0, tau_hi=1.0)
    img_casteer = pipe_wrapper.generate(
        prompt, full_schedule, seed=seed,
        guidance_scale=guidance_scale,
        num_steps=num_steps, height=height, width=width,
    )

    # ---- col 3: Vocab Ortho PCI ----
    if include_vocab_ortho:
        log.info("[3/4] Vocab Ortho PCI …")
        img_vocab_ortho = _gen_vocab_ortho(
            prompt, model_id, schedule, seed=seed, device=device,
            guidance_scale=guidance_scale, num_steps=num_steps,
            height=height, width=width,
        )
    else:
        log.info("[3/4] Skipping Vocab Ortho PCI (include_vocab_ortho=False)")
        img_vocab_ortho = Image.new("RGB", (width, height), (200, 200, 200))

    # ---- col 4: CASteer + Temporal PCI ----
    log.info("[4/4] CASteer + Temporal PCI …")
    img_hybrid = pipe_wrapper.generate(
        prompt, schedule, seed=seed,
        guidance_scale=guidance_scale,
        num_steps=num_steps, height=height, width=width,
    )

    pipe_wrapper.remove_hooks()

    images = [img_unsafe, img_casteer, img_vocab_ortho, img_hybrid]
    grid = _make_grid(images)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    grid.save(save_path)
    log.info("Saved → %s", save_path)
    return grid


# ---------------------------------------------------------------------------
# Batch runner  (mirrors generate_from_csv.py's structure)
# ---------------------------------------------------------------------------

def _slug(text: str, max_len: int = 30) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len]


def run_batch(
    prompts_path: str,
    sv: SteeringVectors,
    schedule,
    save_dir: str,
    col: str = "target_prompt",
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

    Mirrors the structure of generate_from_csv.py:
      <save_dir>/
        prompt_000_<slug>/
            comparison.png
        summary.png

    Args:
        prompts_path:  Path to CSV (needs --col column) or TXT (one per line).
        sv:            Steering vectors.
        schedule:      TemporalSchedule.
        save_dir:      Output root directory.
        col:           CSV column name for prompts.
        n:             Max prompts to process (None = all).
        seed:          RNG seed (same for all prompts).
        model_id:      SD model.
        device:        CUDA device.
        guidance_scale: CFG scale.
        num_steps:     Denoising steps.
        height, width: Image dimensions.
        beta:          CASteer strength.
        include_vocab_ortho: Whether to run col3.
    """
    # load prompts
    if prompts_path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(prompts_path)
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {df.columns.tolist()}")
        prompts = df[col].dropna().astype(str).tolist()
    else:
        with open(prompts_path) as f:
            prompts = [l.strip() for l in f
                       if l.strip() and not l.startswith("#")]
    if n is not None:
        prompts = prompts[:n]

    log.info("Processing %d prompts → %s", len(prompts), save_dir)
    os.makedirs(save_dir, exist_ok=True)

    summary_rows: List[Image.Image] = []

    for idx, prompt in enumerate(prompts):
        folder = os.path.join(save_dir, f"prompt_{idx:03d}_{_slug(prompt)}")
        out_path = os.path.join(folder, "comparison.png")

        if os.path.exists(out_path):
            log.info("[%d/%d] skip (exists): %s", idx + 1, len(prompts), prompt[:60])
            summary_rows.append(Image.open(out_path))
            continue

        log.info("─" * 60)
        log.info("[%d/%d] %s", idx + 1, len(prompts), prompt[:80])
        os.makedirs(folder, exist_ok=True)

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
        )
        summary_rows.append(grid)

    # summary grid
    if summary_rows:
        from PIL import Image as PILImage
        total_h = sum(r.height for r in summary_rows) + (len(summary_rows) - 1) * PAD
        canvas = PILImage.new("RGB", (summary_rows[0].width, total_h), (180, 180, 180))
        y = 0
        for r in summary_rows:
            canvas.paste(r, (0, y))
            y += r.height + PAD
        summary_path = os.path.join(save_dir, "summary.png")
        canvas.save(summary_path)
        log.info("Summary → %s", summary_path)

    print("\n" + "=" * 60)
    print(f"Done. {len(prompts)} prompts processed.")
    print(f"Results : {save_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CASteer + Temporal PCI  —  temporally-gated CA activation steering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- build-vectors subcommand ----
    bv = sub.add_parser("build-vectors",
                        help="Precompute steering vectors from paired generations.")
    bv.add_argument("--model_id",   default="CompVis/stable-diffusion-v1-4")
    bv.add_argument("--concept",    default="nudity")
    bv.add_argument("--num_pairs",  type=int, default=50)
    bv.add_argument("--save_path",  required=True, help="Output .pt path")
    bv.add_argument("--device",     default="cuda")
    bv.add_argument("--steps",      type=int, default=50)
    bv.add_argument("--seed",       type=int, default=0)

    # ---- single subcommand ----
    sg = sub.add_parser("single",
                        help="Generate a 4-column comparison for one prompt.")
    sg.add_argument("--prompt",     required=True)
    sg.add_argument("--sv_path",    required=True, help="Path to steering vectors .pt")
    sg.add_argument("--save_path",  default="casteer_pci_comparison.png")
    sg.add_argument("--model_id",   default="CompVis/stable-diffusion-v1-4")
    sg.add_argument("--device",     default="cuda")
    sg.add_argument("--seed",       type=int,   default=42)
    sg.add_argument("--steps",      type=int,   default=50)
    sg.add_argument("--guidance",   type=float, default=7.5)
    sg.add_argument("--img_size",   type=int,   default=512)
    sg.add_argument("--tau_lo",     type=float, default=0.3,
                    help="Start of temporal intervention window")
    sg.add_argument("--tau_hi",     type=float, default=0.8,
                    help="End of temporal intervention window")
    sg.add_argument("--beta",       type=float, default=2.0)
    sg.add_argument("--no_vocab_ortho", action="store_true",
                    help="Skip Vocab Ortho PCI column (faster)")

    # ---- batch subcommand ----
    bt = sub.add_parser("batch",
                        help="Run compare_methods over a CSV / TXT prompt file.")
    bt.add_argument("--prompts",    required=True)
    bt.add_argument("--sv_path",    required=True)
    bt.add_argument("--save_dir",   required=True)
    bt.add_argument("--col",        default="target_prompt")
    bt.add_argument("--n",          type=int, default=None)
    bt.add_argument("--model_id",   default="CompVis/stable-diffusion-v1-4")
    bt.add_argument("--device",     default="cuda")
    bt.add_argument("--seed",       type=int,   default=42)
    bt.add_argument("--steps",      type=int,   default=50)
    bt.add_argument("--guidance",   type=float, default=7.5)
    bt.add_argument("--img_size",   type=int,   default=512)
    bt.add_argument("--tau_lo",     type=float, default=0.3)
    bt.add_argument("--tau_hi",     type=float, default=0.8)
    bt.add_argument("--beta",       type=float, default=2.0)
    bt.add_argument("--no_vocab_ortho", action="store_true")

    args = p.parse_args()

    if args.cmd == "build-vectors":
        build_steering_vectors(
            model_id=args.model_id,
            concept=args.concept,
            num_pairs=args.num_pairs,
            save_path=args.save_path,
            device=args.device,
            num_steps=args.steps,
            seed=args.seed,
        )

    elif args.cmd == "single":
        from pci_unlearning.schedule import make_uniform_schedule
        sv = torch.load(args.sv_path, map_location="cpu")
        schedule = make_uniform_schedule(tau_lo=args.tau_lo, tau_hi=args.tau_hi)
        compare_methods(
            prompt=args.prompt,
            sv=sv,
            schedule=schedule,
            save_path=args.save_path,
            seed=args.seed,
            model_id=args.model_id,
            device=args.device,
            guidance_scale=args.guidance,
            num_steps=args.steps,
            height=args.img_size,
            width=args.img_size,
            beta=args.beta,
            include_vocab_ortho=not args.no_vocab_ortho,
        )

    elif args.cmd == "batch":
        from pci_unlearning.schedule import make_uniform_schedule
        sv = torch.load(args.sv_path, map_location="cpu")
        schedule = make_uniform_schedule(tau_lo=args.tau_lo, tau_hi=args.tau_hi)
        run_batch(
            prompts_path=args.prompts,
            sv=sv,
            schedule=schedule,
            save_dir=args.save_dir,
            col=args.col,
            n=args.n,
            seed=args.seed,
            model_id=args.model_id,
            device=args.device,
            guidance_scale=args.guidance,
            num_steps=args.steps,
            height=args.img_size,
            width=args.img_size,
            beta=args.beta,
            include_vocab_ortho=not args.no_vocab_ortho,
        )