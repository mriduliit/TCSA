#!/usr/bin/env python3
"""
ablation/dots_specificity_figure.py — CASteer-Fig.2-style specificity figure
for TCSA's own CA steering signal (raw dot-product heatmaps, no image overlay)
================================================================================

Reproduces the structure of CASteer's Figure 2 ("Heatmaps of dot product
values between CA outputs ... and steering vector ... Top: generated
images, bottom: heatmaps") but for TCSA's own steering vector `v` and CA
output, at a single fixed (layer, denoising step), with a SHARED colorbar
across all prompts -- demonstrating that the dot product

    dots = cond[0] @ v          # [S], reshaped to (H, W)

is large and spatially localized only when the generated image actually
depicts the unsafe concept, and small/uniform otherwise. This is purely a
DIAGNOSTIC: generation is unsteered (mode="none") so the recorded dots are
not influenced by our own suppression -- they show the steering vector's
natural specificity to the concept's visual manifestation.

No edits to merged_unlearning_multi_gpu.py. New file only.

Method:
    `_DiagnosticHook` subclasses `_CASteerHook`. When `capturing=True` it
    computes and logs `dots` at every (step, place, li) exactly like the
    real hook, but with `apply_suppression=False` it NEVER modifies the
    CA output -- a pure observer, even though mode="none" sets w=0 (the
    real hook would otherwise skip the layer entirely).

Run inside the `casteer` conda env, e.g. CUDA_VISIBLE_DEVICES=1.
"""
import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/storage/pk_mridul/TCSA/scripts/evaluation")
from merged_unlearning_multi_gpu import (  # noqa: E402
    MergedUnlearningPipeline, build_steering_vectors, _CASteerHook,
)


class _DiagnosticHook(_CASteerHook):
    """Pure observer: logs `dots` at every layer/step, never edits CA output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capturing = False
        self.apply_suppression = False
        self.captured = defaultdict(lambda: defaultdict(dict))  # [step][place][li] -> dots_hw

    def _hook_fn(self, place: str):
        def fn(module, inp, out):
            w = self._w
            li = self._layer_idx[place]
            self._layer_idx[place] += 1
            if not self.capturing and w <= 0.0:
                return out
            v = self._lookup(place, li)
            if v is None:
                return out
            v = v.to(out.device).to(out.dtype)

            x = out
            orig_shape = x.shape
            if x.dim() == 4:
                B, S, H, D = x.shape
                x = x.reshape(B, S, H * D)
            if x.shape[0] > 1:
                cond = x[x.shape[0] // 2:].clone()
            else:
                cond = x.clone()

            dots = cond[0] @ v   # [S]

            if self.capturing:
                S = dots.shape[0]
                side = int(round(S ** 0.5))
                if side * side == S:
                    self.captured[self._step][place][li] = dots.detach().float().reshape(side, side).cpu()

            if not self.apply_suppression or w <= 0.0:
                return out   # observer-only: never modifies generation

            tcl = self._tcl
            beta_l = (self.beta_layer.get(place, {}).get(li, self.beta))
            alpha = beta_l * w * tcl * dots
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


class _DiagPipeline(MergedUnlearningPipeline):
    def __init__(self, *args, **kwargs):
        self.pipe = args[0]
        sv = args[1]
        concept = kwargs.get("concept", "nudity")
        self.beta = kwargs.get("beta", 2.0)
        self.max_proj_ratio = kwargs.get("max_proj_ratio", 0.5)
        n_components = kwargs.get("n_components", 1)
        layer_align = kwargs.get("layer_align", None)
        self._hook = _DiagnosticHook(sv, beta=self.beta, layer_align=layer_align, clip=True)
        self._hook.register(self.pipe.unet)
        self.text_dirs = self._build_text_direction(concept, n_components)

    @classmethod
    def from_pretrained(cls, model_id, sv, device="cuda", concept="nudity",
                         beta=2.0, n_components=1, max_proj_ratio=0.5,
                         layer_align=None):
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
        sched = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=sched, torch_dtype=torch.float32,
            safety_checker=None,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe, sv, concept=concept, beta=beta,
                   n_components=n_components, max_proj_ratio=max_proj_ratio,
                   layer_align=layer_align)


ap = argparse.ArgumentParser()
ap.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
ap.add_argument("--sv_path", default="/storage/pk_mridul/TCSA/steering_vectors/sv_nudity_beta_1.pt")
ap.add_argument("--save_dir", default="/storage/pk_mridul/TCSA/ablation/results/dots_specificity")
ap.add_argument("--device", default="cuda")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--guidance", type=float, default=7.5)
ap.add_argument("--img_size", type=int, default=512)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--beta", type=float, default=2.0)
ap.add_argument("--diag_step", type=int, default=0, help="denoising step index to visualize (CASteer used step 0)")
ap.add_argument("--min_side", type=int, default=16, help="minimum spatial grid side to consider for layer selection")
args = ap.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

PROMPTS = [
    ("concept present", "a nude woman standing in a photography studio, professional photograph"),
    ("structurally similar, safe", "a clothed woman standing in a photography studio, professional photograph"),
    ("unrelated", "a photo of a mountain landscape at sunset"),
]

sv, layer_align = build_steering_vectors(
    model_id=args.model_id, concept="nudity",
    save_path=args.sv_path, device=args.device)

wrapper = _DiagPipeline.from_pretrained(
    args.model_id, sv, device=args.device, concept="nudity",
    beta=args.beta, n_components=1, max_proj_ratio=0.5, layer_align=layer_align)

# ── pass 1: generate + capture for all prompts ──
captured_per_prompt = []
images = []
for label, prompt in PROMPTS:
    wrapper._hook.captured.clear()
    wrapper._hook.capturing = True
    img = wrapper.generate(prompt=prompt, mode="none", schedule=None,
                            seed=args.seed, guidance_scale=args.guidance,
                            num_steps=args.steps, height=args.img_size, width=args.img_size)
    wrapper._hook.capturing = False
    images.append(img)
    captured_per_prompt.append(dict(wrapper._hook.captured[args.diag_step]))  # {place: {li: hw}}
    print(f"[{label}] '{prompt}' -> captured {sum(len(v) for v in captured_per_prompt[-1].values())} layers at step {args.diag_step}")

# ── pick ONE fixed layer for the comparison: highest beta among layers with side>=min_side ──
best = None  # (beta, place, li, side)
for place, li_dict in captured_per_prompt[0].items():
    for li, hw in li_dict.items():
        side = hw.shape[0]
        if side < args.min_side:
            continue
        beta_l = wrapper._hook.beta_layer.get(place, {}).get(li, wrapper._hook.beta)
        if best is None or beta_l > best[0]:
            best = (beta_l, place, li, side)

if best is None:
    raise RuntimeError("No layer found with the requested minimum spatial resolution.")
beta_l, place, li, side = best
print(f"\nChosen layer: place={place} li={li} side={side}x{side} beta={beta_l:.3f} (most concept-aligned, step={args.diag_step})")

heatmaps = [captured_per_prompt[i][place][li].numpy() for i in range(len(PROMPTS))]
vmin = min(h.min() for h in heatmaps)
vmax = max(h.max() for h in heatmaps)
print(f"Shared color scale: [{vmin:.3f}, {vmax:.3f}]")

# ── figure: top row = generated images, bottom row = raw heatmaps, shared colorbar ──
n = len(PROMPTS)
fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4))
for i, ((label, prompt), img, hm) in enumerate(zip(PROMPTS, images, heatmaps)):
    axes[0, i].imshow(np.array(img.convert("RGB")))
    axes[0, i].set_title(f"{label}\n\"{prompt[:38]}...\"" if len(prompt) > 38 else f"{label}\n\"{prompt}\"", fontsize=9)
    axes[0, i].axis("off")
    im = axes[1, i].imshow(hm, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1, i].axis("off")

fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.025, pad=0.01)
fig.suptitle(f"Heatmaps of dot-product values between TCSA's CA output (place={place}, layer {li}, "
             f"denoising step {args.diag_step}) and the nudity steering vector v.\n"
             f"Top: generated images (unsteered, mode=none). Bottom: heatmaps. "
             f"Dot product is high and localized only where the concept is actually depicted.",
             fontsize=10, wrap=True)
out_path = os.path.join(args.save_dir, "dots_specificity_figure.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {out_path}")

for i, (label, prompt) in enumerate(PROMPTS):
    images[i].save(os.path.join(args.save_dir, f"img_{i}_{label.replace(' ', '_')}.png"))

wrapper.remove_hooks()
print("Done.")
