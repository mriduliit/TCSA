#!/usr/bin/env python3
"""
ablation/ca_attention_visualization.py — TCSA's own CA-steering signal,
visualized spatially (NOT a CASteer-style text<->patch attention map)
================================================================================

What this visualizes:

In `_CASteerHook._hook_fn` (merged_unlearning_multi_gpu.py:824-868), at every
cross-attention layer and every denoising step, TCSA computes:

    dots  = cond[0] @ v          # [S] -- per-spatial-patch dot product
                                  #        between the CA output and the
                                  #        concept steering vector v.
                                  #        This is WHERE (in the spatial
                                  #        feature grid) the unsafe concept
                                  #        is forming.
    alpha = clamp(beta_l * w * tcl * dots, min=0)   # [S]
                                  # the ACTUAL per-patch suppression
                                  # magnitude TCSA applies at that
                                  # location/step (after layer-adaptive
                                  # beta, temporal schedule w, and
                                  # text-concept-load tcl gating).
    cond[0] = cond[0] - alpha * v

`dots`/`alpha` are reshaped to the layer's own spatial grid (S = H*W,
H==W for every BasicTransformerBlock in the SD1.4 UNet) and are TCSA's own
internal detection + suppression signal -- this is what we visualize, not
a generic prompt-token <-> image-patch cross-attention map (that would be
replicating CASteer's / Prompt-to-Prompt's visualization, which we are
explicitly NOT doing).

Method (no edits to merged_unlearning_multi_gpu.py):
    - `_LoggingCASteerHook` subclasses `_CASteerHook`, duplicating the
      forward-hook math exactly, but additionally stashes `dots`/`alpha`
      (already reshaped to (H, W)) into `self.captured[step][place][li]`.
    - `_VizPipeline` subclasses `MergedUnlearningPipeline`, swapping in the
      logging hook (everything else identical).

For each of a few explicit P4D prompts we:
    1. generate the unsteered baseline image (mode="none")
    2. generate the TCSA ("both") image while capturing dots/alpha maps
       at every layer/step
    3. pick one early high-resolution layer (down path) and one late
       high-resolution layer (up path) and render heatmap overlays of
       `dots` (raw concept presence, can be negative) and `alpha`
       (applied suppression, >=0) at steps {5, 15, 25} of 30, upsampled
       onto the final steered image
    4. save a single composite figure: baseline | steered | dots heatmaps
       (3 steps) | alpha heatmaps (3 steps), per chosen layer.

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
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/storage/pk_mridul/TCSA/scripts/evaluation")
from merged_unlearning_multi_gpu import (  # noqa: E402
    MergedUnlearningPipeline, Schedule, build_steering_vectors, _CASteerHook,
)


class _LoggingCASteerHook(_CASteerHook):
    """Identical to `_CASteerHook` but also stores dots/alpha per (step, place, li)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capturing = False
        self.captured = defaultdict(lambda: defaultdict(dict))  # [step][place][li] = (dots_hw, alpha_hw)

    def _hook_fn(self, place: str):
        def fn(module, inp, out):
            w = self._w
            if w <= 0.0:
                return out
            tcl = self._tcl
            li = self._layer_idx[place]
            self._layer_idx[place] += 1
            v = self._lookup(place, li)
            if v is None:
                return out
            v = v.to(out.device).to(out.dtype)

            beta_l = (self.beta_layer.get(place, {}).get(li, self.beta))

            x = out
            orig_shape = x.shape
            if x.dim() == 4:
                B, S, H, D = x.shape
                x = x.reshape(B, S, H * D)

            if x.shape[0] > 1:
                cond = x[x.shape[0] // 2:].clone()
            else:
                cond = x.clone()

            dots = cond[0] @ v                       # [S]

            alpha = beta_l * w * tcl * dots
            if self.clip:
                alpha_clamped = torch.clamp(alpha, min=0.0)
            else:
                alpha_clamped = alpha

            if self.capturing:
                S = dots.shape[0]
                side = int(round(S ** 0.5))
                if side * side == S:
                    dots_hw = dots.detach().float().reshape(side, side).cpu()
                    alpha_hw = alpha_clamped.detach().float().reshape(side, side).cpu()
                    self.captured[self._step][place][li] = (dots_hw, alpha_hw)

            cond[0] = cond[0] - alpha_clamped.unsqueeze(-1) * v.unsqueeze(0)

            if x.shape[0] > 1:
                x = x.clone()
                x[x.shape[0] // 2:] = cond
            else:
                x = cond
            if len(orig_shape) == 4:
                x = x.reshape(orig_shape)
            return x.to(out.dtype)
        return fn


class _VizPipeline(MergedUnlearningPipeline):
    def __init__(self, *args, **kwargs):
        # bypass parent __init__'s hook construction; duplicate minimal setup
        self.pipe = args[0]
        sv = args[1]
        concept = kwargs.get("concept", "nudity")
        self.beta = kwargs.get("beta", 2.0)
        self.max_proj_ratio = kwargs.get("max_proj_ratio", 0.5)
        n_components = kwargs.get("n_components", 1)
        layer_align = kwargs.get("layer_align", None)

        self._hook = _LoggingCASteerHook(sv, beta=self.beta,
                                          layer_align=layer_align, clip=True)
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


def overlay_heatmap(base_img: np.ndarray, heat: torch.Tensor, vmin=None, vmax=None, cmap="inferno"):
    """base_img: HxWx3 uint8 in [0,255]. heat: (h,w) tensor. Returns HxWx3 uint8 blended."""
    H, W, _ = base_img.shape
    heat_up = F.interpolate(heat.unsqueeze(0).unsqueeze(0), size=(H, W),
                             mode="bilinear", align_corners=False)[0, 0].numpy()
    if vmin is None:
        vmin = float(heat_up.min())
    if vmax is None:
        vmax = float(heat_up.max())
    norm = np.clip((heat_up - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    cmap_fn = plt.get_cmap(cmap)
    colored = (cmap_fn(norm)[..., :3] * 255).astype(np.uint8)
    blended = (0.45 * base_img.astype(np.float32) + 0.55 * colored.astype(np.float32)).astype(np.uint8)
    return blended


ap = argparse.ArgumentParser()
ap.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
ap.add_argument("--sv_path", default="/storage/pk_mridul/TCSA/steering_vectors/sv_nudity_beta_1.pt")
ap.add_argument("--p4d_csv", default="/storage/pk_mridul/TCSA/datasets/nudity.csv")
ap.add_argument("--save_dir", default="/storage/pk_mridul/TCSA/ablation/results/ca_visualization")
ap.add_argument("--device", default="cuda")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--guidance", type=float, default=7.5)
ap.add_argument("--img_size", type=int, default=512)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--beta", type=float, default=2.0)
ap.add_argument("--max_proj_ratio", type=float, default=0.5)
ap.add_argument("--n_prompts", type=int, default=3)
ap.add_argument("--viz_steps", default="5,15,25")
args = ap.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
viz_steps = [int(s) for s in args.viz_steps.split(",")]

df = pd.read_csv(args.p4d_csv)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
df = df[df["nudity_percentage"] > 0.0].reset_index(drop=True)
prompts = df["prompt"].astype(str).tolist()[:args.n_prompts]
print(f"Visualizing {len(prompts)} P4D prompts: {prompts}")

sv, layer_align = build_steering_vectors(
    model_id=args.model_id, concept="nudity",
    save_path=args.sv_path, device=args.device)

wrapper = _VizPipeline.from_pretrained(
    args.model_id, sv, device=args.device, concept="nudity",
    beta=args.beta, n_components=1,
    max_proj_ratio=args.max_proj_ratio, layer_align=layer_align)


def to_uint8(img):
    return np.array(img.convert("RGB"))


for pi, prompt in enumerate(prompts):
    print(f"\n=== prompt {pi}: {prompt[:80]} ===")

    # 1) unsteered baseline
    wrapper._hook.capturing = False
    base_img = wrapper.generate(prompt=prompt, mode="none", schedule=None,
                                 seed=args.seed, guidance_scale=args.guidance,
                                 num_steps=args.steps, height=args.img_size,
                                 width=args.img_size)

    # 2) TCSA steered, capturing dots/alpha at every layer/step
    wrapper._hook.captured.clear()
    wrapper._hook.capturing = True
    steered_img = wrapper.generate(prompt=prompt, mode="both", schedule=Schedule(),
                                    seed=args.seed, guidance_scale=args.guidance,
                                    num_steps=args.steps, height=args.img_size,
                                    width=args.img_size)
    wrapper._hook.capturing = False

    captured = wrapper._hook.captured  # [step][place][li] -> (dots_hw, alpha_hw)

    # pick one early high-res "down" layer and one late high-res "up" layer
    def pick_layer(place):
        if place not in captured.get(viz_steps[0], {}):
            return None
        lis = sorted(captured[viz_steps[0]][place].keys())
        if not lis:
            return None
        # prefer the highest spatial resolution (largest grid) available
        best_li, best_side = None, -1
        for li in lis:
            side = captured[viz_steps[0]][place][li][0].shape[0]
            if side > best_side:
                best_li, best_side = li, side
        return best_li

    down_li = pick_layer("down")
    up_li = pick_layer("up")
    chosen = [("down", down_li), ("up", up_li)]
    chosen = [(p, li) for p, li in chosen if li is not None]
    print(f"  chosen layers: {chosen}")

    base_arr = to_uint8(base_img)
    steered_arr = to_uint8(steered_img)

    for place, li in chosen:
        n_cols = 2 + 2 * len(viz_steps)
        fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.2))
        axes[0].imshow(base_arr); axes[0].set_title("baseline\n(mode=none)"); axes[0].axis("off")
        axes[1].imshow(steered_arr); axes[1].set_title("TCSA\n(mode=both)"); axes[1].axis("off")

        col = 2
        for step in viz_steps:
            if step not in captured or place not in captured[step] or li not in captured[step][place]:
                col += 2
                continue
            dots_hw, alpha_hw = captured[step][place][li]
            dots_overlay = overlay_heatmap(steered_arr, dots_hw, cmap="coolwarm")
            alpha_overlay = overlay_heatmap(steered_arr, alpha_hw, vmin=0.0, cmap="inferno")
            axes[col].imshow(dots_overlay)
            axes[col].set_title(f"step {step}\nconcept presence\n(dots, {place}[{li}])")
            axes[col].axis("off")
            axes[col + 1].imshow(alpha_overlay)
            axes[col + 1].set_title(f"step {step}\nsuppression applied\n(alpha, {place}[{li}])")
            axes[col + 1].axis("off")
            col += 2

        plt.tight_layout()
        out_path = os.path.join(args.save_dir, f"prompt{pi}_{place}{li}.png")
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {out_path}")

    base_img.save(os.path.join(args.save_dir, f"prompt{pi}_baseline.png"))
    steered_img.save(os.path.join(args.save_dir, f"prompt{pi}_steered.png"))

wrapper.remove_hooks()
print(f"\nDone. Outputs in {args.save_dir}")
