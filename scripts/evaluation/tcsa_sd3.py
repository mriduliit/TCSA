"""
tcsa_sd3.py  —  TCSA adapted for Stable Diffusion 3 (MM-DiT / JointTransformerBlock)
======================================================================================

Why SD3 needs a separate adapter
----------------------------------
SD v1.4 and SDXL use a UNet with BasicTransformerBlock layers that have a
dedicated cross-attention sub-layer (attn2) taking text as key/value.
Our hooks attach to attn2.register_forward_hook() and steer the output.

SD3 uses a DiT (Diffusion Transformer) with JointTransformerBlock layers.
There is NO separate attn2 for text cross-attention. Instead:

  self.attn  — joint self+cross attention: image tokens AND text tokens
               attend to each other in ONE operation.
               forward() takes (hidden_states, encoder_hidden_states)
               and returns (context_out, image_out) as a tuple.

Architecture key facts (SD3-medium):
  - 24 JointTransformerBlock layers (indices 0-23)
  - layer 23: context_pre_only=True  (no separate context output path)
  - hidden dim: 1536  (image tokens)
  - text context dim: 4096 → projected to 1536 inside the block
  - dual-attention (attn2): only in SD3.5; SD3-medium has attn2=None

Strategy
---------
We hook self.attn (the joint attention) in each JointTransformerBlock.
The hook intercepts the output TUPLE (context_out, image_out) and steers
the IMAGE portion (index 1) along the concept direction v.
The text context portion (index 0) is left untouched.

Text-space orthogonalization uses ONLY the T5 encoder embedding (prompt_embeds,
shape [B, seq, 4096]) because that is the dominant semantic signal in SD3.
The two CLIP encoders contribute pooled embeddings that control style/aesthetics
and are harder to surgically edit; we leave them unchanged.

Usage
------
# Step 1 — build steering vectors once (cached):
python tcsa_sd3.py build-vectors \\
    --model_id stabilityai/stable-diffusion-3-medium-diffusers \\
    --concept nudity --num_pairs 50 \\
    --save_path sv_sd3_nudity.pt --device cuda

# Step 2 — evaluate on red-teaming benchmarks:
python tcsa_sd3.py eval \\
    --model_id stabilityai/stable-diffusion-3-medium-diffusers \\
    --prompts_path prompts/p4d.csv --col prompt \\
    --sv_path sv_sd3_nudity.pt \\
    --save_dir results/sd3_tcsa --device cuda

# Step 3 — single comparison (4 columns: unsafe / text-only / ca-only / both):
python tcsa_sd3.py single \\
    --model_id stabilityai/stable-diffusion-3-medium-diffusers \\
    --prompt "a nude woman on the beach" \\
    --sv_path sv_sd3_nudity.pt \\
    --save_path out_sd3.png --device cuda

Requirements: diffusers>=0.27 transformers torch Pillow pandas tqdm
Model download: ~10 GB (SD3-medium); cached under HF_HOME after first run.
Set HF_HOME or TRANSFORMERS_CACHE to a fast disk if needed.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from PIL import Image, ImageDraw, ImageFont
import textwrap
import re

log = logging.getLogger("tcsa_sd3")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Concept vocabularies (same as TCSA nudity, text-ortho uses T5 space) ──
UNSAFE_VOCAB = [
    "nudity", "nude", "naked", "pornography", "explicit content",
    "erotic", "nsfw", "topless", "genitals", "sex", "sexual",
    "pornographic", "adult content", "bare skin", "undressed",
]
SAFE_VOCAB = [
    "clothed person", "fully dressed", "wearing clothes", "outfit",
    "fashion", "attire", "dressed woman", "dressed man",
    "person in clothes", "portrait", "clothed", "wearing outfit",
    "dressed people", "fashion model", "person wearing shirt",
]
PAIRED_PROMPTS = [
    ("a nude woman lying on a bed",        "a woman lying on a bed"),
    ("a naked man in the shower",          "a man in the shower"),
    ("an explicit erotic photo",           "a portrait photo"),
    ("a topless person on the beach",      "a person on the beach"),
    ("naked body in artistic photography", "a person in artistic photography"),
    ("nude figure in a painting",          "a figure in a painting"),
    ("erotic sculpture with bare skin",    "a sculpture"),
    ("a nude model posing",                "a model posing"),
    ("bare skin and nudity",               "a photo of a person"),
    ("explicit nudity in a bedroom",       "a person in a bedroom"),
]


# ── Schedule (identical to TCSA) ──────────────────────────────────────────
@dataclass
class Schedule:
    mode: str = "front"
    tau_relax: float = 0.2
    w_min: float = 0.5

    def weight(self, tau: float) -> float:
        if self.mode == "always":
            return 1.0
        if tau >= self.tau_relax:
            return 1.0
        return self.w_min + (1.0 - self.w_min) * (tau / max(self.tau_relax, 1e-6))


# ── Text-space direction (T5 token space, dim=4096) ───────────────────────

def build_t5_direction(
    pipe: StableDiffusion3Pipeline,
    device: str,
) -> torch.Tensor:
    """
    Build contrastive concept direction in T5 embedding space (dim=4096).

    SD3 has THREE text encoders:
      text_encoder   — CLIP-L  → pooled [B, 768]
      text_encoder_2 — CLIP-G  → pooled [B, 1280]
      text_encoder_3 — T5-XXL  → sequence [B, seq, 4096]

    We build d in T5 space (the sequence encoder) because:
    1. T5 carries the detailed semantic content used in cross-attention.
    2. It has a sequence dimension that supports per-token projection
       (same as TCSA's text-ortho on CLIP-L for SD v1.4).
    CLIP pooled embeddings control global style; we leave them unchanged.
    """
    enc3 = pipe.text_encoder_3
    tok3 = pipe.tokenizer_3
    enc3.eval()

    def encode_t5(text: str) -> torch.Tensor:
        t = tok3(text, return_tensors="pt", padding="max_length",
                 max_length=tok3.model_max_length,
                 truncation=True).to(device)
        with torch.no_grad():
            out = enc3(t.input_ids)[0]   # [1, seq, 4096]
        return out[0, 1, :].float()      # first content token

    unsafe_embs = torch.stack([encode_t5(w) for w in UNSAFE_VOCAB])  # [N, 4096]
    safe_embs   = torch.stack([encode_t5(w) for w in SAFE_VOCAB])    # [N, 4096]
    raw = unsafe_embs.mean(0) - safe_embs.mean(0)                     # [4096]
    d = (raw / raw.norm().clamp(min=1e-8)).cpu()
    log.info("T5 concept direction built  norm=%.4f  dim=%d", d.norm().item(), d.shape[0])
    return d  # [4096]


def orthogonalize_t5_embedding(
    prompt_embeds: torch.Tensor,   # [B, seq, 4096]
    d: torch.Tensor,               # [4096]
    max_proj_ratio: float = 0.5,
    renormalize: bool = True,
) -> torch.Tensor:
    """Project concept direction out of every T5 token."""
    d = d.to(prompt_embeds.device, dtype=prompt_embeds.dtype)
    orig_norms = prompt_embeds.norm(dim=-1, keepdim=True)
    proj = (prompt_embeds * d).sum(dim=-1, keepdim=True)       # [B, seq, 1]
    if max_proj_ratio is not None:
        ceiling = max_proj_ratio * prompt_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        proj = proj.sign() * proj.abs().clamp(max=ceiling)
    out = prompt_embeds - proj * d
    if renormalize:
        new_norms = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        out = out * (orig_norms / new_norms)
    return out


def compute_tcl(
    prompt_embeds: torch.Tensor,   # [B, seq, 4096]
    d: torch.Tensor,               # [4096]
) -> float:
    """Text Concept Load — max token alignment with concept direction."""
    d = d.to(prompt_embeds.device, dtype=prompt_embeds.dtype)
    token_dots = (prompt_embeds[0] @ d)          # [seq]
    max_dot = token_dots.max().item()
    return float(1.0 / (1.0 + math.exp(-3.0 * max_dot)))


# ── Steering vector collection (hooks on JointTransformerBlock.attn) ──────

# SD3 JointTransformerBlock.forward returns (encoder_hidden_states, hidden_states)
# We need to register a hook that intercepts the OUTPUT of self.attn, which
# returns (context_out, image_out) as a tuple from the processor.
# However, register_forward_hook on Attention gives the full output tuple.
# We record image_out (index 1 of the attn tuple) — the image token activations.

class _SD3CACollector:
    """Hook: records mean-pooled image-token attn outputs per block per step."""

    def __init__(self):
        self._step = 0
        self._block_idx: dict = defaultdict(int)
        self.data: dict = defaultdict(lambda: defaultdict(list))
        self._hooks: list = []

    def reset_step(self):
        self._step += 1
        self._block_idx = defaultdict(int)

    def _hook_fn(self, block_i: int):
        def fn(module, inp, out):
            # JointTransformerBlock.attn returns (context_out, image_out)
            # out is a tuple when encoder_hidden_states is not None
            if isinstance(out, tuple):
                img_out = out[1]    # image tokens  [B, img_seq, dim]
            else:
                img_out = out       # context_pre_only last block

            x = img_out.detach().float()
            if x.shape[0] > 1:
                x = x[x.shape[0] // 2:]   # CFG conditioned half
            vec = x.mean(dim=(0, 1))       # [dim]
            self.data[self._step][block_i].append(vec.cpu())
        return fn

    def register(self, transformer: SD3Transformer2DModel):
        for i, block in enumerate(transformer.transformer_blocks):
            h = block.attn.register_forward_hook(self._hook_fn(i))
            self._hooks.append(h)
        log.info("Collector hooks registered on %d blocks", len(self._hooks))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def build_steering_vectors_sd3(
    model_id: str,
    num_pairs: int = 50,
    save_path: Optional[str] = None,
    device: str = "cuda",
    num_steps: int = 28,
    seed: int = 0,
) -> Tuple[dict, dict]:
    """
    Build per-block steering vectors for SD3 from paired concept generations.

    Returns (sv, layer_align) in the same format as the SD v1.4 version
    so the rest of the pipeline is drop-in compatible.

    sv[step][block_i] = list of 1 tensor (one per block, not split by
    down/mid/up as in UNet — SD3 is a flat stack of 24 blocks).
    """
    if save_path and os.path.exists(save_path):
        log.info("Loading cached SD3 steering vectors from %s", save_path)
        return torch.load(save_path, map_location="cpu", weights_only=False)

    log.info("Building SD3 steering vectors  num_pairs=%d", num_pairs)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # Build T5 direction for text-weighted pair weighting
    d = build_t5_direction(pipe, device)

    # Expand pairs to num_pairs by cycling
    pairs = (PAIRED_PROMPTS * ((num_pairs // len(PAIRED_PROMPTS)) + 1))[:num_pairs]

    # Compute text-space weight for each positive prompt
    def _pair_weight(pos_prompt: str) -> float:
        tok3, enc3 = pipe.tokenizer_3, pipe.text_encoder_3
        t = tok3(pos_prompt, return_tensors="pt", padding="max_length",
                 max_length=tok3.model_max_length, truncation=True).to(device)
        with torch.no_grad():
            emb = enc3(t.input_ids)[0][0].float()  # [seq, 4096]
        d_dev = d.to(device, dtype=emb.dtype)
        return float(torch.sigmoid(3.0 * (emb @ d_dev).max()))

    pair_weights = [_pair_weight(p) for p, _ in pairs]
    log.info("Pair weights — min=%.3f max=%.3f mean=%.3f",
             min(pair_weights), max(pair_weights),
             sum(pair_weights) / len(pair_weights))

    def _collect(prompts: list, weights: list) -> dict:
        collector = _SD3CACollector()
        collector.register(pipe.transformer)

        wsum: dict = defaultdict(lambda: defaultdict(lambda: None))
        wacc: dict = defaultdict(lambda: defaultdict(float))

        for i, (prompt, w) in enumerate(zip(prompts, weights)):
            log.info("  [%d/%d] w=%.3f  %s", i + 1, len(prompts), w, prompt[:60])
            collector._step = 0
            collector._block_idx = defaultdict(int)
            collector.data = defaultdict(lambda: defaultdict(list))

            def _cb(pipe_, step, timestep, kwargs):
                collector.reset_step()
                return kwargs

            pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                generator=torch.Generator(device=device).manual_seed(seed + i),
                callback_on_step_end=_cb,
                output_type="latent",
            )

            for step, block_data in collector.data.items():
                for bi, vecs in block_data.items():
                    vec = vecs[0]   # one vec per block per step
                    if wsum[step][bi] is None:
                        wsum[step][bi] = vec.clone() * w
                        wacc[step][bi] = w
                    else:
                        wsum[step][bi] = wsum[step][bi] + vec * w
                        wacc[step][bi] += w

        collector.remove()

        means = {}
        for step in wsum:
            means[step] = {}
            for bi in wsum[step]:
                if wsum[step][bi] is not None:
                    means[step][bi] = [wsum[step][bi] / max(wacc[step][bi], 1e-8)]
        return means

    log.info("Collecting POSITIVE activations ...")
    pos = _collect([p for p, _ in pairs], pair_weights)
    log.info("Collecting NEGATIVE activations ...")
    neg = _collect([n for _, n in pairs], [1.0] * len(pairs))

    # Build steering vectors + layer alignment scores
    sv: dict = {}
    align_sum: dict = defaultdict(lambda: defaultdict(float))
    align_cnt: dict = defaultdict(lambda: defaultdict(int))

    for step in pos:
        if step not in neg:
            continue
        sv[step] = {}
        for bi in pos[step]:
            if bi not in neg.get(step, {}):
                continue
            diff = pos[step][bi][0] - neg[step][bi][0]
            v = (diff / diff.norm().clamp(min=1e-8)).cpu()
            sv[step][bi] = [v]

            pos_vec = pos[step][bi][0].cpu().float()
            alignment = (v.float() @ pos_vec).abs().item()
            align_sum[bi][step] += alignment
            align_cnt[bi][step] += 1

    layer_align: dict = {}
    for bi in align_sum:
        scores = [align_sum[bi][s] / max(align_cnt[bi][s], 1)
                  for s in sorted(align_sum[bi])]
        layer_align[bi] = sum(scores) / max(len(scores), 1)

    log.info("Layer alignment scores (per block):")
    for bi in sorted(layer_align):
        log.info("  block %02d: %.4f", bi, layer_align[bi])

    del pipe
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    result = {"sv": sv, "layer_align": layer_align, "t5_direction": d}
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        torch.save(result, save_path)
        log.info("Saved -> %s", save_path)
    return sv, layer_align, d


# ── Inference hook ────────────────────────────────────────────────────────

class _SD3SteerHook:
    """
    Forward hook on each JointTransformerBlock.attn for SD3 inference.

    Gate formula (same as TCSA):
        alpha_k = max(beta_b * w * tcl * <v, img_tok_k>, 0)
        img_tok_k_new = img_tok_k - alpha_k * v

    Applied to image tokens only; context tokens untouched.
    beta_b is adaptive per-block (softmax-normalized alignment scores).
    """

    def __init__(
        self,
        sv: dict,
        layer_align: dict,
        beta: float = 2.0,
        align_temp: float = 1.0,
    ):
        self.sv = sv or {}
        self._step = 0
        self._w = 0.0
        self._tcl = 1.0
        self._hooks: list = []

        # Build adaptive beta per block
        self.beta_block: dict = {}
        if layer_align:
            blocks = sorted(layer_align.keys())
            n = len(blocks)
            vals = [layer_align[b] for b in blocks]
            max_v = max(vals)
            exps = [math.exp((v - max_v) / max(align_temp, 1e-8)) for v in vals]
            s = sum(exps)
            for b, e in zip(blocks, exps):
                self.beta_block[b] = beta * (e / s) * n
        else:
            log.info("No layer_align — using global beta=%.2f for all blocks", beta)

        self._block_idx = 0

    def set_step(self, step: int, w: float, tcl: float = 1.0):
        self._step = step
        self._w = w
        self._tcl = tcl
        self._block_idx = 0

    def _lookup(self, block_i: int) -> Optional[torch.Tensor]:
        try:
            return self.sv[self._step][block_i][0]
        except (KeyError, IndexError):
            if not self.sv:
                return None
            closest = min(self.sv.keys(), key=lambda s: abs(s - self._step))
            try:
                return self.sv[closest][block_i][0]
            except (KeyError, IndexError):
                return None

    def _hook_fn(self, block_i: int):
        def fn(module, inp, out):
            w = self._w
            if w <= 0.0:
                return out

            # out is (context_out, image_out) tuple, or just image_out for last block
            is_tuple = isinstance(out, tuple)
            if is_tuple:
                ctx_out, img_out = out
            else:
                # context_pre_only last block — out is just hidden_states
                img_out = out
                ctx_out = None

            v = self._lookup(block_i)
            if v is None:
                return out

            v = v.to(img_out.device, dtype=img_out.dtype)
            beta_b = self.beta_block.get(block_i, 2.0)
            tcl = self._tcl

            # Steer only the conditioned CFG half of image tokens
            x = img_out
            if x.shape[0] > 1:
                cond = x[x.shape[0] // 2:].clone()   # [B/2, img_seq, dim]
            else:
                cond = x.clone()

            # per-patch alignment: [img_seq]
            dots = cond[0] @ v         # [img_seq]

            alpha = beta_b * w * tcl * dots
            alpha = torch.clamp(alpha, min=0.0)

            cond[0] = cond[0] - alpha.unsqueeze(-1) * v.unsqueeze(0)

            if x.shape[0] > 1:
                new_img = x.clone()
                new_img[x.shape[0] // 2:] = cond
            else:
                new_img = cond

            if is_tuple and ctx_out is not None:
                return (ctx_out, new_img.to(out[1].dtype))
            return new_img.to(out.dtype)

        return fn

    def register(self, transformer: SD3Transformer2DModel):
        for i, block in enumerate(transformer.transformer_blocks):
            h = block.attn.register_forward_hook(self._hook_fn(i))
            self._hooks.append(h)
        log.info("Steering hooks registered on %d blocks", len(self._hooks))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── SD3 TCSA Pipeline ─────────────────────────────────────────────────────

class TCSA_SD3:
    """
    TCSA adapted for SD3 — drop-in replacement for the SD v1.4 pipeline.

    Mode options (same as original TCSA):
        'none'  — unsafe baseline
        'text'  — T5 embedding orthogonalization only
        'ca'    — activation steering only
        'both'  — full TCSA
    """

    def __init__(
        self,
        pipe: StableDiffusion3Pipeline,
        sv: dict,
        layer_align: dict,
        t5_direction: torch.Tensor,
        beta: float = 2.0,
        max_proj_ratio: float = 0.5,
        device: str = "cuda",
    ):
        self.pipe = pipe
        self.d = t5_direction.cpu()
        self.max_proj_ratio = max_proj_ratio
        self.device = device
        self._hook = _SD3SteerHook(sv, layer_align, beta=beta)
        self._hook.register(pipe.transformer)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        sv: dict,
        layer_align: dict,
        t5_direction: torch.Tensor,
        device: str = "cuda",
        beta: float = 2.0,
        max_proj_ratio: float = 0.5,
        torch_dtype=torch.float16,
    ) -> "TCSA_SD3":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe, sv, layer_align, t5_direction, beta=beta,
                   max_proj_ratio=max_proj_ratio, device=device)

    def _encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """Encode prompt through all three SD3 encoders."""
        with torch.no_grad():
            (prompt_embeds, neg_embeds,
             pooled_embeds, neg_pooled_embeds) = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                negative_prompt=negative_prompt or "",
                negative_prompt_2=negative_prompt or "",
                negative_prompt_3=negative_prompt or "",
                device=self.device,
            )
        return prompt_embeds, neg_embeds, pooled_embeds, neg_pooled_embeds

    def generate(
        self,
        prompt: str,
        mode: str = "both",
        schedule: Optional[Schedule] = None,
        seed: int = 42,
        negative_prompt: str = "",
        guidance_scale: float = 7.0,
        num_steps: int = 28,
        height: int = 512,
        width: int = 512,
    ) -> Image.Image:
        if mode not in ("none", "text", "ca", "both"):
            raise ValueError(f"Bad mode '{mode}'")
        use_text = mode in ("text", "both")
        use_ca = mode in ("ca", "both")

        # Encode prompt
        prompt_embeds, neg_embeds, pooled_embeds, neg_pooled_embeds = \
            self._encode_prompt(prompt, negative_prompt)

        # Text-space orthogonalization (T5 embedding)
        prompt_embeds_ortho = None
        if use_text:
            prompt_embeds_ortho = orthogonalize_t5_embedding(
                prompt_embeds, self.d,
                max_proj_ratio=self.max_proj_ratio,
                renormalize=True,
            )

        # TCL from original prompt embedding
        tcl = compute_tcl(prompt_embeds, self.d) if use_ca else 1.0
        log.debug("TCL=%.3f  prompt='%s'", tcl, prompt[:60])

        # Set up scheduler
        self.pipe.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        # Initial latents
        shape = (1, self.pipe.transformer.config.in_channels,
                 height // self.pipe.vae_scale_factor,
                 width // self.pipe.vae_scale_factor)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(shape, generator=gen, device=self.device,
                              dtype=self.pipe.transformer.dtype)

        # Denoising loop — manual to inject our interventions at each step
        T_max = float(timesteps[0].item())
        for step_idx, t in enumerate(timesteps):
            tau = float(t.item()) / T_max
            w = schedule.weight(tau) if schedule is not None else 1.0

            # Interpolate text embedding for text-space intervention
            if use_text and prompt_embeds_ortho is not None:
                if w >= 1.0:
                    pe_used = prompt_embeds_ortho
                elif w <= 0.0:
                    pe_used = prompt_embeds
                else:
                    pe_used = (1.0 - w) * prompt_embeds + w * prompt_embeds_ortho
            else:
                pe_used = prompt_embeds

            # Set CA hook gate
            self._hook.set_step(step_idx, w if use_ca else 0.0, tcl)

            # Classifier-free guidance step
            latent_input = torch.cat([latents] * 2)
            # FlowMatchEulerDiscreteScheduler (SD3) does not use scale_model_input

            with torch.no_grad():
                noise_pred = self.pipe.transformer(
                    hidden_states=latent_input,
                    timestep=torch.cat([t.unsqueeze(0)] * 2),
                    encoder_hidden_states=torch.cat([neg_embeds, pe_used]),
                    pooled_projections=torch.cat([neg_pooled_embeds, pooled_embeds]),
                ).sample

            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents = self.pipe.scheduler.step(noise, t, latents).prev_sample

        # Decode
        latents = latents / self.pipe.vae.config.scaling_factor
        with torch.no_grad():
            img = self.pipe.vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((img[0] * 255).astype("uint8"))

    def remove_hooks(self):
        self._hook.remove()


# ── Grid / comparison helpers (same as original TCSA) ─────────────────────

LABEL_H, FONT_SZ, PAD = 55, 13, 4
PROMPT_W = 220
COL_LABELS = [
    "1. Unsafe (SD3)",
    "2. Text-Orth Only",
    "3. CA-Steer Only",
    "4. TCSA (both)",
]


def _font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_comparison(
    wrapper: TCSA_SD3,
    prompt: str,
    schedule: Schedule,
    seed: int = 42,
    guidance_scale: float = 7.0,
    num_steps: int = 28,
    height: int = 512,
    width: int = 512,
) -> List[Image.Image]:
    out = []
    for tag, mode in [("unsafe", "none"), ("text", "text"),
                      ("ca", "ca"), ("both", "both")]:
        log.info("  generating %s ...", tag)
        out.append(wrapper.generate(
            prompt, mode=mode, schedule=schedule, seed=seed,
            guidance_scale=guidance_scale, num_steps=num_steps,
            height=height, width=width))
    return out


def make_grid_row(images: List[Image.Image], prompt_text: str = "") -> Image.Image:
    w, h = images[0].size
    left = PROMPT_W if prompt_text else 0
    n = len(images)
    row = Image.new("RGB",
                    (left + n * w + (n - 1) * PAD, h + LABEL_H),
                    (210, 210, 210))

    if prompt_text:
        panel = Image.new("RGB", (PROMPT_W, h + LABEL_H), (235, 240, 255))
        draw = ImageDraw.Draw(panel)
        font = _font(11)
        y = 8
        for line in textwrap.wrap(prompt_text, width=28):
            draw.text((6, y), line, font=font, fill=(10, 10, 50))
            y += 14
        row.paste(panel, (0, 0))

    for i, (img, lbl) in enumerate(zip(images, COL_LABELS)):
        x = left + i * (w + PAD)
        row.paste(img, (x, 0))
        lab = Image.new("RGB", (w, LABEL_H), (250, 250, 250))
        draw = ImageDraw.Draw(lab)
        font = _font(FONT_SZ)
        for line in textwrap.wrap(lbl, width=max(1, w // 8)):
            bb = draw.textbbox((0, 0), line, font=font)
            draw.text(((w - (bb[2] - bb[0])) // 2, 5), line,
                      font=font, fill=(30, 30, 30))
        row.paste(lab, (x, h))
    return row


def slug(text: str, n: int = 28) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:n]


# ── Evaluation runner (for Table 4 completion) ────────────────────────────

def run_eval(
    prompts_path: str,
    sv: dict,
    layer_align: dict,
    t5_dir: torch.Tensor,
    schedule: Schedule,
    save_dir: str,
    model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
    device: str = "cuda",
    beta: float = 2.0,
    num_steps: int = 28,
    guidance_scale: float = 7.0,
    height: int = 512,
    width: int = 512,
    seed: int = 42,
    col: str = "prompt",
    n: Optional[int] = None,
) -> None:
    """
    Generate one image per prompt as {i}_0.png for ASR evaluation.
    Compatible with Receler / SAFREE evaluation scripts.
    """
    import pandas as pd
    from tqdm import tqdm

    os.makedirs(save_dir, exist_ok=True)

    if prompts_path.endswith(".csv"):
        df = pd.read_csv(prompts_path)
        prompts = df[col].dropna().astype(str).tolist()
        indices = list(df.index)
    else:
        with open(prompts_path) as f:
            prompts = [l.strip() for l in f if l.strip()]
        indices = list(range(len(prompts)))

    if n is not None:
        prompts, indices = prompts[:n], indices[:n]

    log.info("Generating %d images -> %s", len(prompts), save_dir)

    wrapper = TCSA_SD3.from_pretrained(
        model_id, sv, layer_align, t5_dir,
        device=device, beta=beta,
    )

    for global_i, prompt in tqdm(zip(indices, prompts), total=len(indices)):
        out_path = os.path.join(save_dir, f"{global_i}_0.png")
        if os.path.exists(out_path):
            continue
        if not prompt.strip():
            continue
        img = wrapper.generate(
            prompt=prompt, mode="both", schedule=schedule,
            seed=seed, guidance_scale=guidance_scale,
            num_steps=num_steps, height=height, width=width,
        )
        img.save(out_path)

    wrapper.remove_hooks()
    done = sum(1 for i in indices
               if os.path.exists(os.path.join(save_dir, f"{i}_0.png")))
    print(f"\nDone. {done}/{len(indices)} images saved to {save_dir}")
    print("Next: run q16_nudenet_detect.py or ASR evaluation on this folder.")


# ── CLI ───────────────────────────────────────────────────────────────────

def _load_sv(sv_path: str, model_id: str, device: str, num_pairs: int,
             num_steps: int, seed: int):
    if os.path.exists(sv_path):
        data = torch.load(sv_path, map_location="cpu", weights_only=False)
        sv = data["sv"]
        layer_align = data["layer_align"]
        t5_dir = data.get("t5_direction", None)
        if t5_dir is None:
            raise RuntimeError(
                "Old cache file missing 't5_direction'. "
                "Delete the .pt file and re-run build-vectors.")
        return sv, layer_align, t5_dir
    return build_steering_vectors_sd3(
        model_id, num_pairs=num_pairs,
        save_path=sv_path, device=device,
        num_steps=num_steps, seed=seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="TCSA for SD3 — training-free concept erasure via JointTransformerBlock hooks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── build-vectors ──
    bv = sub.add_parser("build-vectors",
        help="Pre-compute and cache SD3 steering vectors.")
    bv.add_argument("--model_id",
        default="stabilityai/stable-diffusion-3-medium-diffusers")
    bv.add_argument("--num_pairs", type=int, default=50)
    bv.add_argument("--save_path", required=True)
    bv.add_argument("--device", default="cuda")
    bv.add_argument("--steps", type=int, default=28)
    bv.add_argument("--seed", type=int, default=0)

    # ── single ──
    sg = sub.add_parser("single",
        help="4-column comparison image for one prompt.")
    sg.add_argument("--model_id",
        default="stabilityai/stable-diffusion-3-medium-diffusers")
    sg.add_argument("--prompt", required=True)
    sg.add_argument("--sv_path", required=True)
    sg.add_argument("--save_path", default="tcsa_sd3_comparison.png")
    sg.add_argument("--num_pairs", type=int, default=50)
    sg.add_argument("--device", default="cuda")
    sg.add_argument("--steps", type=int, default=28)
    sg.add_argument("--guidance", type=float, default=7.0)
    sg.add_argument("--img_size", type=int, default=512)
    sg.add_argument("--seed", type=int, default=42)
    sg.add_argument("--beta", type=float, default=2.0)

    # ── eval ──
    ev = sub.add_parser("eval",
        help="Batch eval — generate {i}_0.png for ASR scoring.")
    ev.add_argument("--model_id",
        default="stabilityai/stable-diffusion-3-medium-diffusers")
    ev.add_argument("--prompts_path", required=True)
    ev.add_argument("--sv_path", required=True)
    ev.add_argument("--save_dir", required=True)
    ev.add_argument("--col", default="prompt")
    ev.add_argument("--n", type=int, default=None)
    ev.add_argument("--num_pairs", type=int, default=50)
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--steps", type=int, default=28)
    ev.add_argument("--guidance", type=float, default=7.0)
    ev.add_argument("--img_size", type=int, default=512)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--beta", type=float, default=2.0)

    args = p.parse_args()

    if args.cmd == "build-vectors":
        build_steering_vectors_sd3(
            model_id=args.model_id, num_pairs=args.num_pairs,
            save_path=args.save_path, device=args.device,
            num_steps=args.steps, seed=args.seed)

    elif args.cmd == "single":
        sv, layer_align, t5_dir = _load_sv(
            args.sv_path, args.model_id, args.device,
            args.num_pairs, args.steps, seed=0)
        wrapper = TCSA_SD3.from_pretrained(
            args.model_id, sv, layer_align, t5_dir,
            device=args.device, beta=args.beta)
        sched = Schedule()
        images = make_comparison(
            wrapper, args.prompt, sched,
            seed=args.seed, guidance_scale=args.guidance,
            num_steps=args.steps, height=args.img_size, width=args.img_size)
        row = make_grid_row(images, prompt_text=args.prompt)
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or ".", exist_ok=True)
        row.save(args.save_path)
        wrapper.remove_hooks()
        print(f"Saved -> {args.save_path}")

    elif args.cmd == "eval":
        sv, layer_align, t5_dir = _load_sv(
            args.sv_path, args.model_id, args.device,
            args.num_pairs, args.steps, seed=0)
        run_eval(
            prompts_path=args.prompts_path,
            sv=sv, layer_align=layer_align, t5_dir=t5_dir,
            schedule=Schedule(),
            save_dir=args.save_dir,
            model_id=args.model_id, device=args.device,
            beta=args.beta, num_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.img_size, width=args.img_size,
            seed=args.seed, col=args.col, n=args.n)
