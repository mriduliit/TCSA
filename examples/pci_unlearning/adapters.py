"""
DiffusionAdapter protocol and reference implementations.

All higher-level PCI code interacts only with the DiffusionAdapter interface.
τ is normalized to [0, 1] where τ=1 is pure noise and τ=0 is clean image.
The normalized τ is derived as τ = native_timestep / 1000.0 for all supported
scheduler families (DDPM, DDIM, Euler, FlowMatch) since diffusers stores all
of them in the [0, 1000] integer range internally.
"""

from __future__ import annotations

import abc
from typing import Any, List, Optional, Tuple

import torch
from PIL import Image


class DiffusionAdapter(abc.ABC):
    """Architecture-agnostic interface for a text-to-image diffusion pipeline.

    All timestep-facing APIs use τ ∈ [0, 1] exclusively.
    τ = 1 → pure Gaussian noise; τ = 0 → clean decoded image.
    """

    # ------------------------------------------------------------------ #
    # Prompt                                                               #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
    ) -> Any:
        """Encode *prompt* into model-ready embeddings.

        Returns a tuple ``(uncond_emb, cond_emb)`` so callers can build
        classifier-free guidance themselves, OR a single stacked tensor
        ``[uncond; cond]`` — the concrete adapter decides; ``denoise_step``
        must accept whatever this method returns.
        """

    # ------------------------------------------------------------------ #
    # Latent initialisation                                                #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def init_latents(
        self,
        seed: int,
        batch_size: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample initial noise latents xT for a given seed.

        Returns a tensor with shape ``[batch, C, H_lat, W_lat]``.
        """

    # ------------------------------------------------------------------ #
    # Scheduler                                                            #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def get_tau_schedule(self) -> List[float]:
        """Return the list of τ values for each inference step.

        The list is descending: ``[τ_0, τ_1, ..., τ_{T-1}]`` where
        ``τ_0 ≈ 1`` (high noise) and ``τ_{T-1} ≈ 0`` (nearly clean).
        Length equals the number of inference steps T.
        """

    @abc.abstractmethod
    def get_native_timesteps(self) -> List[Any]:
        """Return the native scheduler timestep values (one per step).

        These are passed verbatim to ``denoise_step`` as ``native_t``.
        Order mirrors ``get_tau_schedule()``.
        """

    # ------------------------------------------------------------------ #
    # Single denoising step                                                #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def denoise_step(
        self,
        latent: torch.Tensor,
        embedding: Any,
        native_t: Any,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """Apply one denoising step and return the updated latent.

        Args:
            latent:        Current noisy latent x_t, shape ``[B, C, H, W]``.
            embedding:     Output of ``encode_prompt`` (adapter-specific).
            native_t:      Native scheduler timestep from ``get_native_timesteps()``.
            guidance_scale: CFG scale ω.
        Returns:
            Updated latent x_{t-1}, same shape as input.
        """

    # ------------------------------------------------------------------ #
    # Decoding                                                             #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def decode(self, latent: torch.Tensor) -> Image.Image:
        """Decode a clean latent x_0 to a PIL RGB image."""

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    def tau_to_step_index(self, tau: float) -> int:
        """Return the step index k whose τ value is closest to *tau*."""
        taus = self.get_tau_schedule()
        return min(range(len(taus)), key=lambda i: abs(taus[i] - tau))

    def set_num_inference_steps(self, steps: int) -> None:  # optional override
        """Reconfigure the scheduler for a different step count."""
        raise NotImplementedError("Override in subclass if supported.")

    def init_latents_batch(
        self,
        seeds: List[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate batched latents — one element per seed.  Default: sequential."""
        latents = [self.init_latents(s, 1, height, width) for s in seeds]
        return torch.cat(latents, dim=0)

    def decode_batch(self, latent: torch.Tensor) -> List[Image.Image]:
        """Decode a batched latent to a list of PIL images.  Default: sequential."""
        return [self.decode(latent[i : i + 1]) for i in range(latent.shape[0])]


# ======================================================================= #
#  UNet adapter  (SD 1.5, SD 2.1, SDXL)                                  #
# ======================================================================= #

class UNetAdapter(DiffusionAdapter):
    """Adapter for StableDiffusionPipeline / StableDiffusionXLPipeline."""

    def __init__(
        self,
        pipeline: Any,
        num_inference_steps: int = 50,
        device: str = "cuda",
    ) -> None:
        self.pipe = pipeline
        self.device = device
        self._num_steps = num_inference_steps
        self._is_xl = "XL" in type(pipeline).__name__ or "xl" in type(pipeline).__name__.lower()
        # Initialise scheduler timestep list
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # ---- prompt ---- #

    def encode_prompt(self, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(uncond_emb, cond_emb)`` each shape ``[1, seq, dim]``."""
        device = self.device
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        def _encode(text: str) -> torch.Tensor:
            toks = tokenizer(
                text,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                emb = text_encoder(toks.input_ids.to(device))[0]
            return emb

        return _encode(negative_prompt), _encode(prompt)

    # ---- latents ---- #

    def init_latents(
        self,
        seed: int,
        batch_size: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        height = height or (self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor)
        width = width or (self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor)
        shape = (
            batch_size,
            self.pipe.unet.config.in_channels,
            height // self.pipe.vae_scale_factor,
            width // self.pipe.vae_scale_factor,
        )
        # Reset scheduler state (step_index) before each new image — DPMSolver
        # and other stateful schedulers increment an internal counter that must
        # be zeroed between runs.
        self.pipe.scheduler.set_timesteps(self._num_steps, device=self.device)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(shape, generator=gen, device=self.device,
                              dtype=torch.float16 if self._is_xl else torch.float32)
        latents = latents * self.pipe.scheduler.init_noise_sigma
        return latents

    # ---- schedule ---- #

    def get_tau_schedule(self) -> List[float]:
        ts = self.pipe.scheduler.timesteps
        return [float(t.item()) / 1000.0 for t in ts]

    def get_native_timesteps(self) -> List[torch.Tensor]:
        return list(self.pipe.scheduler.timesteps)

    def set_num_inference_steps(self, steps: int) -> None:
        self._num_steps = steps
        self.pipe.scheduler.set_timesteps(steps, device=self.device)

    def init_latents_batch(
        self,
        seeds: List[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        """One set_timesteps call, then generate one latent per seed and stack."""
        height = height or (self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor)
        width = width or (self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor)
        single_shape = (
            1,
            self.pipe.unet.config.in_channels,
            height // self.pipe.vae_scale_factor,
            width // self.pipe.vae_scale_factor,
        )
        self.pipe.scheduler.set_timesteps(self._num_steps, device=self.device)
        dtype = torch.float16 if self._is_xl else torch.float32
        parts = []
        for seed in seeds:
            gen = torch.Generator(device=self.device).manual_seed(seed)
            parts.append(torch.randn(single_shape, generator=gen, device=self.device, dtype=dtype))
        return torch.cat(parts, dim=0) * self.pipe.scheduler.init_noise_sigma

    def decode_batch(self, latent: torch.Tensor) -> List[Image.Image]:
        """Decode all images in a batch with a single VAE forward pass."""
        latent = latent / self.pipe.vae.config.scaling_factor
        with torch.no_grad():
            image = self.pipe.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return [Image.fromarray((image[i] * 255).astype("uint8")) for i in range(image.shape[0])]

    # ---- step ---- #

    def denoise_step(
        self,
        latent: torch.Tensor,
        embedding: Tuple[torch.Tensor, torch.Tensor],
        native_t: Any,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        uncond_emb, cond_emb = embedding
        B = latent.shape[0]
        # Tile embeddings to match batch size so B>1 (batched seeds) works correctly.
        combined_emb = torch.cat([uncond_emb.expand(B, -1, -1), cond_emb.expand(B, -1, -1)])
        latent_input = torch.cat([latent] * 2)
        latent_input = self.pipe.scheduler.scale_model_input(latent_input, native_t)
        with torch.no_grad():
            noise_pred = self.pipe.unet(
                latent_input, native_t, encoder_hidden_states=combined_emb
            ).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        return self.pipe.scheduler.step(noise_pred, native_t, latent).prev_sample

    # ---- decode ---- #

    def decode(self, latent: torch.Tensor) -> Image.Image:
        latent = latent / self.pipe.vae.config.scaling_factor
        with torch.no_grad():
            image = self.pipe.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((image[0] * 255).astype("uint8"))


# ======================================================================= #
#  DiT adapter  (PixArt-α / PixArt-Σ)                                    #
# ======================================================================= #

class DiTAdapter(DiffusionAdapter):
    """Adapter for PixArtAlphaPipeline (DiT architecture)."""

    def __init__(
        self,
        pipeline: Any,
        num_inference_steps: int = 50,
        device: str = "cuda",
    ) -> None:
        self.pipe = pipeline
        self.device = device
        self._num_steps = num_inference_steps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    def encode_prompt(self, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            (
                cond_emb, _, uncond_emb, _
            ) = self.pipe.encode_prompt(
                prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
                num_images_per_prompt=1,
                device=self.device,
            )
        return uncond_emb, cond_emb

    def init_latents(
        self,
        seed: int,
        batch_size: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        height = height or self.pipe.transformer.config.sample_size * self.pipe.vae_scale_factor
        width = width or self.pipe.transformer.config.sample_size * self.pipe.vae_scale_factor
        shape = (
            batch_size,
            self.pipe.transformer.in_channels,
            height // self.pipe.vae_scale_factor,
            width // self.pipe.vae_scale_factor,
        )
        gen = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(shape, generator=gen, device=self.device, dtype=torch.float16)
        latents = latents * self.pipe.scheduler.init_noise_sigma
        return latents

    def get_tau_schedule(self) -> List[float]:
        ts = self.pipe.scheduler.timesteps
        return [float(t.item()) / 1000.0 for t in ts]

    def get_native_timesteps(self) -> List[torch.Tensor]:
        return list(self.pipe.scheduler.timesteps)

    def set_num_inference_steps(self, steps: int) -> None:
        self._num_steps = steps
        self.pipe.scheduler.set_timesteps(steps, device=self.device)

    def denoise_step(
        self,
        latent: torch.Tensor,
        embedding: Tuple[torch.Tensor, torch.Tensor],
        native_t: Any,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        uncond_emb, cond_emb = embedding
        latent_input = torch.cat([latent] * 2)
        latent_input = self.pipe.scheduler.scale_model_input(latent_input, native_t)
        combined_emb = torch.cat([cond_emb, uncond_emb])  # PixArt uses (cond, uncond) order
        bs = latent_input.shape[0]
        t_batch = native_t.expand(bs) if hasattr(native_t, "expand") else \
                  torch.tensor([native_t] * bs, device=self.device)
        with torch.no_grad():
            noise_pred = self.pipe.transformer(
                latent_input,
                encoder_hidden_states=combined_emb,
                timestep=t_batch,
            ).sample
        noise_cond, noise_uncond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        return self.pipe.scheduler.step(noise_pred, native_t, latent).prev_sample

    def decode(self, latent: torch.Tensor) -> Image.Image:
        latent = latent / self.pipe.vae.config.scaling_factor
        with torch.no_grad():
            image = self.pipe.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((image[0] * 255).astype("uint8"))


# ======================================================================= #
#  FlowMatch adapter  (SD3.5, FLUX)                                       #
# ======================================================================= #

class FlowMatchAdapter(DiffusionAdapter):
    """Adapter for StableDiffusion3Pipeline and FluxPipeline.

    FlowMatch models use a rectified-flow scheduler where the native timesteps
    are already normalised to [0, 1000], but conceptually t=1000 is pure noise
    and t=0 is clean — same as DDPM so τ = t/1000 is consistent.
    CIS curves for these models are expected to be sharper and shifted earlier
    compared to UNet/DiT models (paper Sec. 4.2).
    """

    def __init__(
        self,
        pipeline: Any,
        num_inference_steps: int = 50,
        device: str = "cuda",
    ) -> None:
        self.pipe = pipeline
        self.device = device
        self._num_steps = num_inference_steps
        self._is_flux = "Flux" in type(pipeline).__name__
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    def encode_prompt(self, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        if self._is_flux:
            # FLUX uses a dual text encoder; delegate to pipeline helper
            (
                cond_emb, _, pooled_cond, _
            ) = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
            (
                uncond_emb, _, pooled_uncond, _
            ) = self.pipe.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
            return (uncond_emb, pooled_uncond), (cond_emb, pooled_cond)
        else:
            # SD3 encode_prompt returns (prompt_embeds, negative_prompt_embeds, pooled, neg_pooled)
            cond_emb, uncond_emb, pooled_cond, pooled_uncond = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                negative_prompt_3=negative_prompt,
                device=self.device,
                num_images_per_prompt=1,
            )
            return (uncond_emb, pooled_uncond), (cond_emb, pooled_cond)

    def init_latents(
        self,
        seed: int,
        batch_size: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        height = height or 1024
        width = width or 1024
        vae_scale = self.pipe.vae_scale_factor
        channels = self.pipe.transformer.config.in_channels
        shape = (batch_size, channels, height // vae_scale, width // vae_scale)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(shape, generator=gen, device=self.device, dtype=torch.bfloat16)

    def get_tau_schedule(self) -> List[float]:
        ts = self.pipe.scheduler.timesteps
        # FlowMatch sigmas in [0, 1]; map to τ ∈ [0, 1] via t/1000 if integer, else direct
        result = []
        for t in ts:
            val = float(t.item()) if hasattr(t, "item") else float(t)
            result.append(val / 1000.0 if val > 1.0 else val)
        return result

    def get_native_timesteps(self) -> List[Any]:
        return list(self.pipe.scheduler.timesteps)

    def set_num_inference_steps(self, steps: int) -> None:
        self._num_steps = steps
        self.pipe.scheduler.set_timesteps(steps, device=self.device)

    def denoise_step(
        self,
        latent: torch.Tensor,
        embedding: Any,
        native_t: Any,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        uncond_data, cond_data = embedding
        if self._is_flux:
            uncond_emb, pooled_uncond = uncond_data
            cond_emb, pooled_cond = cond_data
            latent_input = torch.cat([latent] * 2)
            combined_emb = torch.cat([uncond_emb, cond_emb])
            pooled = torch.cat([pooled_uncond, pooled_cond])
            bs = latent_input.shape[0]
            t_batch = native_t.expand(bs) if hasattr(native_t, "expand") else \
                      torch.tensor([native_t] * bs, device=self.device)
            with torch.no_grad():
                noise_pred = self.pipe.transformer(
                    hidden_states=latent_input,
                    timestep=t_batch / 1000.0,
                    encoder_hidden_states=combined_emb,
                    pooled_projections=pooled,
                ).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)
        else:
            uncond_emb, pooled_uncond = uncond_data
            cond_emb, pooled_cond = cond_data
            latent_input = torch.cat([latent] * 2)
            combined_emb = torch.cat([uncond_emb, cond_emb])
            pooled = torch.cat([pooled_uncond, pooled_cond])
            bs = latent_input.shape[0]
            t_batch = native_t.expand(bs) if hasattr(native_t, "expand") else \
                      torch.tensor([native_t] * bs, device=self.device)
            with torch.no_grad():
                noise_pred = self.pipe.transformer(
                    hidden_states=latent_input,
                    timestep=t_batch,
                    encoder_hidden_states=combined_emb,
                    pooled_projections=pooled,
                ).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        return self.pipe.scheduler.step(noise_pred, native_t, latent).prev_sample

    def decode(self, latent: torch.Tensor) -> Image.Image:
        latent = (latent / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        with torch.no_grad():
            image = self.pipe.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return Image.fromarray((image[0] * 255).astype("uint8"))


# ======================================================================= #
#  Auto-detection factory                                                  #
# ======================================================================= #

def auto_adapter(
    pipeline: Any,
    num_inference_steps: int = 50,
    device: str = "cuda",
) -> DiffusionAdapter:
    """Auto-detect the pipeline class and return the appropriate adapter.

    Supported pipelines (diffusers):
    - StableDiffusionPipeline / StableDiffusionXLPipeline → UNetAdapter
    - PixArtAlphaPipeline / PixArtSigmaPipeline → DiTAdapter
    - StableDiffusion3Pipeline / FluxPipeline → FlowMatchAdapter
    """
    cls_name = type(pipeline).__name__
    if cls_name in ("StableDiffusionPipeline", "StableDiffusionXLPipeline",
                    "ModifiedStableDiffusionPipeline", "ModifiedStableDiffusionXLPipeline"):
        return UNetAdapter(pipeline, num_inference_steps, device)
    elif "PixArt" in cls_name:
        return DiTAdapter(pipeline, num_inference_steps, device)
    elif "StableDiffusion3" in cls_name or "Flux" in cls_name:
        return FlowMatchAdapter(pipeline, num_inference_steps, device)
    else:
        # Default fallback — try UNet adapter heuristically
        import warnings
        warnings.warn(
            f"Unknown pipeline class '{cls_name}'. Falling back to UNetAdapter. "
            "Override manually if incorrect.",
            RuntimeWarning,
            stacklevel=2,
        )
        return UNetAdapter(pipeline, num_inference_steps, device)
