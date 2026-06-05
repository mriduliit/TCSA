"""
Core PCI engine — Prompt-Conditioned Interventions.

Implements the sampling-trajectory switching described in the ICLR 2026 paper
"Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned Interventions".

Usage pattern::

    adapter = auto_adapter(pipe)
    pci = PromptConditionedIntervention(adapter)

    # CIS direction: insert concept at τ=0.5
    image = pci.run_pci("a photo of a person", "a photo of an old person",
                         switch_tau=0.5, seed=42)

    # CDS direction: try to delete concept at τ=0.5
    image = pci.run_cds("a photo of an old person", "a photo of a person",
                         switch_tau=0.5, seed=42)
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

from .adapters import DiffusionAdapter
from .orthogonal import (
    OrthogonalEmbedding,
    build_orthogonal_embedding,
    build_vocab_orthogonal_embedding,
    CONCEPT_VOCAB,
    SAFE_CONCEPT_VOCAB,
)


# ======================================================================= #
#  Censor image                                                             #
# ======================================================================= #

def _make_censor_image(height: int, width: int) -> Image.Image:
    """Return a solid dark-gray placeholder used when a prompt is too unsafe to generate."""
    img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    lines = ["Content", "Blocked"]
    line_h = 28
    total_h = len(lines) * line_h
    y = (height - total_h) // 2
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font)
        x = (width - (bb[2] - bb[0])) // 2
        draw.text((x, y), line, font=font, fill=(180, 180, 180))
        y += line_h
    return img


# ======================================================================= #
#  Embedding format helpers                                                 #
# ======================================================================= #

def _get_cond_tensor(embedding: Any) -> torch.Tensor:
    """Extract the main conditional tensor from any adapter embedding format.

    UNet/DiT format: (uncond_tensor, cond_tensor) → cond_tensor
    FlowMatch format: ((uncond_main, uncond_pooled), (cond_main, cond_pooled)) → cond_main
    """
    _, cond = embedding
    return cond[0] if isinstance(cond, tuple) else cond


def _replace_cond_tensor(embedding: Any, new_cond: torch.Tensor) -> Any:
    """Return a new embedding with the conditional tensor replaced.

    Preserves the adapter-specific format (flat tensor or nested tuple).
    """
    uncond, cond = embedding
    if isinstance(cond, tuple):
        return uncond, (new_cond,) + cond[1:]
    return uncond, new_cond


class PromptConditionedIntervention:
    """Run PCI experiments against a DiffusionAdapter.

    Timestep convention
    -------------------
    All ``switch_tau`` values are in τ ∈ [0, 1] where τ=1 is pure noise
    and τ=0 is clean image.  The denoising loop runs from high τ to low τ.

    CIS experiment (``run_pci``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Denoise under *base_prompt* while τ > switch_tau.
    Switch to *concept_prompt* when τ ≤ switch_tau and continue to τ=0.

    CDS experiment (``run_cds``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Denoise under *concept_prompt* while τ > switch_tau.
    Switch to *base_prompt* when τ ≤ switch_tau and continue to τ=0.
    """

    def __init__(
        self,
        adapter: DiffusionAdapter,
        guidance_scale: float = 7.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> None:
        self.adapter = adapter
        self.guidance_scale = guidance_scale
        self.height = height
        self.width = width

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_pci(
        self,
        base_prompt: str,
        concept_prompt: str,
        switch_tau: float,
        seed: int,
        negative_prompt: str = "",
    ) -> Image.Image:
        """Run Concept Insertion Success experiment.

        Denoises under *base_prompt* until τ drops to *switch_tau*, then
        continues under *concept_prompt*.  Returns the final decoded image.

        Args:
            base_prompt:    Neutral base description (no target concept).
            concept_prompt: Base prompt augmented with the target concept.
            switch_tau:     Normalised τ ∈ [0, 1] at which the prompt switches.
                            τ=1 → switch immediately (always concept_prompt).
                            τ=0 → never switch (always base_prompt).
            seed:           RNG seed for reproducibility.
            negative_prompt: Optional negative conditioning for CFG.
        Returns:
            PIL.Image.Image of the generated sample.
        """
        return self._run_switching(
            first_prompt=base_prompt,
            second_prompt=concept_prompt,
            switch_tau=switch_tau,
            seed=seed,
            negative_prompt=negative_prompt,
            switch_below=True,   # switch to second when τ ≤ switch_tau
        )

    def run_cds(
        self,
        concept_prompt: str,
        base_prompt: str,
        switch_tau: float,
        seed: int,
        negative_prompt: str = "",
    ) -> Image.Image:
        """Run Concept Deletion Success experiment.

        Denoises under *concept_prompt* until τ drops to *switch_tau*, then
        attempts removal by continuing under *base_prompt*.

        Args:
            concept_prompt: Prompt that contains the concept to be deleted.
            base_prompt:    Neutral base prompt without the concept.
            switch_tau:     Normalised τ ∈ [0, 1] at which the prompt switches.
            seed:           RNG seed for reproducibility.
            negative_prompt: Optional negative conditioning for CFG.
        Returns:
            PIL.Image.Image of the generated sample.
        """
        return self._run_switching(
            first_prompt=concept_prompt,
            second_prompt=base_prompt,
            switch_tau=switch_tau,
            seed=seed,
            negative_prompt=negative_prompt,
            switch_below=True,
        )

    def run_full(
        self,
        prompt: str,
        seed: int,
        negative_prompt: str = "",
    ) -> Image.Image:
        """Generate an image without any prompt switching (baseline)."""
        return self._run_switching(
            first_prompt=prompt,
            second_prompt=prompt,
            switch_tau=0.0,  # never switches
            seed=seed,
            negative_prompt=negative_prompt,
            switch_below=True,
        )

    def run_orthogonal(
        self,
        unsafe_prompt: str,
        base_prompt: str,
        schedule: Any,
        seed: int,
        negative_prompt: str = "",
        n_components: int = 1,
        renormalize: bool = True,
    ) -> Image.Image:
        """Generate a safe image using orthogonal embedding interpolation.

        At each denoising step the conditional embedding is blended between
        the original unsafe embedding and its orthogonalized counterpart:

            e(τ) = (1 − λ) · e_unsafe + λ · e_ortho,   λ = schedule.weight(τ)

        where e_ortho has the unsafe semantic direction(s) removed.  This
        continuously suppresses the unsafe concept during the timesteps where
        it forms (high schedule weight) while preserving safe semantic content.

        Args:
            unsafe_prompt:  The original prompt containing the unsafe concept.
            base_prompt:    A safe reference prompt paired with *unsafe_prompt*.
                            Used only to define the unsafe direction — it is not
                            used for generation.
            schedule:       TemporalSchedule controlling λ per τ.  Use
                            ``make_uniform_schedule(0.3, 0.8)`` for a quick
                            baseline or ``derive_schedule_from_cis`` for a
                            data-driven window.
            seed:           RNG seed for reproducibility.
            negative_prompt: Optional negative conditioning for CFG.
            n_components:   Number of unsafe directions to remove.
                            1 → single mean-pooled direction (recommended).
                            >1 → SVD subspace (more thorough, slower).
            renormalize:    Rescale token norms after orthogonalization.
        Returns:
            PIL.Image.Image of the generated image.
        """
        adapter = self.adapter

        emb_unsafe = adapter.encode_prompt(unsafe_prompt, negative_prompt)
        emb_base = adapter.encode_prompt(base_prompt, negative_prompt)

        cond_unsafe = _get_cond_tensor(emb_unsafe)
        cond_base = _get_cond_tensor(emb_base)

        ortho = build_orthogonal_embedding(
            cond_unsafe, cond_base, n_components=n_components, renormalize=renormalize
        )

        latent = adapter.init_latents(
            seed=seed, batch_size=1, height=self.height, width=self.width
        )

        tau_schedule = adapter.get_tau_schedule()
        native_ts = adapter.get_native_timesteps()

        for tau, native_t in zip(tau_schedule, native_ts):
            lam = schedule.weight(tau)
            if lam == 0.0:
                emb = emb_unsafe
            elif lam == 1.0:
                emb = _replace_cond_tensor(emb_unsafe, ortho.cond_ortho)
            else:
                cond_interp = (1.0 - lam) * cond_unsafe + lam * ortho.cond_ortho
                emb = _replace_cond_tensor(emb_unsafe, cond_interp)

            latent = adapter.denoise_step(
                latent=latent,
                embedding=emb,
                native_t=native_t,
                guidance_scale=self.guidance_scale,
            )

        return adapter.decode(latent)

    def run_orthogonal_vocab(
        self,
        prompt: str,
        schedule: Any,
        seed: int,
        concept: str = "nudity",
        vocab: Optional[List[str]] = None,
        safe_vocab: Optional[List[str]] = None,
        n_components: int = 1,
        renormalize: bool = True,
        max_proj_ratio: Optional[float] = 0.5,
        censor_threshold: Optional[float] = 0.3,
        negative_prompt: str = "",
    ) -> Image.Image:
        """Generate a safe image using vocabulary-based orthogonal suppression.

        Unlike ``run_orthogonal`` (which derives the unsafe direction from a
        paired safe prompt), this method builds the unsafe subspace from a
        vocabulary of concept-related words.  The direction therefore captures
        the concept semantics regardless of what the input prompt looks like —
        including adversarial prompts that contain no explicit unsafe words.

        Bug 1 fix — contrastive direction:
            *safe_vocab* (defaults to SAFE_CONCEPT_VOCAB[concept]) cancels
            the photorealism component shared by both vocab sets, so the
            resulting direction captures nudity-vs-clothed rather than
            photorealism-vs-cartoon.

        Bug 2 fix — projection clamping + censor fallback:
            *max_proj_ratio* (default 0.5) limits how much of each token's
            magnitude can be removed, preventing text-garbage on mildly
            explicit prompts.
            *censor_threshold* (default 0.3) catches prompts that are too
            deeply embedded in unsafe space for suppression to be reliable.
            The score is mean( |proj_i| / ‖token_i‖ ) across all tokens —
            the mean cosine similarity of each token to the unsafe direction.
            When this exceeds *censor_threshold*, a "Content Blocked"
            placeholder is returned instead of generating anything.

        Args:
            prompt:           The user's prompt (safe or unsafe, plain or adversarial).
            schedule:         TemporalSchedule controlling interpolation strength per τ.
            seed:             RNG seed.
            concept:          Key into CONCEPT_VOCAB (e.g. "nudity", "violence").
                              Ignored if *vocab* is provided.
            vocab:            Custom unsafe vocab.  If None, uses CONCEPT_VOCAB[concept].
            safe_vocab:       Custom safe counterpart vocab for contrastive direction.
                              If None, uses SAFE_CONCEPT_VOCAB[concept].
                              Pass an empty list [] to disable contrastive direction.
            n_components:     Number of unsafe directions to remove.
            renormalize:      Rescale token norms after projection.
            max_proj_ratio:   Max fraction of token norm that can be removed per
                              direction.  0.5 = each token keeps ≥50% of its
                              magnitude.  None disables clamping (legacy behaviour).
            censor_threshold: Mean per-token cosine similarity to the unsafe
                              direction above which generation is blocked entirely
                              and a placeholder is returned.  None disables.
            negative_prompt:  Optional negative conditioning for CFG.
        Returns:
            PIL.Image.Image.
        """
        from .orthogonal import compute_vocab_direction

        adapter = self.adapter

        if vocab is None:
            if concept not in CONCEPT_VOCAB:
                raise ValueError(
                    f"Unknown concept '{concept}'. "
                    f"Available: {list(CONCEPT_VOCAB.keys())}. "
                    "Pass a custom vocab list to override."
                )
            vocab = CONCEPT_VOCAB[concept]

        if safe_vocab is None:
            safe_vocab = SAFE_CONCEPT_VOCAB.get(concept, None)

        # Build encode_fn: maps a text string to mean-pooled embedding [dim]
        def _encode_mean(text: str) -> torch.Tensor:
            raw = adapter.encode_prompt(text, "")    # (uncond, cond)
            cond = _get_cond_tensor((None, raw[1]))   # [1, seq, dim]
            return cond.float().mean(dim=1).squeeze(0)  # [dim]

        # Compute vocabulary direction(s) — done once, reused for all steps
        unsafe_dirs = compute_vocab_direction(
            _encode_mean, vocab, n_components,
            safe_vocab=safe_vocab if safe_vocab else None,
        )

        # Encode the input prompt
        emb_input = adapter.encode_prompt(prompt, negative_prompt)
        cond_input = _get_cond_tensor(emb_input)

        # Build orthogonalized embedding
        ortho = build_vocab_orthogonal_embedding(
            cond_input, unsafe_dirs, renormalize,
            max_proj_ratio=max_proj_ratio,
        )

        # Censor check: if the prompt is too deeply embedded in unsafe space,
        # suppression cannot be trusted — return a placeholder instead.
        if censor_threshold is not None:
            token_norms = cond_input.norm(dim=-1).squeeze(0).clamp(min=1e-8)
            mean_cos = (ortho.proj_magnitudes / token_norms).mean().item()
            if mean_cos > censor_threshold:
                h = self.height or 512
                w = self.width or 512
                return _make_censor_image(h, w)

        # Denoising loop with per-step interpolation
        latent = adapter.init_latents(
            seed=seed, batch_size=1, height=self.height, width=self.width
        )
        tau_schedule = adapter.get_tau_schedule()
        native_ts    = adapter.get_native_timesteps()

        for tau, native_t in zip(tau_schedule, native_ts):
            lam = schedule.weight(tau)
            if lam == 0.0:
                emb = emb_input
            elif lam == 1.0:
                emb = _replace_cond_tensor(emb_input, ortho.cond_ortho)
            else:
                cond_interp = (1.0 - lam) * cond_input + lam * ortho.cond_ortho
                emb = _replace_cond_tensor(emb_input, cond_interp)

            latent = adapter.denoise_step(
                latent=latent,
                embedding=emb,
                native_t=native_t,
                guidance_scale=self.guidance_scale,
            )

        return adapter.decode(latent)

    # ------------------------------------------------------------------ #
    # Internal loop                                                        #
    # ------------------------------------------------------------------ #

    def _run_switching(
        self,
        first_prompt: str,
        second_prompt: str,
        switch_tau: float,
        seed: int,
        negative_prompt: str,
        switch_below: bool,
    ) -> Image.Image:
        """Core denoising loop with a single prompt switch at *switch_tau*.

        ``switch_below=True``:  use *second_prompt* when current τ ≤ switch_tau.
        ``switch_below=False``: use *second_prompt* when current τ > switch_tau.
        """
        adapter = self.adapter

        # Encode both prompts once
        emb_first = adapter.encode_prompt(first_prompt, negative_prompt)
        emb_second = adapter.encode_prompt(second_prompt, negative_prompt)

        # Initialise latent noise xT
        latent = adapter.init_latents(
            seed=seed,
            batch_size=1,
            height=self.height,
            width=self.width,
        )

        tau_schedule = adapter.get_tau_schedule()
        native_ts = adapter.get_native_timesteps()

        for tau, native_t in zip(tau_schedule, native_ts):
            if switch_below:
                emb = emb_second if tau <= switch_tau else emb_first
            else:
                emb = emb_second if tau > switch_tau else emb_first

            latent = adapter.denoise_step(
                latent=latent,
                embedding=emb,
                native_t=native_t,
                guidance_scale=self.guidance_scale,
            )

        return adapter.decode(latent)

    # ------------------------------------------------------------------ #
    # Seed-batched helpers (all seeds for one τ in a single GPU pass)    #
    # ------------------------------------------------------------------ #

    def run_pci_batch_seeds(
        self,
        base_prompt: str,
        concept_prompt: str,
        switch_tau: float,
        seeds: List[int],
        negative_prompt: str = "",
    ) -> List[Image.Image]:
        """CIS for multiple seeds in one batched denoising pass."""
        return self._run_switching_batch(
            first_prompt=base_prompt,
            second_prompt=concept_prompt,
            switch_tau=switch_tau,
            seeds=seeds,
            negative_prompt=negative_prompt,
            switch_below=True,
        )

    def run_cds_batch_seeds(
        self,
        concept_prompt: str,
        base_prompt: str,
        switch_tau: float,
        seeds: List[int],
        negative_prompt: str = "",
    ) -> List[Image.Image]:
        """CDS for multiple seeds in one batched denoising pass."""
        return self._run_switching_batch(
            first_prompt=concept_prompt,
            second_prompt=base_prompt,
            switch_tau=switch_tau,
            seeds=seeds,
            negative_prompt=negative_prompt,
            switch_below=True,
        )

    def _run_switching_batch(
        self,
        first_prompt: str,
        second_prompt: str,
        switch_tau: float,
        seeds: List[int],
        negative_prompt: str,
        switch_below: bool,
    ) -> List[Image.Image]:
        """Batched denoising: all seeds share the same switching point, one GPU pass.

        Reduces 64 sequential loops to 8 (one per τ) by stacking seeds into a
        single batched latent tensor and letting the UNet process them together.
        """
        adapter = self.adapter
        emb_first = adapter.encode_prompt(first_prompt, negative_prompt)
        emb_second = adapter.encode_prompt(second_prompt, negative_prompt)

        latent = adapter.init_latents_batch(
            seeds=seeds, height=self.height, width=self.width
        )

        tau_schedule = adapter.get_tau_schedule()
        native_ts = adapter.get_native_timesteps()

        for tau, native_t in zip(tau_schedule, native_ts):
            use_second = (tau <= switch_tau) if switch_below else (tau > switch_tau)
            latent = adapter.denoise_step(
                latent=latent,
                embedding=emb_second if use_second else emb_first,
                native_t=native_t,
                guidance_scale=self.guidance_scale,
            )

        return adapter.decode_batch(latent)

    # ------------------------------------------------------------------ #
    # τ-grid batched helpers (original API)                               #
    # ------------------------------------------------------------------ #

    def run_pci_batch(
        self,
        base_prompt: str,
        concept_prompt: str,
        switch_taus: list[float],
        seeds: list[int],
        negative_prompt: str = "",
    ) -> list[tuple[float, int, Image.Image]]:
        """Run PCI for every (switch_tau, seed) combination.

        Returns a list of ``(tau, seed, image)`` triples, which callers can
        feed directly into a scorer for CIS curve estimation.
        """
        results = []
        for tau in switch_taus:
            for seed in seeds:
                img = self.run_pci(base_prompt, concept_prompt, tau, seed, negative_prompt)
                results.append((tau, seed, img))
        return results

    def run_cds_batch(
        self,
        concept_prompt: str,
        base_prompt: str,
        switch_taus: list[float],
        seeds: list[int],
        negative_prompt: str = "",
    ) -> list[tuple[float, int, Image.Image]]:
        """Run CDS for every (switch_tau, seed) combination."""
        results = []
        for tau in switch_taus:
            for seed in seeds:
                img = self.run_cds(concept_prompt, base_prompt, tau, seed, negative_prompt)
                results.append((tau, seed, img))
        return results
