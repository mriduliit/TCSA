"""
Orthogonal embedding manipulation for training-free concept suppression.

Two ways to define the unsafe subspace:

1. VOCABULARY-BASED (recommended, default for run_orthogonal_vocab):
   Encode a pre-defined list of concept-related words with the text encoder,
   take their mean-pooled embedding, and use that as the unsafe direction.

       vocab = ["nudity", "nude", "naked", ...]
       d = normalize( mean( [mean_pool(encode(w)) for w in vocab] ) )

   This approach is concept-specific, not prompt-specific. It works on ANY
   input embedding — including adversarial prompts that look like random text —
   because the direction is defined by the concept itself, not by the prompt.

2. PAIRED-PROMPT (legacy, weaker):
   Estimate the direction from the mean-pooled difference between the unsafe
   and a paired safe prompt. Diagnostic shows this captures only ~8% of the
   actual concept signal and is dominated by contextual noise between the pair.

For each token in the input embedding, the concept component is removed:

    e_ortho[:, i, :] = e_input[:, i, :] - (e_input[:, i, :] · d) * d

Setting n_components > 1 uses the top-k singular vectors of the vocabulary
embedding matrix (SVD subspace) instead of the mean direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ======================================================================= #
#  Result container                                                         #
# ======================================================================= #

@dataclass
class OrthogonalEmbedding:
    """Container for an orthogonalized conditional embedding.

    Attributes:
        cond_ortho:      Orthogonalized conditional tensor [B, seq, dim].
                         The unsafe subspace has been projected out.
        unsafe_dirs:     Orthonormal basis of the removed unsafe subspace [k, dim].
        proj_magnitudes: Per-token projection magnitude onto unsafe_dirs[0] [seq].
                         High values indicate tokens that strongly express the
                         unsafe concept.  Useful for diagnostics.
    """
    cond_ortho: torch.Tensor
    unsafe_dirs: torch.Tensor
    proj_magnitudes: torch.Tensor


# ======================================================================= #
#  Built-in concept vocabularies                                            #
# ======================================================================= #

CONCEPT_VOCAB: dict[str, List[str]] = {
    "nudity": [
        "nudity", "nude", "naked", "pornography", "explicit content",
        "erotic", "nsfw", "topless", "genitals", "sex", "sexual",
        "pornographic", "adult content", "bare skin", "undressed",
    ],
    "violence": [
        "violence", "gore", "blood", "murder", "killing", "brutal",
        "graphic violence", "death", "wound", "injury", "assault",
    ],
    "hate": [
        "hate speech", "racist", "discrimination", "slur", "bigotry",
        "extremism", "offensive", "derogatory", "harassment",
    ],
}

# Safe counterpart vocabularies for contrastive direction computation (Bug 1 fix).
# Each list encodes the same domain (human bodies, human interaction) but without
# the unsafe concept — so the shared photorealism component cancels when we
# compute d = normalize(mean(unsafe) - mean(safe)).
SAFE_CONCEPT_VOCAB: dict[str, List[str]] = {
    "nudity": [
        "clothed person", "fully dressed", "wearing clothes", "outfit",
        "fashion", "attire", "dressed woman", "dressed man",
        "person in clothes", "portrait", "clothed", "wearing outfit",
        "dressed people", "fashion model", "person wearing shirt",
    ],
    "violence": [
        "peaceful scene", "calm", "harmony", "safe", "gentle",
        "kindness", "helping", "care", "protection", "serenity",
    ],
    "hate": [
        "respect", "inclusion", "diversity", "tolerance", "kindness",
        "acceptance", "equality", "unity", "community", "harmony",
    ],
}


# ======================================================================= #
#  Vocabulary-based direction (primary approach)                            #
# ======================================================================= #

def compute_vocab_direction(
    encode_fn: Callable[[str], torch.Tensor],
    vocab: List[str],
    n_components: int = 1,
    safe_vocab: Optional[List[str]] = None,
) -> torch.Tensor:
    """Compute the unsafe concept subspace from a vocabulary of concept words.

    Encodes each word in *vocab* with *encode_fn* (should return a mean-pooled
    vector [dim]), stacks them, and returns the top-k principal directions via
    SVD (or just the normalised mean for k=1).

    Bug 1 fix — contrastive direction:
        When *safe_vocab* is provided the direction is computed as the
        normalised difference between the unsafe mean and the safe mean:

            d = normalize( mean(unsafe_embs) − mean(safe_embs) )

        The safe vocab should contain clothed/neutral counterparts of the
        concept words (see SAFE_CONCEPT_VOCAB).  The shared photorealism
        component that both vocabs carry cancels in the subtraction, leaving
        a direction that points specifically from "safe human depiction" to
        "unsafe concept" rather than from "cartoon" to "photorealistic nude".

    Args:
        encode_fn:    Callable mapping a text string to a mean-pooled embedding
                      tensor of shape [dim].  Typically wraps the pipeline's
                      text encoder.
        vocab:        List of concept-related words / phrases.
        n_components: Number of directions to return.
                      1 → normalised mean (or contrastive mean) of vocab.
                      >1 → top-k right singular vectors of the vocab matrix.
        safe_vocab:   Optional list of safe counterpart words.  When given,
                      a contrastive direction is computed (recommended for
                      photorealistic models to avoid style degradation).
    Returns:
        Tensor of shape [n_components, dim] with orthonormal rows.
    """
    embs = torch.stack([encode_fn(w) for w in vocab]).float()  # [V, dim]

    if n_components == 1:
        mean_unsafe = embs.mean(dim=0)
        if safe_vocab is not None:
            safe_embs = torch.stack([encode_fn(w) for w in safe_vocab]).float()
            mean_safe = safe_embs.mean(dim=0)
            direction = mean_unsafe - mean_safe  # cancel shared photorealism
        else:
            direction = mean_unsafe
        norm = direction.norm()
        if norm < 1e-8:
            return torch.zeros(1, embs.shape[-1], dtype=embs.dtype, device=embs.device)
        return (direction / norm).unsqueeze(0).to(dtype=embs.dtype)  # [1, dim]
    else:
        if safe_vocab is not None:
            safe_embs = torch.stack([encode_fn(w) for w in safe_vocab]).float()
            # Centre each unsafe embedding by subtracting the safe mean so that
            # SVD finds directions specific to the unsafe concept.
            embs = embs - safe_embs.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(embs, full_matrices=False)
        return Vh[:n_components].to(dtype=embs.dtype)  # [k, dim]


def build_vocab_orthogonal_embedding(
    cond_input: torch.Tensor,
    unsafe_dirs: torch.Tensor,
    renormalize: bool = True,
    max_proj_ratio: Optional[float] = None,
) -> OrthogonalEmbedding:
    """Build OrthogonalEmbedding using pre-computed vocabulary directions.

    Args:
        cond_input:      [1, seq, dim] conditional embedding of any prompt.
        unsafe_dirs:     [k, dim] vocabulary-based unsafe directions from
                         ``compute_vocab_direction``.
        renormalize:     Rescale token norms after projection.
        max_proj_ratio:  See ``orthogonalize_embedding``.
    Returns:
        OrthogonalEmbedding.
    """
    dirs = unsafe_dirs.to(dtype=cond_input.dtype, device=cond_input.device)
    cond_ortho = orthogonalize_embedding(cond_input, dirs, renormalize,
                                         max_proj_ratio=max_proj_ratio)
    proj_mag = (cond_input * dirs[0]).sum(dim=-1).squeeze(0).abs()
    return OrthogonalEmbedding(
        cond_ortho=cond_ortho,
        unsafe_dirs=dirs,
        proj_magnitudes=proj_mag,
    )


# ======================================================================= #
#  Direction computation (legacy paired-prompt approach)                   #
# ======================================================================= #

def compute_unsafe_direction(
    cond_concept: torch.Tensor,
    cond_base: torch.Tensor,
) -> torch.Tensor:
    """Compute the unit vector of the unsafe concept direction.

    Mean-pools both embeddings over the sequence dimension to produce a single
    direction in the embedding space.  This is the direction that most
    differentiates the unsafe prompt from the safe reference.

    Args:
        cond_concept: Conditional embedding of the unsafe prompt [B, seq, dim].
        cond_base:    Conditional embedding of the safe/base prompt [B, seq, dim].
    Returns:
        Unit direction vector of shape [dim], same device and dtype as inputs.
        Returns the zero vector when the two prompts are identical in embedding
        space (norm < 1e-8).
    """
    mean_concept = cond_concept.float().mean(dim=1).squeeze(0)  # [dim]
    mean_base = cond_base.float().mean(dim=1).squeeze(0)        # [dim]
    diff = mean_concept - mean_base
    norm = diff.norm()
    if norm < 1e-8:
        return torch.zeros_like(diff).to(dtype=cond_concept.dtype)
    return (diff / norm).to(dtype=cond_concept.dtype)


def compute_unsafe_subspace(
    cond_concept: torch.Tensor,
    cond_base: torch.Tensor,
    n_components: int,
) -> torch.Tensor:
    """Compute the top-k unsafe directions via SVD on the token-wise difference.

    The right singular vectors of (cond_concept - cond_base) span the subspace
    of the embedding space that captures the most variance introduced by the
    unsafe prompt tokens.

    Args:
        cond_concept: [1, seq, dim]
        cond_base:    [1, seq, dim]
        n_components: Number of directions to retain.
    Returns:
        Orthonormal basis of shape [n_components, dim].
    """
    diff = (cond_concept - cond_base).squeeze(0).float()  # [seq, dim]
    _, _, Vh = torch.linalg.svd(diff, full_matrices=False)
    return Vh[:n_components].to(dtype=cond_concept.dtype)  # [k, dim]


# ======================================================================= #
#  Projection                                                               #
# ======================================================================= #

def orthogonalize_embedding(
    cond: torch.Tensor,
    unsafe_dirs: torch.Tensor,
    renormalize: bool = True,
    max_proj_ratio: Optional[float] = None,
) -> torch.Tensor:
    """Remove the unsafe subspace from each token in a conditional embedding.

    For each direction d in unsafe_dirs (applied sequentially, Gram-Schmidt):

        cond[:, i, :] -= proj_clamped * d

    where proj_clamped = (cond[:, i, :] · d) optionally clamped by
    max_proj_ratio (Bug 2 fix — see below).

    Bug 2 fix — projection clamping:
        When *max_proj_ratio* is set (e.g. 0.5), the amount removed along d
        is capped at max_proj_ratio × ‖token‖.  This prevents semantic
        collapse on fully-explicit prompts where every token projects heavily
        onto the unsafe direction: each token retains at least
        (1 − max_proj_ratio) of its original magnitude and therefore still
        carries a coherent semantic signal.  Without clamping, CFG amplifies
        the resulting near-zero residuals into random/text-like patterns.

    Args:
        cond:            Conditional embedding [B, seq, dim].
        unsafe_dirs:     Unsafe directions to remove [k, dim].  Rows must be
                         orthonormal.
        renormalize:     If True, rescale each token to its original norm after
                         projection.  Recommended: True.
        max_proj_ratio:  If not None, clamp each token's projection removal to
                         at most this fraction of the token's current norm.
                         Recommended value: 0.5.  Set None to disable (legacy
                         behaviour).
    Returns:
        Modified embedding of the same shape as *cond*.
    """
    if unsafe_dirs.dim() == 1:
        unsafe_dirs = unsafe_dirs.unsqueeze(0)

    dirs = unsafe_dirs.to(dtype=cond.dtype, device=cond.device)
    original_norms = cond.norm(dim=-1, keepdim=True)  # [B, seq, 1]
    out = cond

    for d in dirs:                                              # d: [dim]
        proj = (out * d).sum(dim=-1, keepdim=True)             # [B, seq, 1]
        if max_proj_ratio is not None:
            # Clamp removal to max_proj_ratio × current token norm so that
            # no token loses more than that fraction of its magnitude.
            token_norms = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            ceiling = max_proj_ratio * token_norms
            proj = proj.sign() * proj.abs().clamp(max=ceiling)
        out = out - proj * d                                    # remove component

    if renormalize:
        new_norms = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        out = out * (original_norms / new_norms)

    return out


# ======================================================================= #
#  Convenience builder                                                      #
# ======================================================================= #

def build_orthogonal_embedding(
    cond_concept: torch.Tensor,
    cond_base: torch.Tensor,
    n_components: int = 1,
    renormalize: bool = True,
) -> OrthogonalEmbedding:
    """Build an OrthogonalEmbedding from paired conditional embeddings.

    This is the main entry point.  It computes the unsafe subspace from the
    paired embeddings and projects it out of cond_concept.

    Args:
        cond_concept: [1, seq, dim] conditional embedding of the unsafe prompt.
        cond_base:    [1, seq, dim] conditional embedding of the safe/base prompt.
        n_components: Number of unsafe directions to remove.
                      1 → mean-pooled direction (fast, usually sufficient).
                      >1 → SVD subspace (captures more variance, slower).
        renormalize:  Rescale token norms after projection.
    Returns:
        OrthogonalEmbedding with the unsafe subspace removed.
    """
    if n_components == 1:
        d = compute_unsafe_direction(cond_concept, cond_base)
        dirs = d.unsqueeze(0)                                   # [1, dim]
    else:
        dirs = compute_unsafe_subspace(cond_concept, cond_base, n_components)

    cond_ortho = orthogonalize_embedding(cond_concept, dirs, renormalize)

    d0 = dirs[0]
    proj_magnitudes = (cond_concept * d0).sum(dim=-1).squeeze(0).abs()  # [seq]

    return OrthogonalEmbedding(
        cond_ortho=cond_ortho,
        unsafe_dirs=dirs,
        proj_magnitudes=proj_magnitudes,
    )
