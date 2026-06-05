"""
CIS and CDS curve computation.

Provides:
- ``concept_controlled_seed_filter`` — Appendix A.2 seed resampling
- ``compute_cis_curve``             — Concept Insertion Success curve
- ``compute_cds_curve``             — Concept Deletion Success curve
- ``CISResult``                     — dataclass wrapping curve + statistics

Seed resampling (Appendix A.2)
------------------------------
Before computing curves, seeds where the *base* prompt already produces the
target concept are discarded.  If too few valid seeds survive, negative
guidance (concept used as a negative prompt) is added to Pb to push the model
away from the concept before retrying.  This ensures CIS truly measures the
*insertion* effect rather than confirming a pre-existing structural bias.

Statistical reporting
---------------------
CIS/CDS values at each τ are seed-averaged binary VQA outcomes (0 or 1).
95 % Wilson confidence intervals are reported over seeds.
τ₅₀ = first τ where CIS ≥ 0.50; τ₇₀ = first τ where CIS ≥ 0.70.
W₇₀→₅₀ = |τ₇₀ − τ₅₀| (bandwidth / steepness proxy).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .adapters import DiffusionAdapter
from .probe import PromptConditionedIntervention
from .scorer import ConceptScorer, QwenVLScorer, get_question

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Result dataclass                                                        #
# ======================================================================= #

@dataclass
class CISResult:
    """Concept Insertion Success (or Deletion Success) curve + statistics.

    Attributes:
        curve:            Dict mapping τ → mean probability across seeds.
        tau_50:           Earliest τ where curve ≥ 0.50 (None if never).
        tau_70:           Earliest τ where curve ≥ 0.70 (None if never).
        bandwidth:        |τ₇₀ − τ₅₀| (None if either crossing undefined).
        wilson_ci_lower:  Dict mapping τ → lower bound of 95 % Wilson CI.
        wilson_ci_upper:  Dict mapping τ → upper bound of 95 % Wilson CI.
        raw_outcomes:     Dict mapping τ → list of per-seed binary outcomes.
        num_seeds:        Number of seeds actually used (after filtering).
    """

    curve: Dict[float, float]
    tau_50: Optional[float]
    tau_70: Optional[float]
    bandwidth: Optional[float]
    wilson_ci_lower: Dict[float, float]
    wilson_ci_upper: Dict[float, float]
    raw_outcomes: Dict[float, List[int]] = field(default_factory=dict, repr=False)
    num_seeds: int = 0


# ======================================================================= #
#  Wilson confidence interval helper                                       #
# ======================================================================= #

def _wilson_ci(
    n_success: int,
    n_total: int,
    z: float = 1.96,  # 95 %
) -> Tuple[float, float]:
    """Return (lower, upper) Wilson confidence interval for a proportion."""
    if n_total == 0:
        return 0.0, 1.0
    p = n_success / n_total
    denom = 1 + z * z / n_total
    centre = (p + z * z / (2 * n_total)) / denom
    half = z * math.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total * n_total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _crossing_time(
    curve: Dict[float, float],
    level: float,
) -> Optional[float]:
    """Return the minimum τ in *curve* whose value ≥ *level*."""
    taus = sorted(curve.keys())
    for t in taus:
        if curve[t] >= level:
            return t
    return None


def _build_result(
    raw_outcomes: Dict[float, List[int]],
) -> CISResult:
    """Aggregate raw per-seed binary outcomes into a CISResult."""
    curve: Dict[float, float] = {}
    lower: Dict[float, float] = {}
    upper: Dict[float, float] = {}

    for tau in sorted(raw_outcomes.keys()):
        outcomes = raw_outcomes[tau]
        n = len(outcomes)
        n_yes = sum(outcomes)
        p = n_yes / n if n > 0 else 0.0
        curve[tau] = p
        lo, hi = _wilson_ci(n_yes, n)
        lower[tau] = lo
        upper[tau] = hi

    tau_50 = _crossing_time(curve, 0.50)
    tau_70 = _crossing_time(curve, 0.70)
    bw = abs(tau_70 - tau_50) if (tau_50 is not None and tau_70 is not None) else None
    n_seeds = len(next(iter(raw_outcomes.values()))) if raw_outcomes else 0

    return CISResult(
        curve=curve,
        tau_50=tau_50,
        tau_70=tau_70,
        bandwidth=bw,
        wilson_ci_lower=lower,
        wilson_ci_upper=upper,
        raw_outcomes=raw_outcomes,
        num_seeds=n_seeds,
    )


# ======================================================================= #
#  Seed resampling (Appendix A.2)                                          #
# ======================================================================= #

def concept_controlled_seed_filter(
    adapter: DiffusionAdapter,
    base_prompt: str,
    concept: str,
    n_seeds_needed: int,
    scorer: Optional[ConceptScorer] = None,
    max_seeds: int = 500,
    subcategory: Optional[str] = None,
    guidance_scale: float = 7.5,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Tuple[List[int], bool]:
    """Return seeds where the base prompt does NOT already produce *concept*.

    Implements the two-stage strategy from Appendix A.2:
    1. Seed filtering — discard seeds where Pb already generates the concept.
    2. Negative guidance fallback — if too few seeds survive stage 1, re-run
       Pb with *concept* as the negative prompt and re-filter.

    Args:
        adapter:         Configured DiffusionAdapter.
        base_prompt:     Pb (neutral, without the concept).
        concept:         Target concept string.
        n_seeds_needed:  Minimum number of valid seeds required.
        scorer:          VQA scorer instance.  Defaults to QwenVLScorer().
        max_seeds:       Maximum seeds to screen before giving up.
        subcategory:     Optional subcategory for question template selection.
        guidance_scale:  CFG scale for baseline generation.
        height/width:    Image resolution.

    Returns:
        (valid_seeds, used_negative_guidance) where *used_negative_guidance*
        is True when the fallback strategy was triggered.
    """
    if scorer is None:
        scorer = QwenVLScorer()

    question = get_question(concept, subcategory)
    probe = PromptConditionedIntervention(
        adapter, guidance_scale=guidance_scale, height=height, width=width
    )

    # --- Stage 1: plain seed filtering ---
    valid_seeds: List[int] = []
    for seed in range(max_seeds):
        if len(valid_seeds) >= n_seeds_needed:
            break
        img = probe.run_full(base_prompt, seed=seed)
        if not scorer.is_concept_present(img, concept, question):
            valid_seeds.append(seed)

    if len(valid_seeds) >= n_seeds_needed:
        logger.debug(
            "Seed filter: found %d/%d valid seeds with plain base prompt.",
            len(valid_seeds), n_seeds_needed,
        )
        return valid_seeds[:n_seeds_needed], False

    # --- Stage 2: negative guidance fallback ---
    logger.debug(
        "Seed filter: only %d/%d seeds survived plain filter. "
        "Falling back to negative guidance (concept as negative prompt).",
        len(valid_seeds), n_seeds_needed,
    )
    valid_seeds = []
    for seed in range(max_seeds):
        if len(valid_seeds) >= n_seeds_needed:
            break
        img = probe.run_full(base_prompt, seed=seed, negative_prompt=concept)
        if not scorer.is_concept_present(img, concept, question):
            valid_seeds.append(seed)

    logger.debug(
        "Seed filter (neg-guidance): found %d/%d valid seeds.",
        len(valid_seeds), n_seeds_needed,
    )
    return valid_seeds[:n_seeds_needed], True


# ======================================================================= #
#  CIS curve computation                                                   #
# ======================================================================= #

def compute_cis_curve(
    adapter: DiffusionAdapter,
    base_prompt: str,
    concept_prompt: str,
    concept: str,
    scorer: Optional[ConceptScorer] = None,
    num_seeds: int = 100,
    num_taus: int = 20,
    subcategory: Optional[str] = None,
    guidance_scale: float = 7.5,
    height: Optional[int] = None,
    width: Optional[int] = None,
    negative_prompt: str = "",
    seeds: Optional[List[int]] = None,
) -> CISResult:
    """Compute the Concept Insertion Success (CIS) curve.

    For each τ in a grid of *num_taus* evenly-spaced points, runs PCI
    (base→concept prompt switch at that τ) over *num_seeds* seeds and records
    the fraction where the VQA scorer detects the concept.

    Args:
        adapter:         Configured DiffusionAdapter (pre-loaded model).
        base_prompt:     Pb — neutral description without the target concept.
        concept_prompt:  Pc — base prompt augmented with the target concept.
        concept:         Concept string used for VQA question generation.
        scorer:          VQA scorer.  Defaults to ``QwenVLScorer()``.
        num_seeds:       Seeds to average over (paper default: 100).
        num_taus:        Grid resolution along the τ axis (paper default: ~20).
        subcategory:     Optional subcategory key for question templates.
        guidance_scale:  CFG scale.
        height/width:    Image resolution.
        negative_prompt: Optional explicit negative conditioning.
        seeds:           Explicit seed list; overrides *num_seeds* and
                         disables automatic seed resampling.
    Returns:
        ``CISResult`` with the averaged curve and statistics.
    """
    if scorer is None:
        scorer = QwenVLScorer()

    question = get_question(concept, subcategory)
    probe = PromptConditionedIntervention(
        adapter, guidance_scale=guidance_scale, height=height, width=width
    )

    # Seed resampling unless the caller provides explicit seeds
    used_neg_guidance = False
    if seeds is None:
        seeds, used_neg_guidance = concept_controlled_seed_filter(
            adapter=adapter,
            base_prompt=base_prompt,
            concept=concept,
            n_seeds_needed=num_seeds,
            scorer=scorer,
            subcategory=subcategory,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )
        # Pair seed-resampled negative prompt into the run if needed
        _neg = concept if used_neg_guidance else negative_prompt
    else:
        _neg = negative_prompt

    # Build τ grid from the adapter's actual schedule
    tau_schedule = adapter.get_tau_schedule()
    tau_min = min(tau_schedule)
    tau_max = max(tau_schedule)
    tau_grid = np.linspace(tau_min, tau_max, num_taus).tolist()

    raw_outcomes: Dict[float, List[int]] = {tau: [] for tau in tau_grid}

    total = len(tau_grid) * len(seeds)
    done = 0
    for tau in tau_grid:
        logger.debug("CIS [%d/%d] τ=%.3f  batch=%d seeds", done, total, tau, len(seeds))
        imgs = probe.run_pci_batch_seeds(
            base_prompt=base_prompt,
            concept_prompt=concept_prompt,
            switch_tau=tau,
            seeds=seeds,
            negative_prompt=_neg,
        )
        for img in imgs:
            raw_outcomes[tau].append(int(scorer.is_concept_present(img, concept, question)))
        done += len(seeds)

    return _build_result(raw_outcomes)


# ======================================================================= #
#  CDS curve computation                                                   #
# ======================================================================= #

def compute_cds_curve(
    adapter: DiffusionAdapter,
    concept_prompt: str,
    base_prompt: str,
    concept: str,
    scorer: Optional[ConceptScorer] = None,
    num_seeds: int = 100,
    num_taus: int = 20,
    subcategory: Optional[str] = None,
    guidance_scale: float = 7.5,
    height: Optional[int] = None,
    width: Optional[int] = None,
    negative_prompt: str = "",
    seeds: Optional[List[int]] = None,
) -> CISResult:
    """Compute the Concept Deletion Success (CDS) curve.

    Mirrors ``compute_cis_curve`` but in the deletion direction: denoising
    starts under *concept_prompt*, then switches to *base_prompt* at τ.
    A high CDS value means the concept can still be removed at that τ;
    low CDS means it has already locked in and resists deletion.

    For unlearning evaluation, a successfully unlearned concept should have
    low CDS AUC — meaning the concept cannot even be inserted (or if inserted,
    cannot be removed by switching prompts).

    Args:
        adapter:        Configured DiffusionAdapter.
        concept_prompt: Pc — prompt that contains the concept to delete.
        base_prompt:    Pb — neutral base prompt.
        concept:        Concept string for VQA questioning.
        (remaining args same as ``compute_cis_curve``)
    Returns:
        ``CISResult`` with CDS curve values (semantics inverted vs. CIS:
        high value = successful deletion = concept NOT present in output).
    """
    if scorer is None:
        scorer = QwenVLScorer()

    question = get_question(concept, subcategory)
    probe = PromptConditionedIntervention(
        adapter, guidance_scale=guidance_scale, height=height, width=width
    )

    if seeds is None:
        seeds = list(range(num_seeds))

    tau_schedule = adapter.get_tau_schedule()
    tau_min = min(tau_schedule)
    tau_max = max(tau_schedule)
    tau_grid = np.linspace(tau_min, tau_max, num_taus).tolist()

    # For CDS we record 1 = concept ABSENT (deletion succeeded), 0 = present
    raw_outcomes: Dict[float, List[int]] = {tau: [] for tau in tau_grid}

    total = len(tau_grid) * len(seeds)
    done = 0
    for tau in tau_grid:
        logger.debug("CDS [%d/%d] τ=%.3f  batch=%d seeds", done, total, tau, len(seeds))
        imgs = probe.run_cds_batch_seeds(
            concept_prompt=concept_prompt,
            base_prompt=base_prompt,
            switch_tau=tau,
            seeds=seeds,
            negative_prompt=negative_prompt,
        )
        for img in imgs:
            raw_outcomes[tau].append(int(not scorer.is_concept_present(img, concept, question)))
        done += len(seeds)

    return _build_result(raw_outcomes)


# ======================================================================= #
#  Convenience: AUC under a curve                                          #
# ======================================================================= #

def curve_auc(result: CISResult) -> float:
    """Compute the area under a CIS/CDS curve using the trapezoidal rule."""
    taus = sorted(result.curve.keys())
    if len(taus) < 2:
        return float(result.curve.get(taus[0], 0.0)) if taus else 0.0
    vals = [result.curve[t] for t in taus]
    return float(np.trapz(vals, x=taus))


def max_cds(result: CISResult) -> float:
    """Return the peak value of a CDS curve across all τ."""
    if not result.curve:
        return 0.0
    return float(max(result.curve.values()))
