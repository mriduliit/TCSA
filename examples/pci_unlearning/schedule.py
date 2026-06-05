"""
Translating CIS curves into intervention schedules for unlearning methods.

This module bridges the PCI analysis layer to the unlearning execution layer.
Given a CIS curve for a concept-to-forget, it produces a ``TemporalSchedule``
that tells the downstream unlearning method *how strongly* to intervene at each
denoising step.

Schedule modes
--------------
``band``           Active (weight=1) inside [τ₇₀, τ₅₀] — the steep transition
                   window where the concept is forming.  Zero elsewhere.
``slope_weighted`` Per-step weight proportional to |dCIS/dτ|, normalised to
                   [0, 1].  Smooth, concentrates mass where CIS rises fastest.
``early_lock``     Active for all τ ≥ τ₇₀ (catches global/style concepts that
                   commit early in the trajectory).

Usage::

    from pci_unlearning.schedule import derive_schedule_from_cis

    schedule = derive_schedule_from_cis(cis_result, mode="band")
    weight = schedule.weight(current_tau)   # float in [0, 1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from .curves import CISResult

ScheduleMode = Literal["band", "slope_weighted", "early_lock"]


# ======================================================================= #
#  TemporalSchedule                                                        #
# ======================================================================= #

@dataclass
class TemporalSchedule:
    """Wraps a per-τ intervention weight function.

    The primary interface is ``weight(tau) -> float``, which the unlearning
    method calls at each denoising step to scale its intervention strength.

    Attributes:
        mode:       The derivation mode used (informational).
        tau_50:     τ crossing at 50 % CIS (from source CISResult).
        tau_70:     τ crossing at 70 % CIS (from source CISResult).
        bandwidth:  |τ₇₀ − τ₅₀| (from source CISResult).
        _weights:   Internal τ → weight dict used for interpolation.
    """

    mode: ScheduleMode
    tau_50: Optional[float]
    tau_70: Optional[float]
    bandwidth: Optional[float]
    _weights: Dict[float, float]

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def weight(self, tau: float) -> float:
        """Return the intervention weight at normalised timestep *tau*.

        Linearly interpolates between stored grid points.  Returns 0.0
        outside the range of stored τ values.

        Args:
            tau: Normalised τ ∈ [0, 1].  τ=1 is pure noise, τ=0 is clean.
        Returns:
            Float weight in [0, 1] indicating intervention strength at *tau*.
        """
        if not self._weights:
            return 0.0

        taus = sorted(self._weights.keys())
        if tau <= taus[0]:
            return self._weights[taus[0]]
        if tau >= taus[-1]:
            return self._weights[taus[-1]]

        # Linear interpolation
        for i in range(len(taus) - 1):
            t0, t1 = taus[i], taus[i + 1]
            if t0 <= tau <= t1:
                alpha = (tau - t0) / (t1 - t0)
                return (1 - alpha) * self._weights[t0] + alpha * self._weights[t1]

        return 0.0

    def active_range(self) -> Tuple[float, float]:
        """Return ``(tau_low, tau_high)`` where weight > 0.

        Useful for logging and sanity checks.
        """
        taus = sorted(self._weights.keys())
        active = [t for t in taus if self._weights[t] > 0.0]
        if not active:
            return (0.0, 0.0)
        return (min(active), max(active))

    def summary(self) -> str:
        """Return a human-readable one-line summary of the schedule."""
        lo, hi = self.active_range()
        return (
            f"TemporalSchedule(mode={self.mode!r}, "
            f"τ₅₀={self.tau_50}, τ₇₀={self.tau_70}, "
            f"bandwidth={self.bandwidth}, active=[{lo:.3f}, {hi:.3f}])"
        )


# ======================================================================= #
#  Derivation                                                              #
# ======================================================================= #

def derive_schedule_from_cis(
    cis_result: CISResult,
    mode: ScheduleMode = "band",
    band_margin: float = 0.0,
    fallback_tau_50: float = 0.4,
    fallback_tau_70: float = 0.7,
) -> TemporalSchedule:
    """Derive a ``TemporalSchedule`` from a pre-computed CIS curve.

    Args:
        cis_result:      Output of ``compute_cis_curve`` for the concept.
        mode:            Schedule derivation strategy (see module docstring).
        band_margin:     Extra τ margin added on both sides of the band for
                         ``"band"`` mode.  Default: 0 (strict band).
        fallback_tau_50: Used when τ₅₀ is undefined (concept never reaches 50 %).
        fallback_tau_70: Used when τ₇₀ is undefined (concept never reaches 70 %).
    Returns:
        A ``TemporalSchedule`` instance.
    """
    tau_50 = cis_result.tau_50 if cis_result.tau_50 is not None else fallback_tau_50
    tau_70 = cis_result.tau_70 if cis_result.tau_70 is not None else fallback_tau_70

    if mode == "band":
        weights = _derive_band(cis_result, tau_50, tau_70, band_margin)
    elif mode == "slope_weighted":
        weights = _derive_slope_weighted(cis_result)
    elif mode == "early_lock":
        weights = _derive_early_lock(cis_result, tau_70)
    else:
        raise ValueError(f"Unknown schedule mode '{mode}'. Choose from: band, slope_weighted, early_lock")

    bw = abs(tau_70 - tau_50) if (cis_result.tau_50 and cis_result.tau_70) else None

    return TemporalSchedule(
        mode=mode,
        tau_50=tau_50,
        tau_70=tau_70,
        bandwidth=bw,
        _weights=weights,
    )


# ======================================================================= #
#  Mode implementations                                                    #
# ======================================================================= #

def _derive_band(
    result: CISResult,
    tau_50: float,
    tau_70: float,
    margin: float,
) -> Dict[float, float]:
    """Band mode: weight=1 inside [τ₇₀−margin, τ₅₀+margin], 0 elsewhere.

    For unlearning the *deletion* direction is relevant: we want to intervene
    inside the window where the concept is actively forming (steep CIS slope).
    This is [τ₇₀, τ₅₀] (note: τ₇₀ > τ₅₀ because higher τ = earlier in the
    trajectory = noisier).
    """
    lo = max(0.0, min(tau_50, tau_70) - margin)
    hi = min(1.0, max(tau_50, tau_70) + margin)

    weights: Dict[float, float] = {}
    for tau in result.curve:
        weights[tau] = 1.0 if lo <= tau <= hi else 0.0

    # Ensure the boundary τ values are represented even if not in curve keys
    if lo not in weights:
        weights[lo] = 1.0
    if hi not in weights:
        weights[hi] = 1.0
    # Add just outside boundaries
    eps = 1e-6
    weights[lo - eps] = 0.0
    weights[hi + eps] = 0.0

    return weights


def _derive_slope_weighted(
    result: CISResult,
) -> Dict[float, float]:
    """Slope-weighted mode: w(τ) ∝ |dCIS/dτ|, normalised to [0, 1].

    Concentrates the intervention budget on timesteps where the concept
    is rising fastest — the most leveraged intervention points.
    """
    taus = sorted(result.curve.keys())
    if len(taus) < 2:
        return {tau: 1.0 for tau in taus}

    # Finite-difference gradient
    slopes: Dict[float, float] = {}
    for i, tau in enumerate(taus):
        if i == 0:
            slope = abs(result.curve[taus[1]] - result.curve[taus[0]])
        elif i == len(taus) - 1:
            slope = abs(result.curve[taus[-1]] - result.curve[taus[-2]])
        else:
            slope = abs(result.curve[taus[i + 1]] - result.curve[taus[i - 1]]) / 2.0
        slopes[tau] = slope

    max_slope = max(slopes.values()) or 1.0
    return {tau: v / max_slope for tau, v in slopes.items()}


def _derive_early_lock(
    result: CISResult,
    tau_70: float,
) -> Dict[float, float]:
    """Early-lock mode: weight=1 for all τ ≥ τ₇₀, 0 below.

    Appropriate for global factors (style, time of day, weather) that lock
    in early in the denoising trajectory.  The unlearning method intervenes
    throughout the high-noise regime where these concepts are first committed.
    """
    weights: Dict[float, float] = {}
    for tau in result.curve:
        weights[tau] = 1.0 if tau >= tau_70 else 0.0
    # Ensure boundary
    eps = 1e-6
    if tau_70 not in weights:
        weights[tau_70] = 1.0
    weights[tau_70 - eps] = 0.0
    return weights


# ======================================================================= #
#  Convenience: build a schedule without a CIS curve                      #
# ======================================================================= #

def make_uniform_schedule(
    tau_lo: float = 0.0,
    tau_hi: float = 1.0,
    n_points: int = 50,
) -> TemporalSchedule:
    """Create a flat schedule that applies uniform weight on [tau_lo, tau_hi].

    Useful as a baseline comparison or when a CIS curve is unavailable.
    """
    taus = np.linspace(0.0, 1.0, n_points).tolist()
    weights = {tau: 1.0 if tau_lo <= tau <= tau_hi else 0.0 for tau in taus}
    eps = 1e-6
    weights[tau_lo - eps] = 0.0
    weights[tau_hi + eps] = 0.0

    dummy_result = CISResult(
        curve={tau: float(tau >= tau_lo) for tau in taus},
        tau_50=tau_lo,
        tau_70=tau_hi,
        bandwidth=abs(tau_hi - tau_lo),
        wilson_ci_lower={},
        wilson_ci_upper={},
    )

    return TemporalSchedule(
        mode="band",
        tau_50=tau_lo,
        tau_70=tau_hi,
        bandwidth=abs(tau_hi - tau_lo),
        _weights=weights,
    )
