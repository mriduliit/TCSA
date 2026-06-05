"""
Hook PCI temporal schedules into existing unlearning methods.

``PCIGuidedIntervention`` is a context manager that wraps an existing
unlearning function.  At each denoising step it:
  1. Computes the current normalised τ from the adapter's schedule.
  2. Queries ``schedule.weight(τ)`` to get a step-specific strength.
  3. Scales the unlearning function's intervention by that weight.

The existing unlearning method is treated as a **black box** — its internals
are not modified.  The context manager patches the function's ``strength``
(or ``scale``, or ``alpha``) keyword argument at call time.

For SAFREE specifically, a ``SAFREEScheduledDict`` helper dynamically adjusts
the ``alpha`` and ``re_attn_t`` fields of the ``safree_dict`` that SAFREE's
``__call__`` method receives, using the PCI schedule to gate temporal windows.

Usage::

    from pci_unlearning.integrate import PCIGuidedIntervention

    schedule = derive_schedule_from_cis(cis_result, mode="band")

    # Generic wrapper (replaces any 'strength'/'scale'/'alpha' kwarg)
    with PCIGuidedIntervention(schedule, my_fn, strength_kwarg="strength") as guided:
        result = guided(latent, emb, t, strength=1.0)

    # SAFREE-specific helper
    from pci_unlearning.integrate import PCIGuidedSAFREE
    guided_dict = PCIGuidedSAFREE(schedule, base_safree_dict)
    imgs = pipe(..., safree_dict=guided_dict, ...)
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Dict, List, Optional

from .adapters import DiffusionAdapter
from .schedule import TemporalSchedule

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Generic context manager                                                 #
# ======================================================================= #

class PCIGuidedIntervention:
    """Context manager that wraps an unlearning function with PCI scheduling.

    The wrapped function receives a ``strength`` multiplier keyword argument
    equal to ``base_strength * schedule.weight(current_tau)`` at each call.
    If the function does not accept a ``strength`` kwarg, use *strength_kwarg*
    to specify an alternative parameter name (e.g., ``"alpha"`` for SAFREE).

    Args:
        adapter:         DiffusionAdapter whose τ schedule drives step timing.
        schedule:        TemporalSchedule from ``derive_schedule_from_cis``.
        unlearning_fn:   The existing unlearning callable to wrap.
        strength_kwarg:  Name of the strength/scale keyword argument in
                         *unlearning_fn*.  Set ``None`` to inject a new
                         ``_pci_weight`` kwarg (function must accept it).
        base_strength:   Nominal strength value that gets multiplied by the
                         schedule weight.  Default: 1.0.

    Example::

        with PCIGuidedIntervention(adapter, schedule, my_fn, "alpha", 0.01) as guided:
            for i, (tau, t) in enumerate(zip(adapter.get_tau_schedule(),
                                             adapter.get_native_timesteps())):
                latent = guided(latent, emb, t)
    """

    def __init__(
        self,
        adapter: DiffusionAdapter,
        schedule: TemporalSchedule,
        unlearning_fn: Callable,
        strength_kwarg: Optional[str] = "strength",
        base_strength: float = 1.0,
    ) -> None:
        self.adapter = adapter
        self.schedule = schedule
        self._fn = unlearning_fn
        self._kwarg = strength_kwarg
        self._base_strength = base_strength
        self._step_index: int = 0
        self._tau_schedule: List[float] = []

    def __enter__(self) -> "PCIGuidedIntervention":
        self._step_index = 0
        self._tau_schedule = self.adapter.get_tau_schedule()
        logger.debug("PCIGuidedIntervention: active range = %s", self.schedule.active_range())
        return self

    def __exit__(self, *args: Any) -> None:
        self._step_index = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function with the schedule-scaled strength.

        Increments the internal step counter each time this is called.
        """
        if self._step_index < len(self._tau_schedule):
            tau = self._tau_schedule[self._step_index]
        else:
            tau = 0.0

        w = self.schedule.weight(tau)
        effective_strength = self._base_strength * w

        if self._kwarg is not None:
            # Multiply the existing kwarg value if provided, else set directly
            if self._kwarg in kwargs:
                kwargs[self._kwarg] = kwargs[self._kwarg] * w
            else:
                kwargs[self._kwarg] = effective_strength
        else:
            kwargs["_pci_weight"] = w

        logger.debug(
            "PCIGuidedIntervention step=%d τ=%.3f weight=%.3f",
            self._step_index, tau, w,
        )
        self._step_index += 1
        return self._fn(*args, **kwargs)

    def current_weight(self) -> float:
        """Return the schedule weight at the current step index."""
        if self._step_index < len(self._tau_schedule):
            return self.schedule.weight(self._tau_schedule[self._step_index])
        return 0.0


# ======================================================================= #
#  SAFREE-specific scheduled dict                                          #
# ======================================================================= #

class PCIGuidedSAFREEDict:
    """A drop-in replacement for SAFREE's ``safree_dict`` that applies PCI scheduling.

    SAFREE Architecture Notes
    -------------------------
    SAFREE reads ``safree_dict`` in two distinct phases:

    **Before the denoising loop** (once per image):
    - ``sf["safree"]``  → enables projection
    - ``sf["alpha"]``   → threshold for token removal in ``safree_projection``
    - ``sf["svf"]``     → enables adaptive ``beta_adjusted`` timing heuristic
    - ``sf["lra"]``     → enables latent re-attention
    - ``sf["logger"]``, ``sf["up_t"]``, ``sf["category"]``

    **Inside the denoising loop** (once per step) — ONLY when ``svf=False``:
    - ``sf["re_attn_t"]`` → step-index range ``[lo, hi]`` controlling which
      steps apply the rescaled embeddings

    Critical: when ``svf=True``, SAFREE uses its own sigmoid heuristic
    (``f_beta``) and completely ignores ``re_attn_t``.  This class therefore
    **overrides ``svf`` to False** and replaces it with the PCI-derived step
    range in ``re_attn_t``.

    The ``alpha`` value cannot be scheduled per-step (it's read once before
    the loop).  We keep ``alpha`` at the base value; the temporal gating is
    achieved entirely through ``re_attn_t``.

    Integration
    -----------
    Pass an instance as ``safree_dict=...`` in any
    ``ModifiedStableDiffusionPipeline.__call__``.  Set ``svf=False`` in the
    base dict (or let this class override it automatically).

    Args:
        schedule:    TemporalSchedule from ``derive_schedule_from_cis``.
        base_dict:   Original ``safree_dict`` (will not be mutated).
        adapter:     DiffusionAdapter — used to convert τ band → step indices.

    Example::

        base_dict = {
            "safree": True, "svf": False, "lra": False,
            "re_attn_t": [-1, 1001], "alpha": 0.01,
            "logger": logger, "up_t": 10, "category": "nudity"
        }
        guided_dict = PCIGuidedSAFREEDict(schedule, base_dict, adapter)
        imgs = pipe(..., safree_dict=guided_dict)
    """

    def __init__(
        self,
        schedule: TemporalSchedule,
        base_dict: Dict[str, Any],
        adapter: DiffusionAdapter,
    ) -> None:
        self._schedule = schedule
        self._base = dict(base_dict)
        self._adapter = adapter

        # Pre-compute the step index window once from the τ active range.
        # This is the only per-step control mechanism that works with SAFREE.
        self._re_attn_override = self._compute_step_range()

        logger.debug(
            "PCIGuidedSAFREEDict: τ active range=%s → step range=%s",
            schedule.active_range(),
            self._re_attn_override,
        )

    def _compute_step_range(self) -> List[int]:
        """Convert the schedule's active τ range into SAFREE step indices.

        SAFREE applies rescaled embeddings for steps where
        ``re_attn_t[0] <= step_index <= re_attn_t[1]``.
        Step index 0 = highest τ (pure noise), last index = τ ≈ 0 (clean).

        We map the schedule's active τ band [τ_lo, τ_hi] (τ_lo < τ_hi since
        τ_lo is closer to 0 / clean side) to the corresponding step indices.
        """
        taus = self._adapter.get_tau_schedule()  # descending: [~1.0, ..., ~0.0]
        tau_lo, tau_hi = self._schedule.active_range()

        if tau_lo == 0.0 and tau_hi == 0.0:
            # Degenerate schedule (e.g. no CIS data) — disable intervention
            return [999999, 999999]

        step_lo = None  # first step entering the active band (tau <= tau_hi)
        step_hi = None  # last step still in the active band (tau >= tau_lo)

        for i, tau in enumerate(taus):
            if tau <= tau_hi and step_lo is None:
                step_lo = i
            if tau >= tau_lo:
                step_hi = i

        if step_lo is None:
            step_lo = 0
        if step_hi is None:
            step_hi = len(taus) - 1

        return [step_lo, step_hi]

    # ---- dict-like interface ---- #

    def __getitem__(self, key: str) -> Any:
        # Override svf → False so that re_attn_t (not f_beta) controls timing.
        if key == "svf":
            return False

        # Return the PCI-derived step range instead of the original window.
        if key == "re_attn_t":
            return self._re_attn_override

        return self._base[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._base[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._base

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self._base.keys()

    def values(self):
        return (self[k] for k in self._base)

    def items(self):
        return ((k, self[k]) for k in self._base)


# ======================================================================= #
#  Convenience factory                                                     #
# ======================================================================= #

def wrap_safree_dict(
    schedule: TemporalSchedule,
    base_safree_dict: Dict[str, Any],
    adapter: DiffusionAdapter,
) -> PCIGuidedSAFREEDict:
    """Convenience wrapper around ``PCIGuidedSAFREEDict``.

    Returns a ``PCIGuidedSAFREEDict`` ready to be passed directly as
    ``safree_dict=...`` in ``ModifiedStableDiffusionPipeline.__call__``.

    The returned dict:
    - Overrides ``svf=False`` so PCI controls the timing, not SAFREE's heuristic.
    - Sets ``re_attn_t`` to the step range corresponding to the PCI τ band.
    - Leaves ``alpha``, ``lra``, ``safree``, ``logger`` etc. unchanged.
    """
    return PCIGuidedSAFREEDict(
        schedule=schedule,
        base_dict=base_safree_dict,
        adapter=adapter,
    )
