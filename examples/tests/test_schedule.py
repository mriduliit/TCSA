"""
Unit tests for the TemporalSchedule and derive_schedule_from_cis functions.

Tests all three schedule modes (band, slope_weighted, early_lock) and verifies
that weight(τ) returns values in [0, 1] and respects boundary conditions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from pci_unlearning.curves import CISResult
from pci_unlearning.schedule import (
    TemporalSchedule,
    derive_schedule_from_cis,
    make_uniform_schedule,
)


def _make_cis_result(taus, probs):
    """Helper: build a CISResult from parallel lists."""
    curve = dict(zip(taus, probs))
    lower = {t: max(0.0, p - 0.1) for t, p in zip(taus, probs)}
    upper = {t: min(1.0, p + 0.1) for t, p in zip(taus, probs)}
    tau_50 = next((t for t, p in sorted(zip(taus, probs)) if p >= 0.5), None)
    tau_70 = next((t for t, p in sorted(zip(taus, probs)) if p >= 0.7), None)
    bw = abs(tau_70 - tau_50) if (tau_50 and tau_70) else None
    return CISResult(
        curve=curve,
        tau_50=tau_50,
        tau_70=tau_70,
        bandwidth=bw,
        wilson_ci_lower=lower,
        wilson_ci_upper=upper,
        num_seeds=100,
    )


# Realistic CIS-like curve: low for small τ, rises to 1 for high τ
TAUS  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PROBS = [0.0, 0.0, 0.0, 0.1, 0.3, 0.55, 0.72, 0.85, 0.95, 1.0, 1.0]


class TestTemporalScheduleWeightBand(unittest.TestCase):

    def setUp(self):
        self.cis = _make_cis_result(TAUS, PROBS)
        self.sched = derive_schedule_from_cis(self.cis, mode="band")

    def test_weight_in_band_is_one(self):
        """Weights inside [τ₇₀, τ₅₀] should be 1."""
        lo = min(self.sched.tau_50, self.sched.tau_70)
        hi = max(self.sched.tau_50, self.sched.tau_70)
        mid = (lo + hi) / 2
        w = self.sched.weight(mid)
        self.assertAlmostEqual(w, 1.0, places=3)

    def test_weight_range(self):
        """All weight values should be in [0, 1]."""
        for tau in TAUS:
            w = self.sched.weight(tau)
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w, 1.0)

    def test_weight_below_band_is_zero(self):
        """Weight below τ₅₀ should be 0 in band mode."""
        w = self.sched.weight(0.0)
        self.assertAlmostEqual(w, 0.0, places=3)

    def test_active_range_within_01(self):
        """Active range should be within [0, 1]."""
        lo, hi = self.sched.active_range()
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLessEqual(lo, hi)


class TestTemporalScheduleWeightSlopeWeighted(unittest.TestCase):

    def setUp(self):
        self.cis = _make_cis_result(TAUS, PROBS)
        self.sched = derive_schedule_from_cis(self.cis, mode="slope_weighted")

    def test_weight_range(self):
        """All weight values should be in [0, 1]."""
        for tau in TAUS:
            w = self.sched.weight(tau)
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w, 1.0 + 1e-6,
                                  msg=f"weight={w} at τ={tau} exceeds 1.0")

    def test_peak_near_steepest_slope(self):
        """The peak weight should be near the steepest CIS slope (τ≈0.5-0.6)."""
        peak_tau = max(TAUS, key=lambda t: self.sched.weight(t))
        self.assertGreater(peak_tau, 0.3)
        self.assertLess(peak_tau, 0.8)

    def test_flat_curve_gives_uniform_weights(self):
        """A flat CIS curve (all same probability) should give equal weights."""
        flat_taus  = [0.2, 0.4, 0.6, 0.8]
        flat_probs = [0.5, 0.5, 0.5, 0.5]
        flat_cis   = _make_cis_result(flat_taus, flat_probs)
        sched = derive_schedule_from_cis(flat_cis, mode="slope_weighted")
        weights = [sched.weight(t) for t in flat_taus]
        self.assertAlmostEqual(min(weights), max(weights), places=3)


class TestTemporalScheduleWeightEarlyLock(unittest.TestCase):

    def setUp(self):
        self.cis = _make_cis_result(TAUS, PROBS)
        self.sched = derive_schedule_from_cis(self.cis, mode="early_lock")

    def test_weight_above_tau70_is_one(self):
        """Weight at τ > τ₇₀ should be 1 in early_lock mode."""
        w = self.sched.weight(0.95)
        self.assertAlmostEqual(w, 1.0, places=3)

    def test_weight_below_tau70_is_zero(self):
        """Weight below τ₇₀ should be 0."""
        w = self.sched.weight(0.1)
        self.assertAlmostEqual(w, 0.0, places=3)

    def test_weight_range(self):
        for tau in TAUS:
            w = self.sched.weight(tau)
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w, 1.0)


class TestTemporalScheduleInterpolation(unittest.TestCase):

    def test_interpolation_between_grid_points(self):
        """Weight should be linearly interpolated at non-grid τ values."""
        cis = _make_cis_result([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
        sched = derive_schedule_from_cis(cis, mode="slope_weighted")
        # Should not raise, should be in [0, 1]
        w = sched.weight(0.25)
        self.assertGreaterEqual(w, 0.0)
        self.assertLessEqual(w, 1.0)

    def test_weight_at_tau_zero(self):
        """Weight at τ=0 should not raise."""
        cis = _make_cis_result(TAUS, PROBS)
        sched = derive_schedule_from_cis(cis, mode="band")
        w = sched.weight(0.0)
        self.assertIsInstance(w, float)

    def test_weight_at_tau_one(self):
        """Weight at τ=1.0 should not raise."""
        cis = _make_cis_result(TAUS, PROBS)
        sched = derive_schedule_from_cis(cis, mode="band")
        w = sched.weight(1.0)
        self.assertIsInstance(w, float)


class TestUniformSchedule(unittest.TestCase):

    def test_uniform_schedule_weight_in_range(self):
        """Uniform schedule weight should be 1 inside range, 0 outside."""
        sched = make_uniform_schedule(tau_lo=0.3, tau_hi=0.7, n_points=20)
        self.assertAlmostEqual(sched.weight(0.5), 1.0, places=3)
        self.assertAlmostEqual(sched.weight(0.0), 0.0, places=3)
        self.assertAlmostEqual(sched.weight(1.0), 0.0, places=3)

    def test_uniform_schedule_active_range(self):
        sched = make_uniform_schedule(tau_lo=0.3, tau_hi=0.7)
        lo, hi = sched.active_range()
        self.assertAlmostEqual(lo, 0.3, delta=0.05)
        self.assertAlmostEqual(hi, 0.7, delta=0.05)


class TestScheduleSummary(unittest.TestCase):

    def test_summary_returns_string(self):
        cis = _make_cis_result(TAUS, PROBS)
        sched = derive_schedule_from_cis(cis, mode="band")
        s = sched.summary()
        self.assertIsInstance(s, str)
        self.assertIn("band", s)

    def test_unknown_mode_raises(self):
        cis = _make_cis_result(TAUS, PROBS)
        with self.assertRaises(ValueError):
            derive_schedule_from_cis(cis, mode="nonexistent_mode")  # type: ignore


if __name__ == "__main__":
    unittest.main()
