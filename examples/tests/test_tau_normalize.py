"""
Unit tests for τ normalisation logic.

Tests that τ = t/1000 is applied correctly for all adapter types and that
the schedule's τ values are monotonically decreasing (high noise first).
All tests run without a GPU and without loading any model weights.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, PropertyMock
import torch


def _make_mock_scheduler(timesteps_tensor: torch.Tensor):
    """Return a minimal mock scheduler with .timesteps and .set_timesteps."""
    sched = MagicMock()
    sched.timesteps = timesteps_tensor
    sched.init_noise_sigma = 1.0
    sched.set_timesteps = MagicMock()
    sched.scale_model_input = lambda latent, t: latent
    return sched


def _make_unet_adapter_with_timesteps(timesteps: list):
    """Build a UNetAdapter using mocked pipeline internals."""
    from pci_unlearning.adapters import UNetAdapter

    ts_tensor = torch.tensor(timesteps, dtype=torch.long)
    sched = _make_mock_scheduler(ts_tensor)

    unet = MagicMock()
    unet.config.sample_size = 64
    unet.in_channels = 4

    vae = MagicMock()
    vae.config.scaling_factor = 0.18215
    vae.scale_factor = 8

    text_encoder = MagicMock()
    tokenizer = MagicMock()
    tokenizer.model_max_length = 77

    pipe = MagicMock()
    pipe.scheduler = sched
    pipe.unet = unet
    pipe.vae = vae
    pipe.vae_scale_factor = 8
    pipe.text_encoder = text_encoder
    pipe.tokenizer = tokenizer
    type(pipe).__name__ = "StableDiffusionPipeline"

    # Patch set_timesteps to not do anything since timesteps already set
    sched.set_timesteps.side_effect = lambda n, device=None: None

    adapter = UNetAdapter.__new__(UNetAdapter)
    adapter.pipe = pipe
    adapter.device = "cpu"
    adapter._num_steps = len(timesteps)
    adapter._is_xl = False
    return adapter


class TestTauNormalise(unittest.TestCase):

    def test_ddpm_tau_range(self):
        """τ values should all be in [0, 1]."""
        # Typical DDPM/DDIM timestep sequence: 999, 979, ..., 19, 1
        timesteps = list(range(999, 0, -20))[:50]  # 50 steps
        adapter = _make_unet_adapter_with_timesteps(timesteps)
        taus = adapter.get_tau_schedule()
        for tau in taus:
            self.assertGreaterEqual(tau, 0.0, "τ should be ≥ 0")
            self.assertLessEqual(tau, 1.0, "τ should be ≤ 1")

    def test_ddpm_tau_monotone_decreasing(self):
        """τ values should be monotonically decreasing (high noise first)."""
        timesteps = list(range(999, 0, -20))[:50]
        adapter = _make_unet_adapter_with_timesteps(timesteps)
        taus = adapter.get_tau_schedule()
        for i in range(len(taus) - 1):
            self.assertGreater(taus[i], taus[i + 1],
                               f"τ[{i}]={taus[i]:.3f} should be > τ[{i+1}]={taus[i+1]:.3f}")

    def test_tau_equals_t_over_1000(self):
        """τ = t / 1000 for integer timesteps in [0, 1000] range."""
        timesteps = [999, 799, 599, 399, 199, 1]
        adapter = _make_unet_adapter_with_timesteps(timesteps)
        taus = adapter.get_tau_schedule()
        expected = [t / 1000.0 for t in timesteps]
        for tau, exp in zip(taus, expected):
            self.assertAlmostEqual(tau, exp, places=5)

    def test_tau_to_step_index(self):
        """tau_to_step_index should return the closest step."""
        timesteps = [1000, 750, 500, 250, 0]
        adapter = _make_unet_adapter_with_timesteps(timesteps)
        # τ=0.5 → closest to 500/1000=0.5 → index 2
        idx = adapter.tau_to_step_index(0.5)
        self.assertEqual(idx, 2)

    def test_tau_to_step_index_high_tau(self):
        """τ close to 1.0 should map to index 0 (first step, pure noise)."""
        timesteps = [999, 799, 599, 399, 199]
        adapter = _make_unet_adapter_with_timesteps(timesteps)
        idx = adapter.tau_to_step_index(0.99)
        self.assertEqual(idx, 0)

    def test_tau_schedule_length_equals_steps(self):
        """Length of τ schedule should equal number of inference steps."""
        n_steps = 30
        timesteps = list(range(999, 0, -(1000 // n_steps)))[:n_steps]
        adapter = _make_unet_adapter_with_timesteps(timesteps)
        taus = adapter.get_tau_schedule()
        self.assertEqual(len(taus), n_steps)

    def test_native_timesteps_length_equals_tau_schedule(self):
        """get_native_timesteps() and get_tau_schedule() must have same length."""
        timesteps = list(range(999, 0, -20))[:50]
        adapter = _make_unet_adapter_with_timesteps(timesteps)
        self.assertEqual(len(adapter.get_tau_schedule()),
                         len(adapter.get_native_timesteps()))


class TestAutoAdapter(unittest.TestCase):

    def test_auto_adapter_unet_detection(self):
        """auto_adapter should return UNetAdapter for StableDiffusionPipeline."""
        from pci_unlearning.adapters import auto_adapter, UNetAdapter

        pipe = MagicMock()
        type(pipe).__name__ = "StableDiffusionPipeline"
        pipe.scheduler.timesteps = torch.tensor([999, 799, 599, 399, 199])
        pipe.scheduler.init_noise_sigma = 1.0
        pipe.scheduler.set_timesteps = MagicMock()
        pipe.unet.config.sample_size = 64
        pipe.unet.in_channels = 4
        pipe.vae_scale_factor = 8

        adapter = auto_adapter(pipe, num_inference_steps=5, device="cpu")
        self.assertIsInstance(adapter, UNetAdapter)

    def test_auto_adapter_flowmatch_detection(self):
        """auto_adapter should return FlowMatchAdapter for FluxPipeline."""
        from pci_unlearning.adapters import auto_adapter, FlowMatchAdapter

        pipe = MagicMock()
        type(pipe).__name__ = "FluxPipeline"
        pipe.scheduler.timesteps = torch.tensor([999, 750, 500, 250, 1])
        pipe.scheduler.init_noise_sigma = 1.0
        pipe.scheduler.set_timesteps = MagicMock()
        pipe.transformer.config.in_channels = 16
        pipe.vae_scale_factor = 8

        adapter = auto_adapter(pipe, num_inference_steps=5, device="cpu")
        self.assertIsInstance(adapter, FlowMatchAdapter)


if __name__ == "__main__":
    unittest.main()
