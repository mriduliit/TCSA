"""
Unit tests for the DiffusionAdapter protocol compliance.

Verifies that all three adapter types satisfy the required interface:
  - get_tau_schedule() / get_native_timesteps() consistent length
  - tau_to_step_index() returns valid index
  - encode_prompt() returns tuple (uncond, cond)
  - auto_adapter() maps pipeline class names correctly

No model weights are loaded; all tests use mocks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, patch
import torch

from pci_unlearning.adapters import (
    DiffusionAdapter,
    UNetAdapter,
    DiTAdapter,
    FlowMatchAdapter,
    auto_adapter,
)


# ======================================================================= #
#  Helpers                                                                 #
# ======================================================================= #

def _mock_scheduler(timesteps):
    sched = MagicMock()
    sched.timesteps = torch.tensor(timesteps, dtype=torch.long)
    sched.init_noise_sigma = 1.0
    sched.set_timesteps = MagicMock()
    sched.scale_model_input = lambda x, t: x
    sched.step = MagicMock(return_value=MagicMock(prev_sample=torch.zeros(1, 4, 64, 64)))
    return sched


def _make_unet_pipe(timesteps, name="StableDiffusionPipeline"):
    pipe = MagicMock()
    type(pipe).__name__ = name
    pipe.scheduler = _mock_scheduler(timesteps)
    pipe.unet.config.sample_size = 64
    pipe.unet.in_channels = 4
    pipe.vae_scale_factor = 8
    pipe.vae.config.scaling_factor = 0.18215
    # text_encoder returns (hidden_states,)
    hidden = torch.zeros(1, 77, 768)
    pipe.text_encoder.return_value = (hidden,)
    pipe.tokenizer.model_max_length = 77
    pipe.tokenizer.return_value = MagicMock(input_ids=torch.zeros(1, 77, dtype=torch.long))
    return pipe


def _make_unet_adapter(steps=10):
    ts = list(range(990, 0, -990 // steps))[:steps]
    pipe = _make_unet_pipe(ts)
    adapter = UNetAdapter.__new__(UNetAdapter)
    adapter.pipe = pipe
    adapter.device = "cpu"
    adapter._num_steps = steps
    adapter._is_xl = False
    return adapter


# ======================================================================= #
#  DiffusionAdapter ABC compliance                                         #
# ======================================================================= #

class TestDiffusionAdapterInterface(unittest.TestCase):

    def test_unet_adapter_is_subclass(self):
        self.assertTrue(issubclass(UNetAdapter, DiffusionAdapter))

    def test_dit_adapter_is_subclass(self):
        self.assertTrue(issubclass(DiTAdapter, DiffusionAdapter))

    def test_flowmatch_adapter_is_subclass(self):
        self.assertTrue(issubclass(FlowMatchAdapter, DiffusionAdapter))

    def test_cannot_instantiate_base_directly(self):
        with self.assertRaises(TypeError):
            DiffusionAdapter()  # type: ignore


# ======================================================================= #
#  UNetAdapter                                                             #
# ======================================================================= #

class TestUNetAdapter(unittest.TestCase):

    def setUp(self):
        self.adapter = _make_unet_adapter(steps=10)

    def test_get_tau_schedule_length(self):
        taus = self.adapter.get_tau_schedule()
        self.assertEqual(len(taus), 10)

    def test_get_tau_schedule_in_01(self):
        taus = self.adapter.get_tau_schedule()
        for t in taus:
            self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(t, 1.0)

    def test_get_tau_schedule_descending(self):
        taus = self.adapter.get_tau_schedule()
        for i in range(len(taus) - 1):
            self.assertGreater(taus[i], taus[i + 1])

    def test_get_native_timesteps_length_matches(self):
        self.assertEqual(
            len(self.adapter.get_tau_schedule()),
            len(self.adapter.get_native_timesteps()),
        )

    def test_tau_to_step_index_valid(self):
        n = len(self.adapter.get_tau_schedule())
        idx = self.adapter.tau_to_step_index(0.5)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, n)

    def test_tau_to_step_index_high_tau(self):
        """τ ≈ 1.0 should map to step 0 (highest noise)."""
        idx = self.adapter.tau_to_step_index(0.99)
        self.assertEqual(idx, 0)

    def test_tau_to_step_index_low_tau(self):
        """τ ≈ 0.0 should map to last step."""
        n = len(self.adapter.get_tau_schedule())
        idx = self.adapter.tau_to_step_index(0.0)
        self.assertEqual(idx, n - 1)

    def test_encode_prompt_returns_tuple(self):
        """encode_prompt should return (uncond, cond) tuple."""
        result = self.adapter.encode_prompt("a photo of a dog", "")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


# ======================================================================= #
#  auto_adapter factory                                                    #
# ======================================================================= #

class TestAutoAdapter(unittest.TestCase):

    def _make_pipe_with_name(self, name: str):
        pipe = MagicMock()
        type(pipe).__name__ = name
        pipe.scheduler.timesteps = torch.tensor([999, 750, 500, 250, 1])
        pipe.scheduler.init_noise_sigma = 1.0
        pipe.scheduler.set_timesteps = MagicMock()
        pipe.unet = MagicMock()
        pipe.unet.config.sample_size = 64
        pipe.unet.in_channels = 4
        pipe.vae_scale_factor = 8
        pipe.transformer = MagicMock()
        pipe.transformer.config.in_channels = 16
        pipe.transformer.config.sample_size = 64
        return pipe

    def test_sd15_maps_to_unet(self):
        pipe = self._make_pipe_with_name("StableDiffusionPipeline")
        adapter = auto_adapter(pipe, num_inference_steps=5, device="cpu")
        self.assertIsInstance(adapter, UNetAdapter)

    def test_sdxl_maps_to_unet(self):
        pipe = self._make_pipe_with_name("StableDiffusionXLPipeline")
        adapter = auto_adapter(pipe, num_inference_steps=5, device="cpu")
        self.assertIsInstance(adapter, UNetAdapter)

    def test_pixart_maps_to_dit(self):
        pipe = self._make_pipe_with_name("PixArtAlphaPipeline")
        adapter = auto_adapter(pipe, num_inference_steps=5, device="cpu")
        self.assertIsInstance(adapter, DiTAdapter)

    def test_flux_maps_to_flowmatch(self):
        pipe = self._make_pipe_with_name("FluxPipeline")
        adapter = auto_adapter(pipe, num_inference_steps=5, device="cpu")
        self.assertIsInstance(adapter, FlowMatchAdapter)

    def test_sd3_maps_to_flowmatch(self):
        pipe = self._make_pipe_with_name("StableDiffusion3Pipeline")
        adapter = auto_adapter(pipe, num_inference_steps=5, device="cpu")
        self.assertIsInstance(adapter, FlowMatchAdapter)

    def test_unknown_pipeline_falls_back_to_unet(self):
        import warnings
        pipe = self._make_pipe_with_name("SomeUnknownPipeline")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adapter = auto_adapter(pipe, num_inference_steps=5, device="cpu")
        self.assertIsInstance(adapter, UNetAdapter)
        self.assertTrue(any("Unknown pipeline" in str(warning.message) for warning in w))


# ======================================================================= #
#  FlowMatchAdapter τ normalisation for sub-1000 range                    #
# ======================================================================= #

class TestFlowMatchTauNormalise(unittest.TestCase):

    def test_flowmatch_sub1_timesteps_treated_as_direct(self):
        """Timestep values ≤ 1 should be used directly as τ (not divided by 1000)."""
        from pci_unlearning.adapters import FlowMatchAdapter

        pipe = MagicMock()
        pipe.scheduler.timesteps = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        pipe.scheduler.init_noise_sigma = 1.0
        pipe.scheduler.set_timesteps = MagicMock()
        pipe.transformer.config.in_channels = 16
        pipe.vae_scale_factor = 8
        type(pipe).__name__ = "FluxPipeline"

        adapter = FlowMatchAdapter.__new__(FlowMatchAdapter)
        adapter.pipe = pipe
        adapter.device = "cpu"
        adapter._num_steps = 5
        adapter._is_flux = True

        taus = adapter.get_tau_schedule()
        self.assertAlmostEqual(taus[0], 1.0, places=4)
        self.assertAlmostEqual(taus[-1], 0.0, places=4)

    def test_flowmatch_integer_timesteps_divided(self):
        """Timestep values > 1 should be divided by 1000."""
        from pci_unlearning.adapters import FlowMatchAdapter

        pipe = MagicMock()
        pipe.scheduler.timesteps = torch.tensor([1000, 750, 500, 250, 0])
        pipe.scheduler.init_noise_sigma = 1.0
        pipe.scheduler.set_timesteps = MagicMock()
        pipe.transformer.config.in_channels = 16
        pipe.vae_scale_factor = 8
        type(pipe).__name__ = "FluxPipeline"

        adapter = FlowMatchAdapter.__new__(FlowMatchAdapter)
        adapter.pipe = pipe
        adapter.device = "cpu"
        adapter._num_steps = 5
        adapter._is_flux = True

        taus = adapter.get_tau_schedule()
        self.assertAlmostEqual(taus[0], 1.0, places=4)
        self.assertAlmostEqual(taus[-1], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
