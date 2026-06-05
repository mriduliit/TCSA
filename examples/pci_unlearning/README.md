# pci_unlearning

Temporally-precise training-free concept unlearning for diffusion models via **Prompt-Conditioned Interventions (PCI)**.

Based on the ICLR 2026 paper:
> *Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned Interventions*  
> Ada Görgün, Fawaz Sammani, Nikos Deligiannis, Bernt Schiele, Jonas Fischer

This module wraps existing unlearning methods (e.g. **SAFREE**) to make their interventions temporally precise using data-driven CIS/CDS curves.  Nothing in the existing unlearning code is modified.

---

## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────┐
│                      Your pipeline                          │
│   (StableDiffusionPipeline, SDXL, SD3.5, FLUX, PixArt)     │
└────────────────────┬────────────────────────────────────────┘
                     │ wrapped by
┌────────────────────▼────────────────────────────────────────┐
│                  DiffusionAdapter                           │
│  (UNetAdapter / DiTAdapter / FlowMatchAdapter)              │
│  Normalised τ API:  τ=1 → pure noise,  τ=0 → clean image   │
└──┬─────────────┬───────────────────────┬────────────────────┘
   │             │                       │
   ▼             ▼                       ▼
probe.py      scorer.py             schedule.py
PCI engine    Qwen2.5-VL-3B         TemporalSchedule
run_pci()     is_concept_present()  weight(τ) → [0,1]
run_cds()     question templates
   │             │                       │
   └──────┬──────┘                       │
          ▼                              │
       curves.py                         │
  compute_cis_curve()                    │
  compute_cds_curve()   ────────────────►integrate.py
  seed_resampling()                     PCIGuidedSAFREEDict
  CISResult dataclass                    wrap_safree_dict()
          │
          ├──────────► evaluate.py    (before/after CDS comparison)
          └──────────► audit.py       (pre-deployment residue check)
```

---

## Installation

No separate install step — the module lives alongside the SAFREE code.  Required pip packages:

```bash
pip install diffusers transformers accelerate torch numpy
pip install qwen-vl-utils   # for the default VQA scorer
```

---

## Quick start

### 1. Load pipeline and create adapter

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from pci_unlearning import auto_adapter

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    scheduler=DPMSolverMultistepScheduler.from_pretrained(..., subfolder="scheduler"),
    torch_dtype=torch.float32,
).to("cuda")

adapter = auto_adapter(pipe, num_inference_steps=50, device="cuda")
```

### 2. Compute the CIS curve for the concept to forget

```python
from pci_unlearning import compute_cis_curve, QwenVLScorer

scorer = QwenVLScorer(device="cuda")

cis_result = compute_cis_curve(
    adapter=adapter,
    base_prompt="a realistic photo of a person at the beach",
    concept_prompt="a realistic photo of a nude person at the beach",
    concept="nudity",
    scorer=scorer,
    num_seeds=100,   # paper default
    num_taus=20,     # paper default
    subcategory="nudity",
)

print(f"τ₅₀ = {cis_result.tau_50}")
print(f"τ₇₀ = {cis_result.tau_70}")
print(f"bandwidth W = {cis_result.bandwidth}")
```

### 3. Derive a temporal intervention schedule

```python
from pci_unlearning import derive_schedule_from_cis

schedule = derive_schedule_from_cis(cis_result, mode="band")
# Other modes: "slope_weighted", "early_lock"

print(schedule.summary())
# → TemporalSchedule(mode='band', τ₅₀=0.45, τ₇₀=0.65, bandwidth=0.20, active=[0.45, 0.65])
```

### 4a. Run SAFREE with PCI-guided temporal schedule

```python
from pci_unlearning import wrap_safree_dict

base_safree_dict = {
    "safree": True, "svf": True, "lra": False,
    "re_attn_t": [-1, 1001], "alpha": 0.01,
    "logger": logger, "up_t": 10, "category": "nudity",
}

guided_dict = wrap_safree_dict(schedule, base_safree_dict, adapter)

# Use exactly like a normal safree_dict — no other changes needed
imgs = pipe(
    "a photo of a person swimming",
    safree_dict=guided_dict,
    negative_prompt="nudity",
    negative_prompt_space=["nudity", "explicit content"],
    ...
)
```

### 4b. Generic function wrapper

```python
from pci_unlearning import PCIGuidedIntervention

with PCIGuidedIntervention(adapter, schedule, my_unlearning_fn, "alpha", 0.01) as guided:
    for tau, t in zip(adapter.get_tau_schedule(), adapter.get_native_timesteps()):
        latent = guided(latent, emb, t)
```

### 5. Evaluate unlearning quality (CDS curves)

```python
from pci_unlearning import evaluate_unlearning

report = evaluate_unlearning(
    adapter=adapter,                      # unlearned pipeline
    forgotten_concepts=[
        ("nudity", concept_prompt, base_prompt),
    ],
    retained_concepts=[
        ("smile", smile_concept_prompt, base_prompt),
    ],
    original_adapter=original_adapter,    # optional: pre-unlearning baseline
    num_seeds=100,
)
print(report.summary())
```

### 6. Pre-deployment audit

```python
from pci_unlearning import audit_concept_residue

audit = audit_concept_residue(
    adapter=adapter,
    forbidden_concept="nudity",
    contexts=[
        "a photo of a person at the beach",
        "a painting of a mythological scene",   # indirect
        "a horse in a living room",              # OOD — locks in earlier
    ],
    threshold=0.1,    # max allowed CIS at any τ
)
print(audit.summary())
```

---

## Module reference

### `adapters.py` — DiffusionAdapter protocol

| Class | Pipelines |
|---|---|
| `UNetAdapter` | SD 1.5, SD 2.1, SDXL |
| `DiTAdapter` | PixArt-α, PixArt-Σ |
| `FlowMatchAdapter` | SD 3.5, FLUX |
| `auto_adapter(pipe, ...)` | auto-detects from class name |

**Key methods** (all adapters):

| Method | Returns | Description |
|---|---|---|
| `encode_prompt(str, str)` | `(uncond_emb, cond_emb)` | Encode prompt for CFG |
| `init_latents(seed, ...)` | `Tensor[B,C,H,W]` | Sample initial noise xT |
| `get_tau_schedule()` | `list[float]` | Normalised τ per step, descending |
| `get_native_timesteps()` | `list` | Native scheduler timestep values |
| `denoise_step(lat, emb, t)` | `Tensor` | Single denoising step |
| `decode(latent)` | `PIL.Image` | Decode latent to RGB image |
| `tau_to_step_index(tau)` | `int` | Closest step index for given τ |

**Timestep normalisation:** τ = native_t / 1000.0 for all scheduler types.
τ=1 is pure noise; τ=0 is clean image.

---

### `probe.py` — PromptConditionedIntervention

```python
pci = PromptConditionedIntervention(adapter, guidance_scale=7.5)

image = pci.run_pci(base_prompt, concept_prompt, switch_tau=0.5, seed=42)
image = pci.run_cds(concept_prompt, base_prompt, switch_tau=0.5, seed=42)
image = pci.run_full(prompt, seed=42)   # no switching (baseline)
```

---

### `scorer.py` — VQA concept detection

```python
scorer = QwenVLScorer(device="cuda")
present = scorer.is_concept_present(image, "nudity")

# Question template registry (Table A1 from paper)
from pci_unlearning import QUESTION_TEMPLATES, get_question
q = get_question("old", subcategory="age_group")
# → "Is the person in the image a old?"

# Custom scorer
from pci_unlearning import register_scorer, ConceptScorer
class MyScorer(ConceptScorer):
    def is_concept_present(self, image, concept, question=None): ...
register_scorer("my_scorer", MyScorer)
scorer = get_scorer("my_scorer")
```

---

### `curves.py` — CIS/CDS curves

```python
from pci_unlearning import (
    compute_cis_curve, compute_cds_curve,
    concept_controlled_seed_filter,
    CISResult, curve_auc, max_cds,
)

# CIS: measures concept insertion success
cis: CISResult = compute_cis_curve(adapter, base_prompt, concept_prompt, concept, ...)

# CDS: measures concept deletion success (for unlearning evaluation)
cds: CISResult = compute_cds_curve(adapter, concept_prompt, base_prompt, concept, ...)

# Seed resampling (Appendix A.2) — called automatically inside compute_*_curve
valid_seeds, used_neg_guidance = concept_controlled_seed_filter(
    adapter, base_prompt, concept, n_seeds_needed=100, scorer=scorer
)
```

**`CISResult` fields:**

| Field | Type | Description |
|---|---|---|
| `curve` | `dict[float, float]` | τ → mean probability |
| `tau_50` | `float \| None` | First τ where curve ≥ 0.50 |
| `tau_70` | `float \| None` | First τ where curve ≥ 0.70 |
| `bandwidth` | `float \| None` | `\|τ₇₀ − τ₅₀\|` |
| `wilson_ci_lower/upper` | `dict[float, float]` | 95% Wilson CI per τ |
| `num_seeds` | `int` | Seeds used after filtering |

---

### `schedule.py` — Temporal schedule

```python
from pci_unlearning import derive_schedule_from_cis, TemporalSchedule

schedule = derive_schedule_from_cis(cis_result, mode="band")

w = schedule.weight(tau)          # float in [0, 1]
lo, hi = schedule.active_range()  # τ range where weight > 0
print(schedule.summary())
```

**Schedule modes:**

| Mode | Behaviour | Best for |
|---|---|---|
| `"band"` | weight=1 inside `[τ₇₀, τ₅₀]` | Most concepts (default) |
| `"slope_weighted"` | w(τ) ∝ `\|dCIS/dτ\|` | Smooth, gradient-like targeting |
| `"early_lock"` | weight=1 for all τ ≥ τ₇₀ | Global/style concepts (lock in early) |

---

### `integrate.py` — Hook into existing unlearning

**For SAFREE** (drop-in `safree_dict` replacement):

```python
from pci_unlearning import wrap_safree_dict
guided_dict = wrap_safree_dict(schedule, base_safree_dict, adapter)
pipe(..., safree_dict=guided_dict)
```

**Generic wrapper** (for any unlearning function):

```python
from pci_unlearning import PCIGuidedIntervention
with PCIGuidedIntervention(adapter, schedule, fn, strength_kwarg="alpha") as guided:
    result = guided(latent, emb, t)   # alpha is scaled by schedule.weight(τ)
```

---

### `evaluate.py` — Unlearning evaluation

```python
from pci_unlearning import evaluate_unlearning, EvalReport

report: EvalReport = evaluate_unlearning(
    adapter=unlearned_adapter,
    forgotten_concepts=[("nudity", concept_prompt, base_prompt)],
    retained_concepts=[("smile", smile_prompt, base_prompt)],
    original_adapter=original_adapter,   # optional baseline
    cds_auc_threshold=0.2,   # lower = stricter unlearning requirement
    cds_peak_threshold=0.3,
)
print(report.summary())
```

---

### `audit.py` — Pre-deployment safety check

```python
from pci_unlearning import audit_concept_residue, AuditReport

report: AuditReport = audit_concept_residue(
    adapter=unlearned_adapter,
    forbidden_concept="nudity",
    contexts=[...],   # include OOD contexts — they lock in earlier!
    threshold=0.1,
)
# report.passed  → True if max CIS ≤ threshold in ALL contexts
```

---

## Running the tests

```bash
cd /storage/s25017/TRANING_FREE_UNLEARNING
python -m pytest tests/ -v
# or individually:
python tests/test_tau_normalize.py
python tests/test_schedule.py
python tests/test_adapters.py
```

All tests run without GPU and without loading model weights (pure mocks).

---

## Running the example

```bash
cd /storage/s25017/TRANING_FREE_UNLEARNING
python examples/run_pci_unlearning.py
```

The example loads SD1.4, computes a CIS curve for nudity, derives a PCI schedule, runs PCI-guided SAFREE generation, computes CDS evaluation, and runs a pre-deployment audit.  Results are saved to `results/pci_example/`.

---

## Key design decisions

1. **Architecture-agnostic**: All code goes through the `DiffusionAdapter` interface.
   UNet, DiT, and FlowMatch models are all supported by swapping the adapter.

2. **τ normalisation**: τ = t/1000 for all scheduler families.
   τ=1 is pure noise, τ=0 is clean — consistent with the paper's Appendix B.1.

3. **Seed resampling (Appendix A.2)**: Automatic two-stage filtering before every
   curve computation.  Stage 1: discard seeds where the base prompt already
   produces the concept.  Stage 2 (fallback): add the concept as a negative
   prompt and re-filter.

4. **No modification of existing code**: `pci_unlearning/` is a completely separate
   module.  The only touchpoint is `PCIGuidedSAFREEDict`, which is a drop-in
   replacement for SAFREE's `safree_dict` dict argument.

5. **VLM scorer is pluggable**: Implement `ConceptScorer` and register it with
   `register_scorer()` to use any VLM.
