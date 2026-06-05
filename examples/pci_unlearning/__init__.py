"""
pci_unlearning — Temporally-precise training-free unlearning via PCI.

Implements the Prompt-Conditioned Intervention (PCI) framework from:
  "Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned
  Interventions", ICLR 2026.

Wraps around existing unlearning methods (e.g. SAFREE) to make them
temporally precise using data-driven CIS/CDS curves.

Public API
----------
Adapters:
    auto_adapter        Auto-detect pipeline and create adapter.
    UNetAdapter         For SD1.5/2.1/SDXL.
    DiTAdapter          For PixArt-α.
    FlowMatchAdapter    For SD3.5/FLUX.

Probe:
    PromptConditionedIntervention    Core PCI engine (run_pci / run_cds).

Scorer:
    QwenVLScorer        Default VQA scorer (Qwen2.5-VL-3B-Instruct).
    QUESTION_TEMPLATES  Template registry from Table A1.
    get_question        Resolve concept + subcategory to full question.
    get_scorer          Instantiate a scorer by registry name.

Curves:
    compute_cis_curve                 CIS curve computation.
    compute_cds_curve                 CDS curve computation.
    concept_controlled_seed_filter    Appendix A.2 seed resampling.
    CISResult                         Curve + statistics dataclass.
    curve_auc                         Area under curve.
    max_cds                           Peak CDS value.

Schedule:
    TemporalSchedule           Intervention weight function.
    derive_schedule_from_cis   Build schedule from CIS curve.
    make_uniform_schedule      Flat schedule baseline.

Integrate:
    PCIGuidedIntervention      Generic unlearning fn wrapper.
    PCIGuidedSAFREEDict        SAFREE-specific safree_dict proxy.
    wrap_safree_dict           Convenience factory for SAFREE.

Evaluate:
    evaluate_unlearning        Full before/after evaluation via CDS.
    EvalReport                 Evaluation results dataclass.

Audit:
    audit_concept_residue      Pre-deployment concept residue check.
    batch_audit                Multi-concept batch audit.
    AuditReport                Audit results dataclass.
"""

from .adapters import (
    DiffusionAdapter,
    UNetAdapter,
    DiTAdapter,
    FlowMatchAdapter,
    auto_adapter,
)
from .probe import PromptConditionedIntervention
from .scorer import (
    ConceptScorer,
    QwenVLScorer,
    NudeNetScorer,
    QUESTION_TEMPLATES,
    get_question,
    get_scorer,
    register_scorer,
)
from .curves import (
    CISResult,
    compute_cis_curve,
    compute_cds_curve,
    concept_controlled_seed_filter,
    curve_auc,
    max_cds,
)
from .schedule import (
    TemporalSchedule,
    derive_schedule_from_cis,
    make_uniform_schedule,
)
from .integrate import (
    PCIGuidedIntervention,
    PCIGuidedSAFREEDict,
    wrap_safree_dict,
)
from .orthogonal import (
    OrthogonalEmbedding,
    CONCEPT_VOCAB,
    SAFE_CONCEPT_VOCAB,
    compute_vocab_direction,
    build_vocab_orthogonal_embedding,
    build_orthogonal_embedding,
    orthogonalize_embedding,
    compute_unsafe_direction,
    compute_unsafe_subspace,
)
from .evaluate import (
    EvalReport,
    ConceptEvalResult,
    evaluate_unlearning,
)
from .audit import (
    AuditReport,
    audit_concept_residue,
    batch_audit,
)

__all__ = [
    # adapters
    "DiffusionAdapter", "UNetAdapter", "DiTAdapter", "FlowMatchAdapter", "auto_adapter",
    # probe
    "PromptConditionedIntervention",
    # scorer
    "ConceptScorer", "QwenVLScorer", "NudeNetScorer", "QUESTION_TEMPLATES",
    "get_question", "get_scorer", "register_scorer",
    # curves
    "CISResult", "compute_cis_curve", "compute_cds_curve",
    "concept_controlled_seed_filter", "curve_auc", "max_cds",
    # schedule
    "TemporalSchedule", "derive_schedule_from_cis", "make_uniform_schedule",
    # integrate
    "PCIGuidedIntervention", "PCIGuidedSAFREEDict", "wrap_safree_dict",
    # evaluate
    "EvalReport", "ConceptEvalResult", "evaluate_unlearning",
    # audit
    "AuditReport", "audit_concept_residue", "batch_audit",
    # orthogonal
    "OrthogonalEmbedding", "CONCEPT_VOCAB", "SAFE_CONCEPT_VOCAB",
    "compute_vocab_direction", "build_vocab_orthogonal_embedding",
    "build_orthogonal_embedding", "orthogonalize_embedding",
    "compute_unsafe_direction", "compute_unsafe_subspace",
]

__version__ = "0.1.0"
