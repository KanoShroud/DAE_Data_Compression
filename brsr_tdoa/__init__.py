"""独立的 BRSR-TDOA 阶段 A/B 研究模块。"""

from .brsr_core import (
    BayesianSpectralModel,
    BitLedger,
    DecisionAction,
    DecisionState,
    LikelihoodContext,
    NestedComplexQuantizer,
    PolicyResult,
    build_default_model,
    build_likelihood_context,
    initialize_state,
    run_dynamic_policy,
)

__all__ = [
    "BayesianSpectralModel",
    "BitLedger",
    "DecisionAction",
    "DecisionState",
    "LikelihoodContext",
    "NestedComplexQuantizer",
    "PolicyResult",
    "build_default_model",
    "build_likelihood_context",
    "initialize_state",
    "run_dynamic_policy",
]
