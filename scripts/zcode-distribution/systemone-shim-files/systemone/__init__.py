"""systemone: local, open System One decision models.

A Jev-style typed-decision layer over local GLiClass checkpoints *or* an
SGLang-served decision model:

- api.SystemOne / api.systemone  -> choice / score / noul in one batched call
- sglang_backend.SGLangBackend   -> same shapes via SGLang /v1/decisions
- sglang_backend.HybridBackend   -> local-first, SGLang escalation
- jev_backend.JevDecideBackend   -> same shapes via a JEV decision model
- loop.DecisionLoop              -> See > Decide > Act, the one agent loop
- calibration.*                 -> temperature / Platt / isotonic calibration
- mcp_server                     -> MCP tools (verify_claims, screen_content,
                                   rank_candidates) over stdio
- distill.*                      -> teacher -> tiny student pipeline
- tune.*                         -> domain fine-tuning helper

Quickstart:
    from systemone import SystemOne
    eng = SystemOne()  # loads gliclass-edge, one model at a time
    out = eng.systemone("The server is on fire", [
        {"name": "urgency", "type": "score", "levels": ["low", "medium", "high"]},
        {"name": "page", "type": "noul", "statement": "Should I page the on-call engineer?"},
    ])

SGLang-only quickstart (no torch needed):
    from systemone import SGLangBackend
    eng = SGLangBackend()  # talks to SGLANG_BASE_URL's /v1/decisions
    out = eng.systemone("The server is on fire", [...])  # same question shapes

Import weight: this package root stays importable on a slim install (no
torch / transformers / gliclass). Heavy names (api.*, shim local engine)
resolve lazily and raise a helpful error telling you to install the
``local`` extra when the heavy deps are absent.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.2.0"
"""Package version; mirrors [project] version in pyproject.toml."""

# Light modules stay eager: calibration/metrics need only numpy, and
# sglang_backend is stdlib-only — so the SGLang + shim-client paths work
# on a slim install with no torch.
from .calibration import (
    DECISION_TYPES,
    MIN_ROWS_PER_TYPE,
    CalibratedScorer,
    CalibrationExample,
    IsotonicCalibrator,
    PerTypeTemperatureCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    conformal_set,
    expected_calibration_error,
    fit_conformal_threshold,
    fit_temperature_by_type,
    load_type_calibration,
    multiclass_ece,
    quality_cost_frontier,
    route_threshold_for_target,
)
from .metrics import (
    aurc,
    brier_decomposition,
    brier_score,
    estimated_accuracy,
    failure_auroc,
    format_table,
    macro_f1,
    mcnemar,
    ndcg_at_k,
    nll_score,
    paired_bootstrap_ci,
    psi,
    reciprocal_rank,
    reliability_curve,
    selective_accuracy,
    summarize,
    top_label_ece,
)
from .patterns import (
    ABSTAIN_LABEL,
    MAX_STATE_CHARS,
    LatencyStats,
    StallGuard,
    SystemOneError,
    build_decision_prompts,
    choice_confidence,
    make_questions,
    noul_confidence,
    score_confidence,
    validate_choice,
    validate_distribution,
    with_abstain,
)
from .jev_backend import JevDecideBackend, JevError
from .jevk5_backend import JevK5Error, JevK5ServerBackend
from .kev_backend import KevBackend, KevError
from .clef_backend import ClefBackend, ClefError
from .rotation import (
    ConformalChoice,
    EnsembleBackend,
    RotationAveraged,
    SelfConsistent,
)
from .loop import ActResult, DecisionLoop, LoopResult, Observation, Step, run_loop
from .rerank_backend import OnnxCrossEncoder, RerankBackend
from .sglang_backend import (
    CascadeBackend,
    HybridBackend,
    SGLangBackend,
    SGLangError,
    decide_fn_for,
)

__all__ = [
    "SystemOne",
    "SystemOneError",
    "SGLangBackend",
    "SGLangError",
    "HybridBackend",
    "CascadeBackend",
    "decide_fn_for",
    "JevK5ServerBackend",
    "JevK5Error",
    "KevBackend",
    "KevError",
    "ClefBackend",
    "ClefError",
    "RotationAveraged",
    "ConformalChoice",
    "SelfConsistent",
    "EnsembleBackend",
    "JevDecideBackend",
    "JevError",
    "DecisionLoop",
    "run_loop",
    "Observation",
    "ActResult",
    "Step",
    "LoopResult",
    "RerankBackend",
    "OnnxCrossEncoder",
    "MAX_STATE_CHARS",
    "ABSTAIN_LABEL",
    "LatencyStats",
    "StallGuard",
    "default_device",
    "make_questions",
    "validate_choice",
    "validate_distribution",
    "with_abstain",
    "serve_shim",
    "TemperatureCalibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "CalibratedScorer",
    "CalibrationExample",
    "expected_calibration_error",
    "multiclass_ece",
    "PerTypeTemperatureCalibrator",
    "fit_temperature_by_type",
    "load_type_calibration",
    "fit_conformal_threshold",
    "conformal_set",
    "route_threshold_for_target",
    "quality_cost_frontier",
    "DECISION_TYPES",
    "MIN_ROWS_PER_TYPE",
    "choice_confidence",
    "score_confidence",
    "noul_confidence",
    "build_decision_prompts",
    "top_label_ece",
    "brier_score",
    "nll_score",
    "aurc",
    "selective_accuracy",
    "summarize",
    "format_table",
    "ndcg_at_k",
    "reciprocal_rank",
    "macro_f1",
    "failure_auroc",
    "brier_decomposition",
    "reliability_curve",
    "paired_bootstrap_ci",
    "mcnemar",
    "psi",
    "estimated_accuracy",
]

# Heavy names resolve on first access so `import systemone` never requires
# torch. Maps attribute -> (submodule, name in submodule).
_LAZY: dict[str, tuple[str, str]] = {
    "SystemOne": ("api", "SystemOne"),
    "default_device": ("api", "default_device"),
    "serve_shim": ("shim", "serve"),
}

_LOCAL_EXTRA_HINT = (
    "pip install 'systemone[local]' for the torch/GLiClass engine, or use "
    "SGLangBackend / SYSTEMONE_ENGINE=sglang for the stdlib-only SGLang path"
)


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'systemone' has no attribute {name!r}")
    module_name, attr = target
    # Static imports only: the lazy table can name just "api"/"shim", and
    # anything else fails closed here instead of reaching importlib.
    try:
        if module_name == "api":
            from . import api as module
        elif module_name == "shim":
            from . import shim as module
        else:  # pragma: no cover - unreachable via the _LAZY table above
            raise ImportError(f"unknown lazy module {module_name!r}")
        return getattr(module, attr)
    except ImportError as exc:
        raise ImportError(
            f"systemone.{name} needs the local-engine dependencies "
            f"({exc.name or 'torch'} is not installed). {_LOCAL_EXTRA_HINT}."
        ) from None


def __dir__() -> list[str]:
    return sorted(__all__)
