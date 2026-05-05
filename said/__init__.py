"""said — reference implementation of the SAID filter for unsupervised RAG
metric reliability filtering."""

from said.algorithm import (
    CellData,
    SAIDResult,
    aggregate_pipeline_scores,
    compute_refusal_mask,
    gold_judge_pipeline_scores,
    said_filter,
    signal_a,
    signal_b,
)

__all__ = [
    "CellData",
    "SAIDResult",
    "aggregate_pipeline_scores",
    "compute_refusal_mask",
    "gold_judge_pipeline_scores",
    "said_filter",
    "signal_a",
    "signal_b",
]
