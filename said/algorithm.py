"""
said/algorithm.py — SAID: Structural Adversarial Invariance Discrimination

Reference implementation of the SAID filter (Algorithm 1 in the paper).

SAID is an unsupervised, training-free filter that selects quality-tracking
metrics for LLM-judged RAG evaluation by checking two retrieval-system
invariants:

  1. Order randomness  (Signal A): non-shuffled pipelines should rank above
     shuffled-retrieval pipelines, since shuffling chunks cannot improve
     retrieval quality.

  2. Information monotonicity (Signal B): top-5 BM25 should rank at least as
     high as top-1 BM25, since top-5 chunks are a strict superset of top-1
     and therefore contain weakly more information.

A refusal-aware masking step removes refusal-questions (any shuffled-pipeline
answer below 50 chars) before computing the signals, since refusal templates
contaminate metric statistics on adversarial pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Default thresholds (paper Section 4.4)
# -----------------------------------------------------------------------------

DEFAULT_THETA_A = 0.85          # Signal A threshold (fraction of pipeline pairs)
DEFAULT_REFUSAL_LEN = 50        # chars; below this, treat as refusal
DEFAULT_DISABLE_THRESHOLD = 0.7  # if > 70% of questions are refusals, disable mask
DEFAULT_FALLBACK_K = 3          # if < K metrics pass, keep top-K by combined score


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class CellData:
    """Per-cell view of metric scores. Wraps one entry from metric_scores_compact.json.

    Attributes:
        dataset: e.g. "HotpotQA"
        generator: e.g. "claude-sonnet-4-6"
        judge: e.g. "gpt-5"
        pipelines: dict of pipeline name -> {
            "answer_lengths": [int, ...],
            "metric_scores": {metric_name: [float | None, ...]},
        }
    """
    dataset: str
    generator: str
    judge: str
    pipelines: Dict[str, Dict]

    @classmethod
    def from_compact_dict(cls, cell: Dict) -> "CellData":
        return cls(
            dataset=cell["dataset"],
            generator=cell["generator"],
            judge=cell["judge"],
            pipelines={
                name: {
                    "answer_lengths": pdata["answer_lengths"],
                    "metric_scores": pdata["metric_scores"],
                }
                for name, pdata in cell["pipelines"].items()
            },
        )

    @property
    def n_samples(self) -> int:
        any_pipe = next(iter(self.pipelines.values()))
        return len(any_pipe["answer_lengths"])

    @property
    def shuffled_pipelines(self) -> List[str]:
        return [n for n in self.pipelines if "shuffled" in n]

    @property
    def non_shuffled_pipelines(self) -> List[str]:
        return [n for n in self.pipelines if "shuffled" not in n]


@dataclass
class SAIDResult:
    """Output of SAID for one cell."""
    kept_metrics: List[str]
    fallback_used: bool
    refusal_mask_active: bool
    n_refusal_questions: int
    n_total_questions: int
    signal_a_scores: Dict[str, float] = field(default_factory=dict)
    signal_b_scores: Dict[str, int] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _safe_mean(values: Sequence) -> float:
    """Mean over non-None / non-NaN entries; NaN-safe."""
    arr = np.asarray([v for v in values if v is not None and not _is_nan(v)],
                     dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _is_nan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


def _pipeline_metric_mean(cell: CellData, pipeline: str, metric: str,
                          mask: Optional[np.ndarray] = None) -> float:
    """Mean of a metric on a pipeline, optionally restricted to a question subset.

    Args:
        mask: bool array of length n_samples; True = include this question.
              If None, use all questions.
    """
    scores = cell.pipelines[pipeline]["metric_scores"].get(metric)
    if scores is None:
        return float("nan")
    if mask is None:
        return _safe_mean(scores)
    selected = [s for s, keep in zip(scores, mask) if keep]
    return _safe_mean(selected)


# -----------------------------------------------------------------------------
# Refusal masking
# -----------------------------------------------------------------------------

def compute_refusal_mask(cell: CellData,
                         refusal_len: int = DEFAULT_REFUSAL_LEN) -> np.ndarray:
    """Mark a question as 'refusal' if any shuffled pipeline's answer for it
    has length < refusal_len characters.

    Returns a bool array of length n_samples; True = NON-refusal (keep).
    """
    n = cell.n_samples
    is_refusal = np.zeros(n, dtype=bool)
    for shuf in cell.shuffled_pipelines:
        lens = cell.pipelines[shuf]["answer_lengths"]
        for i, length in enumerate(lens):
            if length < refusal_len:
                is_refusal[i] = True
    return ~is_refusal  # True = keep (non-refusal)


# -----------------------------------------------------------------------------
# Signal A — order randomness
# -----------------------------------------------------------------------------

def signal_a(cell: CellData, metric: str,
             mask: Optional[np.ndarray] = None) -> float:
    """Signal A: fraction of (non-shuffled, shuffled) pipeline pairs where the
    non-shuffled pipeline has a higher mean metric score.

    Score range: [0, 1]. 1 = always non-shuffled > shuffled (perfect agreement
    with order-randomness invariant). 0.5 = chance.
    """
    non_shuf = cell.non_shuffled_pipelines
    shuf = cell.shuffled_pipelines
    if not non_shuf or not shuf:
        return float("nan")

    total = 0
    wins = 0
    for ns_pipe in non_shuf:
        ns_mean = _pipeline_metric_mean(cell, ns_pipe, metric, mask)
        if _is_nan(ns_mean):
            continue
        for sh_pipe in shuf:
            sh_mean = _pipeline_metric_mean(cell, sh_pipe, metric, mask)
            if _is_nan(sh_mean):
                continue
            total += 1
            if ns_mean > sh_mean:
                wins += 1
    return wins / total if total > 0 else float("nan")


# -----------------------------------------------------------------------------
# Signal B — information monotonicity
# -----------------------------------------------------------------------------

def signal_b(cell: CellData, metric: str,
             mask: Optional[np.ndarray] = None,
             top1_pipeline: str = "bm25_top1_direct",
             top5_pipeline: str = "bm25_top5_direct") -> int:
    """Signal B: 1 if mean(metric on BM25 top-5) >= mean(metric on BM25 top-1),
    else 0. Returns NaN-handled int (0 or 1).

    The default uses BM25 because BM25 is the only retriever family in the
    paper's pipeline pool that exercises top-1 (see paper Appendix C).
    """
    if (top1_pipeline not in cell.pipelines or
            top5_pipeline not in cell.pipelines):
        return 0  # missing pipeline -> conservative: fail
    m_top1 = _pipeline_metric_mean(cell, top1_pipeline, metric, mask)
    m_top5 = _pipeline_metric_mean(cell, top5_pipeline, metric, mask)
    if _is_nan(m_top1) or _is_nan(m_top5):
        return 0
    return int(m_top5 >= m_top1)


# -----------------------------------------------------------------------------
# Main SAID filter — Algorithm 1
# -----------------------------------------------------------------------------

def said_filter(cell: CellData,
                metric_names: Sequence[str],
                theta_a: float = DEFAULT_THETA_A,
                refusal_len: int = DEFAULT_REFUSAL_LEN,
                disable_threshold: float = DEFAULT_DISABLE_THRESHOLD,
                fallback_k: int = DEFAULT_FALLBACK_K,
                ) -> SAIDResult:
    """Apply SAID to one evaluation cell.

    Args:
        cell: CellData with pipelines and metric scores.
        metric_names: list of metric names to filter (e.g. the 10 LLM-judged
            metrics, EXCLUDING gt_judge — gt_judge is the oracle, not a
            candidate metric for the filter).
        theta_a: Signal A threshold (default 0.85).
        refusal_len: refusal-mask threshold in chars (default 50).
        disable_threshold: if > this fraction of questions are masked as
            refusals, the mask is disabled (default 0.7).
        fallback_k: if fewer than this many metrics pass both signals, keep
            the top-K by combined score (default 3).

    Returns:
        SAIDResult with kept_metrics and diagnostic fields.
    """
    # Step 1-2: refusal masking
    keep_mask = compute_refusal_mask(cell, refusal_len)
    n_total = len(keep_mask)
    n_kept = int(keep_mask.sum())
    n_refusal = n_total - n_kept

    if (1 - n_kept / n_total) > disable_threshold if n_total else False:
        # Too many refusals — disable mask
        mask_active = False
        keep_mask_used: Optional[np.ndarray] = None
    else:
        mask_active = True
        keep_mask_used = keep_mask

    # Step 3-6: compute signals per metric
    sa_scores: Dict[str, float] = {}
    sb_scores: Dict[str, int] = {}
    for m in metric_names:
        sa_scores[m] = signal_a(cell, m, keep_mask_used)
        sb_scores[m] = signal_b(cell, m, keep_mask_used)

    kept = [m for m in metric_names
            if (not _is_nan(sa_scores[m]))
            and sa_scores[m] >= theta_a
            and sb_scores[m] == 1]

    fallback = False
    if len(kept) < fallback_k:
        # Step 7-9: fallback — keep top-K by combined score
        fallback = True
        combined = []
        for m in metric_names:
            sa = sa_scores[m] if not _is_nan(sa_scores[m]) else 0.0
            sb = sb_scores[m]
            combined.append((m, sa + 0.3 * sb))
        combined.sort(key=lambda x: x[1], reverse=True)
        kept = [m for m, _ in combined[:fallback_k]]

    return SAIDResult(
        kept_metrics=kept,
        fallback_used=fallback,
        refusal_mask_active=mask_active,
        n_refusal_questions=n_refusal,
        n_total_questions=n_total,
        signal_a_scores=sa_scores,
        signal_b_scores=sb_scores,
    )


# -----------------------------------------------------------------------------
# Aggregation: kept metrics -> pipeline ranking
# -----------------------------------------------------------------------------

def aggregate_pipeline_scores(cell: CellData,
                              kept_metrics: Sequence[str]) -> Dict[str, float]:
    """Aggregate kept metrics into one score per pipeline (simple mean of
    pipeline-level means).

    Returns: {pipeline_name: aggregated_score}
    """
    scores: Dict[str, float] = {}
    for pipe in cell.pipelines:
        per_metric = []
        for m in kept_metrics:
            mean = _pipeline_metric_mean(cell, pipe, m, mask=None)
            if not _is_nan(mean):
                per_metric.append(mean)
        scores[pipe] = float(np.mean(per_metric)) if per_metric else float("nan")
    return scores


def gold_judge_pipeline_scores(cell: CellData,
                               oracle_metric: str = "gt_judge"
                               ) -> Dict[str, float]:
    """Pipeline-level mean of the gold-judge oracle (gt_judge)."""
    scores: Dict[str, float] = {}
    for pipe in cell.pipelines:
        mean = _pipeline_metric_mean(cell, pipe, oracle_metric, mask=None)
        scores[pipe] = mean
    return scores
