"""baselines/unsupervised.py — Unsupervised baseline filters.

Implements the unsupervised baselines compared against SAID in Section 5.1
of the paper:

  - Uniform: simple mean of all metrics (RAGAS-style; the de facto standard).
  - DropConciseness: drop the single metric most correlated with answer length.
  - PMA (Pairwise Metric Agreement): a consistency-based filter that retains
    metrics whose pipeline rankings agree with most other metrics.
    Self-constructed; motivated by the observation that LLM-judge biases are
    often correlated across metrics.
  - LengthFilter: drop any metric whose pipeline-level Kendall tau to mean
    answer length exceeds 0.3 in absolute value. Self-constructed; motivated
    by length-bias documentation in the LLM-judge literature.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import kendalltau

from said.algorithm import CellData, _is_nan, _pipeline_metric_mean


# -----------------------------------------------------------------------------
# Uniform (no filter)
# -----------------------------------------------------------------------------

def uniform_filter(cell: CellData, metric_names: Sequence[str]) -> List[str]:
    """Trivial baseline: keep all metrics. Equivalent to RAGAS-style averaging.
    """
    return list(metric_names)


# -----------------------------------------------------------------------------
# DropConciseness
# -----------------------------------------------------------------------------

def drop_conciseness_filter(cell: CellData,
                            metric_names: Sequence[str]) -> List[str]:
    """Drop the single metric most correlated with answer length on this cell.

    "DropConciseness" is the simplest possible response to length bias:
    identify the one metric most correlated with length and drop it. In our
    experience this is almost always `conciseness`, but the filter is
    data-driven (computed on the cell) rather than hard-coded.
    """
    # pipeline-level mean answer length
    pipe_lens = {
        p: float(np.mean(d["answer_lengths"]))
        for p, d in cell.pipelines.items()
    }
    pipe_order = sorted(pipe_lens.keys())
    lens_vec = np.array([pipe_lens[p] for p in pipe_order])

    # find metric with largest |Kendall tau| to length
    worst_metric = None
    worst_abs_tau = -np.inf
    for m in metric_names:
        scores = []
        for p in pipe_order:
            scores.append(_pipeline_metric_mean(cell, p, m))
        scores = np.array(scores, dtype=float)
        if np.isnan(scores).all():
            continue
        valid = ~np.isnan(scores)
        if valid.sum() < 3:
            continue
        try:
            tau, _ = kendalltau(lens_vec[valid], scores[valid])
        except Exception:
            continue
        if not _is_nan(tau) and abs(tau) > worst_abs_tau:
            worst_abs_tau = abs(tau)
            worst_metric = m
    return [m for m in metric_names if m != worst_metric]


# -----------------------------------------------------------------------------
# Length filter (self-constructed baseline; threshold 0.3)
# -----------------------------------------------------------------------------

def length_filter(cell: CellData,
                  metric_names: Sequence[str],
                  tau_threshold: float = 0.3) -> List[str]:
    """Drop metrics with |pipeline-level Kendall tau to length| > threshold.

    Self-constructed baseline that operationalizes the principle of dropping
    length-correlated metrics. We use this as a comparison point in the human
    evaluation (Section 5.7); it is NOT attributed to prior work.
    """
    pipe_lens = {
        p: float(np.mean(d["answer_lengths"]))
        for p, d in cell.pipelines.items()
    }
    pipe_order = sorted(pipe_lens.keys())
    lens_vec = np.array([pipe_lens[p] for p in pipe_order])

    kept: List[str] = []
    for m in metric_names:
        scores = np.array(
            [_pipeline_metric_mean(cell, p, m) for p in pipe_order],
            dtype=float,
        )
        valid = ~np.isnan(scores)
        if valid.sum() < 3:
            kept.append(m)  # not enough data to judge — keep
            continue
        try:
            tau, _ = kendalltau(lens_vec[valid], scores[valid])
        except Exception:
            kept.append(m)
            continue
        if _is_nan(tau) or abs(tau) <= tau_threshold:
            kept.append(m)
    return kept if kept else list(metric_names)


# -----------------------------------------------------------------------------
# PMA — Pairwise Metric Agreement (self-constructed baseline)
# -----------------------------------------------------------------------------

def pma_filter(cell: CellData,
               metric_names: Sequence[str],
               agreement_quantile: float = 0.5) -> List[str]:
    """Pairwise Metric Agreement: keep metrics whose pipeline rankings agree
    with most other metrics.

    Self-constructed baseline. For each metric, compute Kendall tau to every
    other metric's pipeline ranking; the metric's "agreement score" is the
    mean of those tau values. We then keep the metrics with above-median
    agreement scores (default: top half).

    This is the natural strawman for any consistency-based filtering idea.
    The paper shows it fails on cells where bias is shared across metrics,
    because the biased cluster agrees with itself.
    """
    pipe_order = sorted(cell.pipelines.keys())
    metric_vecs: Dict[str, np.ndarray] = {}
    for m in metric_names:
        scores = np.array(
            [_pipeline_metric_mean(cell, p, m) for p in pipe_order],
            dtype=float,
        )
        if not np.isnan(scores).all():
            metric_vecs[m] = scores

    if len(metric_vecs) < 2:
        return list(metric_vecs.keys())

    # pairwise kendall tau
    metrics = list(metric_vecs.keys())
    agreement: Dict[str, float] = {}
    for i, m_i in enumerate(metrics):
        taus = []
        for j, m_j in enumerate(metrics):
            if i == j:
                continue
            v_i, v_j = metric_vecs[m_i], metric_vecs[m_j]
            valid = ~(np.isnan(v_i) | np.isnan(v_j))
            if valid.sum() < 3:
                continue
            try:
                tau, _ = kendalltau(v_i[valid], v_j[valid])
            except Exception:
                continue
            if not _is_nan(tau):
                taus.append(tau)
        agreement[m_i] = float(np.mean(taus)) if taus else 0.0

    # keep top half by agreement
    cutoff = float(np.quantile(list(agreement.values()), agreement_quantile))
    kept = [m for m, a in agreement.items() if a >= cutoff]
    return kept if kept else metrics
