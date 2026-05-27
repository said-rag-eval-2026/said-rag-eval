"""baselines/supervised.py — Supervised oracle baselines.

These baselines USE gold-judge labels and serve as upper bounds in Table 1.

  - Best fixed subset: choose ONE subset of metrics globally (across all
    cells) that maximizes mean Kendall tau to the gold-judge ranking. Uses
    gold labels for selection. NOT a deployable method.

  - Ridge LODO: leave-one-dataset-out non-negative ridge regression of metric
    means against gold-judge scores. Trained per held-out dataset; uses gold
    labels.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau

from said.algorithm import (
    CellData,
    _pipeline_metric_mean,
    aggregate_pipeline_scores,
    gold_judge_pipeline_scores,
)


# -----------------------------------------------------------------------------
# Best fixed subset (oracle, global)
# -----------------------------------------------------------------------------

def find_best_fixed_subset(cells: List[CellData],
                           metric_names: Sequence[str],
                           min_size: int = 1,
                           max_size: int = None) -> Tuple[List[str], float]:
    """Search over all metric subsets and return the one with highest mean
    Kendall tau to the gold-judge across cells.

    Note: this is exponential in len(metric_names). For 10 metrics, 2^10 - 1
    = 1023 subsets — tractable.

    Returns: (best_subset, mean_tau)
    """
    if max_size is None:
        max_size = len(metric_names)

    best_subset: List[str] = list(metric_names)
    best_mean_tau = -np.inf

    for size in range(min_size, max_size + 1):
        for subset in combinations(metric_names, size):
            taus = []
            for cell in cells:
                t = _kendall_to_gold(cell, list(subset))
                if not np.isnan(t):
                    taus.append(t)
            if not taus:
                continue
            mean_tau = float(np.mean(taus))
            if mean_tau > best_mean_tau:
                best_mean_tau = mean_tau
                best_subset = list(subset)

    return best_subset, best_mean_tau


def _kendall_to_gold(cell: CellData, metrics: Sequence[str]) -> float:
    """Kendall tau between aggregated-metric ranking and gold-judge ranking
    on one cell."""
    if not metrics:
        return float("nan")
    agg = aggregate_pipeline_scores(cell, metrics)
    gold = gold_judge_pipeline_scores(cell)
    pipes = sorted(set(agg.keys()) & set(gold.keys()))
    a = np.array([agg[p] for p in pipes], dtype=float)
    g = np.array([gold[p] for p in pipes], dtype=float)
    valid = ~(np.isnan(a) | np.isnan(g))
    if valid.sum() < 3:
        return float("nan")
    tau, _ = kendalltau(a[valid], g[valid])
    return float(tau) if not np.isnan(tau) else float("nan")


# -----------------------------------------------------------------------------
# Ridge LODO (oracle, leave-one-dataset-out)
# -----------------------------------------------------------------------------

def ridge_lodo_predict(train_cells: List[CellData],
                       test_cell: CellData,
                       metric_names: Sequence[str],
                       alpha: float = 1.0) -> Dict[str, float]:
    """Fit non-negative ridge of pipeline-level metric means against gold-judge
    on train cells, then predict pipeline scores on the test cell.

    For each cell, we form (n_pipelines, n_metrics) feature matrix X and
    (n_pipelines,) target y = gold-judge mean. Stack across train cells and
    fit one ridge.

    Returns: {pipeline_name: predicted_score}
    """
    metrics = list(metric_names)

    # build training data
    X_train, y_train = [], []
    for cell in train_cells:
        gold = gold_judge_pipeline_scores(cell)
        pipes = sorted(cell.pipelines.keys())
        for p in pipes:
            row = [_pipeline_metric_mean(cell, p, m) for m in metrics]
            y = gold[p]
            if any(np.isnan(v) for v in row) or np.isnan(y):
                continue
            X_train.append(row)
            y_train.append(y)

    if not X_train:
        # fallback: uniform
        return aggregate_pipeline_scores(test_cell, metrics)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    # Non-negative ridge: solve min ||y - Xw||^2 + alpha*||w||^2 s.t. w >= 0
    # Use scipy.optimize.nnls on the augmented system (closed-form ridge -> NNLS)
    from scipy.optimize import nnls
    A = np.vstack([X_train, np.sqrt(alpha) * np.eye(X_train.shape[1])])
    b = np.concatenate([y_train, np.zeros(X_train.shape[1])])
    w, _ = nnls(A, b)

    # predict on test cell
    pipes = sorted(test_cell.pipelines.keys())
    out = {}
    for p in pipes:
        row = np.array([_pipeline_metric_mean(test_cell, p, m) for m in metrics],
                       dtype=float)
        if np.isnan(row).any():
            out[p] = float("nan")
        else:
            out[p] = float(row @ w)
    return out


def ridge_lodo_pipeline_scores(all_cells: List[CellData],
                               metric_names: Sequence[str],
                               alpha: float = 1.0
                               ) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """Run leave-one-dataset-out Ridge across all cells.

    Returns: {(dataset, generator, judge): {pipeline: predicted_score}}
    """
    out = {}
    datasets = sorted({c.dataset for c in all_cells})
    for held_out in datasets:
        train = [c for c in all_cells if c.dataset != held_out]
        test = [c for c in all_cells if c.dataset == held_out]
        for tc in test:
            preds = ridge_lodo_predict(train, tc, metric_names, alpha=alpha)
            out[(tc.dataset, tc.generator, tc.judge)] = preds
    return out
