"""said/analysis.py — Reproduce paper Tables 1-4 from a metric-scores file.

Top-level function:
    run_full_analysis(compact_path) -> dict of tables

Outputs match (up to bootstrap noise) the paper's Tables 1-4. Bootstrap CIs
use the paired bootstrap over cells with 5000 resamples and seed 42.
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, wilcoxon

from said.algorithm import (
    CellData,
    aggregate_pipeline_scores,
    gold_judge_pipeline_scores,
    said_filter,
)
from baselines.supervised import (
    find_best_fixed_subset,
    ridge_lodo_pipeline_scores,
)
from baselines.unsupervised import (
    drop_conciseness_filter,
    length_filter,
    pma_filter,
    uniform_filter,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

METRIC_NAMES_DEFAULT = [
    "faithfulness", "hallucination_free", "answer_relevancy",
    "context_precision", "context_utilization", "completeness",
    "conciseness", "coherence", "specificity", "citation_quality",
]

# The 3 frontier judges used in the main 75-cell pool.
FRONTIER_JUDGES = ["claude-sonnet-4-6", "gpt-5", "gemini-2.5-pro"]
# The 5 generators used in the main 75-cell pool.
MAIN_GENERATORS = [
    "claude-sonnet-4-6", "gpt-5", "gemini-2.5-pro",
    "Llama-3.1-8B-Instruct", "Qwen3-8B",
]
# The 5 datasets used in the main 75-cell pool.
MAIN_DATASETS = ["HotpotQA", "MSMARCO", "WikiQA", "PubMedQA", "FinQA"]

BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_compact(path: str) -> List[CellData]:
    """Load metric_scores_compact.json into a list of CellData."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [CellData.from_compact_dict(c) for c in raw["cells"]]


def filter_main_pool(cells: List[CellData]) -> List[CellData]:
    """Keep only the 75 cells in the paper's main pool (5 datasets x 5
    generators x 3 frontier judges).
    """
    return [
        c for c in cells
        if c.dataset in MAIN_DATASETS
        and c.generator in MAIN_GENERATORS
        and c.judge in FRONTIER_JUDGES
    ]


# -----------------------------------------------------------------------------
# Kendall tau between any pipeline-score dict and the gold-judge ranking
# -----------------------------------------------------------------------------

def kendall_to_gold(cell: CellData,
                    pipeline_scores: Dict[str, float]) -> float:
    """Kendall tau between the given pipeline ranking and the gold-judge."""
    gold = gold_judge_pipeline_scores(cell)
    pipes = sorted(set(pipeline_scores.keys()) & set(gold.keys()))
    a = np.array([pipeline_scores[p] for p in pipes], dtype=float)
    g = np.array([gold[p] for p in pipes], dtype=float)
    valid = ~(np.isnan(a) | np.isnan(g))
    if valid.sum() < 3:
        return float("nan")
    tau, _ = kendalltau(a[valid], g[valid])
    return float(tau) if not np.isnan(tau) else float("nan")


# -----------------------------------------------------------------------------
# Per-method tau computation
# -----------------------------------------------------------------------------

def method_tau_per_cell(cell: CellData,
                        method: str,
                        metric_names: Sequence[str],
                        ridge_predictions: Dict = None,
                        best_fixed: List[str] = None,
                        ) -> float:
    """Compute Kendall tau for one method on one cell."""
    if method == "uniform":
        kept = uniform_filter(cell, metric_names)
        scores = aggregate_pipeline_scores(cell, kept)
    elif method == "drop_conciseness":
        kept = drop_conciseness_filter(cell, metric_names)
        scores = aggregate_pipeline_scores(cell, kept)
    elif method == "pma":
        kept = pma_filter(cell, metric_names)
        scores = aggregate_pipeline_scores(cell, kept)
    elif method == "length_filter":
        kept = length_filter(cell, metric_names)
        scores = aggregate_pipeline_scores(cell, kept)
    elif method == "said":
        result = said_filter(cell, metric_names)
        scores = aggregate_pipeline_scores(cell, result.kept_metrics)
    elif method == "best_fixed":
        if best_fixed is None:
            raise ValueError("best_fixed must be provided for method='best_fixed'")
        scores = aggregate_pipeline_scores(cell, best_fixed)
    elif method == "ridge_lodo":
        if ridge_predictions is None:
            raise ValueError("ridge_predictions must be provided")
        key = (cell.dataset, cell.generator, cell.judge)
        scores = ridge_predictions.get(key, {})
    else:
        raise ValueError(f"Unknown method: {method}")
    return kendall_to_gold(cell, scores)


# -----------------------------------------------------------------------------
# Bootstrap CI
# -----------------------------------------------------------------------------

def paired_bootstrap_ci(deltas: np.ndarray,
                        n_resamples: int = BOOTSTRAP_N,
                        seed: int = BOOTSTRAP_SEED,
                        ci: float = 0.95) -> Tuple[float, float]:
    """Non-parametric paired bootstrap CI over cells."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(deltas[idx]))
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return lo, hi


# -----------------------------------------------------------------------------
# Table 1: main results
# -----------------------------------------------------------------------------

def table_1(cells: List[CellData],
            metric_names: Sequence[str] = METRIC_NAMES_DEFAULT) -> Dict:
    """Compute Table 1 (method comparison)."""
    # Pre-compute supervised oracles
    best_fixed, _ = find_best_fixed_subset(cells, metric_names)
    ridge_preds = ridge_lodo_pipeline_scores(cells, metric_names)

    methods = ["uniform", "drop_conciseness", "pma", "said",
               "best_fixed", "ridge_lodo"]

    # tau per cell per method
    tau_matrix: Dict[str, np.ndarray] = {}
    for method in methods:
        taus = []
        for cell in cells:
            t = method_tau_per_cell(cell, method, metric_names,
                                    ridge_predictions=ridge_preds,
                                    best_fixed=best_fixed)
            taus.append(t)
        tau_matrix[method] = np.array(taus, dtype=float)

    uniform_taus = tau_matrix["uniform"]

    rows = []
    for method in methods:
        method_taus = tau_matrix[method]
        valid = ~(np.isnan(method_taus) | np.isnan(uniform_taus))
        deltas = method_taus[valid] - uniform_taus[valid]
        if method == "uniform":
            mean_d = 0.0
            lo, hi = 0.0, 0.0
        else:
            mean_d = float(np.mean(deltas))
            lo, hi = paired_bootstrap_ci(deltas)
        wins = int(np.sum(deltas > 0))
        rows.append({
            "method": method,
            "mean_delta_tau": mean_d,
            "ci_lo": lo,
            "ci_hi": hi,
            "wins": wins,
            "n_cells": int(valid.sum()),
        })

    return {
        "best_fixed_subset": best_fixed,
        "rows": rows,
        "tau_matrix": {m: t.tolist() for m, t in tau_matrix.items()},
    }


# -----------------------------------------------------------------------------
# Table 2: per-generator
# -----------------------------------------------------------------------------

def table_2(cells: List[CellData],
            metric_names: Sequence[str] = METRIC_NAMES_DEFAULT,
            generators: Sequence[str] = MAIN_GENERATORS) -> Dict:
    ridge_preds = ridge_lodo_pipeline_scores(cells, metric_names)
    methods = ["drop_conciseness", "pma", "said", "ridge_lodo"]

    rows = []
    for method in methods:
        per_gen = {}
        for gen in generators:
            sub = [c for c in cells if c.generator == gen]
            uniform_taus = np.array(
                [method_tau_per_cell(c, "uniform", metric_names) for c in sub])
            method_taus = np.array(
                [method_tau_per_cell(c, method, metric_names,
                                     ridge_predictions=ridge_preds) for c in sub])
            valid = ~(np.isnan(uniform_taus) | np.isnan(method_taus))
            deltas = method_taus[valid] - uniform_taus[valid]
            per_gen[gen] = float(np.mean(deltas)) if deltas.size else float("nan")
        rows.append({"method": method, **per_gen})
    return {"rows": rows, "generators": list(generators)}


# -----------------------------------------------------------------------------
# Table 3: per-dataset
# -----------------------------------------------------------------------------

def table_3(cells: List[CellData],
            metric_names: Sequence[str] = METRIC_NAMES_DEFAULT,
            datasets: Sequence[str] = MAIN_DATASETS) -> Dict:
    methods = ["drop_conciseness", "pma", "said"]
    rows = []
    for method in methods:
        per_ds = {}
        for ds in datasets:
            sub = [c for c in cells if c.dataset == ds]
            uniform_taus = np.array(
                [method_tau_per_cell(c, "uniform", metric_names) for c in sub])
            method_taus = np.array(
                [method_tau_per_cell(c, method, metric_names) for c in sub])
            valid = ~(np.isnan(uniform_taus) | np.isnan(method_taus))
            deltas = method_taus[valid] - uniform_taus[valid]
            per_ds[ds] = float(np.mean(deltas)) if deltas.size else float("nan")
        rows.append({"method": method, **per_ds})
    return {"rows": rows, "datasets": list(datasets)}


# -----------------------------------------------------------------------------
# Table 4: ablations
# -----------------------------------------------------------------------------

def table_4(cells: List[CellData],
            metric_names: Sequence[str] = METRIC_NAMES_DEFAULT) -> Dict:
    """Ablation: full SAID vs SAID without each component."""
    # Full SAID
    full = []
    no_refusal = []
    no_signal_b = []
    no_signal_a = []
    uniform = []

    for cell in cells:
        # uniform
        uniform.append(kendall_to_gold(
            cell, aggregate_pipeline_scores(cell, list(metric_names))))

        # full
        r_full = said_filter(cell, metric_names)
        full.append(kendall_to_gold(
            cell, aggregate_pipeline_scores(cell, r_full.kept_metrics)))

        # without refusal mask
        r_nr = said_filter(cell, metric_names, refusal_len=0,
                            disable_threshold=2.0)
        no_refusal.append(kendall_to_gold(
            cell, aggregate_pipeline_scores(cell, r_nr.kept_metrics)))

        # without Signal B (ignore monotonicity)
        from said.algorithm import signal_a as _sa, _is_nan as _nan
        keep_mask = None
        sa_scores = {m: _sa(cell, m, keep_mask) for m in metric_names}
        kept_no_b = [m for m in metric_names
                     if not _nan(sa_scores[m]) and sa_scores[m] >= 0.85]
        if len(kept_no_b) < 3:
            kept_no_b = sorted(metric_names,
                               key=lambda m: -1 if _nan(sa_scores[m]) else sa_scores[m],
                               reverse=True)[:3]
        no_signal_b.append(kendall_to_gold(
            cell, aggregate_pipeline_scores(cell, kept_no_b)))

        # without Signal A (only Signal B)
        from said.algorithm import signal_b as _sb
        sb_scores = {m: _sb(cell, m, keep_mask) for m in metric_names}
        kept_no_a = [m for m in metric_names if sb_scores[m] == 1]
        if len(kept_no_a) < 3:
            kept_no_a = list(metric_names)[:3]
        no_signal_a.append(kendall_to_gold(
            cell, aggregate_pipeline_scores(cell, kept_no_a)))

    arr = lambda x: np.array(x, dtype=float)
    uniform = arr(uniform)
    rows = []
    for label, vals in [("Full SAID", full),
                        ("without refusal masking", no_refusal),
                        ("without Signal B", no_signal_b),
                        ("without Signal A", no_signal_a)]:
        vals = arr(vals)
        valid = ~(np.isnan(vals) | np.isnan(uniform))
        deltas = vals[valid] - uniform[valid]
        rows.append({
            "config": label,
            "mean_delta_tau": float(np.mean(deltas)),
            "wins": int(np.sum(deltas > 0)),
            "n_cells": int(valid.sum()),
        })
    return {"rows": rows}


# -----------------------------------------------------------------------------
# Top-level
# -----------------------------------------------------------------------------

def run_full_analysis(compact_path: str,
                      restrict_to_main_pool: bool = True) -> Dict:
    cells = load_compact(compact_path)
    if restrict_to_main_pool:
        cells = filter_main_pool(cells)
    return {
        "n_cells": len(cells),
        "table_1": table_1(cells),
        "table_2": table_2(cells),
        "table_3": table_3(cells),
        "table_4": table_4(cells),
    }
