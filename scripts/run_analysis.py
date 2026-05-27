#!/usr/bin/env python3
"""
run_analysis.py — Reproduce paper Tables 1-4 from a metric-scores file.

Usage:
  python scripts/run_analysis.py --input metric_scores_compact.json
  python scripts/run_analysis.py --input metric_scores_compact.json --table 1
  python scripts/run_analysis.py --input metric_scores_compact.json --skip-supervised

The --skip-supervised flag skips the slowest computations (Ridge LODO and
exhaustive Best Fixed Subset search), which together take ~10 minutes on
75 cells. Without it, the full Table 1 takes ~12 minutes total.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the parent directory importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from said.analysis import (
    METRIC_NAMES_DEFAULT,
    filter_main_pool,
    load_compact,
    method_tau_per_cell,
    paired_bootstrap_ci,
    table_2,
    table_3,
    table_4,
)


def fast_table_1(cells, metric_names, skip_supervised: bool = False):
    """Run Table 1 with optional skip of supervised oracles."""
    methods = ["uniform", "drop_conciseness", "pma", "length_filter", "said"]
    if not skip_supervised:
        methods += ["best_fixed", "ridge_lodo"]

    # Pre-compute supervised oracles only if needed
    best_fixed = None
    ridge_preds = None
    if not skip_supervised:
        from baselines.supervised import (find_best_fixed_subset,
                                           ridge_lodo_pipeline_scores)
        print("[oracle] Searching best fixed subset (this may take several minutes)...")
        t0 = time.time()
        best_fixed, _ = find_best_fixed_subset(cells, metric_names)
        print(f"  -> {best_fixed}  ({time.time()-t0:.0f}s)")
        print("[oracle] Computing Ridge LODO predictions...")
        t0 = time.time()
        ridge_preds = ridge_lodo_pipeline_scores(cells, metric_names)
        print(f"  -> Ridge LODO done ({time.time()-t0:.0f}s)")

    tau_matrix = {}
    for method in methods:
        taus = []
        for cell in cells:
            t = method_tau_per_cell(
                cell, method, metric_names,
                ridge_predictions=ridge_preds,
                best_fixed=best_fixed,
            )
            taus.append(t)
        tau_matrix[method] = np.array(taus, dtype=float)

    return tau_matrix, best_fixed


def print_table_1(tau_matrix, best_fixed):
    uniform = tau_matrix["uniform"]
    print()
    print("=" * 78)
    print("Table 1 — Method comparison vs Uniform aggregation (RAGAS-style)")
    print("=" * 78)
    if best_fixed is not None:
        print(f"Best fixed subset (oracle): {best_fixed}")
    print()
    print(f"{'Method':<26} {'Mean Δτ':>10} {'95% CI':>22} {'Wins':>10}")
    print("-" * 78)
    for method in tau_matrix:
        if method == "uniform":
            print(f"{'Uniform (RAGAS-style)':<26} {0.000:>+10.3f} {'—':>22} {'—':>10}")
            continue
        m_t = tau_matrix[method]
        valid = ~(np.isnan(m_t) | np.isnan(uniform))
        deltas = m_t[valid] - uniform[valid]
        mean_d = float(np.mean(deltas))
        lo, hi = paired_bootstrap_ci(deltas)
        wins = int(np.sum(deltas > 0))
        ci_str = f"[{lo:+.3f}, {hi:+.3f}]"
        n = int(valid.sum())
        print(f"{method:<26} {mean_d:>+10.3f} {ci_str:>22} {wins:>5}/{n:<3}")


def print_table_2(t2):
    print()
    print("=" * 78)
    print("Table 2 — Per-generator mean Δτ over Uniform")
    print("=" * 78)
    gens = t2["generators"]
    header = f"{'Method':<22}" + "".join(f"{g[:14]:>15}" for g in gens)
    print(header)
    print("-" * len(header))
    for row in t2["rows"]:
        line = f"{row['method']:<22}"
        for g in gens:
            v = row.get(g, float("nan"))
            line += f"{v:>+15.3f}"
        print(line)


def print_table_3(t3):
    print()
    print("=" * 78)
    print("Table 3 — Per-dataset mean Δτ over Uniform")
    print("=" * 78)
    dss = t3["datasets"]
    header = f"{'Method':<22}" + "".join(f"{d:>14}" for d in dss)
    print(header)
    print("-" * len(header))
    for row in t3["rows"]:
        line = f"{row['method']:<22}"
        for d in dss:
            v = row.get(d, float("nan"))
            line += f"{v:>+14.3f}"
        print(line)


def print_table_4(t4):
    print()
    print("=" * 78)
    print("Table 4 — SAID ablations")
    print("=" * 78)
    print(f"{'Configuration':<35} {'Mean Δτ':>10} {'Wins':>10}")
    print("-" * 60)
    for row in t4["rows"]:
        print(f"{row['config']:<35} {row['mean_delta_tau']:>+10.3f} "
              f"{row['wins']:>5}/{row['n_cells']:<3}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="path to metric_scores_compact.json (download from HF)")
    p.add_argument("--table", default="all", choices=["all", "1", "2", "3", "4"],
                   help="which table(s) to print")
    p.add_argument("--skip-supervised", action="store_true",
                   help="skip best-fixed-subset and Ridge LODO oracles")
    p.add_argument("--save-json", default=None,
                   help="optional path to dump all results as JSON")
    args = p.parse_args()

    print(f"[load] {args.input}")
    cells = load_compact(args.input)
    cells = filter_main_pool(cells)
    print(f"[load] {len(cells)} cells in main pool")

    out = {"n_cells": len(cells)}

    if args.table in ("all", "1"):
        tau_matrix, best_fixed = fast_table_1(cells, METRIC_NAMES_DEFAULT,
                                              skip_supervised=args.skip_supervised)
        print_table_1(tau_matrix, best_fixed)
        out["table_1"] = {
            "best_fixed_subset": best_fixed,
            "tau_matrix": {m: t.tolist() for m, t in tau_matrix.items()},
        }
    if args.table in ("all", "2"):
        t2 = table_2(cells)
        print_table_2(t2)
        out["table_2"] = t2
    if args.table in ("all", "3"):
        t3 = table_3(cells)
        print_table_3(t3)
        out["table_3"] = t3
    if args.table in ("all", "4"):
        t4 = table_4(cells)
        print_table_4(t4)
        out["table_4"] = t4

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n[save] results written to {args.save_json}")


if __name__ == "__main__":
    main()
