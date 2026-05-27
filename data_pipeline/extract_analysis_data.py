#!/usr/bin/env python3
"""
extract_analysis_data.py — Compact extraction of Stage B outputs.

Stage B produces per-cell exp_v6_*.json files containing full raw answers
plus all metric scores. For analysis purposes (and for the released
metric_scores_compact.json artifact), we extract only the per-pipeline
per-metric score arrays plus answer-length statistics — no raw answer text.

This is the format that drives all paper analyses (Tables 1-4) and that
ships as v1.0 of the released benchmark.

Output: ./cache_v2/analyze_raw_compact.json (and .json.gz)

  Structure:
  {
    "metadata": {n_cells, datasets, generators, judges, metric_names, ...},
    "cells": [
      {
        "dataset": ..., "generator": ..., "judge": ...,
        "n_answers_total": int,
        "pipelines": {
          "pipeline_name": {
            "n_samples": int,
            "answer_length_stats": {mean, std, min, max, p50, p95},
            "answer_lengths": [int, ...],
            "metric_scores": {metric_name: [float | null, ...], ...}
          },
          ...
        }
      },
      ...
    ]
  }
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import json
import gzip
import numpy as np
from pathlib import Path

CACHE_DIR = Path(os.environ.get("SAID_CACHE_DIR", "./cache_v2"))

DATASETS   = ["HotpotQA", "MSMARCO", "WikiQA", "PubMedQA", "FinQA"]
GENERATORS = ["claude-sonnet-4-6", "gpt-5", "gemini-2.5-pro",
              "Llama-3.1-8B-Instruct", "Qwen3-8B"]
JUDGES     = ["claude-sonnet-4-6", "gpt-5", "gemini-2.5-pro"]

METRIC_NAMES = [
    "faithfulness", "hallucination_free",
    "answer_relevancy", "context_precision", "context_utilization",
    "completeness", "conciseness", "coherence",
    "specificity", "citation_quality",
    "gt_judge",
]

def safe_name(m): return m.replace('.', '-').replace('/', '_')


def extract_cell(dataset, generator, judge):
    """
    1 cell (ds, gen, judge) → compact dict.
    Per-pipeline per-metric score arrays (100 per pipeline × 32 pipelines).
    """
    fname = f"exp_v6_{dataset}_{safe_name(generator)}_{safe_name(judge)}.json"
    fp = CACHE_DIR / fname
    if not fp.exists():
        return None
    
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [skip] {fname}: {e}")
        return None
    
    answers = data.get('answers_with_scores', [])
    if not answers:
        return None
    
    # Group by pipeline
    pipe_data = {}
    for rec in answers:
        pipe = rec.get('pipeline')
        if not pipe: continue
        
        ans_text = rec.get('answer', '') or ''
        if not ans_text.strip():
            continue
        
        ms = rec.get('metric_scores', {}) or {}
        
        if pipe not in pipe_data:
            pipe_data[pipe] = {
                'answer_lengths': [],
                'scores': {m: [] for m in METRIC_NAMES},
            }
        
        pipe_data[pipe]['answer_lengths'].append(len(ans_text))
        for m in METRIC_NAMES:
            v = ms.get(m)
            pipe_data[pipe]['scores'][m].append(
                float(v) if isinstance(v, (int, float)) and not np.isnan(v) else None
            )
    
    # Compact form — length stats + per-sample score arrays
    pipelines_compact = {}
    for pipe, pd in pipe_data.items():
        lens = pd['answer_lengths']
        if not lens: continue
        pipelines_compact[pipe] = {
            'n_samples': len(lens),
            'answer_length_stats': {
                'mean': float(np.mean(lens)),
                'std':  float(np.std(lens)),
                'min':  int(min(lens)),
                'max':  int(max(lens)),
                'p50':  float(np.percentile(lens, 50)),
                'p95':  float(np.percentile(lens, 95)),
            },
            'answer_lengths': lens,  # per-sample
            'metric_scores': pd['scores'],  # per-sample
        }
    
    return {
        'dataset': dataset,
        'generator': generator,
        'judge': judge,
        'n_answers_total': len(answers),
        'pipelines': pipelines_compact,
    }


def main():
    print("="*70)
    print("Raw Data Compact Extraction")
    print("="*70)
    
    cells = []
    missing = []
    for ds in DATASETS:
        for gen in GENERATORS:
            for judge in JUDGES:
                cell = extract_cell(ds, gen, judge)
                if cell is None:
                    missing.append((ds, gen, judge))
                    continue
                cells.append(cell)
                print(f"  ✓ {ds} × {gen} × {judge}: {len(cell['pipelines'])} pipes")
    
    if missing:
        print(f"\n[WARN] {len(missing)} cells missing")
    
    output = {
        'metadata': {
            'n_cells': len(cells),
            'n_expected': len(DATASETS) * len(GENERATORS) * len(JUDGES),
            'datasets': DATASETS,
            'generators': GENERATORS,
            'judges': JUDGES,
            'metric_names': METRIC_NAMES,
            'missing_cells': missing,
        },
        'cells': cells,
    }
    
    # Save as gzipped JSON (smaller)
    out_path_gz = CACHE_DIR / 'analyze_raw_compact.json.gz'
    with gzip.open(out_path_gz, 'wt', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)
    size_mb = out_path_gz.stat().st_size / 1024 / 1024
    print(f"\n✓ Saved (gzipped): {out_path_gz}")
    print(f"  Size: {size_mb:.1f} MB")
    
    out_path = CACHE_DIR / 'analyze_raw_compact.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)
    size_mb_plain = out_path.stat().st_size / 1024 / 1024
    print(f"\n✓ Saved (plain): {out_path}")
    print(f"  Size: {size_mb_plain:.1f} MB")
    
    print(f"\n{'='*70}")
    print("Claude[...] :")
    if size_mb < 20:
        print(f"  {out_path_gz.name} ({size_mb:.1f} MB) ← [...] ")
    else:
        print(f"  [...] {size_mb:.1f}MB. [...] :")
        print(f"    - specific dataset only: python extract_analysis_data.py --datasets HotpotQA")
        print(f"    - or sample[...] : head -N per pipeline (script [...] )")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
