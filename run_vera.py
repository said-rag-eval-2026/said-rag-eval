#!/usr/bin/env python3
"""
run_vera.py — Apply VERA (Ding et al. 2024, arXiv 2409.03759) to existing
              metric-scored data.

VERA method (Section 3 of the VERA paper):
  1. For each (question, answer, metric_scores) record:
     - Build an "enriched answer" by appending each metric's textual
       description and numeric value to the original answer.
  2. Feed (question, enriched_answer) to a pretrained cross-encoder
     (cross-encoder/ms-marco-MiniLM-L-12-v2, MS MARCO-pretrained).
  3. The cross-encoder logit is the VERA score for that record.
  4. Aggregate to pipeline level by mean logit.

Input:
  - cache_v2/exp_v6_{dataset}_{gen}_{judge}.json
    (output of rescore_v6.py + gt_judge_only.py)

Output:
  - vera_output_{dataset}_{gen}_{judge}.json

Usage:
  pip install sentence-transformers torch
  python run_vera.py --input-dir ./cache_v2 --output-dir ./vera_out
  # specific dataset:
  python run_vera.py --input-dir ./cache_v2 --output-dir ./vera_out --dataset HotpotQA
  # GPU auto-detected; --device cpu to force CPU
"""
import sys
import os
import json
import argparse
import time
import re
from pathlib import Path
from typing import List, Dict

import numpy as np

try:
    from sentence_transformers import CrossEncoder
    import torch
except ImportError:
    print("ERROR: Install dependencies first:")
    print("  pip install sentence-transformers torch numpy")
    sys.exit(1)


# VERA metric descriptions (Section 3.2 Text Enhancement, in the VERA paper).
# Wording follows the format in Ding et al. 2024 Appendix
# "Enhanced Document Context".
METRIC_DESCRIPTIONS = {
    'faithfulness': (
        "Faithfulness measures the factual consistency of the generated "
        "answer against the given context. It is considered faithful if all "
        "the claims that are made in the answer can be inferred from the "
        "given context. It is measured between 0 and 1; where a lower score "
        "is given to answers consisting of claims that are not in the "
        "context; and a higher score indicates that the answer is using "
        "information from the contexts."
    ),
    'answer_relevancy': (
        "Answer Relevancy assesses how pertinent the actual answer is to the "
        "given question. It is measured between 0 and 1; where a lower score "
        "is given to answers that are incomplete or contain redundant "
        "information; and a higher score indicates better relevancy."
    ),
    'context_precision': (
        "Context Precision assesses how relevant is every context towards "
        "answering the question. Ideally all of the text in all of the "
        "contexts should be relevant to the question. It is measured between "
        "0 and 1; where a lower score is given to lower precision contexts; "
        "and a higher score indicates more precision."
    ),
    'context_utilization': (
        "Context Utilization measures the extent to which the generated "
        "answer draws on the retrieved context. It is measured between 0 "
        "and 1; where a lower score indicates the answer ignored the "
        "provided context, and a higher score indicates effective use of "
        "the context information."
    ),
    'hallucination_free': (
        "Hallucination-free measures whether the answer avoids generating "
        "information not present in or supported by the given context. It "
        "is measured between 0 and 1; where a lower score indicates more "
        "fabricated content, and a higher score indicates the answer "
        "remains grounded in the provided context."
    ),
    'completeness': (
        "Completeness measures whether the answer covers all aspects of "
        "the question. It is measured between 0 and 1; where a lower score "
        "indicates missing information, and a higher score indicates the "
        "answer addresses the full question."
    ),
    'conciseness': (
        "Conciseness measures whether the answer is direct and free of "
        "unnecessary elaboration. It is measured between 0 and 1; where a "
        "lower score indicates verbosity or redundancy, and a higher score "
        "indicates an appropriately brief answer."
    ),
    'coherence': (
        "Coherence measures the logical flow and readability of the "
        "answer. It is measured between 0 and 1; where a lower score "
        "indicates fragmented or contradictory content, and a higher score "
        "indicates a well-structured, logical answer."
    ),
    'specificity': (
        "Specificity measures the level of detail in the answer. It is "
        "measured between 0 and 1; where a lower score indicates a vague "
        "or generic answer, and a higher score indicates a precise, "
        "detail-rich answer."
    ),
    'citation_quality': (
        "Citation Quality measures whether the answer properly references "
        "or grounds claims in the context. It is measured between 0 and 1; "
        "where a lower score indicates unsupported claims, and a higher "
        "score indicates well-grounded, verifiable assertions."
    ),
}


def build_enriched_answer(answer: str, metric_scores: Dict[str, float]) -> str:
    """VERA Text Enhancement (Section 3.2). Append each metric's description
    and numeric score to the answer."""
    parts = [answer or ""]
    for metric_name, description in METRIC_DESCRIPTIONS.items():
        score = metric_scores.get(metric_name)
        if score is None:
            continue
        parts.append(
            f"\n\n{description} For the given question, context and answer, "
            f"the {metric_name.replace('_', ' ')} score is: {float(score):.4f}."
        )
    return " ".join(parts)


def process_file(in_path: Path, out_path: Path, ce: CrossEncoder,
                 batch_size: int = 32, max_length: int = 512,
                 overwrite: bool = False) -> Dict:
    """Read one exp_v6_*.json file, compute VERA logits per answer,
    aggregate to pipeline level, and save the result."""
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path.name} already exists")
        with open(out_path) as f:
            return json.load(f)

    with open(in_path, encoding='utf-8') as f:
        data = json.load(f)

    answers = data.get('answers_with_scores', [])
    if not answers:
        print(f"  [skip] {in_path.name}: no answers")
        return None

    # Build (question, enriched_answer) pairs
    pairs = []
    idx_map = []
    for rec in answers:
        q = rec.get('question', '')
        a = rec.get('answer', '')
        ms = rec.get('metric_scores', {}) or {}
        if not q or not a or rec.get('failed'):
            continue
        enriched = build_enriched_answer(a, ms)
        pairs.append((q, enriched))
        idx_map.append((rec.get('pipeline', 'unknown'),
                        rec.get('sample_id', '?')))

    if not pairs:
        print(f"  [skip] {in_path.name}: no valid pairs")
        return None

    print(f"  [{in_path.name}] {len(pairs)} (q, enriched_a) pairs")
    t0 = time.time()

    scores = ce.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"  [{in_path.name}] {len(pairs)} VERA scores in {elapsed:.1f}s "
          f"({len(pairs)/elapsed:.0f}/s)")

    # Aggregate per pipeline
    from collections import defaultdict
    pipe_scores = defaultdict(list)
    per_sample = []
    for (pipe, sid), score in zip(idx_map, scores):
        sc = float(score)
        pipe_scores[pipe].append(sc)
        per_sample.append({
            'pipeline': pipe,
            'sample_id': sid,
            'vera_logit': sc,
        })

    pipeline_mean = {p: float(np.mean(ss)) for p, ss in pipe_scores.items()}
    pipeline_stats = {
        p: {
            'mean': float(np.mean(ss)),
            'std': float(np.std(ss)),
            'n': len(ss),
        } for p, ss in pipe_scores.items()
    }

    result = {
        'dataset': data.get('dataset'),
        'generator': data.get('generator'),
        'judge': data.get('judge'),
        'n_samples': len(per_sample),
        'cross_encoder_model': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'pipeline_vera_mean': pipeline_mean,
        'pipeline_vera_stats': pipeline_stats,
        'per_sample': per_sample,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  [{in_path.name}] saved {out_path.name}")
    return result


def parse_filename(fname: str):
    """exp_v6_{dataset}_{gen}_{judge}.json -> (dataset, gen, judge)."""
    stem = fname.replace('.json', '')
    m = re.match(r'exp_v6_([^_]+)_(.+?)_(.+)$', stem)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


def main():
    ap = argparse.ArgumentParser(description="Apply VERA to rescore_v6 outputs")
    ap.add_argument('--input-dir', default='./cache_v2',
                    help='directory with exp_v6_*.json files')
    ap.add_argument('--output-dir', default='./vera_out',
                    help='directory to save VERA results')
    ap.add_argument('--dataset', default=None,
                    help='only process files for this dataset')
    ap.add_argument('--model', default='cross-encoder/ms-marco-MiniLM-L-12-v2',
                    help='cross-encoder model (HuggingFace)')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-length', type=int, default=512)
    ap.add_argument('--device', default='auto', help='cuda, cpu, or auto')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"[VERA] Device: {device}")
    print(f"[VERA] Model:  {args.model}")
    print(f"[VERA] Loading cross-encoder...")
    ce = CrossEncoder(args.model, max_length=args.max_length, device=device)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob('exp_v6_*.json'))
    if args.dataset:
        files = [f for f in files if parse_filename(f.name)[0] == args.dataset]
    print(f"[VERA] Found {len(files)} input files")

    t0_all = time.time()
    for i, in_path in enumerate(files, 1):
        ds, gen, judge = parse_filename(in_path.name)
        if not ds:
            print(f"[{i}/{len(files)}] skip (unparsed): {in_path.name}")
            continue
        print(f"\n[{i}/{len(files)}] {ds} / {gen} / {judge}")
        safe_gen = gen.replace('/', '_')
        safe_judge = judge.replace('/', '_')
        out_path = out_dir / f'vera_output_{ds}_{safe_gen}_{safe_judge}.json'
        try:
            process_file(in_path, out_path, ce,
                         batch_size=args.batch_size,
                         max_length=args.max_length,
                         overwrite=args.overwrite)
        except Exception as e:
            print(f"  ERROR on {in_path.name}: {type(e).__name__}: {e}")
            continue

    total = time.time() - t0_all
    print(f"\n[VERA] All done in {total/60:.1f} min")
    print(f"[VERA] Output -> {out_dir.absolute()}")


if __name__ == '__main__':
    main()
