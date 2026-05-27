#!/usr/bin/env python3
"""
gt_judge_only.py — Add gt_judge (gold-judge oracle) scores to existing
                    Stage B output files.

If Stage B was run without computing gt_judge (the gold-reference oracle),
this script reads the per-cell exp_v6_*.json files and adds gt_judge to
each (question, pipeline) record using the same judge LLM as the cell.

Output: each exp_v6_{ds}_{gen}_{judge}.json file is updated in place to
include "gt_judge" in metric_scores and metric_names.

Usage:
  python gt_judge_only.py                        # all cells
  python gt_judge_only.py --datasets HotpotQA    # subset
  python gt_judge_only.py --overwrite            # recompute even if present
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import re
import json
import time
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))
from rescore_v6 import (
    _AsyncClientPool, _load_api_keys, llm, embed,
    CACHE_DIR, CONCURRENT_PER_KEY, APIKEY_FILE_DEFAULT,
    GENERATORS, JUDGES, DATASETS, MODEL_PROVIDER, safe_name,
)
import rescore_v6
import metric_functions

OPENSOURCE_GENERATORS = ["Llama-3.1-8B-Instruct", "Qwen3-8B"]
ALL_GENERATORS = GENERATORS + OPENSOURCE_GENERATORS


# ═══════════════════════ GT JUDGE ONE ANSWER ═══════════════════════

async def gt_score_one(answer_record: Dict) -> float:
    """
    [...] answer[...] gt_judge [...] .
    EVAL_MODEL([...] global)[...] judge[...] .
    """
    q = answer_record.get('question', '')
    a = answer_record.get('answer', '')
    gold = answer_record.get('ground_truth', '') or answer_record.get('gold', '')
    
    if not gold or not str(gold).strip():
        return None
    if not a or not str(a).strip():
        return None
    
    try:
        return await metric_functions.gt_judge(q, a, gold, t=0.0)
    except Exception as e:
        print(f"    [gt_judge] error: {type(e).__name__}: {str(e)[:80]}")
        return 0.5


# ═══════════════════════ PROCESS ONE FILE ═══════════════════════

async def process_stage_b_file(file_path: Path, overwrite: bool = False) -> bool:
    """
    exp_v6_{ds}_{gen}_{judge}.json [...] gt_judge [...] .
    Judge[...] .
    """
    # Parse filename: exp_v6_HotpotQA_claude-sonnet-4-6_gpt-5-4.json
    # Format: exp_v6_{dataset}_{generator}_{judge}.json
    name = file_path.stem  # remove .json
    parts = name.split('_')
    if len(parts) < 4 or parts[0] != 'exp' or parts[1] != 'v6':
        print(f"  [skip] [...] : {file_path.name}")
        return False
    
    # parts = ['exp', 'v6', 'HotpotQA', 'claude-sonnet-4-6', 'gpt-5-4']
    # or    ['exp', 'v6', 'HotpotQA', 'gpt-5-4', 'claude-sonnet-4-6']
    
    ds = None
    rest = parts[2:]
    for candidate in DATASETS:
        if rest[0] == candidate:
            ds = candidate
            rest = rest[1:]
            break
    
    if ds is None or len(rest) < 2:
        print(f"  [skip] dataset/model [...] : {file_path.name}")
        return False
    
    gen_part = rest[0]
    judge_part = '_'.join(rest[1:])
    
    def find_original(safe_str, candidates):
        for c in candidates:
            if safe_name(c) == safe_str:
                return c
        return None
    
    gen_model = find_original(gen_part, ALL_GENERATORS)
    judge_model = find_original(judge_part, JUDGES)
    
    if not gen_model or not judge_model:
        print(f"  [skip] model not in GENERATORS/JUDGES: gen={gen_part}, judge={judge_part}")
        return False
    
    # Load existing file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [error] {file_path.name} read [...] : {e}")
        return False
    
    answers = data.get('answers_with_scores', [])
    if not answers:
        print(f"  [skip] {file_path.name}: answers_with_scores [...] ")
        return False
    
    # Check if gt_judge already present
    sample_scores = answers[0].get('metric_scores', {})
    has_gt = 'gt_judge' in sample_scores
    if has_gt and not overwrite:
        print(f"  [skip] {file_path.name}: gt_judge [...] (--overwrite[...] )")
        return True
    
    # Set global EVAL_MODEL
    rescore_v6.EVAL_MODEL = judge_model
    provider = MODEL_PROVIDER.get(judge_model, 'openai')
    
    # Filter valid answers
    valid_indices = [i for i, a in enumerate(answers) 
                     if a.get('answer') and a.get('ground_truth') 
                     and not a.get('failed')]
    
    t0 = time.time()
    print(f"  [{file_path.name}] judge={judge_model} via {provider}, "
          f"valid={len(valid_indices)}/{len(answers)}")
    
    # Run all gt_judge tasks in full parallel (semaphore controls actual concurrency)
    all_tasks = [gt_score_one(answers[i]) for i in valid_indices]
    all_results = await asyncio.gather(*all_tasks, return_exceptions=True)

    for idx, score in zip(valid_indices, all_results):
        if isinstance(score, Exception):
            print(f"    err @ idx {idx}: {type(score).__name__}")
            score = 0.5
        ms = answers[idx].setdefault('metric_scores', {})
        ms['gt_judge'] = score

    elapsed = time.time() - t0
    rate = len(valid_indices) / elapsed if elapsed > 0 else 0
    print(f"    {len(valid_indices)}/{len(valid_indices)} ({rate:.1f}/s, {elapsed:.0f}s elapsed)")
    
    # Update metric_names in data
    mn = data.get('metric_names', metric_functions.METRIC_NAMES)
    if 'gt_judge' not in mn:
        data['metric_names'] = mn + ['gt_judge']
    data['answers_with_scores'] = answers
    
    # Save back
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - t0
    print(f"  [{file_path.name}] ✓ gt_judge added ({elapsed:.0f}s)")
    return True


# ═══════════════════════ MAIN ═══════════════════════

async def main_async(args):
    keys = _load_api_keys(Path(args.apikey))
    print(f"[ClientPool] Loaded {len(keys)} keys × {CONCURRENT_PER_KEY} slots "
          f"= {len(keys)*CONCURRENT_PER_KEY} concurrent")
    
    rescore_v6._client_pool = _AsyncClientPool(keys, concurrent=CONCURRENT_PER_KEY)
    metric_functions.set_llm_functions(llm, embed)
    
    print("="*70)
    print("gt_judge only — Stage B [...] gt_judge[...] ")
    print("="*70)
    print(f"  Datasets: {args.datasets}")
    print(f"  Overwrite: {args.overwrite}")
    print("="*70)
    
    # Find all Stage B files matching datasets
    files_to_process = []
    for ds in args.datasets:
        for gen in args.generators:
            for judge in JUDGES:
                fname = f"exp_v6_{ds}_{safe_name(gen)}_{safe_name(judge)}.json"
                fp = CACHE_DIR / fname
                if fp.exists():
                    files_to_process.append(fp)
    
    print(f"\n{len(files_to_process)} files to process")
    
    if not files_to_process:
        print("No Stage B files found. Run rescore_v6.py first.")
        return
    
    t_total = time.time()
    success_count = 0
    for i, fp in enumerate(files_to_process):
        print(f"\n[{i+1}/{len(files_to_process)}]")
        try:
            ok = await process_stage_b_file(fp, overwrite=args.overwrite)
            if ok:
                success_count += 1
        except Exception as e:
            import traceback
            print(f"  [error] {fp.name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"ALL DONE — {success_count}/{len(files_to_process)} files updated in {(time.time()-t_total)/60:.1f}min")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--generators', nargs='+', default=ALL_GENERATORS,
                        help="[...] generator [...] ([...] : all)")
    parser.add_argument('--overwrite', action='store_true',
                        help="[...] gt_judge [...] ")
    parser.add_argument('--apikey', default=str(APIKEY_FILE_DEFAULT))
    args = parser.parse_args()

    invalid = [d for d in args.datasets if d not in DATASETS]
    if invalid:
        print(f"Invalid datasets: {invalid}"); sys.exit(1)
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
