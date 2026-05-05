#!/usr/bin/env python3
"""
pipeline_configs_v2.py — 32 production-realistic RAG pipelines.

Pipelines vary along three axes:
  - retriever: bm25 / dense_bge / dense_openai / hybrid / *_rerank
  - chunk_spec: top1 / top3 / top5 / top-all / shuffled (k=3 or 5)
  - prompt_style: direct / cot / cite / minimal

The 32 pipelines are organized into 6 layers covering the dimensions a
reviewer might question. Two of the 32 are adversarial (shuffled-retrieval),
which SAID uses to probe the order-randomness invariant.

Naming convention: {retriever}_{chunk_spec}_{prompt_style}

Usage:
  # Dry-run (no LLM calls; verify prompt construction):
  python pipeline_configs_v2.py --dry-run HotpotQA
  
  # List all 32 pipelines:
  python pipeline_configs_v2.py --list
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
try:
    from retrieval_v2 import RetrievalCache
except ImportError:
    print("ERROR: retrieval_v2.py not found in same directory")
    sys.exit(1)


CACHE_DIR = Path(os.environ.get("SAID_CACHE_DIR", "./cache_v2"))


# ═══════════════════════ PIPELINE DEFINITIONS ═══════════════════════

# Each pipeline: (name, retriever, chunk_spec, prompt_style)
#   retriever: key in RetrievalCache.rankings, or 'shuffled'/'all'
#   chunk_spec: int (top-k) or 'all' or ('shuffled', k)

PIPELINES = [
    # ═══ Layer 1 — Primary grid (16) ═══
    # BM25
    ("bm25_top3_direct",         "bm25",         3,  "direct"),
    ("bm25_top3_cot",            "bm25",         3,  "cot"),
    ("bm25_top5_direct",         "bm25",         5,  "direct"),
    ("bm25_top5_cot",            "bm25",         5,  "cot"),
    # Dense BGE
    ("dense_bge_top3_direct",    "dense_bge",    3,  "direct"),
    ("dense_bge_top3_cot",       "dense_bge",    3,  "cot"),
    ("dense_bge_top5_direct",    "dense_bge",    5,  "direct"),
    ("dense_bge_top5_cot",       "dense_bge",    5,  "cot"),
    # Dense OpenAI
    ("dense_openai_top3_direct", "dense_openai", 3,  "direct"),
    ("dense_openai_top3_cot",    "dense_openai", 3,  "cot"),
    ("dense_openai_top5_direct", "dense_openai", 5,  "direct"),
    ("dense_openai_top5_cot",    "dense_openai", 5,  "cot"),
    # Hybrid
    ("hybrid_top3_direct",       "hybrid",       3,  "direct"),
    ("hybrid_top3_cot",          "hybrid",       3,  "cot"),
    ("hybrid_top5_direct",       "hybrid",       5,  "direct"),
    ("hybrid_top5_cot",          "hybrid",       5,  "cot"),

    # ═══ Layer 2 — Rerank (4) ═══
    ("dense_bge_rerank_top5_direct", "dense_bge_rerank", 5, "direct"),
    ("dense_bge_rerank_top5_cot",    "dense_bge_rerank", 5, "cot"),
    ("hybrid_rerank_top5_direct",    "hybrid_rerank",    5, "direct"),
    ("hybrid_rerank_top5_cot",       "hybrid_rerank",    5, "cot"),

    # ═══ Layer 3 — Prompt diversity (6) ═══
    # BM25 (cite, minimal) — already has direct/cot
    ("bm25_top5_cite",           "bm25",       5,  "cite"),
    ("bm25_top5_minimal",        "bm25",       5,  "minimal"),
    # Dense BGE
    ("dense_bge_top5_cite",      "dense_bge",  5,  "cite"),
    ("dense_bge_top5_minimal",   "dense_bge",  5,  "minimal"),
    # Hybrid
    ("hybrid_top5_cite",         "hybrid",     5,  "cite"),
    ("hybrid_top5_minimal",      "hybrid",     5,  "minimal"),

    # ═══ Layer 4 — Best production setup (2) ═══
    ("hybrid_rerank_top5_cite",    "hybrid_rerank", 5, "cite"),
    ("hybrid_rerank_top5_minimal", "hybrid_rerank", 5, "minimal"),

    # ═══ Layer 5 — Chunk count extremes (2) ═══
    ("bm25_top1_direct", "bm25", 1,     "direct"),
    ("bm25_all_direct",  "bm25", "all", "direct"),

    # ═══ Layer 6 — Shuffled baselines (2) ═══
    ("bm25_shuffled5_direct", "shuffled", 5, "direct"),
    ("bm25_shuffled3_direct", "shuffled", 3, "direct"),
]


# ═══════════════════════ PROMPT TEMPLATES ═══════════════════════
# Length spread analysis (from previous 10-dataset study):
#   direct:  ~170 chars avg
#   cot:     ~800 chars avg  (5x)
#   cite:    ~500 chars avg  (3x)
#   minimal: ~50 chars avg   (0.3x)
# Overall spread: ~14.5x (max/min), confirming length-confounding susceptibility

PROMPT_TEMPLATES = {
    "direct": """You are an expert assistant. Answer the question based on the provided context.

Context:
{contexts}

Question: {question}

Answer:""",

    "cot": """You are an expert assistant. Think step by step to answer the question based on the provided context.

Context:
{contexts}

Question: {question}

Let's reason step by step to arrive at the answer:""",

    "cite": """You are an expert assistant. Answer the question based on the provided context. Cite specific context passages using [1], [2], etc. to support your answer.

Context:
{contexts}

Question: {question}

Answer (cite sources with [n]):""",

    "minimal": """Context: {contexts}

Q: {question}
A:""",
}


# ═══════════════════════ CONTEXT FORMATTING ═══════════════════════

def format_contexts(contexts: List[Dict[str, Any]], prompt_style: str) -> str:
    """
    Prompt style[...] context [...] .
    - cite: [1], [2] [...]     - others: [...] concatenation
    """
    if not contexts:
        return "(no context retrieved)"
    
    if prompt_style == "cite":
        parts = []
        for i, c in enumerate(contexts):
            parts.append(f"[{i+1}] {c['text']}")
        return "\n\n".join(parts)
    else:
        return "\n\n".join(c['text'] for c in contexts)


# ═══════════════════════ RETRIEVAL DISPATCHER ═══════════════════════

def retrieve_for_pipeline(sample: Dict[str, Any], 
                          retriever: str, 
                          chunk_spec, 
                          cache: RetrievalCache) -> List[Dict[str, Any]]:
    """
    Pipeline[...] retriever + chunk_spec → contexts
    
    Args:
        retriever: 'bm25', 'dense_bge', 'dense_openai', 'hybrid', 
                   'dense_bge_rerank', 'hybrid_rerank', 'shuffled', 'all'
        chunk_spec: int (top-k), 'all', or k for shuffled
    """
    if retriever == "all":
        return cache.get_all(sample)
    
    if retriever == "shuffled":
        return cache.get_shuffled_k(sample, k=chunk_spec)
    
    if chunk_spec == "all":
        # Retrieve all docs in ranked order (still uses the retriever)
        ranked = cache.pre_ranked[sample['id']]['rankings'].get(retriever, [])
        pool = sample['extended_pool']
        return [pool[i] for i in ranked]
    
    # Default: top-k
    return cache.get_top_k(sample, retriever, k=chunk_spec)


# ═══════════════════════ PROMPT BUILDER ═══════════════════════

def build_prompt(sample: Dict[str, Any], 
                 pipeline_cfg: Tuple[str, str, Any, str],
                 cache: RetrievalCache) -> Dict[str, Any]:
    """
    Pipeline config → (prompt, retrieved_contexts, metadata)
    """
    name, retriever, chunk_spec, prompt_style = pipeline_cfg
    
    # 1. Retrieval
    contexts = retrieve_for_pipeline(sample, retriever, chunk_spec, cache)
    
    # 2. Format contexts
    ctx_str = format_contexts(contexts, prompt_style)
    
    # 3. Fill template
    template = PROMPT_TEMPLATES[prompt_style]
    prompt = template.format(
        contexts=ctx_str,
        question=sample['question'],
    )
    
    return {
        "pipeline": name,
        "prompt": prompt,
        "contexts": contexts,
        "context_titles": [c['title'] for c in contexts],
        "retriever": retriever,
        "chunk_spec": chunk_spec,
        "prompt_style": prompt_style,
    }


# ═══════════════════════ VALIDATION (DRY-RUN) ═══════════════════════

def validate_pipelines(dataset_name: str, n_samples: int = 3):
    """
    1 dataset[...] 32 pipeline[...] n_samples[...] prompt [...] .
    LLM [...] . [...] pipeline[...] prompt [...] context [...] .
    """
    extended_path = CACHE_DIR / f"extended_{dataset_name}.pkl"
    if not extended_path.exists():
        print(f"ERROR: {extended_path} not found. Run M1 first.")
        return False
    
    with open(extended_path, 'rb') as f:
        samples = pickle.load(f)
    
    try:
        cache = RetrievalCache(dataset_name)
    except FileNotFoundError as e:
        print(f"ERROR: {e}. Run M2 (retrieval_v2.py) first.")
        return False
    
    print(f"\n{'='*90}")
    print(f"DRY-RUN: {dataset_name} × {n_samples} samples × {len(PIPELINES)} pipelines")
    print(f"{'='*90}")
    print(f"\n{'Pipeline':<35} {'Retr':<18} {'k':<8} {'Style':<8} {'nCtx':>5} {'Prompt len':>10}")
    print("-"*90)
    
    sample = samples[0]
    errors = []
    prompt_lengths = {}
    
    for cfg in PIPELINES:
        name, retriever, chunk_spec, prompt_style = cfg
        try:
            result = build_prompt(sample, cfg, cache)
            n_ctx = len(result['contexts'])
            p_len = len(result['prompt'])
            prompt_lengths[name] = p_len
            chunk_str = str(chunk_spec)
            print(f"{name:<35} {retriever:<18} {chunk_str:<8} {prompt_style:<8} {n_ctx:>5} {p_len:>10}")
        except Exception as e:
            errors.append((name, str(e)))
            print(f"{name:<35} {retriever:<18} {str(chunk_spec):<8} {prompt_style:<8} {'ERR':>5} {type(e).__name__}")
    
    if errors:
        print(f"\n[ERRORS] {len(errors)} pipelines failed:")
        for name, err in errors:
            print(f"  {name}: {err}")
        return False
    
    # Statistics
    print(f"\n{'='*60}")
    print("Prompt length statistics")
    print(f"{'='*60}")
    by_style = {}
    for name, plen in prompt_lengths.items():
        style = name.split('_')[-1]
        by_style.setdefault(style, []).append(plen)
    
    for style in ['minimal', 'direct', 'cite', 'cot']:
        if style in by_style:
            vals = by_style[style]
            print(f"  {style:<10} avg={sum(vals)/len(vals):>7.0f}  min={min(vals):>6}  max={max(vals):>6}  (n={len(vals)})")
    
    # Spread ratio (length confounding indicator)
    all_lens = list(prompt_lengths.values())
    spread = max(all_lens) / min(all_lens) if min(all_lens) > 0 else 0
    print(f"\n  Overall spread (max/min): {spread:.2f}x")
    print(f"  → Higher spread = more length-confounding susceptibility = MMS-F filter more useful")
    
    # Show sample prompts (first 300 chars)
    print(f"\n{'='*60}")
    print("Sample prompts (first 300 chars each)")
    print(f"{'='*60}")
    for style in ['direct', 'cot', 'cite', 'minimal']:
        for cfg in PIPELINES:
            if cfg[3] == style and cfg[1] == 'bm25':
                result = build_prompt(sample, cfg, cache)
                print(f"\n[{cfg[0]}]")
                print(result['prompt'][:300].replace('\n', ' / '))
                print("...")
                break
    
    print(f"\n{'='*90}")
    print(f"✓ DRY-RUN PASS — all 32 pipelines validated on {dataset_name}")
    print(f"{'='*90}")
    return True


# ═══════════════════════ LISTING / SUMMARY ═══════════════════════

def list_pipelines():
    """Pipeline [...] """
    print(f"\n{'='*90}")
    print(f"Pipeline Configs v2 — {len(PIPELINES)} pipelines across 6 layers")
    print(f"{'='*90}")
    
    layers = [
        ("Layer 1 — Primary grid (16): 4 retrievers × 2 chunks × 2 prompts", 0, 16),
        ("Layer 2 — Rerank (4): 2 retrievers × top5 × 2 prompts", 16, 20),
        ("Layer 3 — Prompt diversity (6): 3 retrievers × top5 × {cite, minimal}", 20, 26),
        ("Layer 4 — Best production (2): hybrid_rerank × {cite, minimal}", 26, 28),
        ("Layer 5 — Chunk extremes (2): BM25 × {top1, all}", 28, 30),
        ("Layer 6 — Shuffled baselines (2): BM25 × shuffled × {5, 3}", 30, 32),
    ]
    
    for desc, start, end in layers:
        print(f"\n{desc}")
        print("-"*90)
        for i in range(start, end):
            name, retriever, chunk_spec, style = PIPELINES[i]
            chunk_str = str(chunk_spec)
            print(f"  {i+1:2d}. {name:<35} retriever={retriever:<18} k={chunk_str:<5} style={style}")
    
    # Retriever diversity
    retrievers = set(p[1] for p in PIPELINES)
    prompt_styles = set(p[3] for p in PIPELINES)
    print(f"\n{'='*90}")
    print(f"Summary:")
    print(f"  Total: {len(PIPELINES)} pipelines")
    print(f"  Retrievers: {sorted(retrievers)}")
    print(f"  Prompt styles: {sorted(prompt_styles)}")
    print(f"{'='*90}\n")


# ═══════════════════════ MAIN ═══════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true', help='Pipeline [...] ')
    parser.add_argument('--dry-run', type=str, metavar='DATASET', 
                        help='[...] dataset[...] dry-run (no LLM calls)')
    parser.add_argument('--n-samples', type=int, default=1, help='dry-run [...] ')
    args = parser.parse_args()
    
    if args.list:
        list_pipelines()
    elif args.dry_run:
        validate_pipelines(args.dry_run, n_samples=args.n_samples)
    else:
        list_pipelines()
        print("\n[...] Example:")
        print("  python pipeline_configs_v2.py --dry-run HotpotQA")
        print("  python pipeline_configs_v2.py --dry-run FinQA")


if __name__ == "__main__":
    main()
