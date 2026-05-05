#!/usr/bin/env python3
"""
retrieval_v2.py — Pre-compute retrieval rankings for all datasets.

Strategy:
  - For each sample in each dataset, pre-compute the full ranking under
    every retriever (BM25, dense BGE, dense OpenAI, hybrid, +reranker).
  - Stage A pipelines (in pipeline_configs_v2.py) then slice the precomputed
    rankings to top-k as needed — no recomputation per pipeline.
  - BGE / OpenAI embeddings are computed once per dataset corpus.

Retrievers (6):
  1. bm25              — rank_bm25, local, full ranking
  2. dense_bge         — BGE-large-en-v1.5, local GPU, full ranking
  3. dense_openai      — text-embedding-3-small (via OpenAI), full ranking
  4. hybrid            — BM25 + BGE via reciprocal rank fusion (k=60)
  5. dense_bge_rerank  — BGE top-50 + cross-encoder rerank (50 retained)
  6. hybrid_rerank     — Hybrid top-50 + cross-encoder rerank (50 retained)

Input:  ./cache_v2/extended_{Dataset}.pkl  (from data_loaders_v2.py / M1)
Output: ./cache_v2/pre_ranked_{Dataset}.pkl
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import re
import json
import time
import asyncio
import pickle
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np

# ═══════════════════════ CONFIG ═══════════════════════

CACHE_DIR = Path(os.environ.get("SAID_CACHE_DIR", "./cache_v2"))
EMBEDDING_CACHE_DIR = Path(os.environ.get("SAID_EMB_CACHE_DIR", "./emb_cache"))
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
OPENAI_EMB_MODEL = "text-embedding-3-small"

# BGE query prefix (asymmetric retrieval)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

OPENAI_BASE_URL = "https://api.YOUR-PROVIDER-DOMAIN/hchat-in/api/v3"
OPENAI_API_VERSION = "2025-04-01-preview"
APIKEY_FILE = Path("apikey_emb.txt")   # plain format: one key per line

# Batch sizes
BGE_BATCH = 32         # Local GPU
CROSS_ENC_BATCH = 32
OPENAI_BATCH = 64
CONCURRENT_PER_KEY = 10

# Retrieval params
RRF_K = 60
RERANK_TOP_N = 50

DATASETS = ["HotpotQA", "MSMARCO", "WikiQA", "QASPER", "PubMedQA", "BarExamQA", "FinQA", "LegalRAG"]


# ═══════════════════════ MODEL LAZY LOADERS ═══════════════════════

_bge_model = None
_cross_encoder = None

def get_bge_model():
    global _bge_model
    if _bge_model is None:
        print("[Model] Loading BGE...")
        from sentence_transformers import SentenceTransformer
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _bge_model = SentenceTransformer(BGE_MODEL_NAME, device=device)
        print(f"  BGE loaded on {device}")
    return _bge_model


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        print("[Model] Loading cross-encoder...")
        from sentence_transformers import CrossEncoder
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, device=device, max_length=512)
        print(f"  Cross-encoder loaded on {device}")
    return _cross_encoder


# ═══════════════════════ OPENAI ASYNC EMBEDDING POOL ═══════════════════════

_openai_keys = None
_openai_clients = None
_per_key_sems = None
_dead_keys: set = set()

def _load_openai_keys():
    """apikey_emb.txt[...] API [...] """
    global _openai_keys
    if _openai_keys is not None:
        return _openai_keys

    if not APIKEY_FILE.exists():
        raise FileNotFoundError(
            f"{APIKEY_FILE} not found. Create it with one API key per line."
        )

    keys = [line.strip() for line in APIKEY_FILE.read_text().splitlines()
            if line.strip() and not line.strip().startswith('#')]
    if not keys:
        raise ValueError(f"No API keys found in {APIKEY_FILE}")

    _openai_keys = keys
    print(f"[OpenAI] Loaded {len(keys)} API keys  (slots={len(keys)*CONCURRENT_PER_KEY})")
    return keys


def _get_openai_clients():
    """Async Azure OpenAI [...] Semaphore(CONCURRENT_PER_KEY)"""
    global _openai_clients, _per_key_sems
    if _openai_clients is None:
        from openai import AsyncAzureOpenAI
        keys = _load_openai_keys()
        _openai_clients = [
            AsyncAzureOpenAI(
                azure_endpoint=OPENAI_BASE_URL,
                api_key=k,
                api_version=OPENAI_API_VERSION,
            )
            for k in keys
        ]
        _per_key_sems = [asyncio.Semaphore(CONCURRENT_PER_KEY) for _ in keys]
    return _openai_clients, _per_key_sems


async def _embed_openai_batch(texts: List[str], client_idx: int = 0) -> List[List[float]]:
    """Single batch — [...] + [...]     - 401 ([...] /[...] ): [...] dead [...] → [...]     - 429 (rate limit): [...]     - [...] : [...] exponential backoff ([...] 3[...] )
    - [...] RuntimeError
    """
    global _dead_keys
    clients, sems = _get_openai_clients()
    n = len(clients)
    idx = client_idx % n
    tried_keys = 0

    while tried_keys < n:
        if idx in _dead_keys:
            idx = (idx + 1) % n
            tried_keys += 1
            continue

        client = clients[idx]
        sem = sems[idx]
        move_next = False

        for attempt in range(4):
            try:
                async with sem:
                    resp = await client.embeddings.create(
                        input=texts,
                        model=OPENAI_EMB_MODEL,
                    )
                return [item.embedding for item in resp.data]
            except Exception as e:
                status = getattr(e, 'status_code', None)
                if status == 401:
                    _dead_keys.add(idx)
                    print(f"  [Key {idx}] 401 [...] → dead [...] ({len(_dead_keys)}/{n}[...] )")
                    move_next = True
                    break
                elif status == 429:
                    print(f"  [Key {idx}] 429 rate limit → [...] ")
                    move_next = True
                    break
                else:
                    if attempt == 3:
                        print(f"  [Key {idx}] {type(e).__name__} 4[...] → [...] ")
                        move_next = True
                    else:
                        await asyncio.sleep(2 ** attempt)  # 1s → 2s → 4s

        idx = (idx + 1) % n
        tried_keys += 1
        if not move_next:
            break

    raise RuntimeError(
        f"[...] (dead={len(_dead_keys)}/{n}, tried={tried_keys})"
    )


async def embed_openai_many(texts: List[str], desc: str = "") -> np.ndarray:
    """Batch + concurrent OpenAI embedding — [...] gather"""
    clients, _ = _get_openai_clients()
    n_clients = len(clients)

    batches = [texts[i:i+OPENAI_BATCH] for i in range(0, len(texts), OPENAI_BATCH)]
    total_slots = n_clients * CONCURRENT_PER_KEY
    print(f"  [OpenAI emb] {desc}: {len(texts)} texts / {len(batches)} batches "
          f"/ {n_clients} keys × {CONCURRENT_PER_KEY} = {total_slots} slots")

    tasks = [_embed_openai_batch(batch, client_idx=bi)
             for bi, batch in enumerate(batches)]

    t0 = time.time()
    gathered = await asyncio.gather(*tasks)
    results = [emb for batch_embs in gathered for emb in batch_embs]

    elapsed = time.time() - t0
    print(f"  [OpenAI emb] Done in {elapsed:.1f}s ({len(results)/elapsed:.0f} emb/s)")
    return np.array(results, dtype=np.float32)


# ═══════════════════════ BM25 ═══════════════════════

_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9'-]*\b")

def tokenize(text: str) -> List[str]:
    if not text: return []
    return _TOKEN_RE.findall(str(text).lower())


def rank_bm25_full(query: str, pool_texts: List[str]) -> List[int]:
    """BM25 all ranking: query vs pool_texts, [...] pool index [...] """
    from rank_bm25 import BM25Okapi
    corpus_tokens = [tokenize(t) for t in pool_texts]
    bm25 = BM25Okapi(corpus_tokens)
    q_tokens = tokenize(query)
    if not q_tokens:
        # Fallback: identity ranking
        return list(range(len(pool_texts)))
    scores = bm25.get_scores(q_tokens)
    return np.argsort(scores)[::-1].tolist()


# ═══════════════════════ Dense embedding via BGE ═══════════════════════

def encode_bge(texts: List[str], is_query: bool = False) -> np.ndarray:
    """BGE embedding (normalized)"""
    model = get_bge_model()
    if is_query:
        texts = [f"{BGE_QUERY_PREFIX}{t}" for t in texts]
    embs = model.encode(
        texts,
        batch_size=BGE_BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32)


def rank_dense_full(query_emb: np.ndarray, pool_embs: np.ndarray) -> List[int]:
    """Dense embedding cosine similarity ranking (embeddings are normalized)"""
    scores = pool_embs @ query_emb.T  # (N,)
    return np.argsort(scores)[::-1].tolist()


# ═══════════════════════ Hybrid RRF ═══════════════════════

def rrf_fuse(ranked_lists: List[List[int]], k: int = RRF_K) -> List[int]:
    """
    Reciprocal Rank Fusion (Cormack et al. 2009)
    rrf_score[doc] = Σ 1 / (k + rank)
    """
    scores = defaultdict(float)
    for rlist in ranked_lists:
        for rank, doc_idx in enumerate(rlist):
            scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda d: -scores[d])


# ═══════════════════════ Cross-encoder rerank ═══════════════════════

def cross_encoder_rerank(query: str, pool_texts: List[str], 
                          candidate_indices: List[int]) -> List[int]:
    """
    candidate_indices (e.g., top-50 from BGE/hybrid)[...] cross-encoder[...] rerank.
    [...] : candidate_indices[...] reranked [...] .
    """
    if not candidate_indices:
        return []
    model = get_cross_encoder()
    pairs = [(query, pool_texts[i]) for i in candidate_indices]
    scores = model.predict(pairs, batch_size=CROSS_ENC_BATCH, show_progress_bar=False)
    # sort candidates by score descending
    order = np.argsort(scores)[::-1]
    return [candidate_indices[i] for i in order]


# ═══════════════════════ Per-dataset Pre-compute ═══════════════════════

async def precompute_dataset(dataset_name: str, skip_openai: bool = False) -> bool:
    """
    [...] dataset[...] retriever ranking pre-compute.
    """
    input_path = CACHE_DIR / f"extended_{dataset_name}.pkl"
    output_path = CACHE_DIR / f"pre_ranked_{dataset_name}.pkl"
    
    if not input_path.exists():
        print(f"[{dataset_name}] ✗ INPUT MISSING: {input_path}")
        return False
    
    if output_path.exists():
        print(f"[{dataset_name}] Already computed: {output_path}")
        return True
    
    print(f"\n{'━'*70}")
    print(f"[{dataset_name}] Pre-computing rankings...")
    print(f"{'━'*70}")
    
    with open(input_path, 'rb') as f:
        samples = pickle.load(f)
    
    print(f"[{dataset_name}] Loaded {len(samples)} samples")
    
    global_docs = {}  # title → text (text-level dedup)
    for s in samples:
        for doc in s['extended_pool']:
            key = doc['title']
            if key not in global_docs:
                global_docs[key] = doc['text']
    
    unique_titles = list(global_docs.keys())
    unique_texts = [global_docs[t] for t in unique_titles]
    title_to_global_idx = {t: i for i, t in enumerate(unique_titles)}
    
    print(f"[{dataset_name}] Global corpus: {len(unique_texts)} unique docs")
    
    t0 = time.time()
    print(f"[{dataset_name}] Encoding docs with BGE...")
    doc_embs_bge = encode_bge(unique_texts, is_query=False)
    print(f"  Done in {time.time()-t0:.1f}s, shape={doc_embs_bge.shape}")
    
    # Query embedding (BGE)
    queries = [s['question'] for s in samples]
    print(f"[{dataset_name}] Encoding queries with BGE...")
    query_embs_bge = encode_bge(queries, is_query=True)
    print(f"  shape={query_embs_bge.shape}")
    
    # Step 3: OpenAI embedding (optional)
    doc_embs_oai = None
    query_embs_oai = None
    if not skip_openai:
        print(f"[{dataset_name}] OpenAI embedding docs + queries...")
        # Cache check: if this corpus is already embedded, load from disk
        cache_file = EMBEDDING_CACHE_DIR / f"oai_{dataset_name}_docs.npy"
        if cache_file.exists():
            doc_embs_oai = np.load(cache_file)
            print(f"  Loaded doc embeddings from cache: {cache_file}")
        else:
            doc_embs_oai = await embed_openai_many(unique_texts, desc=f"{dataset_name} docs")
            np.save(cache_file, doc_embs_oai)
        
        cache_q = EMBEDDING_CACHE_DIR / f"oai_{dataset_name}_queries.npy"
        if cache_q.exists():
            query_embs_oai = np.load(cache_q)
            print(f"  Loaded query embeddings from cache: {cache_q}")
        else:
            query_embs_oai = await embed_openai_many(queries, desc=f"{dataset_name} queries")
            np.save(cache_q, query_embs_oai)
        
        # Normalize (OpenAI embeddings not pre-normalized)
        doc_embs_oai = doc_embs_oai / np.linalg.norm(doc_embs_oai, axis=1, keepdims=True).clip(min=1e-8)
        query_embs_oai = query_embs_oai / np.linalg.norm(query_embs_oai, axis=1, keepdims=True).clip(min=1e-8)
    
    # Step 4: Per-sample ranking
    print(f"[{dataset_name}] Per-sample rankings...")
    pre_ranked = {}
    t0 = time.time()
    
    for si, s in enumerate(samples):
        pool = s['extended_pool']
        pool_texts = [d['text'] for d in pool]
        pool_titles = [d['title'] for d in pool]
        query = s['question']
        
        pool_to_global = [title_to_global_idx[t] for t in pool_titles]
        
        bm25_rank = rank_bm25_full(query, pool_texts)
        
        # === Dense BGE (pool only) ===
        pool_embs_bge_local = doc_embs_bge[pool_to_global]  # (pool_size, emb_dim)
        dense_bge_rank = rank_dense_full(query_embs_bge[si], pool_embs_bge_local)
        
        # === Dense OpenAI ===
        if doc_embs_oai is not None:
            pool_embs_oai_local = doc_embs_oai[pool_to_global]
            dense_oai_rank = rank_dense_full(query_embs_oai[si], pool_embs_oai_local)
        else:
            dense_oai_rank = None
        
        # === Hybrid (BM25 + BGE RRF) ===
        hybrid_rank = rrf_fuse([bm25_rank, dense_bge_rank])
        
        # === Rerank variants (cross-encoder) ===
        # BGE top-50 → rerank
        bge_top_candidates = dense_bge_rank[:RERANK_TOP_N]
        dense_bge_rerank = cross_encoder_rerank(query, pool_texts, bge_top_candidates)
        
        # Hybrid top-50 → rerank
        hybrid_top_candidates = hybrid_rank[:RERANK_TOP_N]
        hybrid_rerank = cross_encoder_rerank(query, pool_texts, hybrid_top_candidates)
        
        rankings = {
            'bm25': bm25_rank,
            'dense_bge': dense_bge_rank,
            'hybrid': hybrid_rank,
            'dense_bge_rerank': dense_bge_rerank,
            'hybrid_rerank': hybrid_rerank,
        }
        if dense_oai_rank is not None:
            rankings['dense_openai'] = dense_oai_rank
        
        pre_ranked[s['id']] = {
            'pool_size': len(pool),
            'rankings': rankings,
        }
        
        if (si+1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (si+1) / elapsed
            eta = (len(samples) - si - 1) / rate
            print(f"  {si+1}/{len(samples)} ({rate:.1f} samples/s, ETA {eta:.0f}s)")
    
    print(f"[{dataset_name}] Rankings done in {time.time()-t0:.1f}s")
    
    print(f"[{dataset_name}] Validation — gold recall@5:")
    stats = defaultdict(lambda: {'hit': 0, 'total': 0})
    for s in samples:
        pool = s['extended_pool']
        pool_titles = [d['title'] for d in pool]
        sup = set(s['supporting_facts_titles'])
        if not sup: continue
        
        r = pre_ranked[s['id']]['rankings']
        for ret_name, rank_list in r.items():
            stats[ret_name]['total'] += 1
            top5_titles = {pool_titles[i] for i in rank_list[:5]}
            if sup & top5_titles:
                stats[ret_name]['hit'] += 1
    
    print(f"  {'Retriever':<22} {'Recall@5':>10}")
    for ret_name in sorted(stats.keys()):
        hit, total = stats[ret_name]['hit'], stats[ret_name]['total']
        rec = hit / total if total > 0 else 0
        print(f"  {ret_name:<22} {rec:>9.2%}   ({hit}/{total})")
    
    # Step 6: Save
    with open(output_path, 'wb') as f:
        pickle.dump(pre_ranked, f)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[{dataset_name}] ✓ Saved: {output_path} ({size_mb:.1f} MB)")
    return True


# ═══════════════════════ Retrieval function (for pipelines) ═══════════════════════

class RetrievalCache:
    """M3/M4[...] pipeline [...] retrieval accessor"""
    def __init__(self, dataset_name: str):
        path = CACHE_DIR / f"pre_ranked_{dataset_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Pre-ranked file missing: {path}")
        with open(path, 'rb') as f:
            self.pre_ranked = pickle.load(f)
        self.dataset_name = dataset_name
    
    def get_top_k(self, sample: Dict[str, Any], retriever: str, k: int) -> List[Dict[str, Any]]:
        """
        sample + retriever + k → top-k contexts (list of dicts from extended_pool)
        """
        sid = sample['id']
        if sid not in self.pre_ranked:
            raise KeyError(f"Sample {sid} not in pre-ranked cache")
        rankings = self.pre_ranked[sid]['rankings']
        if retriever not in rankings:
            raise KeyError(f"Retriever '{retriever}' not available. Have: {list(rankings.keys())}")
        indices = rankings[retriever][:k]
        pool = sample['extended_pool']
        return [pool[i] for i in indices]
    
    def get_shuffled_k(self, sample: Dict[str, Any], k: int, seed: int = 0) -> List[Dict[str, Any]]:
        """
        Shuffled baseline: pool[...] supporting facts [...] random k[...]         """
        import random
        rng = random.Random(seed + hash(sample['id']) % 10000)
        pool = sample['extended_pool']
        sup = set(sample.get('supporting_facts_titles', []))
        non_gold = [d for d in pool if d['title'] not in sup]
        rng.shuffle(non_gold)
        return non_gold[:k]
    
    def get_all(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(sample['extended_pool'])


# ═══════════════════════ Main ═══════════════════════

async def main_async(datasets: List[str], skip_openai: bool):
    print("="*70)
    print("Retrieval v2 — Pre-compute Rankings")
    print(f"  Datasets: {datasets}")
    print(f"  Skip OpenAI: {skip_openai}")
    print(f"  BGE: {BGE_MODEL_NAME}")
    print(f"  Cross-encoder: {CROSS_ENCODER_NAME}")
    if not skip_openai:
        print(f"  OpenAI: {OPENAI_EMB_MODEL} (Azure)")
    print("="*70)
    
    t_total = time.time()
    for ds in datasets:
        try:
            await precompute_dataset(ds, skip_openai=skip_openai)
        except Exception as e:
            print(f"\n[{ds}] FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"ALL DONE in {(time.time()-t_total)/60:.1f}min")
    print(f"{'='*70}")
    
    # Summary
    for ds in datasets:
        p = CACHE_DIR / f"pre_ranked_{ds}.pkl"
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"  {ds}: {p.name} ({size_mb:.1f} MB) ✓")
        else:
            print(f"  {ds}: MISSING ✗")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                       help=f"Which datasets to process (default: all 7)")
    parser.add_argument('--skip-openai', action='store_true',
                       help="Skip OpenAI embedding (BGE only)")
    args = parser.parse_args()
    
    # Validate dataset names
    invalid = [d for d in args.datasets if d not in DATASETS]
    if invalid:
        print(f"Invalid datasets: {invalid}. Valid: {DATASETS}")
        sys.exit(1)
    
    asyncio.run(main_async(args.datasets, args.skip_openai))


if __name__ == "__main__":
    main()
