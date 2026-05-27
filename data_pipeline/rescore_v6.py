#!/usr/bin/env python3
"""
rescore_v6.py — Full RAG benchmark generation + metric scoring.

Two stages:
  Stage A: generate answers from each (generator x 32 pipelines x dataset).
  Stage B: score each answer with 10 metrics under each (judge).

Total scale (full main pool):
  Stage A: 32 pipelines x 5 datasets x 5 generators x 100 samples = 80,000 answers
  Stage B: 80,000 x 10 metrics x 3 frontier judges ~= 2.4M LLM judge calls

Usage:
  python rescore_v6.py                             # full run
  python rescore_v6.py --skip-stage-b              # Stage A only
  python rescore_v6.py --skip-stage-a              # Stage B only
  python rescore_v6.py --datasets HotpotQA FinQA   # subset
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import re
import json
import time
import asyncio
import pickle
import argparse
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

try:
    from openai import AsyncAzureOpenAI
    from anthropic import AsyncAnthropic
    from google import genai
    from google.genai.types import HttpOptions, GenerateContentConfig
except ImportError as e:
    print(f"ERROR: Missing package. Run:")
    print(f"  pip install openai anthropic google-genai")
    sys.exit(1)

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_configs_v2 import PIPELINES, build_prompt
from retrieval_v2 import RetrievalCache
import metric_functions


# ═══════════════════════ CONFIG ═══════════════════════

# API endpoints. Set these via env vars or override in your fork.
# Defaults assume direct access to provider APIs.
import os
OPENAI_OPENAI_BASE_URL = os.environ.get("OPENAI_OPENAI_BASE_URL", None)  # None = default OpenAI
ANTHROPIC_OPENAI_BASE_URL = os.environ.get("ANTHROPIC_OPENAI_BASE_URL", None)
GOOGLE_OPENAI_BASE_URL = os.environ.get("GOOGLE_OPENAI_BASE_URL", None)
API_VERSION = os.environ.get("OPENAI_API_VERSION", "2025-04-01-preview")
USE_AZURE = os.environ.get("USE_AZURE_OPENAI", "0") == "1"
CONCURRENT_PER_KEY = int(os.environ.get("CONCURRENT_PER_KEY", "10"))

# Cache directory (relative to repo root by default).
CACHE_DIR = Path(os.environ.get("SAID_CACHE_DIR", "./cache_v2"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model registry
GENERATORS = [
    "claude-sonnet-4-6",
    "gpt-5",
    "gemini-2.5-pro",
]
JUDGES = [
    "claude-sonnet-4-6",
    "gpt-5",
    "gemini-2.5-pro",
]

MODEL_PROVIDER = {
    "gpt-5":             "openai",
    "claude-sonnet-4-6": "anthropic",
    "gemini-2.5-pro":    "google",
}

DATASETS = ["HotpotQA", "MSMARCO", "WikiQA", "PubMedQA", "FinQA"]

# Filename-safe version (replace dots with hyphens)
def safe_name(m: str) -> str:
    return m.replace('.', '-').replace('/', '_')

GEN_TEMPERATURE = 0
METRIC_TEMPERATURE = 0.0

# Global eval model — swapped per judge in Stage B
EVAL_MODEL = "gpt-5"  # initial

APIKEY_FILE_DEFAULT = Path("apikey.txt")


# ═══════════════════════ API KEY LOADING (v4 pattern) ═══════════════════════

def _load_api_keys(path: Path) -> List[str]:
    """Extract 64-char hex API keys from any text format."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    src = path.read_text(encoding="utf-8")
    keys = re.findall(r'\b[0-9a-f]{64}\b', src)
    if not keys:
        raise ValueError(f"No 64-char hex API keys found in {path}")
    return keys


# ═══════════════════════ ASYNC CLIENT POOL (v4 pattern) ═══════════════════════

class _AsyncClientPool:
    """
    Async client pool with N keys × CONCURRENT_PER_KEY slots per provider.
    Three providers (OpenAI/Anthropic/Gemini) with native async clients.
    Round-robin with per-key semaphores.
    """
    def __init__(self, keys: List[str], concurrent: int = CONCURRENT_PER_KEY):
        self._keys = keys
        self._concurrent = concurrent
        self._initialized = False
        self._lock = None
        self._idx_oai = 0
        self._idx_ant = 0
        self._idx_gem = 0
        self._sems_oai: list = []
        self._sems_ant: list = []
        self._sems_gem: list = []
        self._oai: list = []
        self._ant: list = []
        self._gem: list = []

    def _ensure_init(self):
        if self._initialized:
            return
        self._lock = asyncio.Lock()
        self._sems_oai = [asyncio.Semaphore(self._concurrent) for _ in self._keys]
        self._sems_ant = [asyncio.Semaphore(self._concurrent) for _ in self._keys]
        self._sems_gem = [asyncio.Semaphore(self._concurrent) for _ in self._keys]
        # OpenAI / Azure OpenAI client
        if USE_AZURE and OPENAI_OPENAI_BASE_URL:
            self._oai = [
                AsyncAzureOpenAI(azure_endpoint=OPENAI_OPENAI_BASE_URL,
                                 api_key=k, api_version=API_VERSION)
                for k in self._keys
            ]
        else:
            from openai import AsyncOpenAI
            self._oai = [
                AsyncOpenAI(api_key=k, base_url=OPENAI_OPENAI_BASE_URL)
                for k in self._keys
            ]
        # Anthropic client
        ant_kwargs = {}
        if ANTHROPIC_OPENAI_BASE_URL:
            ant_kwargs["base_url"] = ANTHROPIC_OPENAI_BASE_URL
        self._ant = [
            AsyncAnthropic(api_key=k, **ant_kwargs)
            for k in self._keys
        ]
        # Google Gemini client
        gem_http = None
        if GOOGLE_OPENAI_BASE_URL:
            gem_http = HttpOptions(api_version="v1", base_url=GOOGLE_OPENAI_BASE_URL,
                                    headers={"Authorization": "Bearer " + k})
        self._gem = [
            genai.Client(api_key=k, http_options=gem_http)
            for k in self._keys
        ]
        self._initialized = True

    @asynccontextmanager
    async def oai_slot(self):
        self._ensure_init()
        async with self._lock:
            i = self._idx_oai % len(self._keys)
            self._idx_oai += 1
        async with self._sems_oai[i]:
            yield self._oai[i]

    @asynccontextmanager
    async def ant_slot(self):
        self._ensure_init()
        async with self._lock:
            i = self._idx_ant % len(self._keys)
            self._idx_ant += 1
        async with self._sems_ant[i]:
            yield self._ant[i]

    @asynccontextmanager
    async def gem_slot(self):
        self._ensure_init()
        async with self._lock:
            i = self._idx_gem % len(self._keys)
            self._idx_gem += 1
        async with self._sems_gem[i]:
            yield self._gem[i]


# Global pool (initialized in main())
_client_pool: Optional[_AsyncClientPool] = None


# ═══════════════════════ LLM CALLS (v4 pattern, 3 providers) ═══════════════════════

async def call_openai_async(prompt, temperature=0.0, model=None):
    m = model or EVAL_MODEL
    for a in range(5):
        try:
            async with _client_pool.oai_slot() as client:
                kwargs = dict(model=m, messages=[{"role": "user", "content": prompt}])
                if m != "gpt-5":  # gpt-5 only supports default temperature (1)
                    kwargs["temperature"] = temperature
                r = await client.chat.completions.create(**kwargs)
                return r.choices[0].message.content
        except Exception as e:
            w = min(5 * (a + 1), 30)
            if a >= 2: print(f"    ↻ OpenAI #{a+1} ({type(e).__name__}) — {w}s")
            if getattr(e, "status_code", 0) == 400:
                break
            await asyncio.sleep(w)
    return ""


async def call_anthropic_async(prompt, temperature=0.0, model=None):
    m = model or EVAL_MODEL
    for a in range(5):
        try:
            async with _client_pool.ant_slot() as client:
                r = await client.messages.create(
                    model=m, max_tokens=8192, temperature=temperature,
                    system="You are a helpful assistant.",
                    messages=[{"role": "user", "content": prompt}])
                return r.content[0].text if r.content else ""
        except Exception as e:
            w = min(5 * (a + 1), 30)
            if a >= 2: print(f"    ↻ Anthropic #{a+1} ({type(e).__name__}): {e} — {w}s")
            if "BadRequestError" in type(e).__name__ or getattr(e, "status_code", 0) == 400:
                break
            await asyncio.sleep(w)
    return ""


async def call_gemini_async(prompt, temperature=0.0, model=None):
    m = model or EVAL_MODEL
    for a in range(5):
        try:
            async with _client_pool.gem_slot() as client:
                r = await client.aio.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=GenerateContentConfig(temperature=temperature),
                )
                return r.text.strip() if r.text else ""
        except Exception as e:
            w = min(5 * (a + 1), 30)
            if a >= 2: print(f"    ↻ Gemini #{a+1} ({type(e).__name__}: {e}) — {w}s")
            await asyncio.sleep(w)
    return ""


async def llm(prompt, temp=0.0, model=None):
    """LLM router: dispatches to the correct provider based on model name."""
    m = model or EVAL_MODEL
    provider = MODEL_PROVIDER.get(m, "openai")
    if provider == "anthropic":
        return await call_anthropic_async(prompt, temp, m)
    elif provider == "google":
        return await call_gemini_async(prompt, temp, m)
    else:
        return await call_openai_async(prompt, temp, m)


async def embed(texts):
    """OpenAI embedding (used by answer_relevancy)."""
    for a in range(3):
        try:
            async with _client_pool.oai_slot() as c:
                r = await c.embeddings.create(input=texts, model="text-embedding-3-small")
                return [i.embedding for i in r.data]
        except Exception as e:
            if a >= 1: print(f"    ↻ Embed #{a+1} ({type(e).__name__}) — {5*(a+1)}s")
            await asyncio.sleep(5 * (a + 1))
    return []


# ═══════════════════════ STAGE A — ANSWER GENERATION ═══════════════════════

async def generate_answer(gen_model: str, sample: Dict, pipe_cfg: Tuple,
                           cache: RetrievalCache) -> Dict:
    """Single answer: (gen_model, pipeline, sample) -> answer record."""
    prompt_info = build_prompt(sample, pipe_cfg, cache)
    
    # Stage A uses generator model (not EVAL_MODEL)
    answer_text = await llm(
        prompt_info['prompt'],
        temp=GEN_TEMPERATURE,
        model=gen_model,
    )
    
    return {
        "sample_id": sample['id'],
        "pipeline": pipe_cfg[0],
        "generator": gen_model,
        "retriever": pipe_cfg[1],
        "chunk_spec": str(pipe_cfg[2]),
        "prompt_style": pipe_cfg[3],
        "question": sample['question'],
        "ground_truth": sample['answer'],
        "answer": answer_text or "",
        "context_titles": prompt_info['context_titles'],
        "contexts_text": "\n\n".join(c['text'] for c in prompt_info['contexts']),
        "gold_titles": sample.get('supporting_facts_titles', []),
        "failed": not answer_text,
    }


async def run_stage_a_dataset(dataset: str, generators: List[str] = GENERATORS,
                               overwrite: bool = False):
    """1 dataset x N generators x 32 pipelines x 100 samples."""
    print(f"\n{'━'*70}")
    print(f"Stage A: {dataset}")
    print(f"{'━'*70}")
    
    ext_path = CACHE_DIR / f"extended_{dataset}.pkl"
    if not ext_path.exists():
        print(f"  ✗ Missing: {ext_path}. Run M1 first."); return
    with open(ext_path, 'rb') as f:
        samples = pickle.load(f)
    
    try:
        cache = RetrievalCache(dataset)
    except FileNotFoundError:
        print(f"  ✗ Missing pre_ranked. Run M2 first."); return
    
    for gen_model in generators:
        out_path = CACHE_DIR / f"exp_v6_{dataset}_{safe_name(gen_model)}.json"
        if out_path.exists() and not overwrite:
            print(f"  [skip] {out_path.name} exists")
            continue
        
        t0 = time.time()
        total = len(samples) * len(PIPELINES)
        provider = MODEL_PROVIDER.get(gen_model, "openai")
        print(f"  [Gen: {gen_model} via {provider}] {total} answers...")
        
        # Build all tasks
        tasks = []
        for sample in samples:
            for pipe_cfg in PIPELINES:
                tasks.append(generate_answer(gen_model, sample, pipe_cfg, cache))
        
        # Execute all tasks in full parallel (semaphore controls actual concurrency)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        answers = []
        for r in results:
            if isinstance(r, Exception):
                print(f"    ERROR: {type(r).__name__}: {r}")
                continue
            answers.append(r)
        elapsed = time.time() - t0
        rate = len(answers) / elapsed if elapsed > 0 else 0
        fails = sum(1 for a in answers if a.get('failed'))
        print(f"    {len(answers)}/{total} ({rate:.0f}/s, {elapsed:.0f}s elapsed, fails={fails})")
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset": dataset,
                "generator": gen_model,
                "n_samples": len(samples),
                "n_pipelines": len(PIPELINES),
                "answers": answers,
            }, f, ensure_ascii=False, indent=2)
        print(f"  [Gen: {gen_model}] ✓ {out_path.name} ({time.time()-t0:.0f}s)")


# ═══════════════════════ STAGE B — METRIC SCORING ═══════════════════════

async def run_stage_b_dataset(dataset: str, judges: List[str] = JUDGES,
                                overwrite: bool = False,
                                parallel_metrics: bool = True):
    """
    For one dataset:
    - load each generator's answer file
    - score each answer x 10 metrics x N judges
    - judge swap is via global EVAL_MODEL
    - output: exp_v6_{dataset}_{gen}_{judge}.json
    """
    global EVAL_MODEL
    print(f"\n{'━'*70}")
    print(f"Stage B: {dataset}")
    print(f"{'━'*70}")
    
    for gen_model in GENERATORS:
        ans_path = CACHE_DIR / f"exp_v6_{dataset}_{safe_name(gen_model)}.json"
        if not ans_path.exists():
            print(f"  [skip] {ans_path.name} missing — run Stage A first")
            continue
        
        with open(ans_path, 'r', encoding='utf-8') as f:
            ans_data = json.load(f)
        answers = ans_data['answers']
        
        for judge in judges:
            out_path = CACHE_DIR / f"exp_v6_{dataset}_{safe_name(gen_model)}_{safe_name(judge)}.json"
            if out_path.exists() and not overwrite:
                print(f"  [skip] {out_path.name} exists")
                continue
            
            # Switch judge via global
            EVAL_MODEL = judge
            provider = MODEL_PROVIDER.get(judge, "openai")
            
            t0 = time.time()
            valid_answers = [a for a in answers if not a.get('failed') and a.get('answer')]
            print(f"  [{dataset}|gen={gen_model}|judge={judge} via {provider}] "
                  f"{len(valid_answers)} answers × {len(metric_functions.METRIC_NAMES)} metrics...")
            
            # Build QA dicts for metric_functions.score_qa_parallel
            qa_dicts = []
            for ai, ans in enumerate(valid_answers):
                qa_dicts.append({
                    "idx": ai,
                    "original_idx": answers.index(ans),
                    "question": ans['question'],
                    "answer": ans['answer'],
                    "ctx_str": ans.get('contexts_text', ''),
                    "gold": ans.get('ground_truth', ''),
                })
            
            # Score in chunks (to limit concurrency to reasonable level)
            score_fn = metric_functions.score_qa_parallel if parallel_metrics else metric_functions.score_qa
            
            # Execute all tasks in full parallel (semaphore controls actual concurrency)
            all_tasks = [score_fn(qa, t=0.0) for qa in qa_dicts]
            all_results = await asyncio.gather(*all_tasks, return_exceptions=True)
            all_scores = []
            for qa, result in zip(qa_dicts, all_results):
                if isinstance(result, Exception):
                    print(f"    ERROR on {qa['question'][:40]}: {type(result).__name__}")
                    result = {name: 0.5 for name in metric_functions.METRIC_NAMES}
                all_scores.append((qa['original_idx'], result))
            elapsed = time.time() - t0
            rate = len(qa_dicts) / elapsed if elapsed > 0 else 0
            print(f"    {len(qa_dicts)}/{len(qa_dicts)} ({rate:.1f}/s, {elapsed:.0f}s elapsed)")
            
            # Merge back into answer records
            enriched = [dict(a) for a in answers]
            for a in enriched:
                a['metric_scores'] = {}
                a['judge'] = judge
            
            for orig_idx, scores in all_scores:
                enriched[orig_idx]['metric_scores'] = scores
            
            # Save
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "dataset": dataset,
                    "generator": gen_model,
                    "judge": judge,
                    "n_samples": ans_data['n_samples'],
                    "n_pipelines": ans_data['n_pipelines'],
                    "metric_names": metric_functions.METRIC_NAMES,
                    "answers_with_scores": enriched,
                }, f, ensure_ascii=False, indent=2)
            print(f"    ✓ {out_path.name} ({time.time()-t0:.0f}s)")


# ═══════════════════════ MAIN ═══════════════════════

async def main_async(args):
    global _client_pool
    
    # Load API keys
    keys = _load_api_keys(Path(args.apikey))
    print(f"[ClientPool] Loaded {len(keys)} keys × {CONCURRENT_PER_KEY} slots "
          f"= {len(keys)*CONCURRENT_PER_KEY} concurrent (per provider)")
    
    # Init pool + inject into metric_functions
    _client_pool = _AsyncClientPool(keys, concurrent=CONCURRENT_PER_KEY)
    metric_functions.set_llm_functions(llm, embed)
    
    print("="*70)
    print("Rescore v6 — Full RAG benchmark regeneration")
    print("="*70)
    print(f"  Datasets:    {args.datasets}")
    print(f"  Generators:  {GENERATORS}")
    print(f"  Judges:      {JUDGES}")
    print(f"  Pipelines:   {len(PIPELINES)}")
    print(f"  Metrics:     {len(metric_functions.METRIC_NAMES)} (v4)")
    print(f"  Skip A: {args.skip_stage_a}  Skip B: {args.skip_stage_b}")
    print("="*70)
    
    t_total = time.time()
    
    for ds in args.datasets:
        ds_t0 = time.time()
        
        if not args.skip_stage_a:
            try:
                await run_stage_a_dataset(ds, GENERATORS, overwrite=args.overwrite)
            except Exception as e:
                print(f"[{ds}] Stage A FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
        
        if not args.skip_stage_b:
            try:
                await run_stage_b_dataset(ds, JUDGES, overwrite=args.overwrite)
            except Exception as e:
                print(f"[{ds}] Stage B FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
        
        print(f"[{ds}] Completed in {(time.time()-ds_t0)/60:.1f}min")
    
    print(f"\n{'='*70}")
    print(f"ALL DONE in {(time.time()-t_total)/60:.1f}min")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--skip-stage-a', action='store_true')
    parser.add_argument('--skip-stage-b', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--apikey', default=str(APIKEY_FILE_DEFAULT),
                        help='API key file path (default: apikey.txt)')
    args = parser.parse_args()
    
    invalid = [d for d in args.datasets if d not in DATASETS]
    if invalid:
        print(f"Invalid datasets: {invalid}"); sys.exit(1)
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
