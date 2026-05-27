#!/usr/bin/env python3
"""
metric_functions.py — 10 LLM-judged metrics for RAG evaluation.

This module implements the 10 metrics scored on each (question, answer)
pair in the SAID benchmark. The llm() / embed() functions are injected from
the caller via set_llm_functions().

10 METRICS:
  RAGAS (3, exact verbatim from the RAGAS paper):
    1. faithfulness       — 2-step: statement extraction + NLI verdict
    2. answer_relevancy   — 3 calls (question generation) + embedding cosine
    3. context_precision  — per-chunk usefulness with Precision@K
  
  G-Eval (7, Liu et al. 2023 style):
    4. hallucination_free — contradiction check (5-point)
    5. context_utilization — answer uses context info
    6. completeness       — question coverage
    7. conciseness        — brevity
    8. coherence          — logical flow
    9. specificity        — concrete facts/numbers
    10. citation_quality  — attribution to context
"""
import re
import json
import asyncio
from typing import Callable, Optional, List

import numpy as np


# ═══════════════════════ LLM INJECTION ═══════════════════════
# Caller injects these functions

_llm: Optional[Callable] = None
_embed: Optional[Callable] = None


def set_llm_functions(llm_fn, embed_fn):
    """
    Inject llm / embed functions from the caller.
    
    Args:
        llm_fn: async function(prompt, temp=0.0, max_tok=1024, model=None) -> str
        embed_fn: async function(texts: list) -> list of embeddings
    """
    global _llm, _embed
    _llm = llm_fn
    _embed = embed_fn


def _ensure_llm():
    if _llm is None:
        raise RuntimeError("llm function not set. Call metric_functions.set_llm_functions() first.")


# ═══════════════════════ PARSERS ═══════════════════════

R5 = {5: 1.0, 4: 0.75, 3: 0.5, 2: 0.25, 1: 0.0}


def pg(raw):
    """Parse G-Eval JSON → 0-1"""
    if not raw: return 0.5
    try:
        m = re.search(r'"score"\s*:\s*([1-5])', raw)
        if m: return R5[int(m.group(1))]
    except: pass
    try:
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            s = json.loads(m.group()).get("score", 3)
            return R5.get(int(round(s)), 0.5) if isinstance(s, (int, float)) and s > 1 else float(s)
    except: pass
    return 0.5


def p_claims(raw):
    if not raw: return []
    for pat in [r'\{.*\}', r'\[.*\]']:
        try:
            m = re.search(pat, raw, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                return obj.get("statements", obj) if isinstance(obj, dict) else obj
        except: pass
    return []


def p_verdicts(raw):
    if not raw: return []
    try:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m: return [int(x.get("verdict", 0)) for x in json.loads(m.group()) if isinstance(x, dict)]
    except: pass
    return []


def p_arel(raw):
    if not raw: return "", 0
    try:
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m:
            o = json.loads(m.group())
            return o.get("question", ""), int(o.get("noncommittal", 0))
    except: pass
    return "", 0


def p_cprec(raw):
    if not raw: return 0
    try:
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        if m: return int(json.loads(m.group()).get("verdict", 0))
    except: pass
    return 1 if '"verdict": 1' in raw or '"verdict":1' in raw else 0


# ═══════════════════════════════════════════════════════════
# 10 METRICS — Exact RAGAS + G-Eval (v4 verbatim)
# ═══════════════════════════════════════════════════════════

# RAGAS faithfulness (2-step, exact few-shot from source)
async def m_faith(q, a, ctx, t=0.0):
    _ensure_llm()
    r1 = await _llm(
        'Create one or more statements from each sentence in the given answer.\n\n'
        'question: "Who was Albert Einstein and what is he best known for?"\n'
        'answer: "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest '
        'and most influential physicists of all time. He was best known for developing the theory of relativity, '
        'he also made important contributions to the development of the theory of quantum mechanics."\n'
        'statements: {"statements": ["Albert Einstein was a German-born theoretical physicist.", '
        '"Albert Einstein is recognized as one of the greatest and most influential physicists of all time.", '
        '"Albert Einstein was best known for developing the theory of relativity.", '
        '"Albert Einstein also made important contributions to the development of the theory of quantum mechanics."]}\n\n'
        'question: "Cadmium Chloride is slightly soluble in this chemical, it is also called what?"\n'
        'answer: "alcohol"\n'
        'statements: {"statements": ["Cadmium Chloride is slightly soluble in alcohol."]}\n\n'
        f'question: "{q}"\nanswer: "{a}"\nstatements:', t)
    claims = p_claims(r1)
    if not claims: return 0.5
    r2 = await _llm(
        'Your task is to judge the faithfulness of a series of statements based on a given context. '
        'For each statement you must return verdict as 1 if the statement can be directly inferred '
        'based on the context or 0 if the statement can not be directly inferred based on the context.\n\n'
        'context: "John is a student at XYZ University. He is pursuing a degree in Computer Science. '
        'He is enrolled in several courses this semester, including Data Structures, Algorithms, '
        'and Database Management. John is a diligent student and spends a significant amount of time '
        'studying and completing assignments. He often stays late in the library to work on his projects."\n'
        'statements: ["John is majoring in Biology.", "John is taking a course on Artificial Intelligence.", '
        '"John is a dedicated and diligent student.", "John has a part-time job."]\n'
        'answer: [{"statement": "John is majoring in Biology.", '
        '"reason": "John\'s major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.", '
        '"verdict": 0}, {"statement": "John is taking a course on Artificial Intelligence.", '
        '"reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned.", '
        '"verdict": 0}, {"statement": "John is a dedicated and diligent student.", '
        '"reason": "The context states that he spends a significant amount of time studying and completing assignments, and he often stays late in the library.", '
        '"verdict": 1}, {"statement": "John has a part-time job.", '
        '"reason": "There is no information about John having a part-time job.", '
        '"verdict": 0}]\n\n'
        f'context: "{ctx[:3000]}"\nstatements: {json.dumps(claims)}\nanswer:', t)
    v = p_verdicts(r2)
    return sum(v) / len(v) if v else 0.5


# RAGAS answer_relevancy (exact few-shot, 3 calls + embedding)
async def m_arel(q, a, ctx, t=0.0):
    _ensure_llm()
    qs = []
    for _ in range(3):
        r = await _llm(
            'Generate a question for the given answer and Identify if answer is noncommittal. '
            'Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. '
            'A noncommittal answer is one that is evasive, vague, or ambiguous. '
            'For example, "I don\'t know" or "I\'m not sure" are noncommittal answers\n\n'
            'answer: "Everest"\n'
            'context: "The tallest mountain on Earth, measured from sea level, is a renowned peak located in the Himalayas."\n'
            'output: {"question": "What is the tallest mountain on Earth?", "noncommittal": 0}\n\n'
            'answer: "I don\'t know about the groundbreaking feature of the smartphone invented in 2023 '
            'as am unaware of information beyond 2022."\n'
            'context: "In 2023, a groundbreaking invention was announced: a smartphone with a battery life '
            'of one month, revolutionizing the way people use mobile technology."\n'
            'output: {"question": "What was the groundbreaking feature of the smartphone invented in 2023?", "noncommittal": 1}\n\n'
            f'answer: "{a}"\ncontext: "{ctx[:2000]}"\noutput:', t)
        gq, nc = p_arel(r)
        if nc == 1: return 0.0
        if gq: qs.append(gq)
    if not qs: return 0.5
    try:
        embs = await _embed([q] + qs)
        if not embs or len(embs) < 2: return 0.5
        o = np.array(embs[0])
        sims = [float(np.dot(o, np.array(embs[i + 1])) / (np.linalg.norm(o) * np.linalg.norm(np.array(embs[i + 1])) + 1e-10))
                for i in range(len(qs))]
        return float(np.mean(sims))
    except:
        return 0.5


# RAGAS context_precision (exact few-shot, per-chunk)
async def m_cprec(q, a, chunks, t=0.0):
    _ensure_llm()
    if not chunks: return 0.0
    vs = []
    for ch in chunks[:10]:
        r = await _llm(
            'Given question, answer and context verify if the context was useful in arriving at '
            'the given answer. Give verdict as "1" if useful and "0" if not with json output.\n\n'
            'question: "What is the tallest mountain in the world?"\n'
            'context: "The Andes is the longest continental mountain range in the world, located in '
            'South America. It stretches across seven countries and features many of the highest peaks '
            'in the Western Hemisphere. The range is known for its diverse ecosystems, including the '
            'high-altitude Andean Plateau and the Amazon rainforest."\n'
            'answer: "Mount Everest."\n'
            'output: {"reason": "the provided context discusses the Andes mountain range, which, '
            'while impressive, does not include Mount Everest or directly relate to the question about '
            'the world\'s tallest mountain.", "verdict": 0}\n\n'
            f'question: "{q}"\ncontext: "{ch[:500]}"\nanswer: "{a}"\noutput:', t)
        vs.append(p_cprec(r))
    if not vs: return 0.5
    nr = sum(vs)
    if nr == 0: return 0.0
    pak = [sum(vs[:k]) / k * vs[k - 1] for k in range(1, len(vs) + 1)]
    return sum(pak) / nr


# G-Eval metrics (5-point rubric + CoT)
async def m_geval(body, t=0.0):
    _ensure_llm()
    return pg(await _llm(body, t))


async def m_hallu(q, a, ctx, t=0.0):
    return await m_geval(
        f"Evaluate whether the answer contradicts the provided context.\n\n"
        f"Evaluation Steps:\n1. Identify key facts in context.\n2. Identify claims in answer.\n3. Compare.\n\n"
        f"Context:\n{ctx[:3000]}\n\nAnswer: {a}\n\n"
        f"Score 1-5:\n  5: No contradictions\n  4: Minor unsupported claims\n  3: Mixed\n"
        f"  2: Significant contradictions\n  1: Major fabrications\n\n"
        f'Output JSON: {{"evaluation_steps": [...], "score": <1-5>}}', t)


async def m_cutil(q, a, ctx, t=0.0):
    return await m_geval(
        f"Evaluate how effectively the answer uses context information.\n\n"
        f"Evaluation Steps:\n1. Identify key info units.\n2. Check reflection in answer.\n3. Assess utilization.\n\n"
        f"Context:\n{ctx[:2000]}\n\nAnswer: {a}\n\n"
        f"Score 1-5:\n  5: Excellent\n  4: Good\n  3: Moderate\n  2: Low\n  1: None\n\n"
        f'Output JSON: {{"evaluation_steps": [...], "score": <1-5>}}', t)


async def m_comp(q, a, ctx, t=0.0):
    return await m_geval(
        f"Evaluate whether the answer completely addresses the question.\n\n"
        f"Evaluation Steps:\n1. What does question ask?\n2. What does context provide?\n3. Assess coverage.\n\n"
        f"Question: {q}\nContext: {ctx[:2000]}\nAnswer: {a}\n\n"
        f"Score 1-5:\n  5: Comprehensive\n  4: Mostly complete\n  3: Partial\n  2: Incomplete\n  1: Minimal\n\n"
        f'Output JSON: {{"evaluation_steps": [...], "score": <1-5>}}', t)


async def m_conc(q, a, ctx, t=0.0):
    return await m_geval(
        f"Evaluate whether the answer is concise.\n\n"
        f"Evaluation Steps:\n1. Check redundancy.\n2. Check tangents.\n3. Assess brevity.\n\n"
        f"Question: {q}\nAnswer: {a}\n\n"
        f"Score 1-5:\n  5: Perfectly concise\n  4: Mostly concise\n  3: Somewhat verbose\n"
        f"  2: Verbose\n  1: Very verbose\n\n"
        f'Output JSON: {{"evaluation_steps": [...], "score": <1-5>}}', t)


async def m_cohe(q, a, ctx, t=0.0):
    return await m_geval(
        f"Evaluate the answer's logical coherence.\n\n"
        f"Evaluation Steps:\n1. Assess logical flow.\n2. Check contradictions.\n3. Evaluate organization.\n\n"
        f"Answer: {a}\n\n"
        f"Score 1-5:\n  5: Perfectly coherent\n  4: Mostly coherent\n  3: Somewhat\n"
        f"  2: Poorly coherent\n  1: Incoherent\n\n"
        f'Output JSON: {{"evaluation_steps": [...], "score": <1-5>}}', t)


async def m_spec(q, a, ctx, t=0.0):
    return await m_geval(
        f"Evaluate how specific the answer is.\n\n"
        f"Evaluation Steps:\n1. Check for concrete facts/numbers/names.\n2. Beyond surface-level?\n3. Detail appropriate?\n\n"
        f"Question: {q}\nAnswer: {a}\n\n"
        f"Score 1-5:\n  5: Highly specific\n  4: Mostly specific\n  3: Moderate\n"
        f"  2: Mostly vague\n  1: Very vague\n\n"
        f'Output JSON: {{"evaluation_steps": [...], "score": <1-5>}}', t)


async def m_cite(q, a, ctx, t=0.0):
    return await m_geval(
        f"Evaluate whether the answer grounds claims in context.\n\n"
        f"Evaluation Steps:\n1. Check references to context.\n2. Assess traceability.\n3. Evaluate accuracy.\n\n"
        f"Context:\n{ctx[:2000]}\n\nAnswer: {a}\n\n"
        f"Score 1-5:\n  5: Excellent attribution\n  4: Good\n  3: Partial\n  2: Weak\n  1: None\n\n"
        f'Output JSON: {{"evaluation_steps": [...], "score": <1-5>}}', t)


# GT Judge (5-point rubric + CoT, oracle metric)
async def gt_judge(q, a, gold, t=0.0):
    _ensure_llm()
    return pg(await _llm(
        f"Evaluate a RAG answer by comparing to reference.\n\n"
        f"Question: {q}\nGold Answer: {gold}\nSystem Answer: {a}\n\n"
        f"Evaluation Steps:\n1. Key facts in gold.\n2. Which appear in system answer.\n"
        f"3. Incorrect/fabricated info?\n4. Responsive to question?\n\n"
        f"Score 1-5:\n  5: All key facts, no errors\n  4: Most facts, minor omissions\n"
        f"  3: Partially correct\n  2: Mostly incorrect\n  1: Completely wrong\n\n"
        f'Output JSON: {{"evaluation_steps": [...], "score": <1-5>}}', t))


# ═══════════════════════ REGISTRY ═══════════════════════

METRIC_NAMES = [
    "faithfulness", "hallucination_free",
    "answer_relevancy", "context_precision", "context_utilization",
    "completeness", "conciseness", "coherence",
    "specificity", "citation_quality",
]

MFN = {
    "faithfulness":        m_faith,
    "hallucination_free":  m_hallu,
    "answer_relevancy":    m_arel,
    "context_precision":   m_cprec,
    "context_utilization": m_cutil,
    "completeness":        m_comp,
    "conciseness":         m_conc,
    "coherence":           m_cohe,
    "specificity":         m_spec,
    "citation_quality":    m_cite,
}


# ═══════════════════════ SCORING ═══════════════════════

async def score_qa(qa, t=0.0):
    """
    QA dict -> metric scores dict
    
    Args:
        qa: {"question": str, "answer": str, "ctx_str": str}
        t: temperature
    
    Returns:
        {metric_name: score in [0,1]}
    """
    q, a, ctx = qa["question"], qa["answer"], qa.get("ctx_str", "")
    chunks = [ctx[i:i + 500] for i in range(0, min(len(ctx), 5000), 500)] if ctx else []
    res = {}
    for name in METRIC_NAMES:
        fn = MFN[name]
        try:
            res[name] = await (fn(q, a, chunks, t) if name == "context_precision" else fn(q, a, ctx, t))
        except Exception as e:
            print(f"    [{name}] error: {type(e).__name__}: {str(e)[:80]}")
            res[name] = 0.5
    return res


async def score_qa_parallel(qa, t=0.0):
    """
    Compute all metrics in parallel.
    
    """
    q, a, ctx = qa["question"], qa["answer"], qa.get("ctx_str", "")
    chunks = [ctx[i:i + 500] for i in range(0, min(len(ctx), 5000), 500)] if ctx else []
    
    async def _run(name, fn):
        try:
            if name == "context_precision":
                return name, await fn(q, a, chunks, t)
            return name, await fn(q, a, ctx, t)
        except Exception as e:
            print(f"    [{name}] error: {type(e).__name__}: {str(e)[:80]}")
            return name, 0.5
    
    tasks = [_run(name, MFN[name]) for name in METRIC_NAMES]
    results = await asyncio.gather(*tasks)
    return dict(results)


async def score_qa_parallel_with_gt(qa, t=0.0):
    """
    10 metrics + gt_judge in parallel (the core scoring function).
    
    gt_judge is the gold-answer oracle judge (used as the oracle for SAID).
    If no gold answer is available, gt_judge returns None.
    
    Cross-judge matrix: each judge (EVAL_MODEL) runs 10 metrics + gt_judge independently.
    
    Args:
        qa: {"question": str, "answer": str, "ctx_str": str, "gold": str (optional)}
    
    Returns:
        {metric_name: score, ..., "gt_judge": score or None}
    """
    _ensure_llm()
    q, a, ctx = qa["question"], qa["answer"], qa.get("ctx_str", "")
    gold = qa.get("gold", "") or qa.get("ground_truth", "")
    chunks = [ctx[i:i + 500] for i in range(0, min(len(ctx), 5000), 500)] if ctx else []
    
    async def _run_metric(name, fn):
        try:
            if name == "context_precision":
                return name, await fn(q, a, chunks, t)
            return name, await fn(q, a, ctx, t)
        except Exception as e:
            print(f"    [{name}] error: {type(e).__name__}: {str(e)[:80]}")
            return name, 0.5
    
    async def _run_gt():
        if not gold or not str(gold).strip():
            return "gt_judge", None
        try:
            return "gt_judge", await gt_judge(q, a, gold, t)
        except Exception as e:
            print(f"    [gt_judge] error: {type(e).__name__}: {str(e)[:80]}")
            return "gt_judge", 0.5
    
    # 10 metrics + gt_judge all in parallel
    tasks = [_run_metric(name, MFN[name]) for name in METRIC_NAMES]
    tasks.append(_run_gt())
    results = await asyncio.gather(*tasks)
    return dict(results)


if __name__ == "__main__":
    print("metric_functions.py — 10 LLM-judged metrics")
    print(f"Metrics: {METRIC_NAMES}")
    print(f"MFN entries: {len(MFN)}")
    print(f"\nUsage:")
    print(f"  import metric_functions")
    print(f"  metric_functions.set_llm_functions(my_llm, my_embed)")
    print(f"  scores = await metric_functions.score_qa_parallel(qa_dict)")
