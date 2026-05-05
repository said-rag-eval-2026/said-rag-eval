#!/usr/bin/env python3
"""
data_loaders_v2.py — Dataset loaders for the SAID benchmark.

Loads each of the 5 QA datasets (HotpotQA, MS MARCO, WikiQA, PubMedQA, FinQA)
into a unified `samples` list. Each sample is a dict with:
  - id: stable identifier from the source dataset
  - question, answer, supporting_facts (where available)
  - extended_pool: list of {title, text} passages to retrieve from
        (= gold passages + a sampled background pool)

Run M1 first to materialize each dataset's `extended_{Dataset}.pkl` cache,
then M2 (retrieval_v2.py) to pre-compute retrieval rankings.

For source-dataset licenses and download URLs see the dataset card on
Hugging Face. We do not redistribute raw dataset content; users must obtain
the source datasets directly.
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import re
import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("ERROR: rank_bm25 not installed. Run: pip install rank_bm25")
    sys.exit(1)

# ═══════════════════════ CONFIG ═══════════════════════

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_QUESTIONS = 100
N_CROSS_DISTRACTOR = 15       # cross-question distractors per sample
N_HARD_NEG = 10               # BM25 hard negatives per sample

HOTPOTQA_VAL   = Path(os.environ.get("SAID_DATA_DIR", "./data") + "/hotpotqa/validation-00000-of-00001.parquet")
PUBMEDQA_TRAIN = Path(os.environ.get("SAID_DATA_DIR", "./data") + "/pubmedqa/train-00000-of-00001.parquet")
BAREXAM_QA_CSV = Path(os.environ.get("SAID_DATA_DIR", "./data") + "/barexam_qa/qa.csv")
FINQA_TRAIN    = Path(os.environ.get("SAID_DATA_DIR", "./data") + "/finqa/train.json")
QASPER_VAL     = Path("./qasper/validation_0000.parquet")
MSMARCO_DIR    = Path("./msmarco")
WIKIQA_DIR     = Path("./wikiqa")
LEGAL_CORPUS   = Path("./legal_rag_qa/0000.parquet")

OUTPUT_DIR     = Path(os.environ.get("SAID_CACHE_DIR", "./cache_v2"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════ HELPERS ═══════════════════════

_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9'-]*\b")

def tokenize(text: str) -> List[str]:
    """BM25[...] word tokenizer"""
    if not text: return []
    return _TOKEN_RE.findall(text.lower())


def clean_legal_text(text: str) -> str:
    """Bar Exam QA CSV[...] """
    if not text:
        return ""
    s = str(text)
    s = re.sub(r'[[...] ]+', ' ', s)
    s = re.sub(r'[?]{2,}', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def table_row_to_text(row) -> str:
    """FinQA table row[...] markdown format text[...] """
    cells = [str(c) if c is not None else '' for c in row]
    return "| " + " | ".join(cells) + " |"


# ═══════════════════════ DATA LOADERS ═══════════════════════

def load_hotpotqa(n=NUM_QUESTIONS):
    """HotpotQA: multi-hop Wikipedia"""
    print(f"[Parquet] Loading {HOTPOTQA_VAL.name} ...")
    df = pd.read_parquet(HOTPOTQA_VAL)
    samples = []
    for _, row in df.iterrows():
        if len(samples) >= n: break
        contexts = [{"title": t, "text": " ".join(s)}
                    for t, s in zip(row["context"]["title"], row["context"]["sentences"])]
        samples.append({
            "id": row["id"], "question": row["question"], "answer": row["answer"],
            "contexts": contexts,
            "supporting_facts_titles": list(row["supporting_facts"]["title"]),
        })
    print(f"[HotpotQA] {len(samples)} samples loaded")
    return samples


def load_pubmedqa(n=NUM_QUESTIONS):
    """PubMedQA: biomedical yes/no/maybe"""
    print(f"[Parquet] Loading {PUBMEDQA_TRAIN.name} ...")
    df = pd.read_parquet(PUBMEDQA_TRAIN)
    samples = []
    for _, row in df.iterrows():
        if len(samples) >= n: break
        question = row.get("question", "") or ""
        answer = row.get("final_decision", "yes") or "yes"
        ctx_dict = row.get("context", {})
        ctx_paragraphs = []
        if isinstance(ctx_dict, dict) and "contexts" in ctx_dict:
            ctx_paragraphs = [str(c) for c in ctx_dict["contexts"] if c]
        if not ctx_paragraphs:
            long_ans = row.get("long_answer", "") or ""
            if long_ans: ctx_paragraphs = [long_ans]
        if not question or not ctx_paragraphs: continue
        contexts = [{"title": f"PubMed_{row.get('pubid', len(samples))}_{i}", "text": p[:2000]}
                    for i, p in enumerate(ctx_paragraphs)]
        samples.append({
            "id": str(row.get("pubid", len(samples))),
            "question": question, "answer": answer,
            "contexts": contexts,
            "supporting_facts_titles": [contexts[0]["title"]],
        })
    print(f"[PubMedQA] {len(samples)} samples loaded")
    return samples


def load_barexam_qa(n=NUM_QUESTIONS):
    """
    Bar Exam QA (Stanford RegLab, Zheng et al., CS&Law 2025)
    Multistate bar exam questions with hand-annotated gold passages.
    
    Transformation (critical for MMS-F):
      - Multiple choice → "Reasoning + Answer" free-form
      - Ensures answer length varies across prompt styles
    
    Stratification:
      - prompt_id [...] ([...] fact pattern [...] )
      - max 2 per prompt_id → diverse case coverage
    """
    print(f"[CSV] Loading {BAREXAM_QA_CSV.name} ...")
    try:
        df = pd.read_csv(BAREXAM_QA_CSV)
    except pd.errors.ParserError:
        df = pd.read_csv(BAREXAM_QA_CSV, sep='\t')
    required = {'example_id', 'prompt_id', 'question', 'choice_a', 'choice_b',
                'choice_c', 'choice_d', 'answer', 'gold_passage', 'gold_idx'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Bar Exam QA CSV missing required columns: {missing}. "
                        f"Found columns: {list(df.columns)}")
    print(f"[BarExamQA] Total rows: {len(df)}")
    
    MAX_PER_PROMPT = 2
    df_sorted = df.sort_values(['prompt_id', 'example_id']).reset_index(drop=True)
    unique_prompts = df_sorted['prompt_id'].unique().tolist()
    rng = random.Random(SEED + 100)
    rng.shuffle(unique_prompts)
    
    selected_rows = []
    for prompt_id in unique_prompts:
        if len(selected_rows) >= n: break
        group = df_sorted[df_sorted['prompt_id'] == prompt_id].to_dict('records')
        rng.shuffle(group)
        for row in group[:MAX_PER_PROMPT]:
            if len(selected_rows) >= n: break
            selected_rows.append(row)
    
    if len(selected_rows) < n:
        selected_ids = {r['example_id'] for r in selected_rows}
        remaining = [r for r in df_sorted.to_dict('records') 
                     if r['example_id'] not in selected_ids]
        rng.shuffle(remaining)
        for r in remaining:
            if len(selected_rows) >= n: break
            selected_rows.append(r)
    
    samples = []
    skipped = 0
    for row in selected_rows[:n]:
        example_id = str(row.get('example_id', len(samples)))
        prompt = clean_legal_text(row.get('prompt', '') or '')
        q_text = clean_legal_text(row.get('question', '') or '')
        choice_a = clean_legal_text(row.get('choice_a', '') or '')
        choice_b = clean_legal_text(row.get('choice_b', '') or '')
        choice_c = clean_legal_text(row.get('choice_c', '') or '')
        choice_d = clean_legal_text(row.get('choice_d', '') or '')
        answer_letter = str(row.get('answer', '') or '').strip().upper()
        gold_passage = clean_legal_text(row.get('gold_passage', '') or '')
        gold_idx = str(row.get('gold_idx', '') or '').strip()
        
        # Validation
        if answer_letter not in ('A', 'B', 'C', 'D'):
            skipped += 1; continue
        if not gold_passage or len(gold_passage.split()) < 5:
            skipped += 1; continue
        if not gold_idx:
            skipped += 1; continue
        if not q_text:
            skipped += 1; continue
        if not all([choice_a, choice_b, choice_c, choice_d]):
            skipped += 1; continue
        
        # Question: fact pattern + question + 4 choices
        parts = []
        if prompt:
            parts.append(prompt)
        parts.append(q_text)
        parts.append(f"(A) {choice_a}")
        parts.append(f"(B) {choice_b}")
        parts.append(f"(C) {choice_c}")
        parts.append(f"(D) {choice_d}")
        question_full = "\n\n".join(parts)
        
        # Answer: "The correct answer is (X) [text]. [legal reasoning]"
        choice_map = {'A': choice_a, 'B': choice_b, 'C': choice_c, 'D': choice_d}
        correct_text = choice_map[answer_letter]
        answer_full = (
            f"The correct answer is ({answer_letter}) {correct_text}. "
            f"{gold_passage}"
        )
        
        # Context: gold passage (title = gold_idx)
        contexts = [{
            "title": gold_idx,
            "text": gold_passage[:2000]
        }]
        
        samples.append({
            "id": example_id,
            "question": question_full,
            "answer": answer_full,
            "contexts": contexts,
            "supporting_facts_titles": [gold_idx],
        })
    
    print(f"[BarExamQA] {len(samples)} samples loaded (skipped {skipped})")
    return samples


def load_finqa(n=NUM_QUESTIONS):
    """
    FinQA (Chen et al., EMNLP 2021): Numerical reasoning over financial reports.
    6,251 train examples from S&P 500 10-K/10-Q filings.
    
    Transformation (critical for MMS-F):
      - Numerical answer → "The answer is X. [reasoning]" free-form
      - pre_text + table + post_text [...] contexts[...] (question[...] )
    
    Supporting facts:
      - gold_inds[...] ("text_1", "table_0" [...] )[...] context title[...]       - Index [...] : text_0~(L-1) = pre_text, text_L~ = post_text, table_0~ = table rows
    """
    print(f"[JSON] Loading {FINQA_TRAIN.name} ...")
    with open(FINQA_TRAIN, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[FinQA] Total examples: {len(data)}")
    
    # Shuffle for diversity
    rng = random.Random(SEED + 200)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    
    samples = []
    skipped = {'no_q': 0, 'no_a': 0, 'no_gold_inds': 0, 'no_contexts': 0, 'no_valid_support': 0}
    
    for idx in indices:
        if len(samples) >= n: break
        ex = data[idx]
        
        pre_text = ex.get('pre_text', []) or []
        post_text = ex.get('post_text', []) or []
        table = ex.get('table', []) or []
        qa = ex.get('qa', {}) or {}
        
        question = str(qa.get('question', '') or '').strip()
        answer = str(qa.get('answer', '') or '').strip()
        explanation = str(qa.get('explanation', '') or '').strip()
        gold_inds = qa.get('gold_inds', {}) or {}
        
        if not question: skipped['no_q'] += 1; continue
        if not answer:   skipped['no_a'] += 1; continue
        if not gold_inds: skipped['no_gold_inds'] += 1; continue
        if not (pre_text or post_text or table):
            skipped['no_contexts'] += 1; continue
        
        # Contexts: pre_text → table → post_text
        # text_0 ~ text_(L_pre-1) = pre_text
        # text_L_pre ~            = post_text (continued numbering)
        # table_0 ~ table_(L_tab-1) = table rows
        contexts = []
        
        for i, txt in enumerate(pre_text):
            t = str(txt).strip()
            if t:
                contexts.append({"title": f"text_{i}", "text": t[:1500]})
        
        for i, row in enumerate(table):
            row_text = table_row_to_text(row)
            # empty row check
            if row_text.strip().replace('|', '').strip():
                contexts.append({"title": f"table_{i}", "text": row_text[:1500]})
        
        offset = len(pre_text)
        for i, txt in enumerate(post_text):
            t = str(txt).strip()
            if t:
                contexts.append({"title": f"text_{offset+i}", "text": t[:1500]})
        
        if not contexts:
            skipped['no_contexts'] += 1; continue
        
        pool_titles = {c["title"] for c in contexts}
        supporting_valid = [t for t in gold_inds.keys() if t in pool_titles]
        if not supporting_valid:
            skipped['no_valid_support'] += 1; continue
        
        filename = str(ex.get('filename', '') or '')
        company_year = filename.split('/')[0] if '/' in filename else ''
        if company_year:
            question_full = f"(From {company_year} financial report): {question}"
        else:
            question_full = question
        
        # Answer: "The answer is X. [reasoning from explanation or gold_inds]"
        answer_parts = [f"The answer is {answer}."]
        if explanation:
            answer_parts.append(explanation)
        else:
            support_texts = [str(v).strip() for v in gold_inds.values() if v]
            if support_texts:
                answer_parts.append("Based on: " + " ".join(support_texts))
        answer_full = " ".join(answer_parts)
        
        samples.append({
            "id": str(ex.get('id', f'finqa_{idx}')),
            "question": question_full,
            "answer": answer_full,
            "contexts": contexts,
            "supporting_facts_titles": supporting_valid,
        })
    
    print(f"[FinQA] {len(samples)} samples loaded (skipped: {dict(skipped)})")
    return samples


def load_qasper(n=NUM_QUESTIONS):
    """QASPER: NLP paper QA"""
    print(f"[Parquet] Loading {QASPER_VAL.name} ...")
    df = pd.read_parquet(QASPER_VAL)
    samples = []
    for _, row in df.iterrows():
        if len(samples) >= n: break
        qas = row["qas"]
        questions = list(qas.get("question", []))
        answers_list = list(qas.get("answers", []))
        ft = row["full_text"]
        sections = list(ft.get("section_name", []))
        paragraphs = list(ft.get("paragraphs", []))
        contexts = []
        for sec, paras in zip(sections, paragraphs):
            text = " ".join(str(p) for p in paras if p) if hasattr(paras, "__iter__") else str(paras)
            if text.strip():
                contexts.append({"title": str(sec), "text": text[:1000]})
        if not contexts: continue
        for i, (q, ans_dict) in enumerate(zip(questions, answers_list)):
            if len(samples) >= n: break
            answer = ""
            inner = list(ans_dict.get("answer", []))
            for ann in inner:
                if not isinstance(ann, dict): continue
                if ann.get("unanswerable"): continue
                fa = ann.get("free_form_answer", "")
                if fa and str(fa).strip():
                    answer = str(fa).strip(); break
                ev = list(ann.get("evidence", []))
                if ev:
                    answer = " ".join(str(e) for e in ev if e).strip(); break
            if not answer or len(answer.split()) < 3: continue
            samples.append({
                "id": f"qasper_{row['id']}_{i}",
                "question": str(q), "answer": answer,
                "contexts": contexts[:10],
                "supporting_facts_titles": [contexts[0]["title"]] if contexts else [],
            })
    print(f"[QASPER] {len(samples)} samples loaded")
    return samples


def load_msmarco(n=NUM_QUESTIONS):
    """MS MARCO: open-domain web QA"""
    files = sorted(MSMARCO_DIR.glob("*.parquet"))
    if not files:
        print(f"[MSMARCO] No parquet files in {MSMARCO_DIR}"); return []
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"[Parquet] MSMARCO: {len(df)} rows from {len(files)} files")

    def _best_answer(row):
        wa = row.get("wellFormedAnswers", [])
        try:
            wa_list = [str(x) for x in wa if x and str(x).strip()] if hasattr(wa, '__iter__') and not isinstance(wa, str) else []
        except: wa_list = []
        if wa_list: return wa_list[0]
        ans = row.get("answers", [])
        try:
            ans_list = [str(x) for x in ans if x and str(x).strip() and str(x) != "No Answer Present"] if hasattr(ans, '__iter__') and not isinstance(ans, str) else []
        except: ans_list = []
        return ans_list[0] if ans_list else ""

    samples = []
    for _, row in df.iterrows():
        if len(samples) >= n: break
        question = str(row.get("query", "") or "").strip()
        gold = _best_answer(row)
        if not question or not gold: continue
        passages_dict = row.get("passages", {})
        contexts = []
        is_sel_list = []
        if isinstance(passages_dict, dict):
            texts = passages_dict.get("passage_text", [])
            is_sel = passages_dict.get("is_selected", [])
            try:
                texts = list(texts) if hasattr(texts, '__iter__') else []
                is_sel_list = list(is_sel) if hasattr(is_sel, '__iter__') else [0]*len(texts)
            except: texts, is_sel_list = [], []
            for i, txt in enumerate(texts):
                if isinstance(txt, str) and len(txt.strip()) > 20:
                    contexts.append({"title": f"{row.get('query_id','q')}_p{i}", "text": txt.strip()[:1000]})
        if not contexts: continue
        gold_lower = gold.lower()[:40]
        sup = []
        for ci, c in enumerate(contexts):
            sel_val = is_sel_list[ci] if ci < len(is_sel_list) else 0
            if sel_val == 1 or gold_lower in c["text"].lower():
                sup.append(c["title"])
        samples.append({
            "id": str(row.get("query_id", len(samples))),
            "question": question, "answer": gold,
            "contexts": contexts[:10],
            "supporting_facts_titles": sup[:3] if sup else [contexts[0]["title"]],
        })
    print(f"[MSMARCO] {len(samples)} samples loaded")
    return samples


def load_legal_rag_qa(n=NUM_QUESTIONS):
    """LegalRAG: [...] parquet corpus (section title → question, text → context/answer)"""
    print(f"[Parquet] Loading {LEGAL_CORPUS.name} ...")
    df = pd.read_parquet(LEGAL_CORPUS)
    df_main = df[~df["is_supplemental"]].reset_index(drop=True)
    if len(df_main) < n:
        df_main = df.reset_index(drop=True)
    rows = df_main.to_dict("records")
    random.shuffle(rows)
    samples = []
    for row in rows:
        if len(samples) >= n: break
        text  = str(row.get("text", "") or "")
        title = str(row.get("title", "") or "")
        if len(text.split()) < 30: continue
        topic = re.sub(r'^\d[\d.]*:\s*', '', title).strip() or title
        question = f"What are the key legal principles regarding: {topic}?"
        answer   = " ".join(text.split()[:80])
        ctx = {"title": title, "text": text[:2000]}
        samples.append({
            "id": str(row.get("id", len(samples))),
            "question": question, "answer": answer,
            "contexts": [ctx],
            "supporting_facts_titles": [title],
        })
    # Add up to 9 cross-section distractors per sample
    all_ctx = [s["contexts"][0] for s in samples]
    for s in samples:
        distractors = [c for c in all_ctx if c["title"] != s["contexts"][0]["title"]]
        random.shuffle(distractors)
        s["contexts"] = s["contexts"] + distractors[:9]
    print(f"[LegalRAG] {len(samples)} legal questions from corpus")
    return samples


def load_wikiqa(n=NUM_QUESTIONS):
    """WikiQA: sentence-level factoid QA"""
    files = sorted(WIKIQA_DIR.glob("*.parquet"))
    if not files:
        print(f"[WikiQA] No parquet files in {WIKIQA_DIR}"); return []
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"[Parquet] WikiQA: {len(df)} rows from {len(files)} files")
    groups = {}
    for _, row in df.iterrows():
        qid = str(row.get("question_id", ""))
        question = str(row.get("question", "") or "").strip()
        doc_title = str(row.get("document_title", "") or "")
        sent = str(row.get("answer", "") or "").strip()
        label = int(row.get("label", 0))
        if not question or not sent: continue
        if qid not in groups:
            groups[qid] = {"question": question, "positive": [], "all_ctx": []}
        groups[qid]["all_ctx"].append({"title": f"{doc_title}_{len(groups[qid]['all_ctx'])}", "text": sent[:800]})
        if label == 1:
            groups[qid]["positive"].append(sent)
            groups[qid].setdefault("positive_titles", []).append(groups[qid]["all_ctx"][-1]["title"])
    valid = [(qid, g) for qid, g in groups.items() if g["positive"]]
    random.Random(SEED).shuffle(valid)
    samples = []
    for qid, g in valid:
        if len(samples) >= n: break
        gold = " ".join(g["positive"])
        question = g["question"]
        contexts = list(g["all_ctx"])
        pos_titles = g.get("positive_titles", [])
        samples.append({
            "id": qid,
            "question": question, "answer": gold,
            "contexts": contexts,
            "supporting_facts_titles": pos_titles if pos_titles else [contexts[0]["title"]],
        })
    print(f"[WikiQA] {len(samples)} samples loaded")
    return samples


# ═══════════════════════ GLOBAL CORPUS & BM25 HARD NEGATIVES ═══════════════════════

def build_global_corpus(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """[...] sample[...] contexts[...] global corpus [...] ([...] title [...] )"""
    seen_titles = {}
    for s in samples:
        sup_set = set(s.get("supporting_facts_titles", []))
        for ctx in s["contexts"]:
            title = ctx["title"]
            if title not in seen_titles:
                seen_titles[title] = {
                    "title": title,
                    "text": ctx["text"],
                    "gold_for": set()
                }
            if title in sup_set:
                seen_titles[title]["gold_for"].add(s["id"])
    corpus = list(seen_titles.values())
    for c in corpus:
        c["gold_for"] = list(c["gold_for"])
    print(f"  Global corpus: {len(corpus)} unique documents")
    return corpus


def compute_bm25_hard_negatives(samples, global_corpus,
                                 k_retrieve: int = 50,
                                 n_hard: int = N_HARD_NEG):
    """Per-sample BM25 hard negatives"""
    print(f"  Building BM25 index on {len(global_corpus)} docs ...")
    corpus_tokens = [tokenize(doc["text"]) for doc in global_corpus]
    bm25 = BM25Okapi(corpus_tokens)

    hard_negs = {}
    for s in samples:
        q_tokens = tokenize(s["question"])
        if not q_tokens:
            hard_negs[s["id"]] = []
            continue
        scores = bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:k_retrieve]

        gold_titles = set(s.get("supporting_facts_titles", []))
        own_titles = {c["title"] for c in s["contexts"]}

        selected = []
        for idx in top_idx:
            doc = global_corpus[idx]
            if doc["title"] in gold_titles: continue
            if doc["title"] in own_titles: continue
            selected.append({"title": doc["title"], "text": doc["text"], "source": "bm25_hard_neg"})
            if len(selected) >= n_hard: break
        hard_negs[s["id"]] = selected
    n_found = np.mean([len(v) for v in hard_negs.values()])
    print(f"  Hard negatives: avg {n_found:.1f} per sample (target {n_hard})")
    return hard_negs


def sample_cross_distractors(samples, n_cross: int = N_CROSS_DISTRACTOR, rng=None):
    """Per-sample cross-question distractors"""
    rng = rng or random.Random(SEED + 1)
    pool = []
    for s in samples:
        for c in s["contexts"]:
            pool.append({
                "title": c["title"],
                "text": c["text"],
                "from_sample": s["id"],
            })

    cross = {}
    for s in samples:
        own_titles = {c["title"] for c in s["contexts"]}
        gold_titles = set(s.get("supporting_facts_titles", []))
        candidates = [p for p in pool
                      if p["from_sample"] != s["id"]
                      and p["title"] not in own_titles
                      and p["title"] not in gold_titles]
        rng.shuffle(candidates)
        selected = []
        seen = set()
        for p in candidates:
            if p["title"] in seen: continue
            seen.add(p["title"])
            selected.append({"title": p["title"], "text": p["text"], "source": "cross_question"})
            if len(selected) >= n_cross: break
        cross[s["id"]] = selected
    n_found = np.mean([len(v) for v in cross.values()])
    print(f"  Cross-distractors: avg {n_found:.1f} per sample (target {n_cross})")
    return cross


def build_extended_pool(sample, hard_negs, cross_distractors):
    """Extended pool = [...] + hard negatives + cross-distractors (title dedup)"""
    seen_titles = set()
    pool = []

    for c in sample["contexts"]:
        if c["title"] in seen_titles: continue
        seen_titles.add(c["title"])
        pool.append({"title": c["title"], "text": c["text"], "source": "original"})

    for hn in hard_negs:
        if hn["title"] in seen_titles: continue
        seen_titles.add(hn["title"])
        pool.append(hn)

    for cd in cross_distractors:
        if cd["title"] in seen_titles: continue
        seen_titles.add(cd["title"])
        pool.append(cd)

    return pool


def extend_samples(samples, dataset_name):
    """Extended pool end-to-end [...] """
    print(f"\n[{dataset_name}] Building extended pools ...")
    global_corpus = build_global_corpus(samples)
    hard_negs = compute_bm25_hard_negatives(samples, global_corpus)
    cross_distractors = sample_cross_distractors(samples)

    pool_sizes = []
    for s in samples:
        s["extended_pool"] = build_extended_pool(
            s,
            hard_negs.get(s["id"], []),
            cross_distractors.get(s["id"], [])
        )
        pool_sizes.append(len(s["extended_pool"]))

    print(f"  Final pool sizes: mean={np.mean(pool_sizes):.1f}, min={min(pool_sizes)}, max={max(pool_sizes)}")
    return samples


# ═══════════════════════ MAIN ═══════════════════════

LOADERS = {
    "HotpotQA":  load_hotpotqa,
    "MSMARCO":   load_msmarco,
    "WikiQA":    load_wikiqa,
    "QASPER":    load_qasper,
    "PubMedQA":  load_pubmedqa,
    "BarExamQA": load_barexam_qa,
    "FinQA":     load_finqa,        # Finance domain
    "LegalRAG":  load_legal_rag_qa, # Legal corpus (section-based pseudo-QA)
}


def validate_extended_samples(samples, dataset_name):
    """Pool [...] sanity check"""
    issues = []
    for s in samples:
        if "extended_pool" not in s:
            issues.append(f"  {s['id']}: no extended_pool"); continue
        pool = s["extended_pool"]
        if len(pool) < 15:
            issues.append(f"  {s['id']}: pool too small ({len(pool)})")
        sup = set(s.get("supporting_facts_titles", []))
        pool_titles = {p["title"] for p in pool}
        missing_gold = sup - pool_titles
        if missing_gold:
            issues.append(f"  {s['id']}: missing gold in pool: {missing_gold}")
    if issues:
        print(f"\n[{dataset_name}] VALIDATION ISSUES:")
        for issue in issues[:5]:
            print(issue)
        if len(issues) > 5:
            print(f"  ... and {len(issues)-5} more")
    else:
        print(f"[{dataset_name}] ✓ Validation passed")
    return len(issues) == 0


def main():
    print("="*70)
    print("Data Loaders v2.2 — Extended Pool Construction (7 datasets)")
    print(f"  Datasets: {list(LOADERS.keys())}")
    print(f"  N questions per dataset: {NUM_QUESTIONS}")
    print(f"  Cross-distractors: {N_CROSS_DISTRACTOR}, Hard negs: {N_HARD_NEG}")
    print(f"  Output: {OUTPUT_DIR}")
    print("="*70)

    for ds_name, loader in LOADERS.items():
        out_file = OUTPUT_DIR / f"extended_{ds_name}.pkl"
        if out_file.exists():
            print(f"\n[{ds_name}] exists, skip ({out_file})")
            continue

        print(f"\n{'━'*60}")
        print(f"[{ds_name}] Loading ...")
        try:
            samples = loader(n=NUM_QUESTIONS)
        except Exception as e:
            print(f"[{ds_name}] LOAD FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

        if not samples:
            print(f"[{ds_name}] SKIP — no samples loaded")
            continue

        samples = extend_samples(samples, ds_name)
        ok = validate_extended_samples(samples, ds_name)

        with open(out_file, "wb") as f:
            pickle.dump(samples, f)
        print(f"[{ds_name}] Saved: {out_file} ({len(samples)} samples, validation={'PASS' if ok else 'WARN'})")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Samples':>8} {'Avg pool':>10} {'File':>40}")
    print("-"*75)
    for ds_name in LOADERS:
        out_file = OUTPUT_DIR / f"extended_{ds_name}.pkl"
        if out_file.exists():
            with open(out_file, "rb") as f:
                samples = pickle.load(f)
            avg_pool = np.mean([len(s["extended_pool"]) for s in samples])
            print(f"{ds_name:<12} {len(samples):>8} {avg_pool:>10.1f} {out_file.name:>40}")
        else:
            print(f"{ds_name:<12} {'MISSING':>8}")


if __name__ == "__main__":
    main()
