# SAID — Some RAG Metrics Don't Measure Quality

Reference implementation of **SAID (Structural Adversarial Invariance Discrimination)**, an unsupervised filter that identifies quality-tracking LLM-judged metrics for RAG pipeline evaluation.

This repository accompanies the NeurIPS 2026 Evaluations & Datasets Track submission *"Some RAG Metrics Don't Measure Quality: Detecting Surface Confounds via Retrieval Invariants"* (under double-blind review).

**Companion benchmark**: https://huggingface.co/datasets/said-rag-eval-2026/said-rag-eval-benchmark

---

## What's in this repo

| Path | Contents |
|---|---|
| `said/` | Core SAID algorithm + analysis (Tables 1–4 reproduction) |
| `baselines/` | All baselines compared in the paper |
| `data_pipeline/` | The full data-generation pipeline (Stage A: answers, Stage B: metrics) |
| `scripts/` | Top-level entry points |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT |

### `said/`
- `algorithm.py` — Algorithm 1 (`said_filter`), Signal A, Signal B, refusal masking
- `analysis.py` — Tables 1, 2, 3, 4 reproduction; bootstrap CIs

### `baselines/`
- `unsupervised.py` — `uniform_filter`, `drop_conciseness_filter`, `pma_filter`, `length_filter`
- `supervised.py` — `find_best_fixed_subset`, `ridge_lodo_pipeline_scores` (oracle upper bounds)

### `data_pipeline/`
- `data_loaders_v2.py` — Loaders for HotpotQA, MS MARCO, WikiQA, PubMedQA, FinQA
- `retrieval_v2.py` — Pre-compute 6 retrievers (BM25, BGE, OpenAI, hybrid, +rerank)
- `pipeline_configs_v2.py` — 32 RAG pipeline configurations
- `metric_functions.py` — 10 LLM-judged metrics (RAGAS-3 + G-Eval-7)
- `rescore_v6.py` — Stage A (answer generation) + Stage B (metric scoring)
- `gt_judge_only.py` — Add gt_judge (gold-judge oracle) post-hoc
- `extract_analysis_data.py` — Compact extraction for analysis

---

## Reproducing the paper's results

### Quick start (use the released benchmark)

The fastest path is to download the compact metric-scores file from the companion HF dataset and run the analysis directly:

```bash
pip install -r requirements.txt
git clone <this-repo-url>
cd said-rag-eval

# Download metric_scores_compact.json from HF (about 19 MB)
wget https://huggingface.co/datasets/said-rag-eval-2026/said-rag-eval-benchmark/resolve/main/metric_scores_compact.json

# Reproduce Tables 1-4
python scripts/run_analysis.py --input metric_scores_compact.json
```

Expected output (matches Table 1 in the paper, up to bootstrap noise):

```
Method                        Mean Δτ                 95% CI       Wins
------------------------------------------------------------------------------
Uniform (RAGAS-style)          +0.000                      —          —
DropConciseness                +0.054     [+0.028, +0.081]      49/75
PMA (ours, baseline)           -0.060     [-0.112, -0.011]      33/75
LengthFilter                   +0.038     [+0.005, +0.072]      44/75
SAID (ours)                    +0.153     [+0.109, +0.200]      58/75
```

Numbers are reproducible to ±0.005 across seeds (paired bootstrap, 5000 resamples, seed 42). Supervised oracle rows (Best Fixed Subset, Ridge LODO) are printed when `--skip-supervised` is not set.

### From scratch (re-run Stage A + Stage B)

To regenerate the entire benchmark (~$3,200 in API costs and ~240 GPU-hours; expect 1–2 days end-to-end):

```bash
# 1. Set environment variables for your API providers
export OPENAI_API_KEY=...   # or set OPENAI_BASE_URL for a custom endpoint
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
# Optional: export USE_AZURE_OPENAI=1 + OPENAI_BASE_URL=<azure-endpoint>
export SAID_CACHE_DIR=./cache_v2
export SAID_DATA_DIR=./data    # path to source dataset files

# 2. Materialize per-dataset extended pools (M1)
python data_pipeline/data_loaders_v2.py

# 3. Pre-compute retrieval rankings (M2)
python data_pipeline/retrieval_v2.py

# 4. Generate answers (Stage A) and score with metrics (Stage B)
python data_pipeline/rescore_v6.py --datasets HotpotQA MSMARCO WikiQA PubMedQA FinQA

# 5. Extract compact form for analysis
python data_pipeline/extract_analysis_data.py

# 6. Run the analysis
python scripts/run_analysis.py --input ./cache_v2/analyze_raw_compact.json
```

---

## SAID algorithm at a glance

```python
from said import CellData, said_filter, aggregate_pipeline_scores

# `cell` is one (dataset, generator, judge) entry from metric_scores_compact.json
cell = CellData.from_compact_dict(raw_cell_dict)

result = said_filter(
    cell,
    metric_names=["faithfulness", "hallucination_free", "answer_relevancy",
                  "context_precision", "context_utilization", "completeness",
                  "conciseness", "coherence", "specificity", "citation_quality"],
    theta_a=0.85,           # Signal A threshold
    refusal_len=50,         # chars; refusal-mask threshold
    fallback_k=3,           # if < K metrics pass, keep top-K by combined score
)

print("Kept metrics:", result.kept_metrics)
print("Fallback used:", result.fallback_used)

# Aggregate kept metrics into per-pipeline scores
pipeline_scores = aggregate_pipeline_scores(cell, result.kept_metrics)
```

---

## Notes for reviewers

**License**: MIT for code; CC BY 4.0 for the released benchmark artifacts (see HF dataset).

**Anonymity**: This is an anonymous double-blind submission. Authors and affiliation will be added on acceptance.

**Comments**: Some scripts in `data_pipeline/` retain a small number of inline comments in Korean from the development process. Functionality is independent of comment language; the public docstrings, headers, and all `said/` and `baselines/` code are in English. We will fully translate inline comments before camera-ready.

**API endpoints**: `data_pipeline/rescore_v6.py` reads provider endpoints from environment variables (`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_BASE_URL`); defaults assume direct provider access. Set `USE_AZURE_OPENAI=1` to route OpenAI calls through an Azure endpoint.

**API keys**: Place OpenAI/Anthropic/Google keys (one per line, or as a Python list) in `apikey.txt` at the repo root. The loader extracts any 64-char hex sequences. Multiple keys are supported for round-robin parallelism.

---

## Repository structure

```
said-rag-eval/
├── README.md                          # this file
├── LICENSE                            # MIT
├── requirements.txt                   # pip dependencies
├── apikey.txt.example                 # template — fill with your provider keys
│
├── said/                              # SAID algorithm + analysis
│   ├── __init__.py
│   ├── algorithm.py                   # Algorithm 1, Signals A/B, refusal mask
│   └── analysis.py                    # Tables 1-4, bootstrap CIs
│
├── baselines/                         # all baselines from the paper
│   ├── __init__.py
│   ├── unsupervised.py                # uniform, drop_conciseness, pma, length
│   └── supervised.py                  # best fixed subset, Ridge LODO
│
├── data_pipeline/                     # data generation (Stage A + B)
│   ├── data_loaders_v2.py             # 5 dataset loaders
│   ├── retrieval_v2.py                # 6 retrievers, pre-ranked corpus
│   ├── pipeline_configs_v2.py         # 32 RAG pipelines
│   ├── metric_functions.py            # 10 LLM-judged metrics
│   ├── rescore_v6.py                  # Stage A + B orchestration
│   ├── gt_judge_only.py               # gold-judge post-hoc add
│   └── extract_analysis_data.py       # compact format for analysis
│
└── scripts/
    └── run_analysis.py                # CLI: reproduce Tables 1-4
```

---

## Citation

This is an anonymous submission. Please refer to the OpenReview entry for citation; the bibtex below will be updated upon acceptance.

```bibtex
@inproceedings{said2026,
  title  = {Some RAG Metrics Don't Measure Quality:
            Detecting Surface Confounds via Retrieval Invariants},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review at NeurIPS 2026 Evaluations \& Datasets Track}
}
```
