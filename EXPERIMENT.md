# EXPERIMENT.md — Pre-registered experiment design

**Status: FROZEN — approved 2026-06-12. No changes after Phase 3 acceptance without a dated amendment in §10.**
**Drafted:** 2026-06-12
**Approved:** 2026-06-12
**Amendments:** *(none yet — append below with date and justification if any change is approved post-Phase-3)*

---

## 1. Research question

Does retrieval-augmented generation (RAG) improve exact-match and token-F1 accuracy of a
7–8B open-weights instruct model on financial QA, and by how much? Does QLoRA fine-tuning
on the same domain improve accuracy, and does it compound with RAG?

A null result (no significant difference) is an acceptable and publishable finding.

---

## 2. Model and configurations

**Base model:** Qwen3-8B-Instruct (fallback: Llama 3.1-8B-Instruct — see `DECISIONS.md`).
One model family used across **all** configurations. The only thing that varies between
configurations is the component under test.

| Config ID | Retrieval | Adapter | Notes |
|---|---|---|---|
| `base` | none | none | Greedy / low-temp; seeds fixed |
| `rag` | Qdrant top-*k* | none | Same decoding as base |
| `ft` | none | QLoRA adapter | Same decoding as base |
| `ft_rag` | Qdrant top-*k* | QLoRA adapter | Same decoding as base |

**Everything held constant across all configurations:**
- Model family and base weights
- Decoding: temperature = 0 (greedy), `max_new_tokens` = 128, seed = 42
- Prompt template structure (only context block is added/removed)
- Evaluation set (same frozen questions, same order)
- Inference precision: 4-bit (`bitsandbytes`)

**Retrieval hyperparameters (RAG and ft+rag only):**
- Embedding model: TBD at Phase 2 (current BGE/GTE-class sentence-transformers — confirm at build time; record in `DECISIONS.md`)
- Chunk size / overlap: TBD at Phase 1 (record in `DECISIONS.md`)
- Top-*k*: k = 5 (may be tuned on a held-out development set before the eval-set is touched; any tuning must not use eval-set questions)

---

## 3. Corpus

**Retrieval corpus:** SEC EDGAR 10-K annual filings (public, non-sensitive).
- Filing selection: TBD at Phase 1 (year range, number of companies — record in `DECISIONS.md`).
- Chunking: fixed-size with overlap (parameters in `configs/corpus.yaml`).
- Raw data `.gitignore`d; manifest (file list + SHA-256 hashes) committed.

---

## 4. Evaluation set

**Source:** FinQA and/or TAT-QA (public financial QA benchmarks).
**Target size:** N ≥ 300 questions (flag to user if source sets force fewer; minimum acceptable is 200 for adequate power — see §6).
**Sampling:** seeded random sample from the test split(s); seed = 0. Sample drawn once and frozen.
**Commit:** eval-set manifest (question IDs, source dataset, split). Full question texts committed only if licensing permits; otherwise loaded at runtime from the source files.
**Human audit:** user manually reviews ~20 randomly sampled eval items for quality and answerability before the set is frozen (Phase 1 `[HUMAN]` gate). Audit notes recorded here (§9).

**Train/eval separation:** no question ID appearing in the eval set may appear in the fine-tuning set. A pytest assertion enforces zero overlap.

---

## 5. Metrics

All metrics are computed **per question**; aggregate statistics are derived from the per-question
vectors (required for bootstrap and permutation tests).

| Metric | Type | Configurations | Notes |
|---|---|---|---|
| Exact Match (EM) | Binary (0/1) | all | Normalised: lowercase, strip punctuation, articles |
| Token F1 | Continuous [0, 1] | all | Token overlap between prediction and reference |
| Faithfulness | Continuous [0, 1] | `rag`, `ft_rag` | Fraction of answer claims entailed by retrieved context |
| Answer Relevance | Continuous [0, 1] | `rag`, `ft_rag` | Similarity of answer to the question |

**Faithfulness / Answer Relevance implementation:** prefer RAGAS if the installed API is stable
(verify at Phase 2); otherwise implement directly in `metrics.py` using a local small model as
judge. Any spend-bearing judge model requires explicit user approval; a local model is the default.
Record implementation choice in `DECISIONS.md`.

---

## 6. Statistical tests

All tests implemented in `eval/stats.py` (numpy/scipy only; no black-box stats libraries).
All tests are **two-sided** unless stated otherwise. Significance level: **α = 0.05** (family-wise,
after multiple-comparison adjustment).

### 6a. Paired bootstrap (primary inference tool)
- **Purpose:** confidence intervals on per-configuration aggregate metrics AND on pairwise differences.
- **Procedure:** resample question indices with replacement (B = 10,000 resamples, seed = 42);
  recompute both systems' aggregate metric on each resample; report 2.5th / 97.5th percentiles as 95% CI.
- Applied to: EM, Token F1 (and faithfulness / answer relevance for RAG configs).

### 6b. McNemar's exact test
- **Purpose:** hypothesis test for paired binary outcomes.
- Applied to: EM comparisons between configurations.

### 6c. Paired permutation test
- **Purpose:** hypothesis test for continuous paired outcomes.
- **Procedure:** 10,000 permutations of the per-question difference signs, seed = 42.
- Applied to: Token F1 (and faithfulness / answer relevance for RAG configs).

### 6d. Effect sizes
- **Primary:** mean paired difference with its bootstrap 95% CI.
- Reported alongside p-values for every contrast.

---

## 7. Pre-registered pairwise contrasts

All five contrasts below are pre-registered and will be tested regardless of intermediate
results. Results reported for all contrasts, including non-significant ones.

| # | Contrast | Metric(s) | Test |
|---|---|---|---|
| C1 | base vs rag | EM, Token F1 | McNemar (EM); permutation (F1) |
| C2 | base vs ft | EM, Token F1 | McNemar (EM); permutation (F1) |
| C3 | base vs ft_rag | EM, Token F1 | McNemar (EM); permutation (F1) |
| C4 | ft vs ft_rag | EM, Token F1 | McNemar (EM); permutation (F1) |
| C5 | rag vs ft_rag | EM, Token F1 | McNemar (EM); permutation (F1) |

**MVP contrasts** (Phases 1–3, before fine-tuning): C1 only (base vs rag).

---

## 8. Multiple-comparison adjustment

**Method:** Holm–Bonferroni applied across all pre-registered contrasts within each metric.
Reported p-values are adjusted; a result is declared significant if the adjusted p-value < 0.05.
C1–C5 are all adjusted together (5 contrasts per metric).

---

## 9. Human audit notes (Phase 1)

Audit date: 2026-06-12
Items reviewed: 20

Findings:
- Questions are relevant to financial QA and document-grounded reasoning.
- Mix of arithmetic, comparison, retrieval, and multi-span extraction tasks.
- Minor textual artifacts observed (typos, duplicated phrases).
- No evidence of corrupted examples or unanswerable questions.
- Several multi-span questions may penalize semantically correct formatting under Exact Match.

Decision:
Approved for frozen evaluation set.
Primary metric remains Token F1; Exact Match retained as secondary endpoint.

| Date | Auditor | Samples reviewed | Issues found | Decision |
|---|---|---|---|---|
| 2026-06-12 | user | 20 | 0 critical | Approved — see notes above |

---

## 10. Amendments

*(Any change after Phase 3 acceptance must be recorded here with date and justification.
No results may have been seen before an amendment is proposed.)*

### 2026-06-12 — Numeric exact-match extraction/normalization

**Change:** `eval/metrics.py::exact_match` now tries a numeric-aware comparison
(`eval/normalize.py::numeric_exact_match`) before falling back to normalized string
equality. It extracts the last number in the prediction (handling currency symbols,
thousands separators, spelled-out numbers, percent signs, and scale words such as
"million"/"billion") and compares it to the reference under a relative tolerance
(default 1e-3), with an exact-equality requirement when both values look like years
(integers in [1900, 2100]) to avoid spurious matches (e.g. 2021 vs 2019).

**Justification:** Without this, EM scoring penalized correct numeric answers that
differed only in formatting (e.g. "$5,735 million" vs "5735"), conflating a scoring
artifact with the model's actual ability to answer. This is a scoring-pipeline fix,
not a change to the eval set, seeds, decoding, or pre-registered contrasts. No
results were inspected before this change was specified (Task 1 of
`pre-phase4-fixes-prompt.md`); the existing base-run predictions were re-scored
under both the old and new logic and the difference reported before this amendment
was written.

### 2026-06-12 — Retrieval corpus: FinQA test + TAT-QA dev source documents

**Change:** §3's retrieval corpus is amended from "SEC EDGAR 10-K annual filings"
to: the source documents of the FinQA test split and TAT-QA dev split (the same
splits the eval set in §4 is sampled from) — 658 unique source pages/tables,
chunked into 1,221 chunks (`chunk_size=400`, `overlap=40`, same chunker as before),
indexed into a new Qdrant collection `ragbench_finqa_tatqa`. The original
EDGAR-derived `ragbench` collection (278K chunks) is retained as a separate
large-scale ingestion demo (see `DECISIONS.md`) but is no longer used by `rag`/`ft_rag`
configs.

**Justification:** The eval questions (§4) are written against FinQA/TAT-QA source
pages, which are not SEC EDGAR 10-K filings and were never in the EDGAR retrieval
corpus — so RAG configs could not retrieve the gold-supporting passage by
construction, regardless of retriever quality. A gold-span-present retrieval
diagnostic on the new corpus measures hit-rate@5 = 0.5100 (153/300) over the frozen
eval set (`reports/retrieval_diagnostics.md`). This is a corpus-construction fix
addressing a corpus/eval-set mismatch; the eval set itself (§4), seeds, decoding,
and pre-registered contrasts (§7) are unchanged.
