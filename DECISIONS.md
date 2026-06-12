# DECISIONS.md

Log of non-obvious choices, library API deviations, and experiment-design decisions.
Entries are dated; newest first within each section.

---

## Phase 0 — 2026-06-12

### Model family: Qwen3-8B (fallback: Llama 3.1-8B-Instruct)
**Chosen:** `Qwen/Qwen3-8B` as the primary model across all configurations (base, rag, ft, ft+rag).
**Fallback:** `meta-llama/Llama-3.1-8B-Instruct` if Qwen3 issues arise.
**Rationale:** Qwen3 dropped the `-Instruct` suffix — `Qwen/Qwen3-8B` is the chat/instruct model (supports thinking and non-thinking modes). `Qwen/Qwen3-8B-Instruct` returns 404. Both fit comfortably in 24 GB VRAM at 4-bit precision.
**Hardware confirmed:** 2× RTX 4090 (24 GB each). QLoRA is viable on this box without external GPU rental.

### Inference precision: 4-bit (bitsandbytes)
**Chosen:** 4-bit loading via `bitsandbytes` for local inference and serving.
**Rationale:** Matches the roadmap default; fits the 4090s for both inference and QLoRA training.

### Vector store: Qdrant in Docker
**Chosen:** Qdrant via Docker Compose (no FAISS fallback needed).
**Rationale:** Hardware is not a bottleneck; this is the planned stack.

### Experiment tracker: MLflow
**Chosen:** MLflow (no change from roadmap assumption).

### Corpus: SEC EDGAR 10-K + FinQA / TAT-QA
**Chosen:** Primary corpus as roadmap-stated. Fallback to Wikipedia + NQ/HotpotQA only if EDGAR ingestion proves painful, with explicit user sign-off.

### Serving path: CPU for MVP; GPU/vLLM documented for later
**Chosen:** CPU path for Phases 0–3 (and the MVP ship point after Phase 3); GPU/vLLM documented in Phase 5 as an alternative but not a hard dependency.

---

**2026-06-12 — Qwen3 model ID:** `Qwen/Qwen3-8B-Instruct` does not exist on HuggingFace. Qwen3 ships with no separate `-Instruct` variant — the single `Qwen/Qwen3-8B` checkpoint handles both base and instruction-following modes (controlled via chat template + `enable_thinking` flag). Updated `configs/base.yaml` and `configs/rag.yaml` accordingly.

## Phase 2 — 2026-06-12

**Corpus chunk ID collision fix:** `build_corpus.py` originally used `p.stem` as the document ID. All EDGAR filings downloaded by `sec-edgar-downloader` are named `full-submission.txt`, so every file had `id = "full-submission"` and every chunk ID was `full-submission_cN`. Qdrant upserts are idempotent by ID, so 278,520 chunks collapsed to ~12,801 points (one set of chunk indices). Fixed by deriving `id = "{TICKER}_{ACCESSION}"` from the path structure (e.g. `AMZN_0001018724-23-000004`), giving 278,520 unique chunk IDs. Corpus chunks rebuilt and index rerun.

## Phase 1 completion — 2026-06-12

**Base-config eval run completed.** Results: N=300, mean EM=0.0033, mean F1=0.0387 (FinQA: EM=0.000, F1=0.000; TAT-QA: EM=0.008, F1=0.097). Low scores are expected — the base model without document context cannot answer table-grounded financial questions. MLflow artifact logged at `mlruns/1/5bf0cc161f0.../artifacts/base_results.jsonl`. Note: metrics/params were not captured in MLflow (script run outside the project venv). The per-question JSONL is intact; re-run `scripts/run_eval.py` inside `uv run` to populate MLflow metrics for the permanent record before Phase 3.

## Library API deviations

**2026-06-12 — TAT-QA split:** `tatqa_dataset_test.json` withholds all answers (competition format — 0 usable items). Using `tatqa_dataset_dev.json` (1,668 questions, full answers) as a held-out eval source instead. The FinQA test set is unaffected (1,133 questions with answers). TAT-QA scale metadata (million, thousand, billion, percent) is appended to numeric reference answers so they match the expected model output format (e.g., `"15 million"` rather than `"15"`).

**2026-06-12 — FinQA / TAT-QA dataset loading:** `datasets>=3.x` dropped support for custom loading scripts (`trust_remote_code` raises `RuntimeError`). Both `ibm/finqa` and `kasnerz/tatqa` on HuggingFace Hub still use deprecated `finqa.py` / `tatqa.py` scripts. Fix: fetch the raw JSON directly from each dataset's GitHub repo (`czyssrs/FinQA` and `NExTplusplus/TAT-QA`) via `urllib.request`. The `datasets` dependency is no longer used by `build_eval_set.py` (kept in `pyproject.toml` for potential future use by other scripts).

**2026-06-12 — sec-edgar-downloader:** Latest available is 5.1.0, not 5.4 as originally spec'd. Version constraint updated to `>=5.0`. API unchanged — `Downloader(company, email, download_dir)` with `.get(type, ticker, after=, before=)` still works as expected.

**2026-06-12 — PyTorch install:** CUDA 12.4 wheels are at `https://download.pytorch.org/whl/cu124`. Install command requires `--index-strategy unsafe-best-match` because the PyTorch index bundles `tqdm` at an older version that would otherwise shadow PyPI's current tqdm. Standard install command: `uv sync --extra dev --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match` (CPU: replace `cu124` with `cpu`).

**2026-06-12 — Faithfulness/answer relevance metrics:** Implemented as local lexical/embedding proxies rather than RAGAS or an LLM judge, to avoid API spend. Faithfulness = bigram overlap between answer and retrieved context. Answer relevance = cosine similarity between pre-computed embeddings. Both are zero-cost and fully reproducible. LLM-based judging documented as an upgrade path if the user wants richer evaluation in Phase 5.

**2026-06-12 — Embedding model:** BAAI/bge-base-en-v1.5 (768-dim, sentence-transformers). Chosen as a current BGE-class model with strong MTEB retrieval scores. Record final choice in DECISIONS.md after Phase 2 eval run.

**2026-06-12 — Qdrant + protobuf:** qdrant-client 1.12.x is incompatible with protobuf ≥6.0. Pinned `protobuf>=4.25,<6` in pyproject.toml. Integration tests (test_retrieval.py) require compatible grpcio+protobuf and are marked `@pytest.mark.integration` (excluded from CI). Run locally: `pytest tests/integration/ -m integration`.

**2026-06-12 — beautifulsoup4 Python 3.13:** bs4 ≥ 4.15 has a type-alias error at module level under Python 3.13. Worked around by lazy-importing `BeautifulSoup` inside `clean_filing_text()` function body rather than at module import time.

**2026-06-12 — uv cache corruption:** After manually running `uv pip uninstall/install`, the uv cache produced mixed-version hard-link installs for numpy and coverage. Resolution: `uv cache clean` then `uv sync`. Do NOT use `uv pip install/uninstall` on individual packages in this project — always use `uv sync`.

## Pre-Phase-4 fixes — 2026-06-12

### Task 1 — numeric answer extraction and tolerance-based EM

**Numeric exact-match scoring (`eval/normalize.py`):** Exact match previously required
normalized-string equality, which penalized correct numeric answers that differed only
in formatting (currency symbols, thousands separators, spelled-out numbers, scale words
like "million"/"billion", percent signs). `eval/metrics.py::exact_match` now tries
`numeric_exact_match` first (relative tolerance 1e-3, applied only when the reference is
itself a bare number via `is_numeric_string`), falling back to the prior string-equality
check. Year-like values (integers in [1900, 2100]) require exact equality regardless of
tolerance, since e.g. 2021 vs 2019 falls within 1e-3 relative tolerance and would
otherwise spuriously match. See `EXPERIMENT.md` §10 amendment (2026-06-12).

### Task 2 — benchmark-aligned retrieval corpus

**Retrieval corpus choice (Option A — FinQA test + TAT-QA dev source documents):**
Chosen over Option B (distractor expansion to ~3,158 docs / 4,677 chunks by mixing
in unrelated EDGAR filings as hard negatives). Option A builds the smallest corpus
that guarantees the gold-supporting page/table for every eval question is indexable
(658 docs → 1,221 chunks via the existing `chunk_documents`, `chunk_size=400`,
`overlap=40`). Option B is recorded as a possible future robustness check (does
retrieval quality degrade with realistic distractor density?) but is out of scope
for the immediate corpus/eval-set mismatch fix. See `EXPERIMENT.md` §10 amendment
(2026-06-12) for the pre-registration-compliant framing.

**EDGAR `ragbench` collection retained as ingestion demo, not part of the eval
pipeline:** The 278K-chunk EDGAR collection built in Phase 2 cannot contain the
gold-supporting passages for the FinQA/TAT-QA eval set (different source documents
entirely), so it cannot be used for the base-vs-RAG comparison. Rather than delete
278K indexed chunks / discard the EDGAR ingestion pipeline work, it is kept in
Qdrant and referenced in the repo as a demonstration of large-scale corpus
ingestion (chunking, embedding, indexing at EDGAR scale), separate from the
FinQA/TAT-QA-grounded `ragbench_finqa_tatqa` collection used by `configs/rag.yaml`.

**Gold-span-present retrieval diagnostic methodology:** For each eval question,
`scripts/build_finqa_corpus.py` records `gold_span_parts` — the exact
gold-supporting text sentences (`finqa_gold_span_parts`, via FinQA's
`qa.ann_text_rows`/`ann_table_rows`) or paragraph/table strings
(`tatqa_gold_span_parts`, via TAT-QA's `rel_paragraphs`/`answer_from`) — per
question ID. `gold_span_present` (in `ragbench/corpus/finqa_tatqa.py`) checks
whether every gold span part is a whitespace-normalized substring of the
concatenation of the top-k retrieved chunk texts. This is strictly stronger than
"gold-doc-present" (right document retrieved) — it requires the specific
row/sentence the answer depends on to be present. `scripts/retrieval_diagnostics.py`
computes hit-rate@{1,3,5} over the full frozen eval set (n=300) and logs to MLflow;
`tests/integration/test_finqa_retrieval.py` asserts hit-rate@5 ≥ 0.30 on a 30-question
seeded sample against the live `ragbench_finqa_tatqa` collection as a regression
floor (well below the measured 0.5100 to avoid sample-noise flakiness).

**2026-06-12 — Qwen3 thinking mode:** `apply_chat_template(..., enable_thinking=False)` disables Qwen3's chain-of-thought reasoning for deterministic, fast inference in the eval configs. This kwarg raises `TypeError` on other models; `generation/base.py` catches this and retries without it.
