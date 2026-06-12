# ROADMAP — `llm-rag-ft-bench`

## Instructions for Claude Code (read this section first, every session)

You are building an LLM evaluation project: a statistically rigorous comparison of
**base** vs **retrieval-augmented (RAG)** vs **QLoRA fine-tuned** configurations of a
small open-weights model on a public corpus. The owner is a PhD statistician; the
statistical evaluation layer (`eval/stats.py`) is the centrepiece of the repo and must
be the highest-quality code in it.

**Operating rules — follow these strictly:**

1. Work phase by phase, in order. Run each phase's verification commands and show output
   before declaring it done. Full test suite + linter must be green at every phase end.
2. **Never fabricate results.** Every number in the README and in `reports/` must come
   from an actual evaluation run logged to the experiment tracker. If a comparison shows
   no significant difference, report exactly that — a null result honestly reported is
   the desired brand of this repo. Placeholder for unrun numbers:
   `TBD — pending eval run`.
3. **Never invent dataset contents.** Use only the real downloaded corpus and real
   benchmark QA pairs. If generating synthetic QA pairs is ever needed, stop and ask;
   the default plan does not require it.
4. Steps marked **[HUMAN]** require the user (GPU provisioning, spend approval, manual
   audit of eval samples, choice confirmations). Stop and ask; never simulate, stub, or
   skip them.
5. **GPU and cost discipline:** never provision or assume paid compute. All fine-tuning
   runs happen only after the user confirms hardware and current rental rates
   (rates move fast — the user must check them the day of provisioning). Everything in
   Phases 0–3 and 5 must run on CPU or free-tier hardware.
6. Model and library churn: this roadmap names model families and libraries, but check
   what is current at build time (Phase 0 **[HUMAN]** gate). If an installed library's
   API differs from what you expect (RAGAS and Evidently-style churn is common in this
   ecosystem), read its installed docs and adapt; log deviations in `DECISIONS.md`.
7. Keep orchestration thin. Do **not** introduce LangChain. Write the retrieval and
   generation loops as plain Python over the vector-store client and `transformers`.
   LlamaIndex is permitted for document ingestion utilities only, and only if it
   demonstrably saves effort — record the decision in `DECISIONS.md`.
8. Ask before adding any dependency not listed here, and before any change to the
   experiment design (eval set, metrics, statistical tests) after Phase 3 is accepted —
   changing the experiment after seeing results is exactly the malpractice this repo
   exists to avoid.
9. Small conventional commits (`feat:`, `test:`, `eval:`, `docs:`, `chore:`).

**Project intent:** hiring portfolio for AI Engineer / LLM Engineer roles, with a
secondary quant-finance signal from the corpus choice. Optimise for honest, legible,
reproducible evaluation over feature count or leaderboard-chasing.

---

## Stack (fixed — substitutions require asking)

| Concern | Tool |
|---|---|
| Python / packaging | Python ≥ 3.11, `uv`, `pyproject.toml` |
| Lint/format/tests | `ruff`, `pre-commit`, `pytest`, `pytest-cov`; `mypy` on `eval/` and `serving/` |
| Models | One small open-weights instruct family (7–8B class, e.g. Qwen or Llama line — confirm current best at Phase 0) used across ALL configurations |
| Generation | Hugging Face `transformers`; 4-bit load via `bitsandbytes` where needed |
| Fine-tuning | `peft` QLoRA (optionally `unsloth` for speed — user's call at Phase 4) |
| Embeddings | A current BGE/GTE-class sentence-transformers model |
| Vector store | Qdrant via Docker (FAISS fallback only if user requests zero-infra) |
| Eval metrics | Custom EM/F1 + faithfulness/relevance (RAGAS-style; verify RAGAS API before depending on it, else implement directly) |
| Statistics | Custom `eval/stats.py` — numpy/scipy only; no black-box stats libraries |
| Tracking | MLflow (consistent with the user's other repo) |
| Serving | FastAPI; generation via `transformers` (document vLLM as the GPU-serving option) |
| CI | GitHub Actions (CPU-only jobs; never run model inference in CI beyond a tiny stub model if needed) |

---

## Corpus (decided, pending Phase 0 confirmation)

**Primary: SEC EDGAR 10-K filings as the retrieval corpus + FinQA and/or TAT-QA as the
QA evaluation sets.** Public, non-sensitive, finance-flavoured. **Fallback** (only if
filings retrieval proves painful in Phase 1, with user sign-off): Wikipedia + Natural
Questions / HotpotQA.

Hard constraint from the user: no proprietary, clinical, biological, or otherwise
sensitive data anywhere in the repo — including in examples and tests.

---

## Target directory structure

```
llm-rag-ft-bench/
├── src/ragbench/
│   ├── __init__.py
│   ├── corpus/          # download, cleaning, chunking
│   ├── retrieval/       # embedding, Qdrant indexing, retriever
│   ├── generation/      # config-driven wrappers: base | rag | ft | ft+rag
│   ├── finetune/        # QLoRA training scripts + configs
│   ├── eval/
│   │   ├── runner.py    # runs a configuration over the eval set, logs to MLflow
│   │   ├── metrics.py   # EM, F1, faithfulness, answer relevance
│   │   └── stats.py     # paired bootstrap, permutation/McNemar, effect sizes  ← centrepiece
│   └── serving/         # FastAPI endpoint, config= parameter selects system
├── configs/             # one YAML per configuration + eval config
├── tests/
│   ├── unit/            # stats.py gets the most thorough tests in the repo
│   └── integration/
├── data/                # .gitignored raw data; committed: tiny fixtures + eval-set manifest
├── reports/             # headline comparison report + figures (committed)
├── scripts/             # build_corpus.py, build_index.py, run_eval.py, finetune.py
├── .github/workflows/ci.yaml
├── DECISIONS.md
├── pyproject.toml
└── README.md
```

---

## Phase 0 — Confirmations **[HUMAN]**

- [ ] **[HUMAN]** Confirm corpus choice (EDGAR + FinQA/TAT-QA vs fallback).
- [ ] **[HUMAN]** Confirm the model family: check what the current best ~7–8B open-weights
      instruct model is **at build time** and whether the user has any local GPU. Record
      the choice and date in `DECISIONS.md`.
- [ ] **[HUMAN]** Confirm tracker (MLflow assumed) and whether Qdrant-in-Docker is
      acceptable on the user's machine.
- [ ] Write `EXPERIMENT.md`: the pre-registered experiment design — configurations to
      compare, eval set definition and size, metrics, statistical tests, significance
      level, and multiple-comparison policy. **This file is frozen after Phase 3
      acceptance**; any later change requires explicit user approval and a dated
      amendment note. (Pre-registration is part of the repo's pitch.)

**Acceptance:** all three confirmations recorded; `EXPERIMENT.md` drafted and approved.

---

## Phase 1 — Scaffold, corpus, baseline (est. ~1 week part-time)

- [ ] `uv` project scaffold, ruff + pre-commit, pytest, CI skeleton (lint + unit tests
      only; CI never downloads models or corpora).
- [ ] `scripts/build_corpus.py`: download and clean the corpus; deterministic chunking
      (record chunk size/overlap in config). Raw data `.gitignore`d; commit a manifest
      (file list + hashes) and a tiny fixture subset for tests.
- [ ] Build the eval set from FinQA/TAT-QA: fixed, seeded sample of N questions
      (N from `EXPERIMENT.md`; target ≥ 300 for usable power — flag to user if the
      source sets force fewer). Commit the eval-set manifest (IDs, not copyrighted
      full texts, if licensing requires).
- [ ] **[HUMAN]** User manually audits ~20 sampled eval items for quality/answerability;
      record audit notes in `EXPERIMENT.md`.
- [ ] `generation/`: config-driven base-model wrapper (greedy/low-temp decoding, seeded,
      max-token caps). `eval/runner.py` + `metrics.py` (EM, token-F1): run the **base**
      configuration over the eval set, log per-question results + aggregates to MLflow.
      Per-question outputs saved as a JSONL artifact — `stats.py` will need them.

**Verification:** `uv run scripts/run_eval.py --config configs/base.yaml` completes on
the fixture set in CI-feasible time locally; a real base-config run is logged in MLflow
with per-question JSONL attached.

**Acceptance:** metrics table for the base configuration exists from a real run.

---

## Phase 2 — RAG (est. ~1 week)

- [ ] Embedding + indexing: `scripts/build_index.py` embeds chunks into Qdrant
      (Docker compose service). Index build is deterministic and idempotent.
- [ ] Retriever: top-k similarity search (k in config); plain-Python retrieval loop;
      prompt template that injects retrieved chunks with source IDs.
- [ ] RAG configuration in `generation/`, identical decoding settings to base — the
      ONLY difference between configs may be the thing being tested.
- [ ] Add faithfulness/relevance metrics for RAG outputs (RAGAS if its installed API is
      sane, otherwise direct implementation — judge model choice requires **[HUMAN]**
      sign-off if it implies API spend; a local small model judge is the no-spend default).
- [ ] Run the RAG configuration over the same frozen eval set; log to MLflow with
      per-question JSONL.
- [ ] Retrieval unit tests on fixtures (known chunk must be retrieved for a planted query).

**Acceptance:** base and RAG runs exist over the identical eval set, same seeds/decoding,
both with per-question artifacts.

---

## Phase 3 — Statistical layer (est. ~1 week) — THE CENTREPIECE

Write this module test-first. It must be readable enough to walk through in an interview.

- [ ] `eval/stats.py`, numpy/scipy only:
      - **Paired bootstrap** over *questions* (resample question indices, recompute both
        systems' aggregate metrics on each resample) → CIs on per-system metrics AND on
        pairwise **differences**. B ≥ 10,000, seeded.
      - **McNemar's exact test** for paired binary outcomes (exact match).
      - **Paired permutation test** for continuous metrics (F1, faithfulness).
      - Effect sizes (mean paired difference with CI; report alongside p-values).
      - Multiple-comparison adjustment across the pairwise contrasts per
        `EXPERIMENT.md` (Holm–Bonferroni default).
      - Pure functions: arrays in, results dataclasses out. No I/O in stats functions.
- [ ] Unit tests: identical inputs → null differences with correct coverage; planted
      known effect → detected; CI coverage sanity-checked by simulation; seeds make
      everything reproducible bit-for-bit.
- [ ] `eval/` reporting: forest-plot-style figure of metric differences with CIs
      (matplotlib, no seaborn dependency needed), and a markdown results-table generator
      that writes into `reports/`.
- [ ] Run the full base-vs-RAG analysis; commit `reports/base_vs_rag.md` + figure.
- [ ] **[HUMAN]** User reviews the statistical write-up for correctness of framing
      before it is referenced in the README.
- [ ] Freeze `EXPERIMENT.md` (rule 8).

**Acceptance:** stats module ≥ 95% test coverage; committed report with CIs and tests
for base vs RAG; user sign-off on the write-up. **This is the MVP ship point** — the
repo is publishable here while Phase 4 proceeds.

---

## Phase 4 — QLoRA fine-tuning (est. ~1–2 weeks) **[HUMAN-gated throughout]**

- [ ] **[HUMAN] Hardware gate:** user picks hardware (free tier: Kaggle/Colab; or hourly
      rental: RunPod/Vast.ai/Lambda) **after checking current prices that day**.
      Recent ballpark only — A10/4090-class ~$0.20–0.60/hr, A100 ~$1–2/hr; treat these
      as stale until verified. Target total spend < $30. A 7–8B QLoRA run fits in 24 GB
      VRAM (16 GB workable with 4-bit + short sequences).
- [ ] Build the fine-tuning set from the QA training split (never from the frozen eval
      set — write a test asserting zero ID overlap between train and eval).
- [ ] `finetune/`: QLoRA config (r, alpha, target modules, LR, epochs in YAML),
      `scripts/finetune.py`, checkpoints + adapter logged to MLflow. Script must be
      runnable as a standalone on the rented box (document the exact remote setup
      commands in `finetune/README.md`).
- [ ] **[HUMAN]** User executes the GPU run(s) (or supervises Claude Code doing so on a
      box the user provisioned) and confirms spend.
- [ ] Evaluate `ft` and `ft+rag` configurations on the frozen eval set, identical
      decoding; extend the statistical analysis to all pre-registered contrasts with the
      multiple-comparison adjustment; regenerate `reports/` (now `full_comparison.md`).

**Acceptance:** all configurations evaluated on the identical frozen eval set; final
report committed; train/eval leakage test green.

---

## Phase 5 — Serving, packaging, README (est. ~1 week)

- [ ] FastAPI `POST /ask` with `config=base|rag|ft|ft_rag` parameter; `GET /health`;
      structured errors; Pydantic v2 validation. Lazy-load models; degrade gracefully
      (clear 503 with message) for configurations whose weights aren't present locally.
- [ ] Dockerfile + compose (api + qdrant). CPU-only inference documented as slow-but-works;
      vLLM documented as the GPU path (do not make it a hard dependency).
- [ ] CI final: lint, mypy (eval, serving), unit tests, fixture-level integration tests.
- [ ] README per the spec below; verify the quickstart verbatim on a clean machine.

---

## README spec (exact order)

1. One-paragraph pitch + badges.
2. **Headline figure:** the forest plot of metric differences with CIs, at the top.
3. The honest finding in one or two sentences — whatever the data actually showed,
   including non-significant contrasts, with effect sizes and adjusted p-values.
4. 3-command quickstart:
   ```bash
   git clone <repo> && cd llm-rag-ft-bench
   docker compose up -d
   curl -X POST 'localhost:8000/ask?config=rag' -H 'Content-Type: application/json' -d @examples/question.json
   ```
5. Methodology: pre-registered design (link `EXPERIMENT.md`), eval set construction,
   paired bootstrap / McNemar / permutation testing, multiple-comparison policy. Written
   for a technical reader; this section is the differentiator — give it real space.
6. Configurations table (model, decoding, retrieval, adapter — showing everything held
   constant).
7. Cost & hardware notes (actual spend, hardware used, with the date rates were checked).
8. Limitations (eval-set size and power, single corpus, single model family, judge-model
   caveats for faithfulness metrics).
9. Reproduce-everything guide.

---

## MVP cut line

**Ship after Phase 3:** base vs RAG with the full statistical layer, no fine-tuning,
$0 GPU spend. Add a visible "Next: QLoRA fine-tuned arm (pre-registered in
EXPERIMENT.md)" roadmap note. The résumé line at MVP must not mention fine-tuning.

## Definition of done (full build)

- All pre-registered configurations evaluated on one frozen eval set; zero train/eval
  leakage (tested).
- `eval/stats.py` at ≥ 95% coverage with simulation-validated CI behaviour.
- Committed report + forest plot whose numbers exactly match MLflow runs.
- Quickstart works verbatim with Docker only (CPU path).
- `EXPERIMENT.md` frozen pre-results, with any amendments dated and justified.
- No fabricated numbers anywhere; non-significant results reported as such.
