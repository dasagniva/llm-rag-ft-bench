# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Statistically rigorous comparison of **base** vs **RAG** vs **QLoRA fine-tuned** configurations of a small open-weights model on SEC EDGAR 10-K filings (primary corpus), evaluated against FinQA / TAT-QA benchmarks. The repo is a hiring portfolio for AI Engineer roles; honest, reproducible evaluation is the differentiator. The statistical layer (`src/ragbench/eval/stats.py`) is the centrepiece and must be the highest-quality code in the repo.

**Confirmed model (Phase 0, 2026-06-12):** Qwen3-8B-Instruct (fallback: Llama 3.1-8B-Instruct). 4-bit inference via `bitsandbytes`. Hardware: 2× RTX 4090 — QLoRA viable locally.

Full phase-by-phase plan: `ROADMAP-repo2-llm-rag-ft-bench.md`. Phase 0 confirmations recorded in `DECISIONS.md`. Pre-registered experiment design: `EXPERIMENT.md` (frozen 2026-06-12 — do not modify without a dated amendment and explicit user approval).

## Operating rules (follow strictly)

1. **Work phase by phase.** Run each phase's verification commands and show output before declaring it done. Full test suite + linter must be green at every phase end.
2. **Never fabricate results.** Every number in `README.md` and `reports/` must come from an actual MLflow-logged run. Placeholder for unrun numbers: `TBD — pending eval run`.
3. **Never invent dataset contents.** Use only the real downloaded corpus and real benchmark QA pairs. If synthetic QA pairs are ever needed, stop and ask first.
4. **Steps marked `[HUMAN]`** require the user (GPU provisioning, spend approval, eval audits, confirmations). Stop and ask; never simulate, stub, or skip them.
5. **No paid compute without explicit user approval.** Phases 0–3 and 5 must run on CPU or free-tier hardware. Fine-tuning (Phase 4) runs only after the user confirms hardware and checks current rental rates that day.
6. **Library churn:** before depending on RAGAS, Evidently, or any rapidly-changing library, read its installed docs and adapt. Log API deviations from expectations in `DECISIONS.md`.
7. **Keep orchestration thin.** No LangChain. Write retrieval and generation loops as plain Python over the vector-store client and `transformers`. LlamaIndex is permitted for document ingestion only if it demonstrably saves effort; record the decision in `DECISIONS.md`.
8. **Ask before** adding any unlisted dependency, and before any change to the experiment design (eval set, metrics, statistical tests) after `EXPERIMENT.md` is frozen (end of Phase 3).
9. Conventional commits: `feat:`, `test:`, `eval:`, `docs:`, `chore:`.

## Stack (substitutions require asking)

| Concern | Tool |
|---|---|
| Python / packaging | Python ≥ 3.11, `uv`, `pyproject.toml` |
| Lint / format / tests | `ruff`, `pre-commit`, `pytest`, `pytest-cov`; `mypy` on `eval/` and `serving/` |
| Models | One 7–8B open-weights instruct family across ALL configurations (confirm choice at Phase 0) |
| Generation | HuggingFace `transformers`; 4-bit load via `bitsandbytes` where needed |
| Fine-tuning | `peft` QLoRA (optionally `unsloth` — user's call at Phase 4) |
| Embeddings | BGE/GTE-class `sentence-transformers` model |
| Vector store | Qdrant via Docker (FAISS only if user requests zero-infra) |
| Eval metrics | Custom EM/F1 + faithfulness/relevance (RAGAS-style; implement directly if RAGAS API is unstable) |
| Statistics | Custom `eval/stats.py` — numpy/scipy only; no black-box stats libraries |
| Experiment tracking | MLflow |
| Serving | FastAPI; `transformers` for inference; vLLM documented as GPU-serving option, not a hard dependency |
| CI | GitHub Actions — CPU-only; never run real model inference in CI (tiny stub model only if needed) |

## Target directory structure

```
src/ragbench/
├── corpus/          # download, cleaning, chunking
├── retrieval/       # embedding, Qdrant indexing, retriever
├── generation/      # config-driven wrappers: base | rag | ft | ft+rag
├── finetune/        # QLoRA training scripts + configs
└── eval/
    ├── runner.py    # runs a config over the eval set, logs to MLflow
    ├── metrics.py   # EM, F1, faithfulness, answer relevance
    └── stats.py     # paired bootstrap, McNemar, permutation, effect sizes ← centrepiece
configs/             # one YAML per configuration + eval config
tests/
├── unit/            # stats.py gets the most thorough tests in the repo
└── integration/
data/                # .gitignored raw data; committed: tiny fixtures + eval-set manifest
reports/             # committed comparison reports + figures
scripts/             # build_corpus.py, build_index.py, run_eval.py, finetune.py
```

## Common commands

```bash
# Install (GPU — RTX 4090 / CUDA 12.4; index is declared in pyproject.toml)
uv sync --extra dev --index-strategy unsafe-best-match

# Install (CPU only — for CI and testing; overrides the CUDA index in pyproject.toml)
uv sync --extra dev --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# WARNING: never use `uv pip install/uninstall` on individual packages — it corrupts the
# uv cache and produces mixed-version hard-link installs. If packages are broken, run:
# uv cache clean && rm -rf .venv && uv sync ...

# Run linter and formatter
uv run ruff check .
uv run ruff format .

# Type-check the centrepiece modules
uv run mypy src/ragbench/eval/ src/ragbench/serving/

# Run all unit tests
uv run pytest tests/unit/

# Run tests with coverage (eval/ must stay ≥ 95%)
uv run pytest tests/unit/ --cov=src/ragbench/eval --cov-report=term-missing

# Run a single test file
uv run pytest tests/unit/test_stats.py -v

# Build eval set (run once, commits data/eval_manifest.jsonl)
uv run scripts/build_eval_set.py --config configs/eval.yaml

# Download EDGAR corpus (requires --email; takes ~30–60 min first run)
uv run scripts/build_corpus.py --config configs/corpus.yaml --email you@example.com

# Build the Qdrant index (requires Docker: `docker compose up -d` first)
uv run scripts/build_index.py --chunks data/raw/chunks.jsonl --config configs/rag.yaml

# Run retrieval integration tests (excluded from CI — requires compatible grpcio/protobuf)
.venv/bin/pytest tests/integration/ -m integration -v

# Run eval for a configuration (requires GPU + downloaded model)
uv run scripts/run_eval.py --config configs/base.yaml

# Smoke-test eval pipeline on first 5 questions
uv run scripts/run_eval.py --config configs/base.yaml --eval-set tests/fixtures/sample_eval.jsonl --limit 5

# Start the API server (Phase 5)
uv run uvicorn ragbench.serving.app:app --reload
```

## `eval/stats.py` contract

Pure functions only — arrays in, result dataclasses out. No I/O inside stats functions. Required capabilities:

- **Paired bootstrap** over questions (resample question indices, recompute both systems' aggregate metrics per resample) → CIs on per-system metrics and on pairwise differences. B ≥ 10,000, seeded.
- **McNemar's exact test** for paired binary outcomes (exact match).
- **Paired permutation test** for continuous metrics (F1, faithfulness).
- Effect sizes: mean paired difference with CI, reported alongside p-values.
- **Holm–Bonferroni** multiple-comparison adjustment across pre-registered pairwise contrasts.

Tests must: verify null differences with correct CI coverage, detect a planted known effect, and be bit-for-bit reproducible via seeds.

## Key constraints

- No proprietary, clinical, biological, or otherwise sensitive data anywhere — including tests and examples.
- Train/eval split leakage must be tested explicitly (zero ID overlap assertion).
- `EXPERIMENT.md` is frozen after Phase 3 acceptance; later amendments require explicit user approval and a dated note.
- Configurations must be held constant across runs except for the one thing being tested (retrieval, adapter, etc.).
- Per-question JSONL artifacts must be logged to MLflow for every eval run — `stats.py` consumes them.

## Important files to create

- `DECISIONS.md` — log library API deviations and any substitution decisions (LlamaIndex, unsloth, judge model, etc.)
- `EXPERIMENT.md` — pre-registered experiment design (configurations, eval-set definition, metrics, statistical tests, significance level, multiple-comparison policy); frozen after Phase 3
