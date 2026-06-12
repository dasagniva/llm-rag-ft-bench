# Commit structure for the pre-Phase-4 corrections — `llm-rag-ft-bench`

We are committing the Task 1 (answer-extraction fix) and Task 2 (corpus realignment)
changes. There is **no prior baseline commit** capturing the Phase 3 state — confirm this
first with `git log --oneline` and tell me what you find before staging anything. If a
pre-correction snapshot already exists in history, say so and we'll adjust.

## The stance (read this before touching git)

We are committing **code and corpus changes now, results later**. That is the correct and
intended order. There is exactly one nuance that governs how you stage:

`reports/base_vs_rag.md` and its forest-plot PNG are **generated artifacts**. The version
currently in the working tree contains the OLD, broken floor-effect numbers (real numbers,
flawed setup). The rule:

- The **broken** report goes into the baseline commit as honest history — it's the
  "before" that the fixes respond to.
- **Do NOT regenerate, hand-edit, or re-run the report generator** as part of any of the
  code/corpus commits below. No commit in this sequence may write new numbers into
  `base_vs_rag.md`. New numbers only appear after the Task 3 re-baseline run actually
  produces them and logs them to MLflow. Writing the regenerated report before that run
  would mean either stale numbers in new prose or fabricated numbers — both forbidden.

In short: code changes the *inputs* to the report; nothing changes the report *file* until
a real run writes real numbers into it.

## Commit sequence (three commits now)

Stage explicitly per commit — use `git add <files>` or `git add -p`, not `git add -A` —
so generated artifacts don't ride along by accident.

**Commit 1 — baseline snapshot (the "before"):**
- The entire end-of-Phase-3 tree as it currently stands, INCLUDING the broken
  `reports/base_vs_rag.md` and its PNG with the floor-effect numbers.
- EXCLUDING every Task 1 and Task 2 file (those go in commits 2 and 3).
- Message: `chore: snapshot Phase 3 state (base-vs-RAG, pre-correction)`

**Commit 2 — Task 1, answer-extraction fix:**
- `normalize.py` (or wherever the numeric extractor lives), the `eval/metrics.py` changes,
  and the normalizer unit tests.
- The EXPERIMENT.md amendment entry for the **scoring** change (tolerance-based EM /
  numeric normalization), dated, with its one-line justification.
- The DECISIONS.md entry for the scoring change.
- Message: `fix: numeric answer extraction and tolerance-based EM for FinQA/TAT-QA`

**Commit 3 — Task 2, corpus realignment:**
- `finqa_tatqa.py`, the `build_corpus.py` / `build_index.py` changes, the tightened
  retrieval sanity test (gold **span** present in a retrieved chunk, not merely gold doc
  present), and the top-k retrieval-hit-rate diagnostic.
- The EXPERIMENT.md amendment entry for the **retrieval-corpus-definition** change (from
  "EDGAR S&P500 sample" to "FinQA test + TAT-QA dev source documents"), dated, with
  justification.
- The DECISIONS.md entry for the corpus change.
- Message: `feat: benchmark-aligned retrieval corpus (FinQA test + TAT-QA dev)`

### Handling the shared files
EXPERIMENT.md and DECISIONS.md are touched by both Task 1 and Task 2. Each gets **two
separate dated entries** — one per task — and each entry commits with its own task. These
are different hunks, so `git add -p` stages them cleanly into the right commit. If a single
paragraph straddles both changes, split the prose into two entries; do not split a commit
to accommodate a run-on entry.

## Later (NOT part of this sequence — do not do these now)

After the Task 3 re-baseline run produces real logged numbers:
- `eval: re-baseline base vs RAG on aligned corpus` — the regenerated `base_vs_rag.md` +
  PNG with real post-fix numbers.
- `docs: correct faithfulness interpretation and report framing` — the Task 4 prose
  corrections.

## Before you commit
Show me `git status` and the planned file list for each of the three commits, and wait for
my confirmation. Do not run the report generator. Do not start the re-baseline run. Small,
explicit, conventional commits only.
