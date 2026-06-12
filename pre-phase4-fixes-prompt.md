# Pre-Phase-4 corrective task — `llm-rag-ft-bench`

**Context:** We are at the end of Phase 3. The base-vs-RAG report
(`reports/base_vs_rag.md`) is internally consistent arithmetically, but the experiment
itself is broken: both systems score near zero (exact match ≈ 0.3% / 0%, token-F1 ≈ 4%),
which is a performance floor, not an informative result. **Do NOT proceed to Phase 4
(QLoRA fine-tuning) until the tasks below are done and a re-baseline produces
non-degenerate numbers.** Fine-tuning on top of a broken eval would waste the GPU run.

Work through these in order. After each numbered task, run the test suite + linter and
show me the output. Stop at any **[STOP]** marker and wait for my confirmation. Do not
fabricate any metric — every number that lands in a report or README must come from an
actual logged run.

---

## Task 1 — Fix answer extraction / normalization (most likely root cause)

The EM≈0 / F1≈4% signature strongly suggests we are comparing free-text generations
against short structured gold answers without normalizing (e.g. model says "approximately
$4.2 million", gold is "4.2").

- Inspect `eval/metrics.py` and the prediction JSONL artifacts from the base run. Show me
  5–10 actual (prediction, gold) pairs so we can confirm the failure mode before changing
  code.
- Implement a dataset-appropriate answer extractor/normalizer for FinQA / TAT-QA:
  - strip currency symbols, units, commas, and surrounding prose;
  - handle scale words ("million", "billion", "thousand", "%");
  - parse the final numeric answer from a free-text generation;
  - for Exact Match on numerics, apply a relative tolerance (default 1e-3, configurable)
    rather than string equality.
- For FinQA specifically, check whether the dataset ships gold programs and consider
  reporting **program/execution accuracy** in addition to EM/F1. If you add it, make it a
  separate metric, not a replacement.
- Unit-test the normalizer on the planted edge cases above (units, scale words, currency,
  tolerance boundary). These tests must pass before moving on.

**[STOP]** Show me the before/after metrics on the existing base predictions (re-scored
with the new extractor — no model re-run needed for this check). I want to confirm base
F1 comes off the floor before we spend anything on re-running.

---

## Task 2 — Fix the corpus / eval-set mismatch

The current interpretation correctly notes that EDGAR provides general background text
while the questions need specific financial tables — meaning the answers are not in the
retrieval corpus, so RAG cannot help by construction. Fix the alignment:

- **Default fix:** build the retrieval corpus from the FinQA / TAT-QA *source documents*
  (the filings/tables the questions were written against), so retrieved context can
  actually contain the answers. Update `scripts/build_corpus.py` and
  `scripts/build_index.py` accordingly.
- Keep deterministic, seeded chunking; record chunk size/overlap in config.
- Add a retrieval sanity test: for a sample of questions, assert the gold-supporting
  passage is retrievable in top-k.

**[STOP]** Before re-running anything, tell me: (a) which corpus you'll index and why,
(b) the resulting corpus/chunk counts, and (c) whether you recommend keeping any EDGAR
framing for the quant story or dropping it. Wait for my go-ahead.

---

## Task 3 — Re-baseline and re-run base vs RAG

Only after Tasks 1–2 are confirmed:

- Re-run the **base** and **rag** configurations over the frozen eval set, identical
  decoding and seeds, logging per-question JSONL to MLflow as before.
- Do NOT change the eval set, decoding settings, seeds, or `EXPERIMENT.md` contrasts.
  These were pre-registered; the only legitimate changes are the extractor (scoring) and
  the corpus (retrieval input). Note both changes as dated amendments in `EXPERIMENT.md`
  with a one-line justification each.
- Regenerate the statistical analysis and `reports/base_vs_rag.md`.

---

## Task 4 — Fix the report's statistical claims and prose

Independent of the re-run, the current report has framing errors. Correct all of these in
the report generator / template so they're right in the regenerated output:

- **Faithfulness direction:** 0.22 on a 0–1 scale is LOW and indicates answers are largely
  ungrounded in retrieved context — the opposite of the current "context IS being used"
  claim. Fix the interpretation, give faithfulness its own table row (RAG-only, labeled as
  such), and report it with a CI, not as a bare point estimate in prose.
- **Drop the "narrow CIs rule out meaningful effects" framing.** The token-F1 Δ CI is
  ±~33% in relative terms against the base rate; it is not narrow. If results are still
  null after re-baselining, state instead that the experiment now has adequate sensitivity
  (or quantify the minimum detectable effect) rather than claiming effects are ruled out.
- **State N** (eval-set size) in the methodology section and the table caption.
- **Flag McNemar's power:** if the number of discordant pairs is tiny, note explicitly
  that the test is uninformative there rather than presenting p=1.0 as evidence of
  equivalence.
- **Holm note:** state that the correction is currently a no-op (one contrast per metric)
  and that it will bind in Phase 4 across the three pairwise contrasts (base/rag/ft/ft+rag).
- **Casing:** make `base`/`rag` vs `BASE`/`RAG` consistent throughout (this becomes README
  material).
- If the floor effect or any degenerate CI (e.g. [0.0000, 0.0000]) persists after the
  re-run, the report must say plainly that the system is at/near the performance floor and
  treat that as a measurement limitation, not a clean null.

---

## Task 5 — Decision gate **[STOP]**

Summarize for me, in a short note:

- base and RAG metrics after the fixes;
- whether base is now off the performance floor;
- whether the base-vs-RAG difference is significant, and the effect size + CI;
- your recommendation: proceed to Phase 4, or iterate further.

If base is still on the floor, we debug further — we do NOT start fine-tuning. If base is
healthy and the result is a genuine, non-degenerate null, that is a publishable finding
and we proceed to Phase 4 as planned. Wait for my decision before touching
`finetune/` or any GPU.

---

**Constraints reminder:** no fabricated numbers; don't alter the pre-registered eval set,
seeds, decoding, or contrasts; log amendments in `EXPERIMENT.md` with dates; keep
orchestration thin (no LangChain); small conventional commits per task.
