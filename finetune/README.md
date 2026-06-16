# Phase 4 — QLoRA Fine-tuning

## Hardware

2× RTX 4090 (24 GB each), CUDA 12.4. Confirmed at Phase 0 (2026-06-12).
QLoRA (4-bit base + 16-bit LoRA adapters) fits comfortably in a single 4090.

## Step 0 — Install finetune extras

```bash
uv sync --extra dev --extra finetune --index-strategy unsafe-best-match
```

This pulls in `unsloth` and `trl` in addition to the standard dev dependencies.
unsloth requires the CUDA-compiled PyTorch wheels (already configured in pyproject.toml
via the pytorch-cu124 index); do NOT replace with the CPU PyTorch build.

## Step 1 — Build the fine-tuning dataset

```bash
uv run scripts/build_finetune_set.py
```

Downloads FinQA train split (~6 k examples) and TAT-QA train split (~11 k examples)
from GitHub, asserts zero overlap with the frozen eval set, and writes:

    data/finetune_train.jsonl

The fine-tuning set uses completely different dataset splits from the eval set
(eval = FinQA test + TAT-QA dev; train = FinQA train + TAT-QA train), so
zero overlap is guaranteed by construction and verified by the script.

## Step 2 — Verify setup (dry run)

```bash
uv run scripts/finetune.py --config configs/finetune.yaml --dry-run
```

Loads the model and dataset without training. Confirms GPU access, unsloth
installation, and data pipeline before committing to the full training run.

## Step 3 — Run QLoRA fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run scripts/finetune.py --config configs/finetune.yaml
```

Uses both RTX 4090s via HuggingFace Accelerate's device_map. Training takes
approximately 1–3 hours depending on the number of examples and epochs.

Key hyperparameters (see `configs/finetune.yaml` for full spec):

| Parameter | Value |
|---|---|
| LoRA r / alpha | 16 / 32 |
| Target modules | q/k/v/o/gate/up/down proj |
| Effective batch size | 32 (4 × 4 grad acc × 2 GPUs) |
| Learning rate | 2e-4 (cosine decay) |
| Epochs | 3 |
| Max seq length | 512 |
| Precision | bf16 (base quantized to 4-bit nf4) |

Outputs:
- `checkpoints/ft_adapter/` — LoRA adapter + tokenizer (NOT the full model)
- MLflow run in `ragbench` experiment with hyperparams and training loss

## Step 4 — Evaluate fine-tuned configurations

After training completes, run eval on the frozen 300-question set:

```bash
# Fine-tuned (no retrieval)
CUDA_VISIBLE_DEVICES=0 uv run scripts/run_eval.py --config configs/ft.yaml

# Fine-tuned + RAG (requires Qdrant: docker compose up -d first)
CUDA_VISIBLE_DEVICES=1 uv run scripts/run_eval.py --config configs/ft_rag.yaml
```

Both configs read the adapter from `checkpoints/ft_adapter` (set via `model.adapter_path`
in the YAML). If you saved to a different path, update the YAML accordingly.

## Step 5 — Regenerate statistical analysis

Once all four configs are evaluated (base, rag, ft, ft_rag):

```bash
uv run scripts/run_stats.py
```

This will need to be updated to add contrasts C2–C5 to `CONTRASTS` in
`scripts/run_stats.py` before running (see `EXPERIMENT.md §7`).

## Remote execution (rented GPU)

If running on a rented box (RunPod / Vast.ai / Lambda):

```bash
# On the remote box — one-time setup
git clone <repo-url> llm-rag-ft-bench
cd llm-rag-ft-bench
pip install uv
uv sync --extra finetune --index-strategy unsafe-best-match

# Copy or build the fine-tuning data
# Option A: build remotely
uv run scripts/build_finetune_set.py

# Option B: scp from local
scp data/finetune_train.jsonl user@remote:llm-rag-ft-bench/data/

# Run training
CUDA_VISIBLE_DEVICES=0 uv run scripts/finetune.py --config configs/finetune.yaml

# Copy adapter back
scp -r checkpoints/ft_adapter user@local:llm-rag-ft-bench/checkpoints/
```

Check current GPU rental rates before provisioning (CLAUDE.md rule 5).
Target total spend < $30 for a single 3-epoch QLoRA run on an A10/4090-class GPU.

## Notes

- The adapter is stored as PEFT delta weights only (~100 MB), not the full model.
- At inference time, `FtGenerator` loads the base model (4-bit) + PEFT adapter
  without merging, preserving quantization. See `src/ragbench/generation/ft.py`.
- `checkpoints/` is gitignored. Back up the adapter separately if needed.
