#!/usr/bin/env python
"""QLoRA fine-tuning script for Qwen3-8B on the financial QA training set.

Uses unsloth for fast 4-bit QLoRA training (2× RTX 4090, CUDA 12.4).
Saves the LoRA adapter to checkpoints/ft_adapter (or the path in finetune.yaml).
Loss is masked to the assistant's response tokens only (question/system prompt
tokens are excluded from the loss computation via train_on_responses_only).

Prerequisites:
    uv sync --extra dev --extra finetune --index-strategy unsafe-best-match
    uv run scripts/build_finetune_set.py   # builds data/finetune_train.jsonl
    docker compose up -d                    # (only needed for ft_rag eval later)

Usage:
    uv run scripts/finetune.py --config configs/finetune.yaml
    uv run scripts/finetune.py --config configs/finetune.yaml --dry-run  # load-only check

Outputs:
    checkpoints/ft_adapter/   — LoRA adapter weights + tokenizer
    MLflow run in the 'ragbench' experiment with hyperparams and train loss
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for ragbench")
    p.add_argument("--config", default="configs/finetune.yaml", help="YAML config file")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model + dataset only; skip training (for setup verification)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- Check unsloth + trl are installed ---
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
    except ImportError:
        logger.error(
            "unsloth not installed. Run:\n"
            "  uv sync --extra dev --extra finetune --index-strategy unsafe-best-match"
        )
        sys.exit(1)

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        logger.error(
            "trl not installed. Run: uv sync --extra finetune --index-strategy unsafe-best-match"
        )
        sys.exit(1)

    import mlflow
    import torch
    from datasets import Dataset

    from ragbench.finetune.dataset import format_for_sft, load_examples

    # --- Config sections ---
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    mlflow_cfg = cfg.get("mlflow", {})
    system_prompt: str = cfg.get("system_prompt", "")

    model_name: str = model_cfg["name"]
    max_seq_length: int = model_cfg.get("max_seq_length", 512)
    output_dir = Path(train_cfg["output_dir"])

    dtype_name = model_cfg.get("dtype", "bfloat16")
    torch_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16

    # --- Load model via unsloth ---
    logger.info(
        "Loading model %s (max_seq_length=%d, dtype=%s)", model_name, max_seq_length, dtype_name
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )

    # --- Apply LoRA ---
    logger.info("Applying LoRA: r=%d, alpha=%d", lora_cfg["r"], lora_cfg["alpha"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing="unsloth",
        random_state=train_cfg.get("seed", 42),
        use_rslora=False,
        loftq_config=None,
    )

    # --- Dataset ---
    train_path = Path(data_cfg["train_path"])
    if not train_path.exists():
        logger.error(
            "Fine-tuning set not found: %s — run scripts/build_finetune_set.py first", train_path
        )
        sys.exit(1)

    examples = load_examples(train_path)
    max_samples: int = data_cfg.get("max_train_samples", 0)
    formatted = format_for_sft(
        examples, tokenizer, system_prompt=system_prompt, max_samples=max_samples
    )
    logger.info("Training examples: %d", len(formatted))

    if args.dry_run:
        logger.info("--dry-run: model and dataset loaded successfully — skipping training.")
        return

    hf_dataset = Dataset.from_list(formatted)

    # --- SFTTrainer ---
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        seed=train_cfg.get("seed", 42),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        logging_steps=train_cfg.get("logging_steps", 25),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=train_cfg.get("save_total_limit", 1),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        args=sft_config,
    )

    # Mask loss on system/user tokens — train on assistant responses only.
    # instruction_part / response_part are the ChatML tokens Qwen3 uses.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # --- MLflow ---
    mlflow.set_experiment(mlflow_cfg.get("experiment", "ragbench"))
    with mlflow.start_run(run_name=mlflow_cfg.get("run_name", "qlora_finetune")) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("lora_r", lora_cfg["r"])
        mlflow.log_param("lora_alpha", lora_cfg["alpha"])
        mlflow.log_param("target_modules", ",".join(lora_cfg["target_modules"]))
        mlflow.log_param("lora_dropout", lora_cfg.get("lora_dropout", 0.05))
        mlflow.log_param("learning_rate", train_cfg.get("learning_rate"))
        mlflow.log_param("num_epochs", train_cfg.get("num_train_epochs"))
        mlflow.log_param("per_device_batch_size", train_cfg.get("per_device_train_batch_size"))
        mlflow.log_param(
            "gradient_accumulation_steps", train_cfg.get("gradient_accumulation_steps")
        )
        mlflow.log_param("max_seq_length", max_seq_length)
        mlflow.log_param("n_train_examples", len(formatted))
        mlflow.log_param("seed", train_cfg.get("seed"))
        mlflow.log_param("dtype", dtype_name)

        logger.info("Starting training (MLflow run: %s) …", run.info.run_id)
        train_result = trainer.train()

        mlflow.log_metric("train_loss", train_result.training_loss)
        train_runtime = train_result.metrics.get("train_runtime", 0.0)
        mlflow.log_metric("train_runtime_seconds", train_runtime)
        logger.info(
            "Training complete — loss=%.4f  runtime=%.0fs",
            train_result.training_loss,
            train_runtime,
        )

        # Save adapter + tokenizer
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        mlflow.log_artifact(str(output_dir), artifact_path="adapter")
        logger.info("Adapter saved → %s", output_dir)

    logger.info(
        "NEXT STEPS:\n"
        "  uv run scripts/run_eval.py --config configs/ft.yaml\n"
        "  uv run scripts/run_eval.py --config configs/ft_rag.yaml  (requires docker compose up -d)\n"
        "  uv run scripts/run_stats.py  (after all four configs are evaluated)"
    )


if __name__ == "__main__":
    main()
