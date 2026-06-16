#!/usr/bin/env python
"""QLoRA fine-tuning script for Qwen3-8B on the financial QA training set.

Uses unsloth for fast 4-bit QLoRA training (2× RTX 4090, CUDA 12.6).
Saves the LoRA adapter to checkpoints/ft_adapter (or the path in finetune.yaml).
Loss is masked to the assistant's response tokens only (question/system prompt
tokens are excluded from the loss computation via train_on_responses_only).

Prerequisites:
    uv sync --extra dev --extra finetune --index-strategy unsafe-best-match
    uv run scripts/build_finetune_set.py   # builds data/finetune_train.jsonl
    docker compose up -d                    # (only needed for ft_rag eval later)

Usage:
    uv run scripts/finetune.py --config configs/finetune.yaml
    uv run scripts/finetune.py --config configs/finetune.yaml --dry-run
    uv run scripts/finetune.py --config configs/finetune.yaml --max-steps 30  # canary

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
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Cap training at N optimizer steps regardless of num_train_epochs "
        "(0 = use epochs from config). Use for canary runs.",
    )
    return p.parse_args()


def _log_gpu_memory(label: str) -> None:
    import torch

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = props.total_memory / 1e9
        logger.info(
            "GPU %d (%s) %s: %.2f GB allocated / %.2f GB reserved / %.2f GB total",
            i,
            props.name,
            label,
            allocated,
            reserved,
            total,
        )


def _checkpoint_round_trip(
    model_name: str,
    adapter_dir: Path,
    system_prompt: str,
    sample_question: str,
) -> None:
    """Load the saved adapter with plain PEFT (no unsloth) and run one inference.

    Runs in a subprocess so unsloth's process-wide monkey-patches to Qwen3Attention
    (which add apply_qkv) are absent — replicating exactly what FtGenerator does.
    The training model must be freed before calling this to have enough VRAM.
    """
    import json
    import subprocess

    logger.info(
        "Checkpoint round-trip: spawning clean subprocess (plain PEFT, no unsloth) "
        "to load adapter from %s …",
        adapter_dir,
    )

    inline = f"""
import json, sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = {json.dumps(model_name)}
adapter_dir = {json.dumps(str(adapter_dir))}
system_prompt = {json.dumps(system_prompt)}
question = {json.dumps(sample_question[:200])}

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True
)
loaded = PeftModel.from_pretrained(base, adapter_dir)
loaded.eval()

messages = [
    {{"role": "system", "content": system_prompt}},
    {{"role": "user", "content": question}},
]
try:
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
except TypeError:
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

device = next(loaded.parameters()).device
inputs = tok(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = loaded.generate(
        **inputs, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id
    )
new_tokens = out[0][inputs["input_ids"].shape[1]:]
answer = tok.decode(new_tokens, skip_special_tokens=True).strip()
print(json.dumps({{"ok": True, "answer": answer}}))
"""

    result = subprocess.run(
        [sys.executable, "-c", inline],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        tail = result.stderr[-3000:] if result.stderr else "(no stderr)"
        raise RuntimeError(f"Round-trip subprocess failed (exit {result.returncode}):\n{tail}")

    last_line = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else ""
    try:
        data = json.loads(last_line)
        answer = data.get("answer", "(empty)")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Round-trip subprocess produced unexpected output:\n{result.stdout[-1000:]}"
        ) from exc

    logger.info("Round-trip OK (plain PEFT, no unsloth)")
    logger.info("  Q: %s", sample_question[:120])
    logger.info("  A: %s", answer[:120])


def main() -> None:
    args = parse_args()
    is_canary = args.max_steps > 0

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
    if is_canary:
        output_dir = output_dir.parent / (output_dir.name + "_canary")

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
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
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

    # For canary runs: log every step and don't waste time saving mid-run checkpoints.
    logging_steps = 1 if is_canary else train_cfg.get("logging_steps", 25)
    save_strategy = "no" if is_canary else train_cfg.get("save_strategy", "epoch")

    # --- SFTTrainer ---
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        # max_steps=-1 means "use epochs"; any positive value overrides epochs.
        max_steps=args.max_steps if is_canary else -1,
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
        logging_steps=logging_steps,
        save_strategy=save_strategy,
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
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    torch.cuda.reset_peak_memory_stats()

    # --- MLflow ---
    run_name = (
        (mlflow_cfg.get("run_name", "qlora_finetune") + "_canary")
        if is_canary
        else mlflow_cfg.get("run_name", "qlora_finetune")
    )
    mlflow.set_experiment(mlflow_cfg.get("experiment", "ragbench"))
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("lora_r", lora_cfg["r"])
        mlflow.log_param("lora_alpha", lora_cfg["alpha"])
        mlflow.log_param("target_modules", ",".join(lora_cfg["target_modules"]))
        mlflow.log_param("lora_dropout", lora_cfg.get("lora_dropout", 0.0))
        mlflow.log_param("learning_rate", train_cfg.get("learning_rate"))
        mlflow.log_param("num_epochs", train_cfg.get("num_train_epochs"))
        mlflow.log_param("max_steps_override", args.max_steps)
        mlflow.log_param("per_device_batch_size", train_cfg.get("per_device_train_batch_size"))
        mlflow.log_param(
            "gradient_accumulation_steps", train_cfg.get("gradient_accumulation_steps")
        )
        mlflow.log_param("max_seq_length", max_seq_length)
        mlflow.log_param("n_train_examples", len(formatted))
        mlflow.log_param("seed", train_cfg.get("seed"))
        mlflow.log_param("dtype", dtype_name)

        logger.info("Starting training (MLflow run: %s) …", run.info.run_id)
        if is_canary:
            logger.info("CANARY MODE: %d steps, logging every step", args.max_steps)

        train_result = trainer.train()

        mlflow.log_metric("train_loss", train_result.training_loss)
        train_runtime = train_result.metrics.get("train_runtime", 0.0)
        mlflow.log_metric("train_runtime_seconds", train_runtime)
        logger.info(
            "Training complete — loss=%.4f  runtime=%.0fs",
            train_result.training_loss,
            train_runtime,
        )

        # GPU peak memory
        for i in range(torch.cuda.device_count()):
            peak_gb = torch.cuda.max_memory_allocated(i) / 1e9
            total_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(
                "GPU %d peak: %.2f GB / %.2f GB (%.0f%%)",
                i,
                peak_gb,
                total_gb,
                100 * peak_gb / total_gb,
            )
            mlflow.log_metric(f"gpu{i}_peak_memory_gb", peak_gb)

        # Save adapter + tokenizer
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        mlflow.log_artifact(str(output_dir), artifact_path="adapter")
        logger.info("Adapter saved → %s", output_dir)

        # Canary-only: reload with plain PEFT (no unsloth) in a subprocess and run
        # one inference to confirm the saved adapter is a complete, loadable checkpoint.
        if is_canary:
            import gc

            # Free the unsloth training model from VRAM before the subprocess loads
            # the model again with plain PEFT — avoids double-occupancy OOM on 24 GB cards.
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
            _checkpoint_round_trip(
                model_name=model_name,
                adapter_dir=output_dir,
                system_prompt=system_prompt,
                sample_question=examples[0]["question"],
            )

    if not is_canary:
        logger.info(
            "NEXT STEPS:\n"
            "  uv run scripts/run_eval.py --config configs/ft.yaml\n"
            "  uv run scripts/run_eval.py --config configs/ft_rag.yaml  (requires docker compose up -d)\n"
            "  uv run scripts/run_stats.py  (after all four configs are evaluated)"
        )


if __name__ == "__main__":
    main()
