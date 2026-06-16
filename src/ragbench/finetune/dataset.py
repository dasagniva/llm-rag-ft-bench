"""Fine-tuning dataset utilities: load JSONL examples and format for SFT.

The formatted text uses the model's own chat template, matching inference
exactly (same system prompt, same template structure). Loss is masked to
the assistant response only via train_on_responses_only in the training script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_examples(path: str | Path) -> list[dict[str, str]]:
    """Load fine-tuning examples from a JSONL file.

    Returns a list of dicts with at minimum: id, question, reference_answer.
    """
    examples: list[dict[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_for_sft(
    examples: list[dict[str, str]],
    tokenizer: Any,
    system_prompt: str,
    max_samples: int = 0,
) -> list[dict[str, str]]:
    """Apply the chat template to each example for SFTTrainer consumption.

    Args:
        examples:      List of dicts with ``question`` and ``reference_answer`` keys.
        tokenizer:     HuggingFace tokenizer with ``apply_chat_template``.
        system_prompt: System-role content; must match the inference-time system prompt.
        max_samples:   If > 0, cap the dataset at this many examples.

    Returns:
        List of ``{"text": formatted_string}`` dicts ready for SFTTrainer.
    """
    if max_samples > 0:
        examples = examples[:max_samples]

    formatted: list[dict[str, str]] = []
    for ex in examples:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["reference_answer"]},
        ]
        try:
            text: str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=False,  # Qwen3-specific; raises TypeError on others
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted.append({"text": text})

    return formatted
