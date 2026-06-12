"""Base-configuration generator: no retrieval, no adapter.

The generator is config-driven and fully reproducible:
- Greedy decoding (do_sample=False) — temperature is irrelevant but documented as 0.
- Fixed seed applied before every forward pass.
- Chat-template prompt so the same template works for Qwen3 and Llama 3.1.

For testing without GPU: pass _model and _tokenizer to bypass weight loading.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ragbench.generation.config import GenerationConfig

logger = logging.getLogger(__name__)


class BaseGenerator:
    """Thin wrapper around a HuggingFace CausalLM for deterministic greedy generation."""

    def __init__(
        self,
        config: GenerationConfig,
        _model: Any = None,
        _tokenizer: Any = None,
    ) -> None:
        self.config = config
        if _model is not None and _tokenizer is not None:
            self.model = _model
            self.tokenizer = _tokenizer
            self._device = next(_model.parameters()).device
        else:
            self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("Loading model: %s (4bit=%s)", self.config.model_name, self.config.load_in_4bit)

        bnb_config = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self._device = next(self.model.parameters()).device
        logger.info("Model loaded on %s", self._device)

    def _build_prompt(self, question: str, context: str | None = None) -> str:
        user_content = question
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # apply_chat_template handles Qwen3, Llama 3.1, and most HF chat models
        # Qwen3 thinking mode disabled via enable_thinking=False for determinism
        try:
            return self.tokenizer.apply_chat_template(  # type: ignore[no-any-return]
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3-specific; silently ignored by other models
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(  # type: ignore[no-any-return]
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(self, question: str, context: str | None = None) -> str:
        """Generate an answer for *question*, optionally injecting *context*.

        Args:
            question: The eval question string.
            context: Retrieved chunks (RAG configs only); None for base/ft configs.

        Returns:
            Decoded answer string (new tokens only, special tokens stripped).
        """
        prompt = self._build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)

        torch.manual_seed(self.config.seed)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
