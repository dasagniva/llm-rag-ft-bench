"""Fine-tuned configuration generator: base model + QLoRA adapter, no retrieval.

Loads the base model in 4-bit precision (bitsandbytes) and overlays the
QLoRA adapter from a local checkpoint directory via PEFT. The adapter is
NOT merged at eval time — it stays as a PEFT overlay to preserve 4-bit
quantization and minimise VRAM usage on the RTX 4090s.

Inherits _build_prompt and generate from BaseGenerator unchanged so decoding
parameters are identical to the base config (EXPERIMENT.md §2 requirement).
"""

from __future__ import annotations

import logging
from typing import Any

from ragbench.generation.base import BaseGenerator
from ragbench.generation.config import GenerationConfig

logger = logging.getLogger(__name__)


class FtGenerator(BaseGenerator):
    """BaseGenerator variant that loads a QLoRA adapter on top of the base model."""

    def __init__(
        self,
        config: GenerationConfig,
        adapter_path: str,
        _model: Any = None,
        _tokenizer: Any = None,
    ) -> None:
        self.config = config
        self.adapter_path = adapter_path
        if _model is not None and _tokenizer is not None:
            self.model = _model
            self.tokenizer = _tokenizer
            self._device = next(_model.parameters()).device
        else:
            self._load_model_with_adapter()

    def _load_model_with_adapter(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(
            "Loading model: %s (4bit=%s) + adapter: %s",
            self.config.model_name,
            self.config.load_in_4bit,
            self.adapter_path,
        )

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
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()
        self._device = next(self.model.parameters()).device
        logger.info("Model + adapter loaded on %s", self._device)
