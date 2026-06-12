"""RAG configuration generator: retrieves context chunks before generation.

Extends BaseGenerator with a Retriever. The only difference from base is that
retrieved context is injected into the prompt. All decoding parameters are identical
to enforce the experiment's single-variable-at-a-time constraint.
"""

from __future__ import annotations

from ragbench.generation.base import BaseGenerator
from ragbench.generation.config import GenerationConfig
from ragbench.retrieval.retriever import Retriever


class RagGenerator:
    """Retrieves top-k chunks for each question, then generates via BaseGenerator."""

    def __init__(self, config: GenerationConfig, retriever: Retriever) -> None:
        self.retriever = retriever
        self._base = BaseGenerator(config)

    def generate(self, question: str) -> tuple[str, str]:
        """Generate an answer with retrieved context.

        Returns:
            (answer, context) tuple so the caller can log both and compute
            faithfulness against the context.
        """
        chunks = self.retriever.retrieve(question)
        context = self.retriever.format_context(chunks)
        answer = self._base.generate(question, context=context)
        return answer, context

    def generate_answer(self, question: str) -> str:
        """Convenience wrapper returning only the answer string (for run_eval compatibility)."""
        answer, _ = self.generate(question)
        return answer
