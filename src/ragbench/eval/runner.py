"""Eval runner: pure orchestration over an eval set, no I/O or MLflow side-effects.

The runner returns structured results; the calling script (scripts/run_eval.py) handles
MLflow logging and artifact serialization. This keeps the runner fully testable without
any infrastructure.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from ragbench.eval.metrics import compute_metrics, compute_rag_metrics

logger = logging.getLogger(__name__)

# Generator callable: question -> answer (base/ft configs)
GeneratorFn = Callable[[str], str]
# RAG generator callable: question -> (answer, context)
RagGeneratorFn = Callable[[str], tuple[str, str]]


@dataclass
class QuestionResult:
    id: str
    question: str
    reference_answer: str
    prediction: str
    exact_match: float
    token_f1: float
    source_dataset: str = ""
    # RAG-only fields; empty string / NaN for non-RAG configs
    context: str = ""
    faithfulness: float = float("nan")
    answer_relevance: float = float("nan")

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "id": self.id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "prediction": self.prediction,
            "exact_match": self.exact_match,
            "token_f1": self.token_f1,
            "source_dataset": self.source_dataset,
        }
        if self.context:
            d["context"] = self.context
        import math

        if not math.isnan(self.faithfulness):
            d["faithfulness"] = self.faithfulness
        if not math.isnan(self.answer_relevance):
            d["answer_relevance"] = self.answer_relevance
        return d


@dataclass
class EvalResult:
    config_name: str
    per_question: list[QuestionResult] = field(default_factory=list)

    @property
    def mean_em(self) -> float:
        if not self.per_question:
            return 0.0
        return sum(r.exact_match for r in self.per_question) / len(self.per_question)

    @property
    def mean_f1(self) -> float:
        if not self.per_question:
            return 0.0
        return sum(r.token_f1 for r in self.per_question) / len(self.per_question)

    @property
    def mean_faithfulness(self) -> float | None:
        import math

        vals = [r.faithfulness for r in self.per_question if not math.isnan(r.faithfulness)]
        return sum(vals) / len(vals) if vals else None

    @property
    def mean_answer_relevance(self) -> float | None:
        import math

        vals = [r.answer_relevance for r in self.per_question if not math.isnan(r.answer_relevance)]
        return sum(vals) / len(vals) if vals else None

    @property
    def n(self) -> int:
        return len(self.per_question)

    def summary(self) -> dict[str, object]:
        d: dict[str, object] = {
            "config_name": self.config_name,
            "n": self.n,
            "mean_em": round(self.mean_em, 4),
            "mean_f1": round(self.mean_f1, 4),
        }
        if self.mean_faithfulness is not None:
            d["mean_faithfulness"] = round(self.mean_faithfulness, 4)
        if self.mean_answer_relevance is not None:
            d["mean_answer_relevance"] = round(self.mean_answer_relevance, 4)
        return d


def run_eval(
    generator: GeneratorFn,
    questions: list[dict[str, str]],
    config_name: str = "base",
    log_every: int = 10,
) -> EvalResult:
    """Run *generator* over *questions* and return per-question results with aggregates.

    Args:
        generator: Callable(question: str, context: str | None = None) -> str.
                   The runner passes context=None for base/ft configs.
        questions: List of dicts with keys: id, question, reference_answer, source_dataset.
        config_name: Identifies this run in results and logs.
        log_every: Log progress every N questions.

    Returns:
        EvalResult with per-question QuestionResult objects and aggregate properties.
    """
    result = EvalResult(config_name=config_name)

    for i, q in enumerate(questions):
        if i > 0 and i % log_every == 0:
            logger.info(
                "[%s] %d/%d — running mean EM=%.3f F1=%.3f",
                config_name,
                i,
                len(questions),
                result.mean_em,
                result.mean_f1,
            )

        prediction = generator(q["question"])
        metrics = compute_metrics(prediction, q["reference_answer"])

        result.per_question.append(
            QuestionResult(
                id=q["id"],
                question=q["question"],
                reference_answer=q["reference_answer"],
                prediction=prediction,
                exact_match=metrics["exact_match"],
                token_f1=metrics["token_f1"],
                source_dataset=q.get("source_dataset", ""),
            )
        )

    logger.info(
        "[%s] Done. n=%d  EM=%.4f  F1=%.4f",
        config_name,
        result.n,
        result.mean_em,
        result.mean_f1,
    )
    return result


def run_rag_eval(
    rag_generator: RagGeneratorFn,
    questions: list[dict[str, str]],
    config_name: str = "rag",
    log_every: int = 10,
) -> EvalResult:
    """Run a RAG generator over *questions*, computing faithfulness alongside EM/F1.

    Args:
        rag_generator: Callable(question: str) -> (answer: str, context: str).
        questions: Same format as run_eval.
        config_name: Identifies this run.
        log_every: Progress logging interval.

    Returns:
        EvalResult with faithfulness and answer_relevance populated per question.
    """
    result = EvalResult(config_name=config_name)

    for i, q in enumerate(questions):
        if i > 0 and i % log_every == 0:
            logger.info(
                "[%s] %d/%d — running mean EM=%.3f F1=%.3f Faith=%.3f",
                config_name,
                i,
                len(questions),
                result.mean_em,
                result.mean_f1,
                result.mean_faithfulness or 0.0,
            )

        prediction, context = rag_generator(q["question"])
        metrics = compute_rag_metrics(prediction, q["reference_answer"], context)

        result.per_question.append(
            QuestionResult(
                id=q["id"],
                question=q["question"],
                reference_answer=q["reference_answer"],
                prediction=prediction,
                exact_match=metrics["exact_match"],
                token_f1=metrics["token_f1"],
                source_dataset=q.get("source_dataset", ""),
                context=context,
                faithfulness=metrics["faithfulness"],
                answer_relevance=metrics.get("answer_relevance", float("nan")),
            )
        )

    logger.info(
        "[%s] Done. n=%d  EM=%.4f  F1=%.4f  Faith=%.4f",
        config_name,
        result.n,
        result.mean_em,
        result.mean_f1,
        result.mean_faithfulness or float("nan"),
    )
    return result
