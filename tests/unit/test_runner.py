"""Unit tests for eval/runner.py using a mock generator (no model required)."""

import pytest

from ragbench.eval.runner import EvalResult, QuestionResult, run_eval

FIXTURE_QUESTIONS = [
    {
        "id": "finqa_001",
        "question": "What was the net income in 2022?",
        "reference_answer": "15 million",
        "source_dataset": "finqa",
    },
    {
        "id": "finqa_002",
        "question": "Did revenue increase from 2021 to 2022?",
        "reference_answer": "yes",
        "source_dataset": "finqa",
    },
    {
        "id": "tatqa_003",
        "question": "What is the total assets?",
        "reference_answer": "500",
        "source_dataset": "tatqa",
    },
    {
        "id": "tatqa_004",
        "question": "What percentage of revenue was operating income?",
        "reference_answer": "12%",
        "source_dataset": "tatqa",
    },
    {
        "id": "finqa_005",
        "question": "What were total liabilities?",
        "reference_answer": "200 million",
        "source_dataset": "finqa",
    },
]


def perfect_generator(question: str) -> str:
    """Returns the reference answer for fixture questions — produces EM=1, F1=1."""
    answers = {
        "What was the net income in 2022?": "15 million",
        "Did revenue increase from 2021 to 2022?": "yes",
        "What is the total assets?": "500",
        "What percentage of revenue was operating income?": "12%",
        "What were total liabilities?": "200 million",
    }
    return answers.get(question, "unknown")


def zero_generator(question: str) -> str:
    return "zzz this is wrong zzz"


def partial_generator(question: str) -> str:
    """Returns answers with partial token overlap."""
    return "15"  # matches number tokens but not full reference


class TestRunEval:
    def test_returns_eval_result(self):
        result = run_eval(perfect_generator, FIXTURE_QUESTIONS)
        assert isinstance(result, EvalResult)

    def test_n_equals_input_length(self):
        result = run_eval(perfect_generator, FIXTURE_QUESTIONS)
        assert result.n == len(FIXTURE_QUESTIONS)

    def test_perfect_generator_mean_em_one(self):
        result = run_eval(perfect_generator, FIXTURE_QUESTIONS)
        assert result.mean_em == pytest.approx(1.0)

    def test_perfect_generator_mean_f1_one(self):
        result = run_eval(perfect_generator, FIXTURE_QUESTIONS)
        assert result.mean_f1 == pytest.approx(1.0)

    def test_zero_generator_mean_em_zero(self):
        result = run_eval(zero_generator, FIXTURE_QUESTIONS)
        assert result.mean_em == pytest.approx(0.0)

    def test_per_question_results_populated(self):
        result = run_eval(perfect_generator, FIXTURE_QUESTIONS)
        assert len(result.per_question) == len(FIXTURE_QUESTIONS)
        for qr in result.per_question:
            assert isinstance(qr, QuestionResult)
            assert qr.id
            assert qr.prediction

    def test_per_question_to_dict_keys(self):
        result = run_eval(perfect_generator, FIXTURE_QUESTIONS[:1])
        d = result.per_question[0].to_dict()
        for key in ("id", "question", "reference_answer", "prediction", "exact_match", "token_f1"):
            assert key in d

    def test_config_name_stored(self):
        result = run_eval(perfect_generator, FIXTURE_QUESTIONS, config_name="base")
        assert result.config_name == "base"

    def test_summary_keys(self):
        result = run_eval(perfect_generator, FIXTURE_QUESTIONS)
        summary = result.summary()
        for key in ("config_name", "n", "mean_em", "mean_f1"):
            assert key in summary

    def test_empty_questions_list(self):
        result = run_eval(perfect_generator, [])
        assert result.n == 0
        assert result.mean_em == pytest.approx(0.0)
        assert result.mean_f1 == pytest.approx(0.0)

    def test_metrics_in_range(self):
        result = run_eval(partial_generator, FIXTURE_QUESTIONS)
        assert 0.0 <= result.mean_em <= 1.0
        assert 0.0 <= result.mean_f1 <= 1.0

    def test_source_dataset_preserved(self):
        result = run_eval(perfect_generator, FIXTURE_QUESTIONS)
        datasets = {qr.source_dataset for qr in result.per_question}
        assert "finqa" in datasets
        assert "tatqa" in datasets
