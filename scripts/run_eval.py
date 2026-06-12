#!/usr/bin/env python
"""Run a configuration over the frozen eval set and log results to MLflow.

Usage:
    uv run scripts/run_eval.py --config configs/base.yaml

Outputs:
    - MLflow run with params, mean EM / mean F1 metrics
    - reports/{config_name}_results.jsonl  (per-question, matches EXPERIMENT.md §5)
    - Printed summary table

The per-question JSONL is the input to eval/stats.py in Phase 3.
Never run with --config pointing to a config whose results have already been used
for comparison (that would change the experiment after seeing results).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run eval for a given configuration")
    p.add_argument("--config", required=True, help="Path to a YAML config (e.g. configs/base.yaml)")
    p.add_argument(
        "--eval-set",
        default="",
        help="Override eval_set_path from config",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Override output directory for per-question JSONL",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run on first N questions only (for development/smoke-testing)",
    )
    return p.parse_args()


def load_questions(path: str) -> list[dict[str, str]]:
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _make_gen_config(cfg: dict):  # type: ignore[return]
    from ragbench.generation.config import GenerationConfig

    model_cfg = cfg.get("model", {})
    return GenerationConfig(
        model_name=model_cfg.get("name", "Qwen/Qwen3-8B-Instruct"),
        max_new_tokens=model_cfg.get("max_new_tokens", 128),
        seed=model_cfg.get("seed", 42),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        system_prompt=cfg.get("system_prompt", ""),
        use_context=model_cfg.get("use_context", False),
        config_name=cfg.get("config_name", "base"),
    )


def build_base_generator(cfg: dict):  # type: ignore[return]
    from ragbench.generation.base import BaseGenerator

    generator = BaseGenerator(_make_gen_config(cfg))
    return generator.generate, False  # (callable, is_rag)


def build_rag_generator(cfg: dict):  # type: ignore[return]
    from qdrant_client import QdrantClient

    from ragbench.generation.rag import RagGenerator
    from ragbench.retrieval.embedder import Embedder
    from ragbench.retrieval.retriever import Retriever

    retrieval_cfg = cfg.get("retrieval", {})
    embedder = Embedder(
        model_name=retrieval_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5"),
    )
    client = QdrantClient(url=retrieval_cfg.get("qdrant_url", "http://localhost:6333"))
    retriever = Retriever(
        client=client,
        embedder=embedder,
        collection=retrieval_cfg.get("collection", "ragbench"),
        top_k=retrieval_cfg.get("top_k", 5),
    )
    generator = RagGenerator(_make_gen_config(cfg), retriever=retriever)
    return generator.generate, True  # (callable, is_rag)


def build_generator(cfg: dict):  # type: ignore[return]
    """Return (callable, is_rag) for the config type."""
    config_name = cfg.get("config_name", "base")
    if config_name in ("base", "ft"):
        return build_base_generator(cfg)
    elif config_name in ("rag", "ft_rag"):
        return build_rag_generator(cfg)
    else:
        raise NotImplementedError(f"Unknown config '{config_name}'")


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    config_name = cfg.get("config_name", "base")
    eval_cfg = cfg.get("eval", {})
    eval_set_path = args.eval_set or eval_cfg.get("eval_set_path", "data/eval_manifest.jsonl")
    output_dir = Path(args.output_dir or eval_cfg.get("output_dir", "reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(eval_set_path)
    if args.limit > 0:
        questions = questions[: args.limit]
    logger.info("Eval set: %d questions from %s", len(questions), eval_set_path)

    generator_fn, is_rag = build_generator(cfg)

    import mlflow

    from ragbench.eval.runner import run_eval, run_rag_eval

    mlflow_exp = eval_cfg.get("mlflow_experiment", "ragbench")
    mlflow_run_name = eval_cfg.get("mlflow_run_name", "") or config_name

    mlflow.set_experiment(mlflow_exp)
    with mlflow.start_run(run_name=mlflow_run_name):
        model_cfg = cfg.get("model", {})
        mlflow.log_param("config_name", config_name)
        mlflow.log_param("model_name", model_cfg.get("name", ""))
        mlflow.log_param("seed", model_cfg.get("seed", 42))
        mlflow.log_param("max_new_tokens", model_cfg.get("max_new_tokens", 128))
        mlflow.log_param("load_in_4bit", model_cfg.get("load_in_4bit", True))
        mlflow.log_param("eval_set", eval_set_path)
        mlflow.log_param("n_questions", len(questions))
        if is_rag:
            retrieval_cfg = cfg.get("retrieval", {})
            mlflow.log_param("embedding_model", retrieval_cfg.get("embedding_model", ""))
            mlflow.log_param("top_k", retrieval_cfg.get("top_k", 5))

        if is_rag:
            result = run_rag_eval(generator_fn, questions, config_name=config_name)
        else:
            result = run_eval(generator_fn, questions, config_name=config_name)

        mlflow.log_metric("mean_em", result.mean_em)
        mlflow.log_metric("mean_f1", result.mean_f1)
        if result.mean_faithfulness is not None:
            mlflow.log_metric("mean_faithfulness", result.mean_faithfulness)
        if result.mean_answer_relevance is not None:
            mlflow.log_metric("mean_answer_relevance", result.mean_answer_relevance)

        artifact_path = output_dir / f"{config_name}_results.jsonl"
        with open(artifact_path, "w") as f:
            for qr in result.per_question:
                f.write(json.dumps(qr.to_dict()) + "\n")

        mlflow.log_artifact(str(artifact_path))
        logger.info("Artifact saved: %s", artifact_path)

    summary = result.summary()
    print("\n" + "=" * 50)
    print(f"  Config:   {summary['config_name']}")
    print(f"  N:        {summary['n']}")
    print(f"  Mean EM:  {summary['mean_em']:.4f}")
    print(f"  Mean F1:  {summary['mean_f1']:.4f}")
    if "mean_faithfulness" in summary:
        print(f"  Mean Faithfulness:      {summary['mean_faithfulness']:.4f}")
    if "mean_answer_relevance" in summary:
        print(f"  Mean Answer Relevance:  {summary['mean_answer_relevance']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
