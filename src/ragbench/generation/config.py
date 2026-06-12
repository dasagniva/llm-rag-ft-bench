"""Configuration dataclasses for generation.

A config is loaded from a YAML file and passed to the appropriate generator class.
All fields that affect output (model_name, seed, decoding) are recorded in MLflow params.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    model_name: str
    # Decoding — greedy by default; do_sample stays False so temperature is unused
    max_new_tokens: int = 128
    seed: int = 42
    # 4-bit is the default for local inference on RTX 4090; set False for CPU-only runs
    load_in_4bit: bool = True
    # Prompt template; {question} is substituted before chat-template wrapping
    system_prompt: str = (
        "You are a financial analyst. Answer the question concisely and precisely. "
        "If the answer is a number, provide only the number with its unit."
    )
    # Whether to include retrieved context chunks in the prompt (set True for RAG configs)
    use_context: bool = False
    # Identifies this config in MLflow and output file names
    config_name: str = "base"


@dataclass
class EvalConfig:
    eval_set_path: str = "data/eval_manifest.jsonl"
    output_dir: str = "reports"
    mlflow_experiment: str = "ragbench"
    mlflow_run_name: str = ""  # defaults to config_name if empty


@dataclass
class CorpusConfig:
    raw_dir: str = "data/raw/edgar"
    manifest_path: str = "data/corpus_manifest.json"
    chunk_size: int = 400
    chunk_overlap: int = 40
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=lambda: [2020, 2021, 2022, 2023])
    edgar_email: str = ""
