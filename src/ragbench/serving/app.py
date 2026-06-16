"""FastAPI serving app for ragbench.

Exposes:
    GET  /health          — liveness probe
    POST /ask?config=X    — answer a question with one of the four configurations

Models are lazy-loaded on the first request per config and then cached for the
lifetime of the process.  If weights are missing the endpoint returns a 503 with
a plain-language explanation rather than a 500 stack trace.

Environment variables (all optional):
    RAGBENCH_CONFIG_DIR   path to the directory containing base.yaml etc.
                          (default: configs/ relative to the working directory)
    QDRANT_URL            overrides the qdrant_url in rag.yaml / ft_rag.yaml
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ragbench.generation.config import GenerationConfig

logger = logging.getLogger(__name__)

CONFIG_NAMES = ["base", "rag", "ft", "ft_rag"]
ConfigName = Literal["base", "rag", "ft", "ft_rag"]

app = FastAPI(
    title="ragbench",
    version="0.1.0",
    description=(
        "Serves four configurations of Qwen3-8B on financial QA: base · rag · ft (QLoRA) · ft_rag."
    ),
)

# ---------------------------------------------------------------------------
# State — lazy-loaded generators, one per config
# ---------------------------------------------------------------------------

# Callable that accepts (question: str) -> str.
# Populated on first request; tests may pre-populate to skip model loading.
_generators: dict[str, Callable[[str], str]] = {}
_load_errors: dict[str, str] = {}  # config → human-readable error if load failed
_locks: dict[str, threading.Lock] = {c: threading.Lock() for c in CONFIG_NAMES}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=4096, examples=["What was the revenue in 2022?"]
    )


class AskResponse(BaseModel):
    answer: str
    config: ConfigName


class HealthResponse(BaseModel):
    status: str
    loaded_configs: list[str]


# ---------------------------------------------------------------------------
# Generator factory
# ---------------------------------------------------------------------------


def _config_dir() -> Path:
    return Path(os.environ.get("RAGBENCH_CONFIG_DIR", "configs"))


def _load_yaml(config_name: str) -> dict[str, Any]:
    path = _config_dir() / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    # Allow QDRANT_URL env var to override the value in YAML
    qdrant_override = os.environ.get("QDRANT_URL")
    if qdrant_override and "retrieval" in cfg:
        cfg["retrieval"]["qdrant_url"] = qdrant_override
    return cfg


def _build_generator(config_name: str) -> Callable[[str], str]:
    """Build and return a (question: str) -> str callable for *config_name*.

    Raises ServiceUnavailableError (subclass of HTTPException) with a 503 status
    and a plain-language message if a prerequisite is missing.
    """
    cfg = _load_yaml(config_name)

    if config_name in ("ft", "ft_rag"):
        adapter_path = cfg.get("model", {}).get("adapter_path", "checkpoints/ft_adapter")
        if not Path(adapter_path).exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"QLoRA adapter not found at '{adapter_path}'. "
                    "Run `uv run scripts/finetune.py --config configs/finetune.yaml` first."
                ),
            )

    # Import generation classes inside the function so that the module can be
    # imported in CPU-only / test environments without heavy GPU dependencies.
    if config_name == "base":
        from ragbench.generation.base import BaseGenerator

        gen = BaseGenerator(_make_gen_config(cfg))
        return gen.generate  # BaseGenerator.generate(question, context=None) -> str

    if config_name == "ft":
        from ragbench.generation.ft import FtGenerator

        adapter_path = cfg.get("model", {}).get("adapter_path", "checkpoints/ft_adapter")
        gen = FtGenerator(_make_gen_config(cfg), adapter_path=adapter_path)
        return gen.generate  # FtGenerator inherits BaseGenerator.generate

    if config_name in ("rag", "ft_rag"):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise HTTPException(
                status_code=503, detail=f"qdrant-client not installed: {exc}"
            ) from exc

        from ragbench.generation.rag import RagGenerator
        from ragbench.retrieval.embedder import Embedder
        from ragbench.retrieval.retriever import Retriever

        retrieval_cfg = cfg.get("retrieval", {})
        qdrant_url = retrieval_cfg.get("qdrant_url", "http://localhost:6333")

        try:
            client = QdrantClient(url=qdrant_url, timeout=5)
            client.get_collections()  # fail fast if Qdrant is unreachable
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Qdrant is not reachable at {qdrant_url}: {exc}. Run `docker compose up -d`.",
            ) from exc

        embedder = Embedder(
            model_name=retrieval_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
        )
        retriever = Retriever(
            client=client,
            embedder=embedder,
            collection=retrieval_cfg.get("collection", "ragbench_finqa_tatqa"),
            top_k=retrieval_cfg.get("top_k", 5),
        )

        if config_name == "rag":
            gen_rag = RagGenerator(_make_gen_config(cfg), retriever=retriever)
            # generate_answer returns str; generate returns (answer, context) tuple
            return gen_rag.generate_answer

        # ft_rag
        from ragbench.generation.ft import FtGenerator

        adapter_path = cfg.get("model", {}).get("adapter_path", "checkpoints/ft_adapter")
        ft_gen = FtGenerator(_make_gen_config(cfg), adapter_path=adapter_path)
        gen_ft_rag = RagGenerator(
            _make_gen_config(cfg), retriever=retriever, _base_generator=ft_gen
        )
        return gen_ft_rag.generate_answer

    raise NotImplementedError(f"Unknown config '{config_name}'")


def _make_gen_config(cfg: dict[str, Any]) -> GenerationConfig:
    from ragbench.generation.config import GenerationConfig

    model_cfg = cfg.get("model", {})
    return GenerationConfig(
        model_name=model_cfg.get("name", "Qwen/Qwen3-8B"),
        max_new_tokens=model_cfg.get("max_new_tokens", 128),
        seed=model_cfg.get("seed", 42),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        system_prompt=cfg.get("system_prompt", ""),
        use_context=model_cfg.get("use_context", False),
        config_name=cfg.get("config_name", "base"),
    )


# ---------------------------------------------------------------------------
# Lazy-load helper
# ---------------------------------------------------------------------------


def _get_generator(config_name: str) -> Callable[[str], str]:
    """Return the cached generator, loading it on the first call (thread-safe)."""
    if config_name in _generators:
        return _generators[config_name]

    with _locks[config_name]:
        # Double-check after acquiring the lock
        if config_name in _generators:
            return _generators[config_name]
        if config_name in _load_errors:
            raise HTTPException(status_code=503, detail=_load_errors[config_name])

        logger.info("Loading generator for config '%s' …", config_name)
        try:
            gen = _build_generator(config_name)
        except HTTPException:
            raise
        except Exception as exc:
            msg = f"Failed to load config '{config_name}': {exc}"
            _load_errors[config_name] = msg
            logger.exception(msg)
            raise HTTPException(status_code=503, detail=msg) from exc

        _generators[config_name] = gen
        logger.info("Generator for '%s' ready.", config_name)
        return gen


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", loaded_configs=sorted(_generators.keys()))


@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    config: Annotated[
        ConfigName, Query(description="Which configuration to use for inference")
    ] = "rag",
) -> AskResponse:
    gen = _get_generator(config)
    try:
        answer = gen(body.question)
    except Exception as exc:
        logger.exception("Generation failed for config '%s'", config)
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}") from exc
    return AskResponse(answer=answer, config=config)
