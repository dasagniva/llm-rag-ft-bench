"""Unit tests for the FastAPI serving app.

All tests inject stub generators directly into the app's _generators dict so that
no model weights or GPU are required.  The stubs are cleaned up after each test.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ragbench.serving import app as serving_module
from ragbench.serving.app import app


@pytest.fixture(autouse=True)
def _clean_generators():
    """Reset lazy-load state before and after every test."""
    serving_module._generators.clear()
    serving_module._load_errors.clear()
    yield
    serving_module._generators.clear()
    serving_module._load_errors.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def stub_base(monkeypatch):
    """Pre-load a stub generator for the 'base' config."""
    serving_module._generators["base"] = lambda q: f"stub answer for: {q}"


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["loaded_configs"], list)


def test_health_reports_loaded_configs(client: TestClient) -> None:
    serving_module._generators["base"] = lambda q: "x"
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "base" in resp.json()["loaded_configs"]


# ---------------------------------------------------------------------------
# /ask — happy path
# ---------------------------------------------------------------------------


def test_ask_base_stub(client: TestClient, stub_base) -> None:
    resp = client.post("/ask?config=base", json={"question": "What was revenue in 2022?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["config"] == "base"
    assert "stub answer" in body["answer"]


def test_ask_response_shape(client: TestClient, stub_base) -> None:
    resp = client.post("/ask?config=base", json={"question": "Revenue?"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"answer", "config"}


def test_ask_default_config_is_rag(client: TestClient) -> None:
    serving_module._generators["rag"] = lambda q: "rag stub"
    resp = client.post("/ask", json={"question": "Revenue?"})
    assert resp.status_code == 200
    assert resp.json()["config"] == "rag"


# ---------------------------------------------------------------------------
# /ask — validation errors (422)
# ---------------------------------------------------------------------------


def test_ask_missing_question_field(client: TestClient) -> None:
    resp = client.post("/ask?config=base", json={})
    assert resp.status_code == 422


def test_ask_empty_question(client: TestClient) -> None:
    resp = client.post("/ask?config=base", json={"question": ""})
    assert resp.status_code == 422


def test_ask_invalid_config(client: TestClient) -> None:
    resp = client.post("/ask?config=invalid", json={"question": "Revenue?"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /ask — 503 when weights are missing or load fails
# ---------------------------------------------------------------------------


def test_ask_503_when_load_raises(client: TestClient, monkeypatch) -> None:
    from fastapi import HTTPException

    def _fail(name: str):
        raise HTTPException(
            status_code=503,
            detail="QLoRA adapter not found",
        )

    monkeypatch.setattr(serving_module, "_build_generator", _fail)
    resp = client.post("/ask?config=ft", json={"question": "Revenue?"})
    assert resp.status_code == 503
    assert "adapter" in resp.json()["detail"].lower()


def test_ask_503_cached_error(client: TestClient) -> None:
    serving_module._load_errors["ft"] = "adapter not found"
    resp = client.post("/ask?config=ft", json={"question": "Revenue?"})
    assert resp.status_code == 503


def test_ask_500_on_generation_error(client: TestClient) -> None:
    def _boom(q: str) -> str:
        raise RuntimeError("CUDA OOM")

    serving_module._generators["base"] = _boom
    resp = client.post("/ask?config=base", json={"question": "Revenue?"})
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Generator caching
# ---------------------------------------------------------------------------


def test_generator_cached_after_first_call(client: TestClient, monkeypatch) -> None:
    call_count = {"n": 0}

    def counting_build(name: str):
        call_count["n"] += 1
        return lambda q: "cached"

    monkeypatch.setattr(serving_module, "_build_generator", counting_build)

    client.post("/ask?config=base", json={"question": "Q1?"})
    client.post("/ask?config=base", json={"question": "Q2?"})
    assert call_count["n"] == 1, "Generator should be built only once"
