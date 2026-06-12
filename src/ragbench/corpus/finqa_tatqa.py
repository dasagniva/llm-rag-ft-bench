"""Build retrieval-corpus documents from FinQA / TAT-QA source pages.

EXPERIMENT.md amendment (2026-06-12): the retrieval corpus for the base-vs-RAG
comparison is built from the FinQA test split and TAT-QA dev split — the exact
source document (10-K page text + table, or table + paragraphs) each eval
question was written against — rather than the generic EDGAR S&P500 sample.
This guarantees the gold-supporting passage for every eval question is present
in the index.

All functions here are pure (no I/O, no network).
"""

from __future__ import annotations

from typing import Any


def render_table(table: list[list[Any]]) -> str:
    """Render a list-of-lists table as pipe-separated rows, one row per line."""
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def finqa_documents(raw: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build one document per unique FinQA source page (deduplicated by filename).

    Document text = pre_text + rendered table + post_text, matching the page
    content the question writers had in front of them.
    """
    docs: dict[str, dict[str, str]] = {}
    for ex in raw:
        filename = ex["filename"]
        if filename in docs:
            continue
        text = "\n".join(
            [
                " ".join(ex["pre_text"]),
                render_table(ex["table"]),
                " ".join(ex["post_text"]),
            ]
        )
        docs[filename] = {
            "id": f"finqa_{filename}",
            "text": text,
            "source_path": filename,
        }
    return list(docs.values())


def tatqa_documents(raw: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build one document per unique TAT-QA table (deduplicated by table UID).

    Document text = paragraphs + rendered table.
    """
    docs: dict[str, dict[str, str]] = {}
    for entry in raw:
        uid = entry["table"]["uid"]
        if uid in docs:
            continue
        text = "\n".join(
            [
                "\n".join(p["text"] for p in entry["paragraphs"]),
                render_table(entry["table"]["table"]),
            ]
        )
        docs[uid] = {
            "id": f"tatqa_{uid}",
            "text": text,
            "source_path": uid,
        }
    return list(docs.values())


def finqa_gold_span_parts(example: dict[str, Any]) -> list[str]:
    """Return the gold-supporting text/table-row strings for a FinQA question.

    These are the exact substrings (after whitespace normalization) that should
    appear in the retrieved context if retrieval found the right page.
    """
    qa = example["qa"]
    lines = list(example["pre_text"]) + list(example["post_text"])
    table = example["table"]

    parts: list[str] = []
    for idx in qa.get("ann_text_rows", []) or []:
        if 0 <= idx < len(lines):
            parts.append(lines[idx])
    for idx in qa.get("ann_table_rows", []) or []:
        if 0 <= idx < len(table):
            parts.append(render_table([table[idx]]))
    return parts


def normalize_whitespace(text: str) -> str:
    """Lowercase and collapse whitespace, for substring comparison across chunking."""
    return " ".join(text.lower().split())


def gold_span_present(gold_span_parts: list[str], retrieved_texts: list[str]) -> bool:
    """True if every gold span part is a substring of the concatenated retrieved text.

    Returns False if *gold_span_parts* is empty (no gold annotation recorded for
    this question — cannot be a "hit" by construction).
    """
    if not gold_span_parts:
        return False
    combined = normalize_whitespace(" ".join(retrieved_texts))
    return all(normalize_whitespace(part) in combined for part in gold_span_parts)


def tatqa_gold_span_parts(entry: dict[str, Any], question: dict[str, Any]) -> list[str]:
    """Return the gold-supporting paragraph/table strings for a TAT-QA question.

    For "table" and "table-text" questions (no specific cell references in the
    dataset), the whole table is included as the gold span.
    """
    parts: list[str] = []
    rel_paragraphs = {str(o) for o in question.get("rel_paragraphs", []) or []}
    for p in entry["paragraphs"]:
        if str(p.get("order")) in rel_paragraphs:
            parts.append(p["text"])

    answer_from = question.get("answer_from", "")
    if answer_from in ("table", "table-text") or not parts:
        parts.append(render_table(entry["table"]["table"]))
    return parts
