"""Unit tests for corpus/finqa_tatqa.py — FinQA/TAT-QA document and gold-span builders.

Fixtures here are small hand-written analogues of the real dataset shapes
(verified against actual FinQA test.json / TAT-QA dev.json structure).
"""

from ragbench.corpus.finqa_tatqa import (
    finqa_documents,
    finqa_gold_span_parts,
    gold_span_present,
    normalize_whitespace,
    render_table,
    tatqa_documents,
    tatqa_gold_span_parts,
)


class TestRenderTable:
    def test_simple_table(self):
        table = [["", "2014", "2015"], ["net revenue", "5735", "5829"]]
        assert render_table(table) == " | 2014 | 2015\nnet revenue | 5735 | 5829"

    def test_single_row(self):
        assert render_table([["a", "b"]]) == "a | b"

    def test_empty_table(self):
        assert render_table([]) == ""


FINQA_EXAMPLE = {
    "filename": "ETR/2016/page_23.pdf",
    "pre_text": ["pre sentence one .", "pre sentence two ."],
    "post_text": ["post sentence one ."],
    "table": [
        ["", "amount ( in millions )"],
        ["2014 net revenue", "$ 5735"],
        ["2015 net revenue", "$ 5829"],
    ],
    "qa": {
        "question": "what is the net change in net revenue?",
        "answer": "94",
        "ann_table_rows": [1, 2],
        "ann_text_rows": [0],
    },
    "id": "ETR/2016/page_23.pdf-2",
}

FINQA_EXAMPLE_DUP = {
    **FINQA_EXAMPLE,
    "id": "ETR/2016/page_23.pdf-3",
    "qa": {
        "question": "another question",
        "answer": "1",
        "ann_table_rows": [],
        "ann_text_rows": [],
    },
}


class TestFinqaDocuments:
    def test_builds_one_doc_per_page(self):
        docs = finqa_documents([FINQA_EXAMPLE])
        assert len(docs) == 1
        doc = docs[0]
        assert doc["id"] == "finqa_ETR/2016/page_23.pdf"
        assert "pre sentence one ." in doc["text"]
        assert "2014 net revenue | $ 5735" in doc["text"]
        assert "post sentence one ." in doc["text"]

    def test_deduplicates_by_filename(self):
        docs = finqa_documents([FINQA_EXAMPLE, FINQA_EXAMPLE_DUP])
        assert len(docs) == 1


class TestFinqaGoldSpanParts:
    def test_returns_text_and_table_rows(self):
        parts = finqa_gold_span_parts(FINQA_EXAMPLE)
        assert "pre sentence one ." in parts
        assert "2014 net revenue | $ 5735" in parts
        assert "2015 net revenue | $ 5829" in parts

    def test_out_of_range_indices_ignored(self):
        ex = {
            **FINQA_EXAMPLE,
            "qa": {"ann_table_rows": [99], "ann_text_rows": [99]},
        }
        assert finqa_gold_span_parts(ex) == []

    def test_no_annotations_returns_empty(self):
        ex = {**FINQA_EXAMPLE, "qa": {"ann_table_rows": [], "ann_text_rows": []}}
        assert finqa_gold_span_parts(ex) == []


TATQA_ENTRY = {
    "table": {
        "uid": "table-uid-1",
        "table": [["", "2019", "2018"], ["Total sales", "1496.5", "1202.9"]],
    },
    "paragraphs": [
        {"order": 1, "text": "Sales by contract type paragraph."},
        {"order": 2, "text": "Fixed-price contract paragraph."},
    ],
}


class TestTatqaDocuments:
    def test_builds_one_doc_per_table(self):
        docs = tatqa_documents([TATQA_ENTRY])
        assert len(docs) == 1
        doc = docs[0]
        assert doc["id"] == "tatqa_table-uid-1"
        assert "Sales by contract type paragraph." in doc["text"]
        assert "Total sales | 1496.5 | 1202.9" in doc["text"]

    def test_deduplicates_by_uid(self):
        docs = tatqa_documents([TATQA_ENTRY, TATQA_ENTRY])
        assert len(docs) == 1


class TestTatqaGoldSpanParts:
    def test_text_answer_uses_rel_paragraphs(self):
        question = {"rel_paragraphs": ["2"], "answer_from": "text"}
        parts = tatqa_gold_span_parts(TATQA_ENTRY, question)
        assert parts == ["Fixed-price contract paragraph."]

    def test_table_answer_includes_whole_table(self):
        question = {"rel_paragraphs": [], "answer_from": "table"}
        parts = tatqa_gold_span_parts(TATQA_ENTRY, question)
        assert any("Total sales | 1496.5 | 1202.9" in p for p in parts)

    def test_table_text_answer_includes_both(self):
        question = {"rel_paragraphs": ["1"], "answer_from": "table-text"}
        parts = tatqa_gold_span_parts(TATQA_ENTRY, question)
        assert "Sales by contract type paragraph." in parts
        assert any("Total sales | 1496.5 | 1202.9" in p for p in parts)

    def test_missing_rel_paragraphs_falls_back_to_table(self):
        question = {"answer_from": "text"}
        parts = tatqa_gold_span_parts(TATQA_ENTRY, question)
        assert any("Total sales | 1496.5 | 1202.9" in p for p in parts)


class TestNormalizeWhitespace:
    def test_lowercases_and_collapses(self):
        assert normalize_whitespace("  Net   Income\n2014  ") == "net income 2014"


class TestGoldSpanPresent:
    def test_all_parts_present(self):
        parts = ["2014 net revenue | $ 5735", "pre sentence one ."]
        retrieved = ["... 2014 net revenue | $ 5735 ... pre sentence one . more text ..."]
        assert gold_span_present(parts, retrieved) is True

    def test_part_split_across_chunks_does_not_match(self):
        parts = ["2014 net revenue | $ 5735"]
        retrieved = ["chunk a text 2014 net revenue extra words", "| $ 5735 chunk b text"]
        # Concatenation of retrieved chunks loses the original adjacency,
        # so a span split across a chunk boundary will NOT match — documented limitation.
        assert gold_span_present(parts, retrieved) is False

    def test_missing_part_is_false(self):
        parts = ["2014 net revenue | $ 5735", "this sentence is not retrieved"]
        retrieved = ["... 2014 net revenue | $ 5735 ..."]
        assert gold_span_present(parts, retrieved) is False

    def test_empty_gold_span_is_false(self):
        assert gold_span_present([], ["anything"]) is False

    def test_case_and_whitespace_insensitive(self):
        parts = ["2014 Net Revenue | $ 5735"]
        retrieved = ["...   2014   net revenue   |   $ 5735  ..."]
        assert gold_span_present(parts, retrieved) is True
