"""Regression for price questions over receipt tables with a Qty column."""
from __future__ import annotations

import json
from pathlib import Path

from parsing.unified_structured_document_builder import build_unified_structured_document
from services.structured_document_retrieval import retrieve_structured_answer


def _row(number: int, *texts: str) -> dict:
    return {
        "table_id": "synthetic_table", "source_row_id": number,
        "cells": [
            {"column_index": i, "text": value, "evidence_item_ids": [f"synthetic:ocr_region:{number * 10 + i}"]}
            for i, value in enumerate(texts)
        ],
    }


def main() -> int:
    path = Path("results/gui_v4/single/Trend Lounge/unified_raw_ocr.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    document = build_unified_structured_document(raw)
    question = "What is the price for Classic White Tee?"
    result = retrieve_structured_answer(document, question)
    selected = result.get("selected_candidate") or {}
    valid_tee = [
        c for c in result["candidates"]
        if c["type_valid"] and c["label"] == "Classic White Tee"
    ]

    conflict = {
        "fields": [],
        "records": [
            _row(1, "Item", "Qty", "Price", "Alternative Price"),
            _row(2, "Classic White Tee", "2", "$40.00", "$45.00"),
        ],
    }
    conflicted = retrieve_structured_answer(conflict, question)

    bare = {
        "fields": [],
        "records": [
            _row(1, "Item", "Price"),
            _row(2, "Black Tea (L)", "45"),
        ],
    }
    bare_result = retrieve_structured_answer(bare, "What is the price for Black Tea (L)?")

    checks = [
        ("white tee price selected", result["status"] == "structured_answer_candidate" and selected.get("value") == "$40.00"),
        ("quantity excluded from money candidates", not any(c["value"] == "2" for c in valid_tee)),
        ("selected price cites item and price OCR", len(selected.get("evidence_item_ids", [])) == 2),
        ("two actual prices remain ambiguous", conflicted["status"] == "ambiguous_structured_evidence"),
        ("two-column currencyless price remains available", bare_result["status"] == "structured_answer_candidate" and bare_result["selected_candidate"]["value"] == "45"),
    ]
    for index, (name, ok) in enumerate(checks, 1):
        print(f"[{'PASS' if ok else 'FAIL'}] {index:02d}. {name}")
    passed = sum(ok for _, ok in checks)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
