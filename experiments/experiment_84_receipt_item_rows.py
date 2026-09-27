"""Experiment 84: row-linked receipt item and price candidates."""
from __future__ import annotations

import json
from pathlib import Path

from parsing.unified_structured_document_builder import build_unified_structured_document
from services.receipt_item_rows import extract_receipt_item_rows

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "results/gui_v4/single/supermarket/unified_raw_ocr.json"


def cell(text: str, evidence_id: str) -> dict:
    return {"text": text, "evidence_item_ids": [evidence_id]}


def main() -> int:
    if not FIXTURE.is_file():
        print(f"[MISSING] {FIXTURE}")
        return 1
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    structured = build_unified_structured_document(raw)
    receipt = structured.get("receipt_semantics") or {}
    items = receipt.get("items", [])
    by_name = {item["observed_name"].rstrip("."): item for item in items}
    expected = {
        "Organic Bananas": "2.99", "Whole Milk": "3.49",
        "Whole Wheat Bread": "2.49", "Grade A Large Eggs": "3.99",
        "Ground Beef": "5.99",
    }
    synthetic = extract_receipt_item_rows([
        {"record_id": "ambiguous", "table_id": "t", "source_row_id": 1,
         "cells": [cell("Milk", "n"), cell("$2.00", "p1"), cell("$3.00", "p2")]},
        {"record_id": "summary", "table_id": "t", "source_row_id": 2,
         "cells": [cell("TOTAL:", "l"), cell("$5.00", "v")]},
    ])
    checks = [
        ("receipt classification retained", receipt.get("status") == "receipt_candidate"),
        ("five item candidates", len(items) == 5),
        ("expected item names", set(by_name) == set(expected)),
        ("all five prices normalized", all(by_name.get(name, {}).get("normalized_price") == price for name, price in expected.items())),
        ("raw price for OCR-noisy bananas preserved", by_name.get("Organic Bananas", {}).get("observed_price") == ".$ 2.99"),
        ("display price cleaned", by_name.get("Organic Bananas", {}).get("display_price") == "$2.99"),
        ("each item cites name and price", all(len(item.get("evidence_item_ids", [])) == 2 for item in items)),
        ("each item retains its row", len({item.get("source_row_id") for item in items}) == 5),
        ("summary rows excluded", not any(name in by_name for name in ("SUBTOTAL", "TOTAL", "CHANGE"))),
        ("payment row excluded", not any("PAID:" in item["observed_name"] for item in items)),
        ("real fixture has no ambiguous item row", receipt.get("ambiguous_item_rows") == []),
        ("receipt total unchanged", receipt.get("fields", {}).get("total", {}).get("normalized_value") == "20.09"),
        ("OCR evidence unchanged", structured.get("evidence") == raw.get("evidence")),
        ("multiple prices not selected", synthetic["items"] == []),
        ("multiple prices reported as ambiguous", len(synthetic["ambiguous_rows"]) == 1 and synthetic["ambiguous_rows"][0]["record_id"] == "ambiguous"),
    ]
    print("EXPERIMENT 84: RECEIPT ITEM ROWS")
    for item in items:
        print(item["observed_name"], "->", item["observed_price"], "=>", item["display_price"])
    for i, (name, ok) in enumerate(checks, 1):
        print(f"[{'PASS' if ok else 'FAIL'}] {i:02d}. {name}")
    passed = sum(ok for _, ok in checks)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
