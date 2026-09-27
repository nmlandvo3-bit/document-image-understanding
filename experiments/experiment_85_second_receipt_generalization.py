"""Experiment 85: second real receipt and existing non-receipt corpus."""
from __future__ import annotations

import json
from pathlib import Path

from parsing.unified_structured_document_builder import build_unified_structured_document
from services.receipt_table_evidence import extract_bare_receipt_table

ROOT = Path(__file__).resolve().parent.parent / "results/gui_v4/single"
NON_RECEIPTS = [
    "ChatGPT Image Sep 24, 2026, 06_13_58 PM", "pic13", "pic16",
    "WhatsApp Image 2026-09-22 at 22.21.24",
]


def build(folder: str) -> dict:
    path = ROOT / folder / "unified_raw_ocr.json"
    return build_unified_structured_document(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    folders = ["receipt", "supermarket", *NON_RECEIPTS]
    for folder in folders:
        if not (ROOT / folder / "unified_raw_ocr.json").is_file():
            print(f"[MISSING] {ROOT / folder / 'unified_raw_ocr.json'}")
            return 1
    campus = build("receipt")
    supermarket = build("supermarket")
    others = {folder: build(folder) for folder in NON_RECEIPTS}
    receipt = campus.get("receipt_semantics") or {}
    fields = receipt.get("fields", {})
    items = receipt.get("items", [])
    names = {item["observed_name"]: item for item in items}
    expected = {
        "Black Tea (L)": "45", "Rice Ball x2": "70", "Notebook A4": "85",
        "Pen Blue 0.5": "30", "Coffee Latte": "65",
    }
    summary = {"subtotal": "295", "tax": "15", "total": "310", "cash_tendered": "500", "change": "190"}
    checks = [
        ("campus store detected as receipt", receipt.get("status") == "receipt_candidate"),
        ("campus merchant heading", fields.get("merchant", {}).get("observed_value") == "NDHU CAMPUS STORE"),
        ("five campus item rows", len(items) == 5 and set(names) == set(expected)),
        ("five bare item prices preserved", all(names.get(name, {}).get("observed_price") == value for name, value in expected.items())),
        ("no currency invented", all(item.get("currency_symbol") is None for item in items)),
        ("item pairs retain two evidence IDs", all(len(item.get("evidence_item_ids", [])) == 2 for item in items)),
        ("all summary amounts recovered", all(fields.get(key, {}).get("observed_value") == value for key, value in summary.items())),
        ("all summary amounts normalized without currency", all(fields.get(key, {}).get("normalized_value") == value and fields.get(key, {}).get("currency_symbol") is None for key, value in summary.items())),
        ("all summary pairs cite two regions", all(len(fields.get(key, {}).get("evidence_item_ids", [])) == 2 for key in summary)),
        ("campus raw evidence unchanged", campus.get("evidence") == json.loads((ROOT / "receipt/unified_raw_ocr.json").read_text(encoding="utf-8")).get("evidence")),
        ("supermarket remains receipt with five items", len((supermarket.get("receipt_semantics") or {}).get("items", [])) == 5),
        ("supermarket total unchanged", (supermarket.get("receipt_semantics") or {}).get("fields", {}).get("total", {}).get("normalized_value") == "20.09"),
        ("four other scans remain non-receipts", all(doc.get("receipt_semantics") is None for doc in others.values())),
        ("bare total alone insufficient", extract_bare_receipt_table([{"record_id": "r", "cells": [{"text": "TOTAL"}, {"text": "310"}]}]) is None),
    ]
    print("EXPERIMENT 85: SECOND RECEIPT GENERALIZATION")
    print("Campus items:", [(item["observed_name"], item["observed_price"]) for item in items])
    print("Campus summary:", {key: fields.get(key, {}).get("observed_value") for key in summary})
    print("Other scans:", {folder: doc.get("receipt_semantics") is not None for folder, doc in others.items()})
    for i, (name, ok) in enumerate(checks, 1):
        print(f"[{'PASS' if ok else 'FAIL'}] {i:02d}. {name}")
    passed = sum(ok for _, ok in checks)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
