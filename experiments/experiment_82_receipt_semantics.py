"""Experiment 82: evidence-linked, unverified receipt fields.

Reads a saved OCR JSON fixture. No OCR, Groq, or network requests.
"""
from __future__ import annotations

import json
from pathlib import Path

from parsing.unified_structured_document_builder import build_unified_structured_document
from services.receipt_semantics import infer_receipt_semantics

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "results/gui_v4/single/supermarket/unified_raw_ocr.json"


def field(key: str, value: str, region: int, relationship: str = "inline_label_value") -> dict:
    return {
        "field_id": f"field_{region}", "label": key, "value": value,
        "relationship": relationship,
        "evidence_item_ids": [f"sample:ocr_region:{region}"],
    }


def main() -> int:
    if not FIXTURE.is_file():
        print(f"[MISSING] {FIXTURE}")
        return 1
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    structured = build_unified_structured_document(raw)
    receipt = structured.get("receipt_semantics") or {}
    fields = receipt.get("fields", {})

    def observed(key: str) -> str | None:
        return fields.get(key, {}).get("observed_value")

    def money(key: str) -> str | None:
        value = observed(key)
        return "".join(value.split()) if value else None

    synthetic = {
        "fields": [
            field("SUBTOTAL:", "$10.00", 1, "same_row_right"),
            field("TOTAL:", "$11.00", 2, "same_row_right"),
            field("TOTAL:", "$12.00", 3, "same_row_right"),
            field("DATE:", "09/27/2026", 4),
            field("ADDRESS", "Jane Smith", 5, "next_row_below"),
            field("CHANGE:", "$0.00", 6, "same_row_right"),
        ],
        "sections": [], "free_text": [],
    }
    ambiguous = infer_receipt_semantics(synthetic) or {}
    ambiguous_total = ambiguous.get("fields", {}).get("total", {})
    unpaired = infer_receipt_semantics({
        "document_id": "sample",
        "fields": [field("SUBTOTAL:", "$10.00", 1), field("CHANGE:", "$0.00", 2)],
        "sections": [], "free_text": [],
        "evidence": [
            {"evidence_type": "ocr_region", "region_id": 3, "text": "TOTAL:", "bbox": {"x": 10, "y": 30, "width": 60, "height": 20}},
            {"evidence_type": "ocr_region", "region_id": 4, "text": "$11.00", "bbox": {"x": 200, "y": 31, "width": 60, "height": 20}},
            {"evidence_type": "ocr_region", "region_id": 5, "text": "$12.00", "bbox": {"x": 200, "y": 60, "width": 60, "height": 20}},
        ],
    }) or {}
    unpaired_total = unpaired.get("fields", {}).get("total", {})
    nonreceipt = infer_receipt_semantics({
        "fields": [field("TOTAL:", "$11.00", 1)],
        "sections": [], "free_text": [],
    })

    checks = [
        ("receipt detected from explicit signals", receipt.get("status") == "receipt_candidate"),
        ("interpretation remains unverified", receipt.get("semantic_verified") is False),
        ("merchant heading preserved", observed("merchant") == "FRESH MART SUPERMARKET"),
        ("merchant has two evidence IDs", len(fields.get("merchant", {}).get("evidence_item_ids", [])) == 2),
        ("date from explicit label", observed("date") == "08/24/2023"),
        ("cashier from explicit label", observed("cashier") == "Jane Smith"),
        ("transaction number from TRN", observed("transaction_number") == "9887"),
        ("subtotal preserved", money("subtotal") == "$18.95"),
        ("tax preserved", money("tax") == "$1.14"),
        ("total preserved", money("total") == "$20.09"),
        ("total cites label and amount regions", len(fields.get("total", {}).get("evidence_item_ids", [])) == 2),
        ("payment method preserved", observed("payment_method") == "DEBIT CARD"),
        ("change preserved", money("change") == "$0.00"),
        ("items sold preserved", observed("items_sold") == "5"),
        ("raw OCR evidence unchanged", structured.get("evidence") == raw.get("evidence")),
        ("conflicting total is marked ambiguous", ambiguous_total.get("status") == "ambiguous"),
        ("conflicting total selects no value", ambiguous_total.get("observed_value") is None),
        ("ambiguous total preserves both candidates", len(ambiguous_total.get("candidates", [])) == 2),
        ("geometric next-row guess excluded", "address" not in ambiguous.get("fields", {})),
        ("one total alone does not imply a receipt", nonreceipt is None),
        ("unpaired total uses same-line OCR evidence", unpaired_total.get("observed_value") == "$11.00" and unpaired_total.get("relationship") == "same_line_ocr_geometry"),
    ]
    print("EXPERIMENT 82: RECEIPT SEMANTIC CANDIDATES")
    print("Extracted:", {key: value.get("observed_value") for key, value in fields.items()})
    for i, (name, ok) in enumerate(checks, 1):
        print(f"[{'PASS' if ok else 'FAIL'}] {i:02d}. {name}")
    passed = sum(ok for _, ok in checks)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
