"""Experiment 83: additive, conservative receipt value normalization."""
from __future__ import annotations

import json
from pathlib import Path

from parsing.unified_structured_document_builder import build_unified_structured_document
from services.receipt_semantics import infer_receipt_semantics
from services.receipt_value_normalization import normalize_receipt_value

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "results/gui_v4/single/supermarket/unified_raw_ocr.json"


def field(label: str, value: str, n: int) -> dict:
    return {"field_id": str(n), "label": label, "value": value,
            "relationship": "same_row_right", "evidence_item_ids": [f"sample:ocr_region:{n}"]}


def main() -> int:
    if not FIXTURE.is_file():
        print(f"[MISSING] {FIXTURE}")
        return 1
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document = build_unified_structured_document(raw)
    receipt = document.get("receipt_semantics") or {}
    fields = receipt.get("fields", {})
    synthetic = infer_receipt_semantics({
        "fields": [field("SUBTOTAL:", "$10.00", 1), field("CHANGE:", "$0.00", 2),
                   field("TOTAL:", "$11.00", 3), field("TOTAL:", "$12.00", 4)],
        "sections": [], "free_text": [],
    }) or {}
    conflicting = synthetic.get("fields", {}).get("total", {})
    money_noise = normalize_receipt_value("total", "'.$ 5.99")
    ambiguous_date = normalize_receipt_value("date", "03/04/2023")
    invalid_date = normalize_receipt_value("date", "02/30/2023")
    identifier = normalize_receipt_value("transaction_number", "0009887")
    checks = [
        ("receipt remains detected", receipt.get("status") == "receipt_candidate"),
        ("subtotal OCR text retained", fields.get("subtotal", {}).get("observed_value") == "$ 18.95"),
        ("subtotal normalized precisely", fields.get("subtotal", {}).get("normalized_value") == "18.95"),
        ("subtotal display has canonical spacing", fields.get("subtotal", {}).get("display_value") == "$18.95"),
        ("total normalized precisely", fields.get("total", {}).get("normalized_value") == "20.09"),
        ("total evidence still has two IDs", len(fields.get("total", {}).get("evidence_item_ids", [])) == 2),
        ("date normalized to ISO", fields.get("date", {}).get("normalized_value") == "2023-08-24"),
        ("time normalized to 24-hour text", fields.get("time", {}).get("normalized_value") == "14:48"),
        ("item count normalized to integer", fields.get("items_sold", {}).get("normalized_value") == 5),
        ("transaction identifier stays a string", fields.get("transaction_number", {}).get("normalized_value") == "9887"),
        ("raw OCR evidence unchanged", document.get("evidence") == raw.get("evidence")),
        ("money OCR punctuation normalized", money_noise.get("normalized_value") == "5.99"),
        ("ambiguous date order remains unresolved", ambiguous_date.get("normalization_status") == "ambiguous_date_order" and ambiguous_date.get("normalized_value") is None),
        ("invalid calendar date remains unresolved", invalid_date.get("normalization_status") == "invalid_date" and invalid_date.get("normalized_value") is None),
        ("leading zeros preserved in identifier", identifier.get("normalized_value") == "0009887"),
        ("conflicting total still ambiguous", conflicting.get("status") == "ambiguous" and conflicting.get("observed_value") is None),
        ("each conflicting value normalized separately", {x.get("normalized_value") for x in conflicting.get("candidates", [])} == {"11.00", "12.00"}),
    ]
    print("EXPERIMENT 83: NORMALIZED RECEIPT VALUES")
    for i, (name, ok) in enumerate(checks, 1):
        print(f"[{'PASS' if ok else 'FAIL'}] {i:02d}. {name}")
    passed = sum(ok for _, ok in checks)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
