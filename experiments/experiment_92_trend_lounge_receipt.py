"""Regression check for the saved Trend Lounge receipt; no OCR or network calls."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from parsing.unified_structured_document_builder import build_unified_structured_document
from services.receipt_semantics import infer_receipt_semantics


def main() -> int:
    path = Path("results/gui_v4/single/Trend Lounge/unified_raw_ocr.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    document = build_unified_structured_document(raw)
    receipt = document.get("receipt_semantics") or {}
    fields = receipt.get("fields", {})
    items = receipt.get("items", [])
    expected = {
        "Slim Fit Denim Jeans": ("1", "59.99"),
        "Classic White Tee": ("2", "40.00"),
        "Leather Bomber Jacket": ("1", "129.99"),
        "Athletic Crew Socks": ("3", "18.00"),
    }
    mapped = {item["observed_name"]: item for item in items}
    without_number = deepcopy(document)
    without_number["fields"] = [
        field for field in without_number["fields"]
        if field.get("label", "").casefold().strip() != "receipt #"
    ]

    checks = [
        ("receipt classified", receipt.get("status") == "receipt_candidate"),
        ("receipt remains unverified", receipt.get("semantic_verified") is False),
        ("receipt number preserved", fields.get("receipt_number", {}).get("observed_value") == "TL-98432"),
        ("four items extracted", set(mapped) == set(expected)),
        ("quantities and line prices retained", all(
            (mapped.get(name, {}).get("observed_quantity"), mapped.get(name, {}).get("normalized_price")) == pair
            for name, pair in expected.items()
        )),
        ("each item grounded in name, quantity and price", all(len(item.get("evidence_item_ids", [])) == 3 for item in items)),
        ("subtotal normalized", fields.get("subtotal", {}).get("normalized_value") == "247.98"),
        ("tax normalized", fields.get("tax", {}).get("normalized_value") == "21.70"),
        ("total normalized", fields.get("total", {}).get("normalized_value") == "269.68"),
        ("summary amounts cite label and value", all(len(fields.get(key, {}).get("evidence_item_ids", [])) >= 2 for key in ("subtotal", "tax", "total"))),
        ("item sum matches subtotal", receipt.get("consistency_checks", {}).get("items_vs_subtotal", {}).get("status") == "matches"),
        ("subtotal plus tax matches total", receipt.get("consistency_checks", {}).get("subtotal_plus_tax_vs_total", {}).get("status") == "matches"),
        ("OCR evidence unchanged", raw["evidence"] == document["evidence"]),
        ("item table alone does not imply receipt", infer_receipt_semantics(without_number) is None),
    ]
    for index, (name, okay) in enumerate(checks, 1):
        print(f"[{'PASS' if okay else 'FAIL'}] {index:02d}. {name}")
    count = sum(okay for _, okay in checks)
    print(f"RESULT: {count}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if count == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
