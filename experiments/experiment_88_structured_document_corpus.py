"""Experiment 88: exact corpus lookup and currency-safe receipt totals."""
from __future__ import annotations

import json
from pathlib import Path

from parsing.unified_structured_document_builder import build_unified_structured_document
from services.structured_document_corpus import (
    compare_receipt_totals, find_documents, index_structured_documents,
    list_receipt_totals,
)

ROOT = Path(__file__).resolve().parent.parent / "results/gui_v4/single"
FOLDERS = [
    "receipt", "supermarket", "pic13", "pic16",
    "ChatGPT Image Sep 24, 2026, 06_13_58 PM",
    "WhatsApp Image 2026-09-22 at 22.21.24",
]


def build(folder: str) -> dict:
    path = ROOT / folder / "unified_raw_ocr.json"
    return build_unified_structured_document(json.loads(path.read_text(encoding="utf-8")))


def minimal_receipt(document_id: str, merchant: str, total: str, currency: str) -> dict:
    return {
        "document_id": document_id,
        "receipt_semantics": {"status": "receipt_candidate", "fields": {
            "merchant": {"status": "candidate", "observed_value": merchant,
                         "evidence_item_ids": [f"{document_id}:ocr_region:1"]},
            "total": {"status": "candidate", "observed_value": currency + total,
                      "normalized_value": total, "currency_symbol": currency,
                      "evidence_item_ids": [f"{document_id}:ocr_region:2"]},
        }},
    }


def raises_value_error(call) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def main() -> int:
    for folder in FOLDERS:
        if not (ROOT / folder / "unified_raw_ocr.json").is_file():
            print(f"[MISSING] {ROOT / folder / 'unified_raw_ocr.json'}")
            return 1
    documents = [build(folder) for folder in FOLDERS]
    index = index_structured_documents(documents)
    campus = documents[0]["document_id"]
    market = documents[1]["document_id"]
    receipt_number = find_documents(index, "receipt_number", "004512")
    merchant = find_documents(index, "merchant", "  ndhu  campus   store ")
    heading = find_documents(index, "heading", "FRESH MART SUPERMARKET")
    transaction = find_documents(index, "transaction_number", "9887")
    missing = find_documents(index, "receipt_number", "NOT-FOUND")
    totals = list_receipt_totals(index)
    comparison = compare_receipt_totals(index)
    same_currency = index_structured_documents([
        minimal_receipt("r1", "Store A", "20.09", "$"),
        minimal_receipt("r2", "Store B", "5.00", "$"),
    ])
    ranked = compare_receipt_totals(same_currency)
    duplicate_merchant = index_structured_documents([
        minimal_receipt("r1", "Same Store", "20.09", "$"),
        minimal_receipt("r2", "Same Store", "5.00", "$"),
    ])
    ambiguity = find_documents(duplicate_merchant, "merchant", "Same Store")
    mismatched_currency = index_structured_documents([
        minimal_receipt("r1", "Store A", "20.09", "$"),
        minimal_receipt("r2", "Store B", "5.00", "€"),
    ])

    checks = [
        ("six documents indexed", len(index["documents"]) == 6 and len(index["by_id"]) == 6),
        ("receipt number finds campus receipt", receipt_number["selected_document_id"] == campus),
        ("receipt lookup keeps OCR provenance", len(receipt_number["matches"][0]["evidence_item_ids"]) == 1),
        ("merchant lookup normalizes spacing and case", merchant["selected_document_id"] == campus),
        ("heading lookup finds supermarket", heading["selected_document_id"] == market),
        ("transaction number finds supermarket", transaction["selected_document_id"] == market),
        ("missing receipt number selects nothing", missing["status"] == "no_match" and missing["selected_document_id"] is None),
        ("only two receipts listed", len(totals) == 2 and {t["document_id"] for t in totals} == {campus, market}),
        ("campus total preserves unknown currency", totals[0]["normalized_value"] == "310" and totals[0]["currency_symbol"] is None),
        ("supermarket total retains currency and evidence", totals[1]["normalized_value"] == "20.09" and totals[1]["currency_symbol"] == "$" and len(totals[1]["evidence_item_ids"]) == 2),
        ("mixed known/unknown currencies not ranked", comparison["status"] == "not_comparable" and comparison["reason"] == "currency_unknown"),
        ("same currency can be ranked", ranked["status"] == "ranked" and ranked["highest_document_ids"] == ["r1"]),
        ("different known currencies not ranked", compare_receipt_totals(mismatched_currency)["reason"] == "currency_mismatch"),
        ("duplicate merchant remains ambiguous", ambiguity["status"] == "ambiguous" and ambiguity["selected_document_id"] is None and len(ambiguity["matches"]) == 2),
        ("duplicate document ID rejected", raises_value_error(lambda: index_structured_documents([documents[0], documents[0]]))),
        ("empty lookup rejected", raises_value_error(lambda: find_documents(index, "merchant", "  "))),
        ("unsupported lookup key rejected", raises_value_error(lambda: find_documents(index, "total", "310"))),
    ]
    print("EXPERIMENT 88: STRUCTURED DOCUMENT CORPUS")
    print("Receipt totals:", totals)
    print("Comparison:", comparison["status"], comparison.get("reason"))
    for i, (label, ok) in enumerate(checks, 1):
        print(f"[{'PASS' if ok else 'FAIL'}] {i:02d}. {label}")
    passed = sum(ok for _, ok in checks)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
