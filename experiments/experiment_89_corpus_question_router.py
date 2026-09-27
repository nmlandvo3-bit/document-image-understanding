"""Experiment 89: grounded, narrow questions over a structured corpus."""
from __future__ import annotations

import json
from pathlib import Path

from parsing.unified_structured_document_builder import build_unified_structured_document
from services.corpus_question_router import answer_corpus_question
from services.structured_document_corpus import index_structured_documents

ROOT = Path(__file__).resolve().parent.parent / "results/gui_v4/single"
FOLDERS = [
    "receipt", "supermarket", "pic13", "pic16",
    "ChatGPT Image Sep 24, 2026, 06_13_58 PM",
    "WhatsApp Image 2026-09-22 at 22.21.24",
]


def minimal_receipt(document_id: str, merchant: str, total: str,
                    receipt_number: str, currency: str = "$") -> dict:
    return {
        "document_id": document_id,
        "receipt_semantics": {"status": "receipt_candidate", "fields": {
            "merchant": {"status": "candidate", "observed_value": merchant,
                         "evidence_item_ids": [f"{document_id}:ocr_region:1"]},
            "receipt_number": {"status": "candidate", "observed_value": receipt_number,
                               "display_value": receipt_number,
                               "evidence_item_ids": [f"{document_id}:ocr_region:2"]},
            "total": {"status": "candidate", "observed_value": currency + total,
                      "normalized_value": total, "display_value": currency + total,
                      "currency_symbol": currency,
                      "evidence_item_ids": [f"{document_id}:ocr_region:3"]},
        }},
    }


def main() -> int:
    for folder in FOLDERS:
        path = ROOT / folder / "unified_raw_ocr.json"
        if not path.is_file():
            print(f"[MISSING] {path}")
            return 1
    documents = [
        build_unified_structured_document(json.loads(
            (ROOT / folder / "unified_raw_ocr.json").read_text(encoding="utf-8")
        )) for folder in FOLDERS
    ]
    index = index_structured_documents(documents)
    campus_id, market_id = (documents[i]["document_id"] for i in (0, 1))
    ask = lambda question: answer_corpus_question(index, question)

    receipt = ask("Which receipt has receipt number 004512?")
    heading = ask("Which document has heading FRESH MART SUPERMARKET?")
    campus_date = ask("What is the date on receipt number 004512?")
    campus_number = ask("What is the receipt number for NDHU CAMPUS STORE?")
    market_total = ask("What is the total for merchant FRESH MART SUPERMARKET?")
    campus_total = ask("What is the total for NDHU CAMPUS STORE?")
    totals = ask("List the totals of all receipts")
    highest = ask("Which receipt has the highest total?")
    unknown = ask("What is the date on receipt number 999999?")
    missing_field = ask("What is the cashier on receipt number 004512?")
    unsupported = ask("Who bought the coffee?")

    same_currency = index_structured_documents([
        minimal_receipt("r1", "Store A", "20.09", "0001"),
        minimal_receipt("r2", "Store B", "5.00", "0002"),
    ])
    ranked = answer_corpus_question(same_currency, "Which receipt has the highest total?")
    duplicate = index_structured_documents([
        minimal_receipt("r1", "Same Store", "20.09", "004512"),
        minimal_receipt("r2", "Same Store", "5.00", "004512"),
    ])
    ambiguous_document = answer_corpus_question(
        duplicate, "What is the total for Same Store?"
    )
    ambiguous_receipt = answer_corpus_question(
        duplicate, "What is the total on receipt number 004512?"
    )
    ambiguous_field_doc = minimal_receipt("r3", "Store C", "9.00", "0003")
    ambiguous_field_doc["receipt_semantics"]["fields"]["total"] = {
        "status": "ambiguous", "candidates": [
            {"observed_value": "$9.00", "evidence_item_ids": ["r3:ocr_region:3"]},
            {"observed_value": "$8.00", "evidence_item_ids": ["r3:ocr_region:4"]},
        ],
    }
    ambiguous_field = answer_corpus_question(
        index_structured_documents([ambiguous_field_doc]),
        "What is the total on receipt number 0003?",
    )

    checks = [
        ("receipt number resolves campus document", receipt["status"] == "one_match" and receipt["answer"] == campus_id),
        ("receipt lookup retains OCR evidence", len(receipt["matches"][0]["evidence_item_ids"]) == 1),
        ("heading resolves supermarket document", heading["status"] == "one_match" and heading["answer"] == market_id),
        ("heading lookup retains OCR evidence", len(heading["matches"][0]["evidence_item_ids"]) == 2),
        ("campus date is normalized and grounded", campus_date["status"] == "answered" and campus_date["answer"] == "2026-09-14" and campus_date["document_id"] == campus_id and len(campus_date["evidence_item_ids"]) == 1),
        ("receipt number retains leading zeros", campus_number["answer"] == "004512" and campus_number["document_id"] == campus_id),
        ("supermarket total carries document and evidence", market_total["answer"] == "$20.09" and market_total["document_id"] == market_id and len(market_total["evidence_item_ids"]) == 2),
        ("campus total does not invent currency", campus_total["answer"] == "310" and campus_total["currency_symbol"] is None),
        ("two receipt totals are listed", totals["status"] == "answered" and len(totals["totals"]) == 2 and {t["document_id"] for t in totals["totals"]} == {campus_id, market_id}),
        ("listed totals retain evidence", all(len(t["evidence_item_ids"]) == 2 for t in totals["totals"])),
        ("highest total blocks unknown currency", highest["status"] == "not_comparable" and highest["answer"] is None and highest["reason"] == "currency_unknown"),
        ("unknown receipt does not select a document", unknown["status"] == "document_not_found" and unknown["answer"] is None),
        ("missing field does not invent a value", missing_field["status"] == "field_not_found" and missing_field["answer"] is None),
        ("unsupported question is explicit", unsupported["status"] == "unsupported_question" and unsupported["answer"] is None),
        ("same-currency synthetic totals rank", ranked["status"] == "answered" and ranked["document_ids"] == ["r1"] and ranked["ranked_totals"][0]["evidence_item_ids"] == ["r1:ocr_region:3"]),
        ("duplicate merchant blocks field selection", ambiguous_document["status"] == "ambiguous_document" and ambiguous_document["answer"] is None and len(ambiguous_document["matches"]) == 2),
        ("duplicate receipt number blocks field selection", ambiguous_receipt["status"] == "ambiguous_document" and ambiguous_receipt["answer"] is None),
        ("ambiguous total keeps both candidates", ambiguous_field["status"] == "ambiguous_field" and ambiguous_field["answer"] is None and len(ambiguous_field["candidates"]) == 2),
        ("answered values remain unverified", all(result["semantic_verified"] is False for result in (campus_date, campus_number, market_total, campus_total))),
    ]

    print("EXPERIMENT 89: GROUNDED CORPUS QUESTIONS")
    print("Campus date:", campus_date)
    print("Market total:", market_total)
    print("Highest total:", highest["status"], highest.get("reason"))
    for number, (label, passed) in enumerate(checks, 1):
        print(f"[{'PASS' if passed else 'FAIL'}] {number:02d}. {label}")
    count = sum(passed for _, passed in checks)
    print(f"RESULT: {count}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if count == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
