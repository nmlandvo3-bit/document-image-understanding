"""Experiment 80: structured heading / document identity retrieval."""
from __future__ import annotations
import json
import re
from pathlib import Path

from parsing.unified_structured_document_builder import build_unified_structured_document
from services.structured_document_retrieval import retrieve_structured_answer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPERMARKET = PROJECT_ROOT / "results/gui_v4/single/supermarket/unified_raw_ocr.json"
SYNTHETIC = (
    PROJECT_ROOT
    / "results/gui_v4/single/ChatGPT Image Sep 24, 2026, 06_13_58 PM/unified_raw_ocr.json"
)

def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def selected(result: dict) -> dict:
    return result.get("selected_candidate") or {}

def selected_value(result: dict) -> str | None:
    return selected(result).get("value")

def money_equivalent(value: str | None, expected: str) -> bool:
    if value is None:
        return False
    return re.sub(r"\s+", "", value) == re.sub(r"\s+", "", expected)

def main() -> int:
    print("=" * 116)
    print("EXPERIMENT 80: STRUCTURED HEADING / DOCUMENT IDENTITY RETRIEVAL")
    print("=" * 116)

    for path in (SUPERMARKET, SYNTHETIC):
        if not path.is_file():
            print(f"[MISSING] {path}")
            return 1

    supermarket = build_unified_structured_document(load(SUPERMARKET))
    synthetic = build_unified_structured_document(load(SYNTHETIC))

    questions = {
        "supermarket": "What is the supermarket name?",
        "store": "What is the store name?",
        "company": "What company issued this receipt?",
        "title": "What is the document title?",
    }
    identity_results = {
        key: retrieve_structured_answer(supermarket, question)
        for key, question in questions.items()
    }

    date_result = retrieve_structured_answer(supermarket, "What is the date?")
    cashier_result = retrieve_structured_answer(
        supermarket, "What is the name of the cashier?"
    )
    customer_result = retrieve_structured_answer(
        supermarket, "What is the name of the customer?"
    )
    subtotal_result = retrieve_structured_answer(supermarket, "What is subtotal?")
    beef_result = retrieve_structured_answer(
        supermarket, "What is the price of Ground Beef?"
    )
    ambiguous_total = retrieve_structured_answer(
        synthetic, "What is the total?"
    )

    for key, result in identity_results.items():
        print()
        print(f"{key.upper()} RESULT:")
        print(result)

    target = "FRESH MART SUPERMARKET"
    checks = []

    for key in ("supermarket", "store", "company", "title"):
        result = identity_results[key]
        candidate = selected(result)
        checks.extend([
            (f"{key} question uses structured answer",
             result.get("status") == "structured_answer_candidate"),
            (f"{key} answer is FRESH MART SUPERMARKET",
             selected_value(result) == target),
            (f"{key} source is structured_heading",
             candidate.get("source") == "structured_heading"),
        ])

    first_heading = selected(identity_results["supermarket"])

    checks.extend([
        ("heading candidate preserves two evidence IDs",
         len(first_heading.get("evidence_item_ids", [])) == 2),
        ("heading candidate uses first heading section",
         first_heading.get("section_id") == "section_001"),
        ("date remains structured and correct",
         date_result.get("status") == "structured_answer_candidate"
         and selected_value(date_result) == "08/24/2023"),
        ("cashier remains Jane Smith",
         selected_value(cashier_result) == "Jane Smith"),
        ("customer remains John Doe",
         selected_value(customer_result) == "John Doe"),
        ("subtotal remains $18.95",
         money_equivalent(selected_value(subtotal_result), "$18.95")),
        ("Ground Beef remains $5.99",
         money_equivalent(selected_value(beef_result), "$5.99")),
        ("synthetic total remains ambiguous",
         ambiguous_total.get("status") == "ambiguous_structured_evidence"),
        ("synthetic ambiguity still selects no value",
         ambiguous_total.get("selected_candidate") is None),
    ])

    print()
    print("-" * 116)
    print("REGRESSION CHECKS")
    print("-" * 116)

    passed = 0
    for index, (label, ok) in enumerate(checks, start=1):
        print(f"[{'PASS' if ok else 'FAIL'}] {index:02d}. {label}")
        if ok:
            passed += 1

    print()
    print("=" * 116)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("=" * 116)
    print("No OCR, Groq, or network requests were made.")
    return 0 if passed == len(checks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
