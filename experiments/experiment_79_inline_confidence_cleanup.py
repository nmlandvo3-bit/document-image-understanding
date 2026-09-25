"""Experiment 79 revision 2: inline-confidence, provenance, and GUI cleanup.

No OCR, Groq, or network calls.

This revision fixes two overly strict test expectations:
- subtotal accepts formatting-equivalent "$ 18.95" / "$18.95";
- GUI wording is checked after collapsing source-code whitespace/string-line layout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from parsing.unified_structured_document_builder import (
    build_unified_structured_document,
)
from services.answer_type_confidence import score_candidate
from services.structured_document_retrieval import retrieve_structured_answer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPERMARKET = PROJECT_ROOT / (
    "results/gui_v4/single/supermarket/unified_raw_ocr.json"
)
SYNTHETIC = PROJECT_ROOT / (
    "results/gui_v4/single/"
    "ChatGPT Image Sep 24, 2026, 06_13_58 PM/"
    "unified_raw_ocr.json"
)
APP_PATH = PROJECT_ROOT / "app.py"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_value(result: dict):
    return (result.get("selected_candidate") or {}).get("value")


def inline_field(document: dict, label: str) -> dict:
    wanted = label.casefold()

    for field in document.get("fields", []):
        if (
            field.get("relationship") == "inline_label_value"
            and str(field.get("label", "")).casefold() == wanted
        ):
            return field

    return {}


def normalize_money_display(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", str(value))


def collapse_source_string_layout(source: str) -> str:
    """Collapse Python source string-literal line layout for text checks."""
    # This does not execute app.py. It simply makes adjacent quoted source
    # fragments comparable as one logical sentence for regression purposes.
    return re.sub(r'"\s*\n\s*"', "", source)


def main() -> int:
    print("=" * 112)
    print("EXPERIMENT 79: INLINE CONFIDENCE + PROVENANCE + GUI CLEANUP")
    print("=" * 112)

    for path in (SUPERMARKET, SYNTHETIC, APP_PATH):
        if not path.is_file():
            print(f"[MISSING] {path}")
            return 1

    supermarket = build_unified_structured_document(load(SUPERMARKET))
    synthetic = build_unified_structured_document(load(SYNTHETIC))

    date_result = retrieve_structured_answer(
        supermarket, "What is the date?"
    )
    cashier_result = retrieve_structured_answer(
        supermarket, "What is the name of the cashier?"
    )
    customer_result = retrieve_structured_answer(
        supermarket, "What is the name of the customer?"
    )
    beef_result = retrieve_structured_answer(
        supermarket, "What is the price of Ground Beef?"
    )
    subtotal_result = retrieve_structured_answer(
        supermarket, "What is subtotal?"
    )
    ambiguous_total = retrieve_structured_answer(
        synthetic, "What is the total?"
    )

    date_field = inline_field(supermarket, "DATE")
    cashier_field = inline_field(supermarket, "CASHIER")
    customer_field = inline_field(supermarket, "CUSTOMER")

    cashier_confidence = score_candidate(
        question="What is the name of the cashier?",
        candidate="Jane Smith",
        relation="inline_label_value",
        ocr_confidence=1.0,
        ambiguous=False,
        candidate_count=1,
    )

    date_confidence = score_candidate(
        question="What is the date?",
        candidate="08/24/2023",
        relation="inline_label_value",
        ocr_confidence=0.985,
        ambiguous=False,
        candidate_count=1,
    )

    app_source = APP_PATH.read_text(encoding="utf-8")
    collapsed_app_source = collapse_source_string_layout(app_source)

    checks = [
        (
            "inline cashier confidence is high",
            cashier_confidence.level == "high",
        ),
        (
            "inline cashier uses explicit relation",
            "relation=inline_label_value:0.97"
            in cashier_confidence.reason,
        ),
        (
            "inline date confidence is high",
            date_confidence.level == "high",
        ),
        (
            "DATE field has one evidence ID",
            len(date_field.get("evidence_item_ids", [])) == 1,
        ),
        (
            "CASHIER field has one evidence ID",
            len(cashier_field.get("evidence_item_ids", [])) == 1,
        ),
        (
            "CUSTOMER field has one evidence ID",
            len(customer_field.get("evidence_item_ids", [])) == 1,
        ),
        (
            "date remains structured",
            date_result.get("status") == "structured_answer_candidate",
        ),
        (
            "date remains 08/24/2023",
            selected_value(date_result) == "08/24/2023",
        ),
        (
            "cashier remains Jane Smith",
            selected_value(cashier_result) == "Jane Smith",
        ),
        (
            "customer remains John Doe",
            selected_value(customer_result) == "John Doe",
        ),
        (
            "Ground Beef remains $5.99",
            normalize_money_display(selected_value(beef_result)) == "$5.99",
        ),
        (
            "subtotal remains $18.95",
            normalize_money_display(selected_value(subtotal_result))
            == "$18.95",
        ),
        (
            "synthetic total remains ambiguous",
            ambiguous_total.get("status")
            == "ambiguous_structured_evidence",
        ),
        (
            "synthetic ambiguity selects no value",
            ambiguous_total.get("selected_candidate") is None,
        ),
        (
            "GUI mentions automatic structure first",
            "New scans use automatic structure first with raw OCR fallback"
            in collapsed_app_source,
        ),
        (
            "old raw-only GUI wording removed",
            "New scans use raw OCR regions, not verified form fields or table structures."
            not in collapsed_app_source,
        ),
        (
            "GUI states generated structure remains unverified",
            "generated structure remains unverified."
            in collapsed_app_source,
        ),
    ]

    print()
    print("Cashier confidence:", cashier_confidence)
    print("Date confidence:", date_confidence)
    print("DATE evidence IDs:", date_field.get("evidence_item_ids"))
    print("CASHIER evidence IDs:", cashier_field.get("evidence_item_ids"))
    print("CUSTOMER evidence IDs:", customer_field.get("evidence_item_ids"))
    print("Subtotal selected value:", repr(selected_value(subtotal_result)))

    print()
    print("-" * 112)
    print("REGRESSION CHECKS")
    print("-" * 112)

    passed = 0
    for index, (label, ok) in enumerate(checks, start=1):
        print(f"[{'PASS' if ok else 'FAIL'}] {index:02d}. {label}")
        if ok:
            passed += 1

    print()
    print("=" * 112)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("=" * 112)
    print("No OCR, Groq, or network requests were made.")

    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
