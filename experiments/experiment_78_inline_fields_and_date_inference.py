"""Experiment 78: inline LABEL: VALUE + generic date inference regression."""

from __future__ import annotations

import json
from pathlib import Path

from services.answer_type_confidence import infer_answer_type
from parsing.unified_structured_document_builder import build_unified_structured_document
from services.structured_document_retrieval import retrieve_structured_answer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPERMARKET = PROJECT_ROOT / "results/gui_v4/single/supermarket/unified_raw_ocr.json"
PIC13 = PROJECT_ROOT / "results/gui_v4/single/pic13/unified_raw_ocr.json"
SYNTHETIC = PROJECT_ROOT / (
    "results/gui_v4/single/"
    "ChatGPT Image Sep 24, 2026, 06_13_58 PM/"
    "unified_raw_ocr.json"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def has_field(document: dict, label: str, value: str) -> bool:
    label_cf = label.casefold()
    value_cf = value.casefold()
    return any(
        label_cf == str(field.get("label", "")).casefold()
        and value_cf == str(field.get("value", "")).casefold()
        for field in document.get("fields", [])
    )


def selected_value(result: dict):
    return (result.get("selected_candidate") or {}).get("value")


def main() -> int:
    print("=" * 112)
    print("EXPERIMENT 78: INLINE LABEL/VALUE + GENERIC DATE INFERENCE")
    print("=" * 112)

    for path in (SUPERMARKET, PIC13, SYNTHETIC):
        if not path.is_file():
            print(f"[MISSING] {path}")
            return 1

    supermarket = build_unified_structured_document(load(SUPERMARKET))
    pic13 = build_unified_structured_document(load(PIC13))
    synthetic = build_unified_structured_document(load(SYNTHETIC))

    date_result = retrieve_structured_answer(supermarket, "What is the date?")
    cashier_result = retrieve_structured_answer(
        supermarket, "What is the name of the cashier?"
    )
    customer_result = retrieve_structured_answer(
        supermarket, "What is the name of the customer?"
    )
    tax_result = retrieve_structured_answer(supermarket, "What is the tax?")
    beef_result = retrieve_structured_answer(
        supermarket, "What is the price of Ground Beef?"
    )
    ambiguous_total = retrieve_structured_answer(synthetic, "What is the total?")
    pic13_dob = retrieve_structured_answer(pic13, "What is the date of birth?")

    checks = [
        ("generic date wording infers date",
         infer_answer_type("What is the date?") == "date"),
        ("what was the Date infers date",
         infer_answer_type("what was the Date?") == "date"),
        ("When was this still infers date",
         infer_answer_type("When was this?") == "date"),
        ("DATE inline field reconstructed",
         has_field(supermarket, "DATE", "08/24/2023")),
        ("TIME inline field reconstructed",
         has_field(supermarket, "TIME", "14:48")),
        ("CASHIER inline field reconstructed",
         has_field(supermarket, "CASHIER", "Jane Smith")),
        ("CUSTOMER inline field reconstructed",
         has_field(supermarket, "CUSTOMER", "John Doe")),
        ("REG inline field reconstructed",
         has_field(supermarket, "REG", "3")),
        ("TRN inline field reconstructed",
         has_field(supermarket, "TRN", "9887")),
        ("date question structured",
         date_result.get("status") == "structured_answer_candidate"),
        ("date answer correct", selected_value(date_result) == "08/24/2023"),
        ("cashier question structured",
         cashier_result.get("status") == "structured_answer_candidate"),
        ("cashier answer correct", selected_value(cashier_result) == "Jane Smith"),
        ("customer question structured",
         customer_result.get("status") == "structured_answer_candidate"),
        ("customer answer correct", selected_value(customer_result) == "John Doe"),
        ("tax remains $1.14", selected_value(tax_result) == "$1.14"),
        ("Ground Beef remains $5.99", selected_value(beef_result) == "$5.99"),
        ("synthetic total remains ambiguous",
         ambiguous_total.get("status") == "ambiguous_structured_evidence"),
        ("synthetic ambiguity selects no value",
         ambiguous_total.get("selected_candidate") is None),
        ("pic13 DOB remains structured",
         pic13_dob.get("status") == "structured_answer_candidate"),
        ("pic13 DOB remains 03/14/1992",
         selected_value(pic13_dob) == "03/14/1992"),
        ("inline fields use single-region provenance",
         all(
             f.get("label_region_id") == f.get("value_region_id")
             for f in supermarket.get("fields", [])
             if f.get("relationship") == "inline_label_value"
         )),
        ("inline fields remain semantically unverified",
         all(
             f.get("semantic_verified") is False
             for f in supermarket.get("fields", [])
             if f.get("relationship") == "inline_label_value"
         )),
    ]

    print()
    print("Inline supermarket fields:")
    for field in supermarket.get("fields", []):
        if field.get("relationship") == "inline_label_value":
            print(field)

    print()
    print("Date retrieval:", date_result)
    print("Cashier retrieval:", cashier_result)
    print("Customer retrieval:", customer_result)

    print()
    print("-" * 112)
    passed = 0
    for index, (label, ok) in enumerate(checks, 1):
        print(f"[{'PASS' if ok else 'FAIL'}] {index:02d}. {label}")
        if ok:
            passed += 1

    print("=" * 112)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("=" * 112)
    print("No OCR, Groq, or network requests were made.")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
