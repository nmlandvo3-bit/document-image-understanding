"""Experiment 73: structured Q&A retrieval integration - revision 4."""

from __future__ import annotations

import json
from pathlib import Path

from parsing.unified_structured_document_builder import (
    build_unified_structured_document,
)
from services.structured_document_retrieval import (
    retrieve_structured_answer,
)


CASES = {
    "receipt": Path(
        "results/gui_v4/single/receipt/unified_raw_ocr.json"
    ),
    "pic13": Path(
        "results/gui_v4/single/pic13/unified_raw_ocr.json"
    ),
    "nhi": Path(
        "results/gui_v4/single/"
        "WhatsApp Image 2026-09-22 at 22.21.24/"
        "unified_raw_ocr.json"
    ),
    "synthetic": Path(
        "results/gui_v4/single/"
        "ChatGPT Image Sep 24, 2026, 06_13_58 PM/"
        "unified_raw_ocr.json"
    ),
}


def load_structured(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return build_unified_structured_document(raw)


def value_of(result: dict) -> str | None:
    selected = result.get("selected_candidate")
    return None if selected is None else selected.get("value")


def print_case(label: str, result: dict) -> None:
    print()
    print(label)
    print(f"  status={result['status']}")
    print(f"  answer_type={result['answer_type']}")
    print(f"  selected={value_of(result)!r}")
    print(f"  fallback={result['fallback_recommended']}")

    if result.get("ambiguities"):
        print(
            "  ambiguity_values="
            + repr(
                [
                    item["value"]
                    for item in result["ambiguities"]
                ]
            )
        )


def main() -> int:
    print("=" * 110)
    print("EXPERIMENT 73: STRUCTURED Q&A RETRIEVAL INTEGRATION - REVISION 4")
    print("=" * 110)

    structured = {}

    for name, path in CASES.items():
        if not path.is_file():
            print(f"[MISSING] {name}: {path}")
            return 1
        structured[name] = load_structured(path)

    tests = [
        (
            "pic13 customer name",
            "pic13",
            "What is the customer name?",
            "Alex Carter",
            "structured_answer_candidate",
            False,
        ),
        (
            "pic13 date of birth",
            "pic13",
            "What is the date of birth?",
            "03/14/1992",
            "structured_answer_candidate",
            False,
        ),
        (
            "pic13 Medicare ID",
            "pic13",
            "What is the Medicare Identification Number?",
            "1EG4-TE5-MK72",
            "structured_answer_candidate",
            False,
        ),
        (
            "receipt total",
            "receipt",
            "What is the total?",
            "310",
            "structured_answer_candidate",
            False,
        ),
        (
            "receipt tax",
            "receipt",
            "What is the tax?",
            "15",
            "structured_answer_candidate",
            False,
        ),
        (
            "NHI payment date",
            "nhi",
            "What is the payment date?",
            "115/09/21",
            "structured_answer_candidate",
            False,
        ),
        (
            "NHI card pickup",
            "nhi",
            "What is the Card Pick-up Number?",
            "0055",
            "structured_answer_candidate",
            False,
        ),
    ]

    results = {}

    for label, doc_name, question, expected, status, fallback in tests:
        result = retrieve_structured_answer(
            structured[doc_name],
            question,
        )
        results[label] = result
        print_case(label, result)

    synthetic_total = retrieve_structured_answer(
        structured["synthetic"],
        "What is the total?",
    )
    print_case("synthetic total ambiguity", synthetic_total)

    absent = retrieve_structured_answer(
        structured["receipt"],
        "What is the date of birth?",
    )
    print_case("receipt absent DOB", absent)

    checks = []

    for label, _, _, expected, expected_status, expected_fallback in tests:
        result = results[label]
        checks.extend(
            [
                (f"{label}: status", result["status"] == expected_status),
                (f"{label}: value", value_of(result) == expected),
                (
                    f"{label}: no fallback",
                    result["fallback_recommended"] == expected_fallback,
                ),
            ]
        )

    checks.extend(
        [
            (
                "synthetic total is ambiguous",
                synthetic_total["status"]
                == "ambiguous_structured_evidence",
            ),
            (
                "synthetic total does not choose a value",
                synthetic_total["selected_candidate"] is None,
            ),
            (
                "synthetic total does not recommend fallback over known ambiguity",
                synthetic_total["fallback_recommended"] is False,
            ),
            (
                "receipt absent DOB recommends raw OCR fallback",
                absent["fallback_recommended"] is True,
            ),
            (
                "receipt absent DOB has no selected structured answer",
                absent["selected_candidate"] is None,
            ),
        ]
    )

    print()
    print("-" * 110)
    print("STRUCTURED Q&A CHECKS")
    print("-" * 110)

    passed = 0

    for index, (label, ok) in enumerate(checks, start=1):
        print(f"[{'PASS' if ok else 'FAIL'}] {index:02d}. {label}")
        if ok:
            passed += 1

    print()
    print("=" * 110)
    print(
        f"RESULT: {passed}/{len(checks)} "
        "structured Q&A checks passed"
    )
    print("=" * 110)
    print("No Groq requests were made.")

    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
