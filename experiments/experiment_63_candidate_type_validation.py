from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from experiments.experiment_62_answer_type_inference import infer_answer_type


DATE_PATTERN = re.compile(
    r"^(?:\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})$",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"^(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?|\d{1,2}\s*[AP]M)$",
    re.IGNORECASE,
)

MONEY_PATTERN = re.compile(
    r"^(?:(?:NT\$|TWD|USD|EUR|GBP|JPY|[$€£¥])\s*)?-?\d[\d,]*(?:\.\d{1,2})?(?:\s*(?:NTD|TWD|USD|EUR|GBP|JPY))?$",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"^-?\d[\d,]*(?:\.\d+)?$"
)

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/-]*$"
)

NAME_TOKEN_PATTERN = re.compile(
    r"^[A-Za-zÀ-ÖØ-öø-ÿ'’-]+$"
)


@dataclass(frozen=True)
class ValidationResult:
    expected_type: str
    candidate: str
    valid: bool
    reason: str


def normalize_candidate(candidate: str) -> str:
    return " ".join(str(candidate or "").strip().split())


def looks_like_label(text: str) -> bool:
    """
    Conservative heuristic for label-like OCR text.

    This is intentionally small. We do not want to reject legitimate free text.
    It is mainly used to stop obvious field labels from being accepted as
    values for typed answers.
    """
    t = normalize_candidate(text).casefold()

    label_phrases = {
        "payment date",
        "alternative date",
        "alternate date",
        "date of birth",
        "reference number",
        "card pick-up",
        "card pickup",
        "card pick-up number",
        "id number",
        "customer",
        "customer name",
        "name",
        "total",
        "subtotal",
        "tax",
        "cash",
        "change",
        "amount",
        "price",
        "cost",
        "balance",
        "quantity",
    }

    return t in label_phrases


def validate_candidate(expected_type: str, candidate: str) -> ValidationResult:
    text = normalize_candidate(candidate)

    if expected_type not in {
        "date",
        "time",
        "money",
        "name",
        "identifier",
        "number",
        "free_text",
    }:
        return ValidationResult(
            expected_type=expected_type,
            candidate=text,
            valid=False,
            reason="unknown_expected_type",
        )

    if not text:
        return ValidationResult(
            expected_type=expected_type,
            candidate=text,
            valid=False,
            reason="empty_candidate",
        )

    if expected_type == "free_text":
        return ValidationResult(
            expected_type=expected_type,
            candidate=text,
            valid=True,
            reason="free_text_accepts_nonempty_text",
        )

    if looks_like_label(text):
        return ValidationResult(
            expected_type=expected_type,
            candidate=text,
            valid=False,
            reason="candidate_looks_like_label",
        )

    if expected_type == "date":
        valid = bool(DATE_PATTERN.fullmatch(text))
        return ValidationResult(
            expected_type=expected_type,
            candidate=text,
            valid=valid,
            reason="date_pattern_match" if valid else "not_date_shaped",
        )

    if expected_type == "time":
        valid = bool(TIME_PATTERN.fullmatch(text))
        return ValidationResult(
            expected_type=expected_type,
            candidate=text,
            valid=valid,
            reason="time_pattern_match" if valid else "not_time_shaped",
        )

    if expected_type == "money":
        # Percentages such as "5%" are rates, not monetary amounts.
        if "%" in text:
            return ValidationResult(
                expected_type=expected_type,
                candidate=text,
                valid=False,
                reason="percentage_is_not_money_amount",
            )

        valid = bool(MONEY_PATTERN.fullmatch(text))
        return ValidationResult(
            expected_type=expected_type,
            candidate=text,
            valid=valid,
            reason="money_pattern_match" if valid else "not_money_shaped",
        )

    if expected_type == "number":
        valid = bool(NUMBER_PATTERN.fullmatch(text))
        return ValidationResult(
            expected_type=expected_type,
            candidate=text,
            valid=valid,
            reason="number_pattern_match" if valid else "not_numeric",
        )

    if expected_type == "identifier":
        if " " in text:
            return ValidationResult(
                expected_type=expected_type,
                candidate=text,
                valid=False,
                reason="identifier_contains_spaces",
            )

        valid = bool(IDENTIFIER_PATTERN.fullmatch(text))
        return ValidationResult(
            expected_type=expected_type,
            candidate=text,
            valid=valid,
            reason="identifier_pattern_match" if valid else "not_identifier_shaped",
        )

    if expected_type == "name":
        parts = text.split()

        if not (1 <= len(parts) <= 6):
            return ValidationResult(
                expected_type=expected_type,
                candidate=text,
                valid=False,
                reason="name_token_count_out_of_range",
            )

        if any(char.isdigit() for char in text):
            return ValidationResult(
                expected_type=expected_type,
                candidate=text,
                valid=False,
                reason="name_contains_digits",
            )

        valid = all(NAME_TOKEN_PATTERN.fullmatch(part) for part in parts)

        return ValidationResult(
            expected_type=expected_type,
            candidate=text,
            valid=valid,
            reason="name_pattern_match" if valid else "not_name_shaped",
        )

    return ValidationResult(
        expected_type=expected_type,
        candidate=text,
        valid=False,
        reason="unhandled_type",
    )


def validate_for_question(question: str, candidate: str) -> ValidationResult:
    expected_type = infer_answer_type(question)
    return validate_candidate(expected_type, candidate)


def run_validation() -> int:
    cases = [
        ("What is the payment date?", "115/09/21", True),
        ("What is the payment date?", "Card Pick-up", False),
        ("What is the date of birth?", "03/14/1992", True),
        ("What time was the payment?", "15:01:23", True),
        ("What time was the payment?", "115/09/21", False),
        ("What is the total?", "310", True),
        ("What is the tax?", "15", True),
        ("What is the tax?", "5%", False),
        ("What is the cash amount?", "NT$500", True),
        ("What is the customer name?", "Alex Carter", True),
        ("What is the customer name?", "03/14/1992", False),
        ("What is the reference number?", "0055", True),
        ("What is the reference number?", "Reference Number", False),
        ("What is the ID number?", "AB-12345", True),
        ("How many items are listed?", "12", True),
        ("How many items are listed?", "twelve", False),
        ("What is the address?", "No. 1, Sec. 2, Da Hsueh Rd.", True),
        ("What is written in the notes section?", "Paid at front desk", True),
    ]

    print("=" * 92)
    print("EXPERIMENT 63: CANDIDATE TYPE VALIDATION")
    print("=" * 92)

    passed = 0

    for i, (question, candidate, expected_valid) in enumerate(cases, start=1):
        result = validate_for_question(question, candidate)
        ok = result.valid == expected_valid

        status = "PASS" if ok else "FAIL"

        print(
            f"[{status}] {i:02d}. "
            f"question={question!r} | "
            f"type={result.expected_type!r} | "
            f"candidate={candidate!r} | "
            f"valid={result.valid} | "
            f"reason={result.reason}"
        )

        if ok:
            passed += 1

    print("=" * 92)
    print(f"RESULT: {passed}/{len(cases)} tests passed")
    print("=" * 92)

    return 0 if passed == len(cases) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 63: deterministic candidate-type validation."
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="Optional natural-language question.",
    )

    parser.add_argument(
        "candidate",
        nargs="?",
        help="Optional candidate answer to validate.",
    )

    args = parser.parse_args()

    if args.question is None and args.candidate is None:
        return run_validation()

    if not args.question or args.candidate is None:
        parser.error("Provide both question and candidate, or neither.")

    result = validate_for_question(args.question, args.candidate)

    print(f"QUESTION: {args.question}")
    print(f"EXPECTED ANSWER TYPE: {result.expected_type}")
    print(f"CANDIDATE: {result.candidate}")
    print(f"VALID: {result.valid}")
    print(f"REASON: {result.reason}")

    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
