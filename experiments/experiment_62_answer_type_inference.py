from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable


ANSWER_TYPES = (
    "date",
    "time",
    "money",
    "name",
    "identifier",
    "number",
    "free_text",
)


@dataclass(frozen=True)
class TypeRule:
    answer_type: str
    patterns: tuple[re.Pattern[str], ...]


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


RULES: tuple[TypeRule, ...] = (
    TypeRule(
        "date",
        _compile(
            r"\bdate of birth\b",
            r"\bdob\b",
            r"\bbirth(?:day)? date\b",
            r"\bpayment date\b",
            r"\bissue date\b",
            r"\bexpiry date\b",
            r"\bexpiration date\b",
            r"\bstart date\b",
            r"\bend date\b",
            r"\barrival date\b",
            r"\bdeparture date\b",
            r"\bwhat date\b",
            r"\bwhich date\b",
            r"\bwhen\b",
        ),
    ),
    TypeRule(
        "time",
        _compile(
            r"\bpayment time\b",
            r"\barrival time\b",
            r"\bdeparture time\b",
            r"\bstart time\b",
            r"\bend time\b",
            r"\bwhat time\b",
            r"\bwhich time\b",
        ),
    ),
    TypeRule(
        "money",
        _compile(
            r"\btotal\b",
            r"\bsubtotal\b",
            r"\btax\b",
            r"\bcash\b",
            r"\bchange\b",
            r"\bamount\b",
            r"\bprice\b",
            r"\bcost\b",
            r"\bbalance\b",
            r"\bfee\b",
            r"\bpayment amount\b",
            r"\bhow much\b",
        ),
    ),
    TypeRule(
        "name",
        _compile(
            r"\bcustomer name\b",
            r"\bpatient name\b",
            r"\bperson(?:'s)? name\b",
            r"\bfull name\b",
            r"\bname\b",
            r"\bwho is\b",
            r"\bwho was\b",
        ),
    ),
    TypeRule(
        "identifier",
        _compile(
            r"\breference number\b",
            r"\bcard pick[- ]?up number\b",
            r"\bid number\b",
            r"\bidentifier\b",
            r"\bserial number\b",
            r"\baccount number\b",
            r"\bmember number\b",
            r"\bpolicy number\b",
            r"\breceipt number\b",
            r"\border number\b",
            r"\binvoice number\b",
            r"\btracking number\b",
            r"\bcode\b",
        ),
    ),
    TypeRule(
        "number",
        _compile(
            r"\bhow many\b",
            r"\bquantity\b",
            r"\bcount\b",
            r"\bnumber of\b",
            r"\bhow old\b",
            r"\bage\b",
        ),
    ),
)


def normalize_question(question: str) -> str:
    return " ".join(str(question or "").strip().split())


def infer_answer_type(question: str) -> str:
    """
    Infer the expected answer type from a natural-language question.

    The function is deterministic and local:
    - no OCR
    - no LLM
    - no API calls

    Rule precedence is intentional. For example:
    - "payment date" -> date, not money
    - "reference number" -> identifier, not generic number
    - "customer name" -> name
    """
    q = normalize_question(question)

    if not q:
        return "free_text"

    # Handle time before date for questions containing both words such as
    # "What time was the payment date recorded?".
    explicit_time = (
        re.search(r"\bwhat time\b", q, re.IGNORECASE)
        or re.search(r"\bwhich time\b", q, re.IGNORECASE)
        or re.search(r"\bpayment time\b", q, re.IGNORECASE)
        or re.search(r"\barrival time\b", q, re.IGNORECASE)
        or re.search(r"\bdeparture time\b", q, re.IGNORECASE)
        or re.search(r"\bstart time\b", q, re.IGNORECASE)
        or re.search(r"\bend time\b", q, re.IGNORECASE)
    )
    if explicit_time:
        return "time"

    for rule in RULES:
        if rule.answer_type == "time":
            continue

        if any(pattern.search(q) for pattern in rule.patterns):
            return rule.answer_type

    return "free_text"


def run_validation() -> int:
    cases: list[tuple[str, str]] = [
        ("What is the payment date?", "date"),
        ("When was the payment made?", "date"),
        ("What is the date of birth?", "date"),
        ("What time was the payment?", "time"),
        ("What is the total?", "money"),
        ("How much cash was paid?", "money"),
        ("What is the tax?", "money"),
        ("What is the customer name?", "name"),
        ("Who is the customer?", "name"),
        ("What is the reference number?", "identifier"),
        ("What is the Card Pick-up Number?", "identifier"),
        ("What is the ID number?", "identifier"),
        ("How many items are listed?", "number"),
        ("What is the quantity?", "number"),
        ("What is the address?", "free_text"),
        ("What is written in the notes section?", "free_text"),
    ]

    print("=" * 84)
    print("EXPERIMENT 62: EXPECTED ANSWER TYPE INFERENCE")
    print("=" * 84)

    passed = 0

    for i, (question, expected) in enumerate(cases, start=1):
        actual = infer_answer_type(question)
        ok = actual == expected
        status = "PASS" if ok else "FAIL"

        print(
            f"[{status}] {i:02d}. "
            f"{question!r} -> {actual!r} "
            f"(expected {expected!r})"
        )

        if ok:
            passed += 1

    print("=" * 84)
    print(f"RESULT: {passed}/{len(cases)} tests passed")
    print("=" * 84)

    return 0 if passed == len(cases) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 62: deterministic expected-answer type inference."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Optional single question to classify. If omitted, run the built-in validation suite.",
    )
    args = parser.parse_args()

    if args.question:
        answer_type = infer_answer_type(args.question)
        print(f"QUESTION: {args.question}")
        print(f"EXPECTED ANSWER TYPE: {answer_type}")
        return 0

    return run_validation()


if __name__ == "__main__":
    raise SystemExit(main())
