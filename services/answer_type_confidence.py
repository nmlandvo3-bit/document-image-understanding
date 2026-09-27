"""
Typed answer inference, candidate validation, and deterministic confidence scoring.

This module is local-only:
- no OCR
- no LLM
- no API calls

The confidence score is a transparent heuristic, not a calibrated probability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


ANSWER_TYPES = (
    "date",
    "time",
    "money",
    "name",
    "identifier",
    "number",
    "free_text",
)

RELATION_BASE = {
    "same_row_right": 0.96,
    "inline_label_value": 0.97,
    "same_row_right_continuation": 0.90,
    "table_column_below": 0.92,
    "below": 0.84,
    "unknown": 0.70,
}

DATE_PATTERN = re.compile(
    r"^(?:\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}|"
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|"
    r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})$",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"^(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?|\d{1,2}\s*[AP]M)$",
    re.IGNORECASE,
)

MONEY_PATTERN = re.compile(
    r"^(?:(?:NT\$|TWD|USD|EUR|GBP|JPY|[$€£¥])\s*)?"
    r"-?\d[\d,]*(?:\.\d{1,2})?"
    r"(?:\s*(?:NTD|TWD|USD|EUR|GBP|JPY))?$",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
NAME_TOKEN_PATTERN = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'’-]+$")


@dataclass(frozen=True)
class CandidateValidation:
    expected_type: str
    candidate: str
    valid: bool
    reason: str


@dataclass(frozen=True)
class ConfidenceResult:
    score: float
    level: str
    expected_type: str
    type_valid: bool
    relation: str
    ocr_confidence: float | None
    ambiguous: bool
    candidate_count: int
    reason: str


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def infer_answer_type(question: str) -> str:
    q = _normalize(question).casefold()

    if not q:
        return "free_text"

    if re.search(
        r"\b(?:what time|which time|payment time|arrival time|"
        r"departure time|start time|end time)\b",
        q,
    ):
        return "time"

    if re.search(
        r"\b(?:date of birth|dob|birth(?:day)? date|payment date|"
        r"issue date|expiry date|expiration date|start date|end date|"
        r"arrival date|departure date|what date|which date|date|when)\b",
        q,
    ):
        return "date"

    if re.search(
        r"\b(?:total|subtotal|tax|cash|change|amount|price|cost|"
        r"balance|fee|payment amount|how much)\b",
        q,
    ):
        return "money"

    if re.search(
        r"\b(?:customer name|patient name|person(?:'s)? name|full name|"
        r"name|who is|who was)\b",
        q,
    ):
        return "name"

    if re.search(
        r"\b(?:reference number|card pick[- ]?up number|id number|"
        r"identifier|serial number|account number|member number|"
        r"policy number|receipt number|order number|invoice number|"
        r"tracking number|code)\b",
        q,
    ):
        return "identifier"

    if re.search(
        r"\b(?:how many|quantity|count|number of|how old|age)\b",
        q,
    ):
        return "number"

    return "free_text"


def looks_like_label(text: str) -> bool:
    t = _normalize(text).casefold()

    return t in {
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


def validate_candidate(expected_type: str, candidate: str) -> CandidateValidation:
    text = _normalize(candidate)

    if expected_type not in ANSWER_TYPES:
        return CandidateValidation(
            expected_type, text, False, "unknown_expected_type"
        )

    if not text:
        return CandidateValidation(
            expected_type, text, False, "empty_candidate"
        )

    if expected_type == "free_text":
        return CandidateValidation(
            expected_type, text, True, "free_text_accepts_nonempty_text"
        )

    if looks_like_label(text):
        return CandidateValidation(
            expected_type, text, False, "candidate_looks_like_label"
        )

    if expected_type == "date":
        valid = bool(DATE_PATTERN.fullmatch(text))
        return CandidateValidation(
            expected_type,
            text,
            valid,
            "date_pattern_match" if valid else "not_date_shaped",
        )

    if expected_type == "time":
        valid = bool(TIME_PATTERN.fullmatch(text))
        return CandidateValidation(
            expected_type,
            text,
            valid,
            "time_pattern_match" if valid else "not_time_shaped",
        )

    if expected_type == "money":
        if "%" in text:
            return CandidateValidation(
                expected_type, text, False, "percentage_is_not_money_amount"
            )

        valid = bool(MONEY_PATTERN.fullmatch(text))
        return CandidateValidation(
            expected_type,
            text,
            valid,
            "money_pattern_match" if valid else "not_money_shaped",
        )

    if expected_type == "number":
        valid = bool(NUMBER_PATTERN.fullmatch(text))
        return CandidateValidation(
            expected_type,
            text,
            valid,
            "number_pattern_match" if valid else "not_numeric",
        )

    if expected_type == "identifier":
        if " " in text:
            return CandidateValidation(
                expected_type, text, False, "identifier_contains_spaces"
            )

        valid = bool(IDENTIFIER_PATTERN.fullmatch(text))
        return CandidateValidation(
            expected_type,
            text,
            valid,
            "identifier_pattern_match" if valid else "not_identifier_shaped",
        )

    if expected_type == "name":
        parts = text.split()

        if not (1 <= len(parts) <= 6):
            return CandidateValidation(
                expected_type, text, False, "name_token_count_out_of_range"
            )

        if any(char.isdigit() for char in text):
            return CandidateValidation(
                expected_type, text, False, "name_contains_digits"
            )

        valid = all(
            NAME_TOKEN_PATTERN.fullmatch(part)
            or (len(part) == 2 and part[0].isalpha() and part[1] == ".")
            for part in parts
        )
        return CandidateValidation(
            expected_type,
            text,
            valid,
            "name_pattern_match" if valid else "not_name_shaped",
        )

    return CandidateValidation(
        expected_type, text, False, "unhandled_type"
    )


def confidence_level(score: float) -> str:
    if score >= 0.90:
        return "high"
    if score >= 0.70:
        return "medium"
    return "low"


def _normalize_ocr_confidence(value: float | None) -> float | None:
    if value is None:
        return None

    value = float(value)

    # Be tolerant of either 0..1 or 0..100 storage.
    if value > 1.0 and value <= 100.0:
        value = value / 100.0

    return max(0.0, min(1.0, value))


def score_candidate(
    *,
    question: str,
    candidate: str,
    relation: str = "unknown",
    ocr_confidence: float | None = None,
    ambiguous: bool = False,
    candidate_count: int = 1,
) -> ConfidenceResult:
    """
    Produce a deterministic confidence heuristic for a retrieved candidate.

    This is NOT a calibrated probability.
    """

    expected_type = infer_answer_type(question)
    validation = validate_candidate(expected_type, candidate)

    relation_score = RELATION_BASE.get(
        relation,
        RELATION_BASE["unknown"],
    )

    score = relation_score
    reasons = [f"relation={relation}:{relation_score:.2f}"]

    if validation.valid:
        score += 0.03
        reasons.append("type_match=+0.03")
    else:
        score -= 0.35
        reasons.append("type_mismatch=-0.35")

    normalized_ocr = _normalize_ocr_confidence(ocr_confidence)

    if normalized_ocr is not None:
        adjustment = (normalized_ocr - 0.85) * 0.20
        score += adjustment
        reasons.append(f"ocr_adjustment={adjustment:+.3f}")

    if candidate_count > 1:
        penalty = min(0.20, 0.07 * (candidate_count - 1))
        score -= penalty
        reasons.append(f"candidate_count_penalty=-{penalty:.2f}")

    if ambiguous:
        score = min(score, 0.49)
        reasons.append("ambiguity_cap=0.49")

    score = round(max(0.0, min(1.0, score)), 3)

    return ConfidenceResult(
        score=score,
        level=confidence_level(score),
        expected_type=expected_type,
        type_valid=validation.valid,
        relation=relation,
        ocr_confidence=normalized_ocr,
        ambiguous=ambiguous,
        candidate_count=max(1, int(candidate_count)),
        reason="; ".join(reasons),
    )
