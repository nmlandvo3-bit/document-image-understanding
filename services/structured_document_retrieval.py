"""Structured document retrieval for Experiment 73 - revision 4.

Final refinement:
- rejects label leakage when a candidate value is just an alphabetic fragment
  already contained in the question (for example, "Number" from a split
  "Card Pick-up / Number" table header);
- preserves genuine identifiers such as "0055";
- keeps cross-source ambiguity detection for cases such as synthetic TOTAL.

No OCR, LLM, API, or document-specific field names are used.
"""

from __future__ import annotations

import re
from typing import Any

from services.answer_type_confidence import (
    infer_answer_type,
    validate_candidate,
)


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "for", "to", "in", "on",
    "was", "were", "are", "and", "please", "give", "me", "show",
}


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_RE.findall(_normalize(text))
        if token.casefold() not in STOPWORDS
    }


def _label_score(question: str, label: str) -> float:
    q = _tokens(question)
    l = _tokens(label)

    if not q or not l:
        return 0.0

    overlap = len(q & l)
    union = len(q | l)
    score = overlap / union if union else 0.0

    q_text = _normalize(question).casefold()
    l_text = _normalize(label).casefold()

    if l_text and l_text in q_text:
        score += 1.0

    return score


def _normalize_typed_candidate_value(expected_type: str, value: str) -> tuple[str, bool]:
    """Normalize only punctuation noise before an explicit currency marker."""
    text = _normalize(value)
    if expected_type != "money" or not text:
        return text, False

    if validate_candidate(expected_type, text).valid:
        return text, False

    working = text.lstrip()
    punctuation = ".,;:|_~`'\"!?*·•"
    removed_prefix = False

    while working and working[0] in punctuation:
        working = working[1:].lstrip()
        removed_prefix = True

    if working.startswith("NT$"):
        prefix = "NT$"
    elif working[:1] in {"$", "€", "£", "¥"}:
        prefix = working[0]
    else:
        return text, False

    if not removed_prefix:
        return text, False

    candidate = prefix + working[len(prefix):].lstrip()
    if not validate_candidate(expected_type, candidate).valid:
        return text, False

    return candidate, candidate != text

def _candidate_type_valid(
    expected_type: str,
    value: str,
) -> tuple[bool, str]:
    validation = validate_candidate(expected_type, value)
    return bool(validation.valid), validation.reason


def _looks_like_question_label_leakage(
    question: str,
    value: str,
) -> bool:
    """Reject short alphabetic answer candidates copied from the question.

    This is useful for split table headers such as:
        Card Pick-up
        Number
        ...
        0055

    "Number" passes a broad identifier regex, but because it is only an
    alphabetic token already present in the question, it is much more likely
    to be a header continuation than the answer.
    """
    text = _normalize(value)

    if not text:
        return False

    if any(ch.isdigit() for ch in text):
        return False

    if not all(ch.isalpha() or ch.isspace() or ch in "-'" for ch in text):
        return False

    value_tokens = _tokens(text)
    question_tokens = _tokens(question)

    if not value_tokens:
        return False

    return value_tokens.issubset(question_tokens)


_DOCUMENT_IDENTITY_CUES = {
    "title", "store", "merchant", "company", "issuer",
    "supermarket", "business", "organization", "document name",
}

def _asks_for_document_identity(question: str) -> bool:
    text = _normalize(question).casefold()
    if "document name" in text:
        return True
    return bool(
        _tokens(question)
        & {
            "title", "store", "merchant", "company", "issuer",
            "supermarket", "business", "organization",
        }
    )

def _heading_identity_candidates(
    document: dict[str, Any],
    question: str,
    expected_type: str,
) -> list[dict[str, Any]]:
    if not _asks_for_document_identity(question):
        return []

    heading_sections = [
        section
        for section in document.get("sections", [])
        if section.get("structure_type") == "heading"
    ]
    if not heading_sections:
        return []

    first_section = min(
        heading_sections,
        key=lambda section: (
            int(section.get("row_start", 10**9)),
            str(section.get("section_id", "")),
        ),
    )

    row_ids = {
        int(row_id)
        for row_id in first_section.get("row_ids", [])
        if row_id is not None
    }

    heading_items = [
        item
        for item in document.get("free_text", [])
        if (
            item.get("structure_type") == "heading"
            and item.get("row_id") is not None
            and int(item.get("row_id")) in row_ids
        )
    ]
    heading_items.sort(
        key=lambda item: (
            int(item.get("row_id", 10**9)),
            int(item.get("region_id", 10**9)),
        )
    )

    texts = [
        _normalize(item.get("text", ""))
        for item in heading_items
        if _normalize(item.get("text", ""))
    ]
    if not texts:
        return []

    value = " ".join(texts)
    valid, reason = _candidate_type_valid(expected_type, value)

    evidence_item_ids = list(
        dict.fromkeys(
            evidence_id
            for item in heading_items
            for evidence_id in item.get("evidence_item_ids", [])
        )
    )

    return [{
        "source": "structured_heading",
        "label": "document_identity_heading",
        "value": value,
        "observed_value": value,
        "value_normalized_for_type": False,
        "lexical_score": 3.0,
        "type_valid": valid,
        "type_validation_reason": reason,
        "evidence_item_ids": evidence_item_ids,
        "section_id": first_section.get("section_id"),
        "relationship": "document_identity_heading",
    }]

def _field_candidates(
    document: dict[str, Any],
    question: str,
    expected_type: str,
) -> list[dict[str, Any]]:
    candidates = []

    for field in document.get("fields", []):
        label = _normalize(field.get("label", ""))
        observed_value = _normalize(field.get("value", ""))
        value, value_normalized = _normalize_typed_candidate_value(
            expected_type, observed_value
        )

        if not label or not value:
            continue

        lexical_score = _label_score(question, label)
        if lexical_score <= 0:
            continue

        valid, reason = _candidate_type_valid(expected_type, value)
        if valid and value_normalized:
            reason = "money_pattern_match_after_ocr_punctuation_normalization"

        if valid and _looks_like_question_label_leakage(question, value):
            valid = False
            reason = "candidate_matches_question_label_fragment"

        candidates.append(
            {
                "source": "structured_field",
                "label": label,
                "value": value,
                "observed_value": observed_value,
                "value_normalized_for_type": value_normalized,
                "lexical_score": round(lexical_score, 6),
                "type_valid": valid,
                "type_validation_reason": reason,
                "evidence_item_ids": list(
                    field.get("evidence_item_ids", [])
                ),
                "field_id": field.get("field_id"),
                "relationship": field.get("relationship"),
            }
        )

    return candidates


def _table_candidates(
    document: dict[str, Any],
    question: str,
    expected_type: str,
) -> list[dict[str, Any]]:
    candidates = []
    records_by_table: dict[str, list[dict[str, Any]]] = {}

    for record in document.get("records", []):
        records_by_table.setdefault(
            str(record.get("table_id")),
            [],
        ).append(record)

    for table_id, records in records_by_table.items():
        # A bare quantity such as "2" can satisfy the broad money validator.
        # Use an explicit Qty/Quantity table header to keep it out of money
        # answers while retaining currencyless prices in two-column receipts.
        quantity_columns: set[int] = set()
        if expected_type == "money":
            for header in records:
                header_cells = header.get("cells", [])
                labels = {_normalize(c.get("text", "")).casefold() for c in header_cells}
                if not ({"item", "price"} <= labels):
                    continue
                quantity_columns.update(
                    int(c.get("column_index", -1)) for c in header_cells
                    if _normalize(c.get("text", "")).casefold() in {"qty", "quantity"}
                )
        for record_index, record in enumerate(records):
            cells = record.get("cells", [])

            for cell in cells:
                label = _normalize(cell.get("text", ""))
                lexical_score = _label_score(question, label)

                if lexical_score <= 0:
                    continue

                label_col = int(cell.get("column_index", -1))

                for other in cells:
                    if other is cell:
                        continue

                    if (expected_type == "money"
                            and int(other.get("column_index", -1)) in quantity_columns):
                        continue

                    observed_value = _normalize(other.get("text", ""))
                    value, value_normalized = _normalize_typed_candidate_value(
                        expected_type, observed_value
                    )
                    if not value:
                        continue

                    valid, reason = _candidate_type_valid(
                        expected_type,
                        value,
                    )
                    if valid and value_normalized:
                        reason = "money_pattern_match_after_ocr_punctuation_normalization"

                    if valid and _looks_like_question_label_leakage(
                        question,
                        value,
                    ):
                        valid = False
                        reason = "candidate_matches_question_label_fragment"

                    candidates.append(
                        {
                            "source": "structured_table_same_row",
                            "label": label,
                            "value": value,
                            "observed_value": observed_value,
                            "value_normalized_for_type": value_normalized,
                            "lexical_score": round(lexical_score, 6),
                            "type_valid": valid,
                            "type_validation_reason": reason,
                            "evidence_item_ids": list(
                                dict.fromkeys(
                                    list(
                                        cell.get(
                                            "evidence_item_ids",
                                            [],
                                        )
                                    )
                                    + list(
                                        other.get(
                                            "evidence_item_ids",
                                            [],
                                        )
                                    )
                                )
                            ),
                            "table_id": table_id,
                            "label_column_index": label_col,
                            "value_column_index": int(
                                other.get("column_index", -1)
                            ),
                            "source_row_id": record.get(
                                "source_row_id"
                            ),
                        }
                    )

                for later_record in records[record_index + 1:]:
                    for below_cell in later_record.get("cells", []):
                        if (
                            int(
                                below_cell.get(
                                    "column_index",
                                    -1,
                                )
                            )
                            != label_col
                        ):
                            continue

                        observed_value = _normalize(
                            below_cell.get("text", "")
                        )
                        value, value_normalized = _normalize_typed_candidate_value(
                            expected_type, observed_value
                        )
                        if not value:
                            continue

                        valid, reason = _candidate_type_valid(
                            expected_type,
                            value,
                        )
                        if valid and value_normalized:
                            reason = "money_pattern_match_after_ocr_punctuation_normalization"

                        if valid and _looks_like_question_label_leakage(
                            question,
                            value,
                        ):
                            valid = False
                            reason = "candidate_matches_question_label_fragment"

                        candidates.append(
                            {
                                "source": "structured_table_same_column",
                                "label": label,
                                "value": value,
                                "observed_value": observed_value,
                                "value_normalized_for_type": value_normalized,
                                "lexical_score": round(
                                    lexical_score,
                                    6,
                                ),
                                "type_valid": valid,
                                "type_validation_reason": reason,
                                "evidence_item_ids": list(
                                    dict.fromkeys(
                                        list(
                                            cell.get(
                                                "evidence_item_ids",
                                                [],
                                            )
                                        )
                                        + list(
                                            below_cell.get(
                                                "evidence_item_ids",
                                                [],
                                            )
                                        )
                                    )
                                ),
                                "table_id": table_id,
                                "label_column_index": label_col,
                                "value_column_index": label_col,
                                "source_row_id": later_record.get(
                                    "source_row_id"
                                ),
                            }
                        )

    return candidates


def _deduplicate(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique = []
    seen = set()

    for candidate in candidates:
        key = (
            candidate["source"],
            candidate["label"].casefold(),
            candidate["value"].casefold(),
            tuple(candidate.get("evidence_item_ids", [])),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(candidate)

    return unique


def _source_rank(source: str) -> int:
    ranks = {
        "structured_table_same_row": 0,
        "structured_field": 1,
        "structured_table_same_column": 2,
    }
    return ranks.get(source, 9)


def retrieve_structured_answer(
    document: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    expected_type = infer_answer_type(question)

    all_candidates = _deduplicate(
        _heading_identity_candidates(
            document,
            question,
            expected_type,
        )
        + _field_candidates(
            document,
            question,
            expected_type,
        )
        + _table_candidates(
            document,
            question,
            expected_type,
        )
    )

    if not all_candidates:
        return {
            "status": "no_structured_evidence",
            "answer_type": expected_type,
            "candidates": [],
            "selected_candidate": None,
            "ambiguities": [],
            "fallback_recommended": True,
        }

    valid = [
        candidate
        for candidate in all_candidates
        if candidate["type_valid"]
    ]

    if not valid:
        return {
            "status": "structured_type_mismatch",
            "answer_type": expected_type,
            "candidates": all_candidates,
            "selected_candidate": None,
            "ambiguities": [],
            "fallback_recommended": True,
        }

    best_lexical = max(
        candidate["lexical_score"]
        for candidate in valid
    )

    top = [
        candidate
        for candidate in valid
        if abs(
            candidate["lexical_score"] - best_lexical
        ) <= 1e-9
    ]

    distinct_values = {
        candidate["value"].casefold()
        for candidate in top
    }

    if len(distinct_values) > 1:
        top.sort(
            key=lambda item: (
                _source_rank(item["source"]),
                item["value"].casefold(),
            )
        )

        return {
            "status": "ambiguous_structured_evidence",
            "answer_type": expected_type,
            "candidates": all_candidates,
            "selected_candidate": None,
            "ambiguities": top,
            "fallback_recommended": False,
        }

    top.sort(
        key=lambda item: (
            _source_rank(item["source"]),
            item["value"].casefold(),
        )
    )

    return {
        "status": "structured_answer_candidate",
        "answer_type": expected_type,
        "candidates": all_candidates,
        "selected_candidate": top[0],
        "ambiguities": [],
        "fallback_recommended": False,
    }
