"""Small, explicit question router for a structured-document corpus.

This is a local grammar, not a general-language QA model. It returns evidence
objects and refuses to guess when a document, field, or comparison is unclear.
"""
from __future__ import annotations

import re
from typing import Any

from services.structured_document_corpus import (
    compare_receipt_totals, find_documents, list_receipt_totals,
)

FIELD_KEY = {
    "receipt number": "receipt_number",
    "cashier name": "cashier",
    "address": "location",
}


def _printed_address(document: dict[str, Any]) -> dict[str, Any] | None:
    """Find distinct full street-address OCR regions, without guessing a city."""
    pattern = re.compile(
        r"\b\d{1,6}\s+[^\n,]{2,80}?\s+"
        r"(?:street|st\.?|avenue|ave\.?|road|rd\.?|"
        r"boulevard|blvd\.?|lane|ln\.?|drive|dr\.?)\b",
        re.I,
    )
    candidates = []
    for region in document.get("evidence", []):
        if region.get("evidence_type") != "ocr_region":
            continue
        text = " ".join(str(region.get("text", "")).split())
        if pattern.search(text):
            candidates.append({
                "observed_value": text,
                "evidence_item_ids": [
                    f"{document['document_id']}:ocr_region:{region['region_id']}"
                ],
            })
    if not candidates:
        return None
    distinct = {candidate["observed_value"].casefold() for candidate in candidates}
    if len(distinct) > 1:
        return {"status": "ambiguous", "candidates": candidates}
    return {
        "status": "candidate",
        "observed_value": candidates[0]["observed_value"],
        "display_value": candidates[0]["observed_value"],
        "evidence_item_ids": list(dict.fromkeys(
            evidence_id for candidate in candidates
            for evidence_id in candidate["evidence_item_ids"]
        )),
    }


def _find_named_document(index: dict[str, Any], name: str) -> dict[str, Any]:
    """Prefer an exact merchant; use an exact heading if none matches."""
    merchant = find_documents(index, "merchant", name)
    if merchant["status"] != "no_match":
        return merchant
    return find_documents(index, "heading", name)


def _answer_field(index: dict[str, Any], match: dict[str, Any], field_name: str) -> dict[str, Any]:
    if match["status"] == "no_match":
        return {"status": "document_not_found", "answer": None, "matches": []}
    if match["status"] == "ambiguous":
        return {"status": "ambiguous_document", "answer": None, "matches": match["matches"]}
    document_id = match["selected_document_id"]
    document = index["by_id"][document_id]
    key = FIELD_KEY.get(field_name, field_name)
    candidate = (
        _printed_address(document) if key == "location"
        else (document.get("receipt_semantics") or {}).get("fields", {}).get(key)
    )
    if not candidate:
        return {"status": "field_not_found", "answer": None, "document_id": document_id}
    if candidate.get("status") == "ambiguous":
        return {"status": "ambiguous_field", "answer": None, "document_id": document_id,
                "candidates": candidate.get("candidates", [])}
    if candidate.get("status") != "candidate" or candidate.get("observed_value") is None:
        return {"status": "field_not_found", "answer": None, "document_id": document_id}
    return {
        "status": "answered", "answer": candidate.get("display_value", candidate["observed_value"]),
        "observed_value": candidate["observed_value"],
        "normalized_value": candidate.get("normalized_value"),
        "currency_symbol": candidate.get("currency_symbol"),
        "document_id": document_id,
        "evidence_item_ids": list(candidate.get("evidence_item_ids", [])),
        "semantic_verified": False,
    }


def answer_corpus_question(index: dict[str, Any], question: str) -> dict[str, Any]:
    """Answer only the supported receipt/document question forms."""
    text = " ".join(str(question).split()).strip().rstrip("?").strip()
    if not text:
        return {"status": "unsupported_question", "answer": None}

    if re.fullmatch(r"(?:list|show) (?:the )?totals (?:of|for) (?:all )?receipts", text, re.I):
        return {"status": "answered", "answer_type": "receipt_totals",
                "totals": list_receipt_totals(index)}
    if re.fullmatch(r"which receipt has (?:the )?highest total", text, re.I):
        comparison = compare_receipt_totals(index)
        if comparison["status"] != "ranked":
            return {"status": "not_comparable", "answer": None,
                    "reason": comparison["reason"], "totals": comparison["totals"]}
        return {"status": "answered", "answer_type": "highest_receipt_total",
                "document_ids": comparison["highest_document_ids"],
                "ranked_totals": comparison["ranked_totals"]}

    lookup = re.fullmatch(
        r"which (?:receipt|document) has (?:receipt )?(?:number|no\.?) ([A-Za-z0-9/-]+)", text, re.I
    )
    if lookup:
        match = find_documents(index, "receipt_number", lookup.group(1))
        return {"status": match["status"], "answer": match["selected_document_id"],
                "matches": match["matches"]}

    field = re.fullmatch(
        r"what is the (date|time|total|merchant|receipt number|cashier|subtotal) "
        r"(?:on|for) receipt (?:number )?([A-Za-z0-9/-]+)", text, re.I
    )
    if field:
        match = find_documents(index, "receipt_number", field.group(2))
        return _answer_field(index, match, field.group(1).casefold())

    natural_receipt = re.fullmatch(
        r"(?:what is the )?(cashier(?: name)?|location|address) "
        r"(?:on|for|of) receipt (?:number )?([A-Za-z0-9/-]+)", text, re.I,
    )
    if natural_receipt:
        match = find_documents(index, "receipt_number", natural_receipt.group(2))
        return _answer_field(index, match, natural_receipt.group(1).casefold())

    by_merchant = re.fullmatch(
        r"what is the (date|time|total|receipt number|cashier|subtotal) "
        r"for (?:merchant )?(.+)", text, re.I
    )
    if by_merchant:
        match = _find_named_document(index, by_merchant.group(2))
        return _answer_field(index, match, by_merchant.group(1).casefold())

    named_field = re.fullmatch(
        r"(?:what is the )?(cashier(?: name)?|location|address) "
        r"(?:for|of|at) (.+)", text, re.I,
    )
    if named_field:
        match = _find_named_document(index, named_field.group(2))
        return _answer_field(index, match, named_field.group(1).casefold())

    where = re.fullmatch(r"where is (.+?) located", text, re.I)
    if where:
        return _answer_field(index, _find_named_document(index, where.group(1)), "location")

    heading = re.fullmatch(r"which document has (?:the )?heading (.+)", text, re.I)
    if heading:
        match = find_documents(index, "heading", heading.group(1))
        return {"status": match["status"], "answer": match["selected_document_id"],
                "matches": match["matches"]}

    return {"status": "unsupported_question", "answer": None}
