"""Conservative multi-document lookup over already-built structured documents.

No OCR, LLM, network, or persistence. Exact identity matching prevents a
similarly named document from being silently selected.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def index_structured_documents(documents: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ordered = []
    by_id = {}
    for document in documents:
        document_id = _clean(document.get("document_id"))
        if not document_id:
            raise ValueError("Every indexed document needs a document_id")
        if document_id in by_id:
            raise ValueError(f"Duplicate document_id: {document_id}")
        ordered.append(document)
        by_id[document_id] = document
    return {"documents": ordered, "by_id": by_id}


def _first_heading(document: dict[str, Any]) -> dict[str, Any] | None:
    headings = [s for s in document.get("sections", []) if s.get("structure_type") == "heading"]
    if not headings:
        return None
    first = min(headings, key=lambda s: (int(s.get("row_start", 10**9)), str(s.get("section_id", ""))))
    row_ids = {int(row) for row in first.get("row_ids", []) if row is not None}
    items = [
        item for item in document.get("free_text", [])
        if item.get("structure_type") == "heading" and item.get("row_id") is not None
        and int(item["row_id"]) in row_ids
    ]
    items.sort(key=lambda item: (int(item["row_id"]), int(item.get("region_id", 10**9))))
    value = " ".join(filter(None, (_clean(item.get("text")) for item in items)))
    if not value:
        return None
    return {"observed_value": value, "evidence_item_ids": list(dict.fromkeys(
        evidence_id for item in items for evidence_id in item.get("evidence_item_ids", [])
    ))}


def find_documents(index: dict[str, Any], key: str, value: str) -> dict[str, Any]:
    """Match merchant, heading, receipt_number, or transaction_number exactly."""
    allowed = {"merchant", "heading", "receipt_number", "transaction_number"}
    if key not in allowed:
        raise ValueError(f"Unsupported lookup key: {key}")
    needle = _clean(value).casefold()
    if not needle:
        raise ValueError("Lookup value cannot be empty")
    matches = []
    for document in index["documents"]:
        if key == "heading":
            candidate = _first_heading(document)
        else:
            candidate = (document.get("receipt_semantics") or {}).get("fields", {}).get(key)
        if not candidate or candidate.get("status") == "ambiguous":
            continue
        observed = _clean(candidate.get("observed_value"))
        if observed.casefold() == needle:
            matches.append({
                "document_id": document["document_id"],
                "match_type": key,
                "observed_value": observed,
                "evidence_item_ids": list(candidate.get("evidence_item_ids", [])),
            })
    return {
        "status": "one_match" if len(matches) == 1 else ("ambiguous" if matches else "no_match"),
        "selected_document_id": matches[0]["document_id"] if len(matches) == 1 else None,
        "matches": matches,
    }


def list_receipt_totals(index: dict[str, Any]) -> list[dict[str, Any]]:
    totals = []
    for document in index["documents"]:
        receipt = document.get("receipt_semantics")
        if not receipt or receipt.get("status") != "receipt_candidate":
            continue
        total = receipt.get("fields", {}).get("total") or {}
        merchant = receipt.get("fields", {}).get("merchant") or {}
        totals.append({
            "document_id": document["document_id"],
            "merchant": merchant.get("observed_value"),
            "status": total.get("status", "missing"),
            "observed_value": total.get("observed_value"),
            "normalized_value": total.get("normalized_value"),
            "currency_symbol": total.get("currency_symbol"),
            "evidence_item_ids": list(total.get("evidence_item_ids", [])),
        })
    return totals


def compare_receipt_totals(index: dict[str, Any]) -> dict[str, Any]:
    """Rank only unambiguous totals with an explicit common currency."""
    totals = list_receipt_totals(index)
    if len(totals) < 2:
        return {"status": "not_comparable", "reason": "fewer_than_two_receipts", "totals": totals}
    if any(t["status"] != "candidate" or t["normalized_value"] is None for t in totals):
        return {"status": "not_comparable", "reason": "missing_or_ambiguous_total", "totals": totals}
    currencies = [t["currency_symbol"] for t in totals]
    if any(currency is None for currency in currencies):
        return {"status": "not_comparable", "reason": "currency_unknown", "totals": totals}
    if len(set(currencies)) != 1:
        return {"status": "not_comparable", "reason": "currency_mismatch", "totals": totals}
    try:
        ordered = sorted(totals, key=lambda t: Decimal(str(t["normalized_value"])), reverse=True)
    except (InvalidOperation, TypeError, ValueError):
        return {"status": "not_comparable", "reason": "invalid_normalized_total", "totals": totals}
    best = Decimal(str(ordered[0]["normalized_value"]))
    winners = [t["document_id"] for t in ordered if Decimal(str(t["normalized_value"])) == best]
    return {
        "status": "ranked",
        "currency_symbol": currencies[0],
        "ranked_totals": ordered,
        "highest_document_ids": winners,
    }
