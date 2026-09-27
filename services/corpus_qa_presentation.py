"""Plain text display for local corpus answers and their OCR provenance."""
from __future__ import annotations

from typing import Any


def present_corpus_answer(result: dict[str, Any]) -> tuple[str, str]:
    """Return (answer_text, evidence_text) without treating candidates as verified."""
    status = result.get("status")
    evidence = [f"Status: {status}", "Generated structure is unverified."]
    if result.get("provider"):
        evidence.append(f"Answer path: {result['provider']}")
    if result.get("interpreted_question"):
        evidence.append(f"Search question: {result['interpreted_question']}")

    if status == "answered" and result.get("answer_type") == "receipt_totals":
        totals = result.get("totals", [])
        if not totals:
            return "No receipt candidates found.", "\n".join(evidence)
        rows = []
        for total in totals:
            doc_id = total.get("document_id", "-")
            value = total.get("observed_value")
            label = value if total.get("status") == "candidate" and value is not None else "[missing or ambiguous]"
            rows.append(f"{doc_id} | {total.get('merchant') or 'Unknown merchant'} | {label}")
            evidence.append(f"\n{doc_id} | total status: {total.get('status')}")
            evidence.extend(f"  {eid}" for eid in total.get("evidence_item_ids", []))
        return "Receipt total candidates:\n" + "\n".join(rows), "\n".join(evidence)

    if status == "answered" and result.get("answer_type") == "highest_receipt_total":
        winners = result.get("document_ids", [])
        for total in result.get("ranked_totals", []):
            evidence.append(f"\n{total.get('document_id')} | {total.get('observed_value')}")
            evidence.extend(f"  {eid}" for eid in total.get("evidence_item_ids", []))
        return "Highest total candidate: " + ", ".join(winners), "\n".join(evidence)

    if status == "answered":
        doc_id = result.get("document_id", "-")
        evidence.append(f"Document: {doc_id}")
        evidence.append(f"Observed OCR value: {result.get('observed_value')}")
        evidence.extend(result.get("evidence_item_ids", []))
        return f"Candidate: {result.get('answer')}\nDocument: {doc_id}", "\n".join(evidence)

    if status == "one_match":
        doc_id = result.get("answer", "-")
        for match in result.get("matches", []):
            evidence.append(f"\n{match.get('document_id')} | {match.get('observed_value')}")
            evidence.extend(f"  {eid}" for eid in match.get("evidence_item_ids", []))
        return f"Matching document: {doc_id}", "\n".join(evidence)

    if status == "not_comparable":
        reason = result.get("reason", "unknown")
        explanations = {
            "currency_unknown": "a receipt has no known currency",
            "currency_mismatch": "receipts have different currencies",
            "missing_or_ambiguous_total": "a receipt total is missing or ambiguous",
            "fewer_than_two_receipts": "fewer than two receipts were found",
            "invalid_normalized_total": "a total cannot be read as a number",
        }
        for total in result.get("totals", []):
            evidence.append(f"\n{total.get('document_id')} | {total.get('observed_value')} | currency: {total.get('currency_symbol')}")
            evidence.extend(f"  {eid}" for eid in total.get("evidence_item_ids", []))
        return "Cannot compare receipt totals: " + explanations.get(reason, reason) + ".", "\n".join(evidence)

    if status in {"ambiguous", "ambiguous_document"}:
        for match in result.get("matches", []):
            evidence.append(f"\n{match.get('document_id')} | {match.get('observed_value')}")
            evidence.extend(f"  {eid}" for eid in match.get("evidence_item_ids", []))
        return "Multiple documents match. No document was selected.", "\n".join(evidence)

    if status == "ambiguous_field":
        evidence.append(f"Document: {result.get('document_id', '-')}")
        for candidate in result.get("candidates", []):
            evidence.append(f"\nCandidate: {candidate.get('observed_value')}")
            evidence.extend(f"  {eid}" for eid in candidate.get("evidence_item_ids", []))
        return "Multiple field values found. No value was selected.", "\n".join(evidence)

    messages = {
        "document_not_found": "No matching document found.",
        "no_match": "No matching document found.",
        "field_not_found": "The matching document has no supported candidate for this field.",
        "unsupported_question": "Unsupported question. Try: What is the total for FRESH MART SUPERMARKET?",
    }
    if result.get("document_id"):
        evidence.append(f"Document: {result['document_id']}")
    return messages.get(status, "No answer available."), "\n".join(evidence)
