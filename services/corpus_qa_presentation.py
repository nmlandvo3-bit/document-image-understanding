"""Plain text display for local corpus answers and their OCR provenance."""
from __future__ import annotations

from typing import Any


def present_corpus_answer(result: dict[str, Any]) -> tuple[str, str]:
    """Return (answer_text, evidence_text) without treating candidates as verified."""
    status = result.get("status")
    evidence = [f"Status: {status}", "Generated structure is unverified."]
    if result.get("failure_reason"):
        evidence.append("Failure reason: " + result["failure_reason"])
    if result.get("model"):
        evidence.append("Requested model: " + str(result["model"]))
    if result.get("provider"):
        evidence.append(f"Answer path: {result['provider']}")
    if result.get("interpreted_question"):
        evidence.append(f"Search question: {result['interpreted_question']}")

    if result.get("provider") in {"groq_corpus_analysis", "ollama_corpus_analysis"}:
        evidence.append("Accepted quotes have source citations; OCR and semantic interpretations remain unverified.")
        if result.get("qa_revision"):
            evidence.append("Q&A revision: " + result["qa_revision"])
        if result.get("scope_description"):
            evidence.append("Requested receipt scope: " + result["scope_description"])
            evidence.append(f"Loaded corpus: {result['corpus_documents_total']} documents; in requested scope: {result['documents_in_scope']}")
        if result.get("synthesis_status"):
            evidence.append("Final synthesis: " + result["synthesis_status"])
        if result.get("batch_reports"):
            evidence.append(f"Documents processed: {result.get('documents_inspected', len(result['batch_reports']))}/{result.get('documents_total')}")
            evidence.append(f"Successful model results: {result.get('documents_analyzed')}/{result.get('documents_total')}")
            evidence.append("All model results passed: " + str(result.get("coverage_complete")))
            evidence.extend(f"  {r['document_id']}: {r['status']}" + (f" — {r['failure_reason']}" if r.get("failure_reason") else "") for r in result['batch_reports'])
        if result.get("receipt_total_listing"):
            listing = result["receipt_total_listing"]
            evidence.append("Selection method: local_receipt_total_listing")
            evidence.append("Model path status: " + result["model_status"])
            if result.get("model_failure_reason"):
                evidence.append("Model path failure: " + result["model_failure_reason"])
            evidence.append("Local receipt list complete: " + str(listing["complete"]))
            evidence.extend(f"  {r['document_id']}: {r['status']}" for r in listing["reports"])
            evidence.append("Receipt classification and OCR geometry remain candidate interpretations.")
            for claim in result.get("claims", []):
                evidence.append(f"\n{claim['document_id']} | {claim['quote']}")
                evidence.extend(claim["evidence_item_ids"])
            prefix = ("LOCAL RECEIPT TOTALS (one printed total per receipt candidate):\n"
                      if listing["complete"] else "INCOMPLETE LOCAL RECEIPT TOTALS (missing or conflicting totals):\n")
            return prefix + result["answer"], "\n".join(evidence)
        if result.get("item_comparison"):
            comparison = result["item_comparison"]
            evidence.append("Selection method: " + result["selection_method"])
            evidence.append("Model returned: " + result.get("model_status", "unknown"))
            if result.get("model_failure_reason"):
                evidence.append("Model path failure: " + result["model_failure_reason"])
            evidence.append("Local item evidence complete: " + str(result.get("local_coverage_complete")))
            evidence.append(comparison["reason"])
            evidence.append(comparison.get("price_basis", "Printed item-row prices."))
            evidence.append("Currency symbols do not establish a verified currency denomination.")
            for claim in result.get("claims", []):
                evidence.append(f"\n{claim['document_id']} | {claim['quote']}")
                evidence.extend(claim["evidence_item_ids"])
            text = "\n".join(
                f"{i['document_id']}: {i['name']} — {i['price']} [printed currency: {i['currency_symbol'] or 'unspecified'}]"
                for i in comparison['items']) or "No complete item comparison available."
            if status == "comparison_ambiguous":
                text = "CURRENCY AMBIGUITY: one global winner cannot be established. Candidates by printed currency group:\n" + text
            elif status == "comparison_incomplete":
                text = "INCOMPLETE ITEM COMPARISON: " + comparison["reason"]
            elif result["selection_method"] == "local_item_price_comparison":
                text = "LOCAL COMPARISON (model answer corrected using OCR item rows):\n" + text
            return text, "\n".join(evidence)
        if status == "partial_analysis":
            for claim in result.get("claims", []):
                evidence.append(f"\n{claim['document_id']} | {claim['quote']}")
                evidence.extend(claim['evidence_item_ids'])
            return ("INCOMPLETE ANALYSIS: document extraction or final synthesis did not complete. "
                    "The following accepted quotes are only partial results.\n"
                    + (result.get('answer') or 'No accepted quotes.')), "\n".join(evidence)
        if status == "answered":
            evidence.append(f"Documents analyzed: {result.get('documents_analyzed')}")
            for claim in result.get("claims", []):
                evidence.append(f"\n{claim['document_id']} | {claim['quote']}")
                evidence.extend(claim["evidence_item_ids"])
            return result["answer"], "\n".join(evidence)
        messages = {
            "local_context_limit": "This document exceeds the local context budget. Waiting will not fix it; use a smaller document or increase the configured context.",
            "request_limit": "Model request exceeds the size/context budget or rate limit. Choose a smaller scan folder; for rate limits, wait before retrying.",
            "authentication_failed": "Groq authentication failed. Check your API key and access.",
            "context_limit": "Corpus exceeds the analysis budget. Choose a smaller scan folder.",
            "local_model_unavailable": "Ollama is unavailable or the requested model is missing. Start Ollama and check ollama list against the requested model shown above.",
            "empty_corpus": "No corpus OCR evidence found.",
            "invalid_corpus": "Duplicate OCR evidence IDs found. No analysis performed.",
            "model_not_found": "The model found insufficient evidence to answer.",
            "model_ambiguous": "The model reported conflicting evidence. No answer selected.",
            "analysis_failed": "Model analysis failed or its evidence validation failed. No answer accepted.",
        }
        return messages.get(status, "No answer accepted."), "\n".join(evidence)

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
