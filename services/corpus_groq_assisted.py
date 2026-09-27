"""Opt-in Groq interpretation, with local corpus selection and validation."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from services.corpus_question_router import answer_corpus_question


def _context(index: dict[str, Any]) -> list[dict[str, Any]]:
    context = []
    for document in index["documents"][:30]:
        fields = (document.get("receipt_semantics") or {}).get("fields", {})
        context.append({
            "document_id": document["document_id"],
            "fields": {key: value.get("observed_value") for key, value in fields.items()
                       if value.get("status") == "candidate"},
            "headings": [item.get("text") for item in document.get("free_text", [])
                         if item.get("structure_type") == "heading"][:10],
            "ocr_regions": [{"region_id": item.get("region_id"), "text": item.get("text")}
                            for item in document.get("evidence", [])[:60]
                            if item.get("evidence_type") == "ocr_region"],
        })
    return context


def _safe_focus(original: str, proposal: Any) -> str:
    if not isinstance(proposal, str) or not proposal.strip():
        return original
    old = set(re.findall(r"[\w]+", original.casefold()))
    new = set(re.findall(r"[\w]+", proposal.casefold()))
    aliases = set()
    if old & {"location", "located", "where", "address"}:
        aliases.update({"location", "address", "located", "where"})
    if "cashier" in old:
        aliases.update({"cashier", "name"})
    filler = {"what", "which", "who", "where", "how", "is", "are", "the", "a",
              "an", "for", "of", "on", "at", "in", "receipt", "document", "number",
              "please", "tell", "me", "show", "located"}
    if not new or not new.issubset(old | aliases | filler):
        return original
    if not (old - filler - aliases).issubset(new):
        return original
    return proposal.strip()


def answer_corpus_question_assisted(index: dict[str, Any], question: str,
                                    client: Any = None, model: str = "openai/gpt-oss-20b") -> dict[str, Any]:
    """Groq interprets and phrases; local code retains final selection authority."""
    if client is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set.")
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])

    focus = question
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Interpret the user's corpus question for local retrieval. "
                    "Do not answer. Return JSON with focused_question only. "
                    "Keep document identity terms from the original question. "
                    "Location/address and cashier/name are allowed synonyms. "
                    "Do not introduce document values absent from the question."
                )},
                {"role": "user", "content": json.dumps({"question": question,
                    "document_context": _context(index)}, ensure_ascii=False)},
            ],
            temperature=0.0, reasoning_effort="low", include_reasoning=False,
            response_format={"type": "json_object"}, max_completion_tokens=120,
            stream=False,
        )
        focus = _safe_focus(question,
                            json.loads(response.choices[0].message.content)["focused_question"])
    except Exception:
        focus = question

    result = answer_corpus_question(index, focus)
    result = {**result, "provider": "local_retrieval_with_groq_interpretation",
              "interpreted_question": focus}
    # Only single candidates can be sent for Groq phrasing.
    if result.get("status") != "answered" or not isinstance(result.get("answer"), str):
        return result
    ids = result.get("evidence_item_ids", [])
    if not ids:
        return result
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Return JSON: {\"answer\": string, \"evidence_item_ids\": array}. "
                    "Use exactly the selected candidate value and cite its supplied OCR IDs. "
                    "Do not add facts or edit OCR text."
                )},
                {"role": "user", "content": json.dumps({
                    "question": question,
                    "selected_candidate": result["answer"],
                    "document_id": result.get("document_id"),
                    "evidence_item_ids": ids,
                }, ensure_ascii=False)},
            ],
            temperature=0.0, reasoning_effort="low", include_reasoning=False,
            response_format={"type": "json_object"}, max_completion_tokens=180,
            stream=False,
        )
        parsed = json.loads(response.choices[0].message.content)
        cited = parsed.get("evidence_item_ids", [])
        if (str(parsed.get("answer", "")).strip() == result["answer"]
                and isinstance(cited, list) and cited
                and all(isinstance(item, str) and item in ids for item in cited)):
            return {**result, "provider": "groq_grounded_in_local_corpus",
                    "evidence_item_ids": cited}
    except Exception:
        pass
    return result
