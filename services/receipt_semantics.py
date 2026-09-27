"""Conservative, unverified receipt interpretation over generic structure.

This layer does not alter OCR evidence or generic fields. It only recognizes
explicit receipt labels and reports conflicting values without choosing one.
"""
from __future__ import annotations

import re
from typing import Any

from services.receipt_value_normalization import normalize_receipt_value
from services.receipt_item_rows import extract_receipt_item_rows


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _semantic_key(label: str) -> str | None:
    words = re.findall(r"[a-z0-9]+", label.casefold())
    if not words:
        return None
    key = " ".join(words)
    exact = {
        "date": "date", "time": "time", "cashier": "cashier",
        "customer": "customer", "reg": "register", "register": "register",
        "trn": "transaction_number", "transaction": "transaction_number",
        "transaction number": "transaction_number", "subtotal": "subtotal",
        "total": "total", "paid": "payment_method",
        "payment method": "payment_method", "change": "change",
        "items sold": "items_sold",
    }
    if key in exact:
        return exact[key]
    if len(words) == 2 and words[0] == "tax" and words[1].isdigit():
        return "tax"
    return None


def _merchant(document: dict[str, Any]) -> dict[str, Any] | None:
    headings = [s for s in document.get("sections", []) if s.get("structure_type") == "heading"]
    if not headings:
        return None
    first = min(headings, key=lambda s: (int(s.get("row_start", 10**9)), str(s.get("section_id", ""))))
    rows = {int(x) for x in first.get("row_ids", []) if x is not None}
    items = [
        item for item in document.get("free_text", [])
        if item.get("structure_type") == "heading"
        and item.get("row_id") is not None
        and int(item["row_id"]) in rows
    ]
    items.sort(key=lambda x: (int(x["row_id"]), int(x.get("region_id", 10**9))))
    value = " ".join(filter(None, (_clean(x.get("text")) for x in items)))
    if not value:
        return None
    return {
        "status": "candidate", "observed_value": value,
        "evidence_item_ids": list(dict.fromkeys(
            evidence_id for item in items for evidence_id in item.get("evidence_item_ids", [])
        )),
        "source_section_id": first.get("section_id"),
    }


def _unpaired_total_candidates(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Pair an exact TOTAL label with money on the same OCR line.

    OCR reading order can place the right-hand amount before the left-hand
    label, so the generic row builder may miss this pair. Geometric evidence
    remains unverified and conflicting amounts remain separate candidates.
    """
    evidence = [e for e in document.get("evidence", []) if e.get("evidence_type") == "ocr_region"]
    labels = [e for e in evidence if re.fullmatch(r"TOTAL\s*[:：]", _clean(e.get("text")), re.I)]
    money = [
        e for e in evidence
        if re.fullmatch(r"[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})", _clean(e.get("text")))
    ]
    document_id = document.get("document_id")
    prefix = f"{document_id}:" if document_id else ""
    result = []
    for label in labels:
        lb = label.get("bbox") or {}
        try:
            lx, ly, lw, lh = (float(lb[k]) for k in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError):
            continue
        for amount in money:
            ab = amount.get("bbox") or {}
            try:
                ax, ay, aw, ah = (float(ab[k]) for k in ("x", "y", "width", "height"))
            except (KeyError, TypeError, ValueError):
                continue
            overlap = max(0.0, min(ly + lh, ay + ah) - max(ly, ay))
            if ax < lx + lw or overlap < 0.7 * min(lh, ah):
                continue
            result.append({
                "observed_value": _clean(amount["text"]),
                "evidence_item_ids": [
                    f"{prefix}ocr_region:{label['region_id']}",
                    f"{prefix}ocr_region:{amount['region_id']}",
                ],
                "source_field_id": None,
                "relationship": "same_line_ocr_geometry",
            })
    return result


def infer_receipt_semantics(document: dict[str, Any]) -> dict[str, Any] | None:
    """Return receipt candidates only with multiple explicit receipt signals.

    None means there is insufficient evidence to classify this as a receipt.
    Each present field has candidate/ambiguous status; raw text is preserved.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for field in document.get("fields", []):
        # Nearby-line guesses are not reliable semantic label/value evidence.
        if field.get("relationship") not in {"inline_label_value", "same_row_right"}:
            continue
        key = _semantic_key(_clean(field.get("label")))
        value = _clean(field.get("value"))
        if key is None or not value:
            continue
        grouped.setdefault(key, []).append({
            "observed_value": value,
            "evidence_item_ids": list(dict.fromkeys(field.get("evidence_item_ids", []))),
            "source_field_id": field.get("field_id"),
            "relationship": field.get("relationship"),
        })

    receipt_signals = {"subtotal", "total", "tax", "payment_method", "change", "items_sold"}
    if (
        len(receipt_signals.intersection(grouped)) < 2
        or not {"payment_method", "change", "items_sold"}.intersection(grouped)
    ):
        return None

    if "total" not in grouped:
        fallback = _unpaired_total_candidates(document)
        if fallback:
            grouped["total"] = fallback

    interpreted: dict[str, dict[str, Any]] = {}
    for key, candidates in grouped.items():
        unique: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            identity = candidate["observed_value"].casefold()
            if identity not in unique:
                unique[identity] = candidate.copy()
            else:
                unique[identity]["evidence_item_ids"] = list(dict.fromkeys(
                    unique[identity]["evidence_item_ids"] + candidate["evidence_item_ids"]
                ))
        values = list(unique.values())
        interpreted[key] = (
            {"status": "candidate", **values[0]}
            if len(values) == 1 else
            {"status": "ambiguous", "observed_value": None, "candidates": values}
        )

    merchant = _merchant(document)
    if merchant is not None:
        interpreted["merchant"] = merchant

    for key, entry in interpreted.items():
        if entry["status"] == "candidate":
            entry.update(normalize_receipt_value(key, entry["observed_value"]))
        elif entry["status"] == "ambiguous":
            for candidate in entry["candidates"]:
                candidate.update(normalize_receipt_value(key, candidate["observed_value"]))
    item_rows = extract_receipt_item_rows(document.get("records", []))
    return {
        "status": "receipt_candidate",
        "semantic_verified": False,
        "fields": interpreted,
        "items": item_rows["items"],
        "ambiguous_item_rows": item_rows["ambiguous_rows"],
    }
