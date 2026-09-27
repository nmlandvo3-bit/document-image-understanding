"""Explicit standalone date, time, and receipt number on known receipts."""
from __future__ import annotations

import re
from typing import Any

from services.receipt_value_normalization import normalize_receipt_value

DATE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}")
TIME = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")
NUMBER = re.compile(r"receipt\s*(?:no\.?|number|#)\s*[:.]?\s*([A-Za-z0-9][A-Za-z0-9/-]*)", re.I)


def standalone_receipt_metadata(document: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Extract only entire-region matches; never guess from adjacent text."""
    result: dict[str, list[dict[str, Any]]] = {}
    prefix = f"{document['document_id']}:" if document.get("document_id") else ""
    for evidence in document.get("evidence", []):
        if evidence.get("evidence_type") != "ocr_region":
            continue
        text = " ".join(str(evidence.get("text", "")).split()).strip()
        if DATE.fullmatch(text):
            key, value = "date", text
        elif TIME.fullmatch(text):
            key, value = "time", text
        else:
            match = NUMBER.fullmatch(text)
            if match is None:
                continue
            key, value = "receipt_number", match.group(1)
        if key in {"date", "time"} and normalize_receipt_value(key, value)["normalized_value"] is None:
            continue
        result.setdefault(key, []).append({
            "observed_value": value,
            "evidence_item_ids": [f"{prefix}ocr_region:{evidence['region_id']}"],
            "source_field_id": None,
            "relationship": "standalone_ocr_pattern",
        })
    return result
