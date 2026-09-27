"""Extract conservative item/price candidates from reconstructed receipt rows."""
from __future__ import annotations

import re
from typing import Any

from services.receipt_value_normalization import normalize_receipt_value

SUMMARY_LABEL = re.compile(r"^(?:SUBTOTAL|TOTAL|TAX(?:\s+\d+(?:\.\d+)?\s*%)?|PAID|CHANGE)\s*[:：]", re.I)


def extract_receipt_item_rows(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Use only one-name/one-money rows; retain uncertain rows separately."""
    items: list[dict[str, Any]] = []
    ambiguous_rows: list[dict[str, Any]] = []
    for record in records:
        cells = [
            cell for cell in record.get("cells", [])
            if " ".join(str(cell.get("text", "")).split()).strip()
        ]
        names: list[dict[str, Any]] = []
        prices: list[tuple[dict[str, Any], dict[str, Any]]] = []
        summary = False
        for cell in cells:
            text = " ".join(str(cell["text"]).split()).strip()
            if SUMMARY_LABEL.match(text):
                summary = True
                break
            # Total and item prices share the same conservative money syntax.
            normalized = normalize_receipt_value("total", text)
            if normalized["normalization_status"] == "normalized":
                prices.append((cell, normalized))
            else:
                names.append(cell)
        if summary or not prices:
            continue
        if len(names) != 1 or len(prices) != 1:
            ambiguous_rows.append({
                "record_id": record.get("record_id"),
                "table_id": record.get("table_id"),
                "source_row_id": record.get("source_row_id"),
                "reason": "expected_one_item_name_and_one_price",
                "observed_cells": [str(c.get("text", "")) for c in cells],
            })
            continue
        name = names[0]
        price, normalized = prices[0]
        items.append({
            "status": "candidate",
            "semantic_verified": False,
            "observed_name": " ".join(str(name["text"]).split()),
            "observed_price": " ".join(str(price["text"]).split()),
            "normalized_price": normalized["normalized_value"],
            "display_price": normalized["display_value"],
            "currency_symbol": normalized["currency_symbol"],
            "evidence_item_ids": list(dict.fromkeys(
                list(name.get("evidence_item_ids", []))
                + list(price.get("evidence_item_ids", []))
            )),
            "record_id": record.get("record_id"),
            "table_id": record.get("table_id"),
            "source_row_id": record.get("source_row_id"),
        })
    return {"items": items, "ambiguous_rows": ambiguous_rows}
