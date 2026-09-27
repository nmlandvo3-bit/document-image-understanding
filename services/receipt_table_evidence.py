"""Conservative receipt evidence from ordered, reconstructed table rows.

Supports receipts with bare numeric amounts and labels without colons. This
path requires item rows plus a specific ordered subtotal/tax/total/cash/change
summary, so generic payment forms do not become receipts from a lone total.
"""
from __future__ import annotations

import re
from typing import Any

AMOUNT = re.compile(r"\d+(?:\.\d{1,2})?")
TAX = re.compile(r"TAX\s+\d+(?:\.\d+)?\s*%", re.I)
SUMMARY = {"SUBTOTAL": "subtotal", "TOTAL": "total", "CASH": "cash_tendered", "CHANGE": "change"}


def _text(cell: dict[str, Any]) -> str:
    return " ".join(str(cell.get("text", "")).split()).strip()


def _summary_key(text: str) -> str | None:
    text = text.strip().rstrip(":：").strip().upper()
    return "tax" if TAX.fullmatch(text) else SUMMARY.get(text)


def extract_bare_receipt_table(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = []
    for record in records:
        cells = [c for c in record.get("cells", []) if _text(c)]
        if len(cells) != 2 or not AMOUNT.fullmatch(_text(cells[1])):
            continue
        rows.append((record, cells[0], cells[1], _summary_key(_text(cells[0]))))

    locations: dict[str, list[int]] = {}
    for index, (_, _, _, key) in enumerate(rows):
        if key:
            locations.setdefault(key, []).append(index)
    required = ("subtotal", "tax", "total", "cash_tendered", "change")
    if any(len(locations.get(key, [])) != 1 for key in required):
        return None
    positions = [locations[key][0] for key in required]
    if positions != sorted(positions):
        return None
    before_summary = rows[:positions[0]]
    if len(before_summary) < 3 or any(key is not None for _, _, _, key in before_summary):
        return None

    items = []
    for record, name, amount, _ in before_summary:
        items.append({
            "status": "candidate", "semantic_verified": False,
            "observed_name": _text(name), "observed_price": _text(amount),
            "normalized_price": _text(amount), "display_price": _text(amount),
            "currency_symbol": None,
            "evidence_item_ids": list(dict.fromkeys(
                list(name.get("evidence_item_ids", [])) + list(amount.get("evidence_item_ids", []))
            )),
            "record_id": record.get("record_id"), "table_id": record.get("table_id"),
            "source_row_id": record.get("source_row_id"),
        })

    fields = {}
    for key in required:
        record, label, amount, _ = rows[locations[key][0]]
        fields[key] = [{
            "observed_value": _text(amount),
            "evidence_item_ids": list(dict.fromkeys(
                list(label.get("evidence_item_ids", [])) + list(amount.get("evidence_item_ids", []))
            )),
            "source_field_id": None, "relationship": "same_table_row",
        }]
    return {"fields": fields, "items": items, "ambiguous_rows": []}
