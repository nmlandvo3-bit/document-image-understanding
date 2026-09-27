"""Conservative machine-readable values for receipt candidates.

Never replaces observed OCR text. Ambiguous date order remains unresolved.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
import re
from typing import Any

MONEY_FIELDS = {"subtotal", "tax", "total", "change", "cash_tendered"}
IDENTIFIER_FIELDS = {"register", "transaction_number"}
MONEY_RE = re.compile(r"^[\s.'‘’]*([$€£¥])\s*(\d+(?:,\d{3})*)\.(\d{2})$")


def normalize_receipt_value(key: str, observed: str, *, allow_currencyless: bool = False) -> dict[str, Any]:
    """Return additive normalization metadata without interpreting uncertainty."""
    text = " ".join(str(observed).split()).strip()
    result: dict[str, Any] = {
        "normalized_value": None,
        "display_value": text,
        "normalization_status": "not_applicable",
    }
    if key in MONEY_FIELDS:
        match = MONEY_RE.fullmatch(text)
        if not match:
            if allow_currencyless and re.fullmatch(r"\d+(?:\.\d{1,2})?", text):
                result.update(normalized_value=text, normalization_status="normalized_currency_unspecified", currency_symbol=None)
                return result
            result["normalization_status"] = "unrecognized_money"
            return result
        symbol, units, cents = match.groups()
        amount = Decimal(f"{units.replace(',', '')}.{cents}")
        result.update(
            normalized_value=f"{amount:.2f}",
            display_value=f"{symbol}{amount:.2f}",
            currency_symbol=symbol,
            normalization_status="normalized",
        )
    elif key == "date":
        iso = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
        numeric = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text)
        try:
            if iso:
                y, m, d = map(int, iso.groups())
            elif numeric:
                first, second, y = map(int, numeric.groups())
                if first <= 12 and second <= 12 and first != second:
                    result["normalization_status"] = "ambiguous_date_order"
                    return result
                m, d = (second, first) if first > 12 else (first, second)
            else:
                result["normalization_status"] = "unrecognized_date"
                return result
            normalized = date(y, m, d).isoformat()
        except ValueError:
            result["normalization_status"] = "invalid_date"
            return result
        result.update(normalized_value=normalized, display_value=normalized, normalization_status="normalized")
    elif key == "time":
        match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
        if not match:
            result["normalization_status"] = "unrecognized_time"
            return result
        h, m, s = match.groups()
        if int(h) > 23 or int(m) > 59 or (s is not None and int(s) > 59):
            result["normalization_status"] = "invalid_time"
            return result
        normalized = f"{int(h):02d}:{m}" + (f":{s}" if s is not None else "")
        result.update(normalized_value=normalized, display_value=normalized, normalization_status="normalized")
    elif key == "items_sold":
        if not re.fullmatch(r"\d+", text):
            result["normalization_status"] = "invalid_count"
            return result
        result.update(normalized_value=int(text), normalization_status="normalized")
    elif key in IDENTIFIER_FIELDS:
        result.update(normalized_value=text, normalization_status="preserved_identifier")
    return result
