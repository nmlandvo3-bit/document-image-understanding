"""Non-authoritative arithmetic comparisons for receipt candidates.

Reports observed agreements or differences; never rewrites source values.
Discounts, fees, and OCR errors can explain a difference, so a mismatch is not
an assertion that any individual field is wrong.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


def _amount(candidate: dict[str, Any] | None, key: str = "normalized_value") -> Decimal | None:
    if not candidate or candidate.get("status") != "candidate":
        return None
    try:
        return Decimal(str(candidate[key]))
    except (KeyError, InvalidOperation, TypeError, ValueError):
        return None


def _ids(*candidates: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        evidence_id for candidate in candidates
        for evidence_id in candidate.get("evidence_item_ids", [])
    ))


def _check(computed: Decimal | None, observed: Decimal | None,
           evidence: list[str], currencies: list[str | None]) -> dict[str, Any]:
    if computed is None or observed is None:
        return {"status": "not_evaluated", "reason": "missing_or_ambiguous_value"}
    if len(set(currencies)) > 1:
        return {"status": "not_evaluated", "reason": "currency_mismatch"}
    return {
        "status": "matches" if computed == observed else "differs",
        "computed_value": format(computed, "f"),
        "observed_value": format(observed, "f"),
        "currency_symbol": currencies[0] if currencies else None,
        "evidence_item_ids": evidence,
        "semantic_verified": False,
    }


def assess_receipt_consistency(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fields = receipt.get("fields", {})
    items = receipt.get("items", [])
    subtotal = fields.get("subtotal") or {}
    tax = fields.get("tax") or {}
    total = fields.get("total") or {}
    cash = fields.get("cash_tendered") or {}
    change = fields.get("change") or {}

    prices = [_amount(item, "normalized_price") for item in items]
    if items and all(price is not None for price in prices):
        item_sum = sum(prices, Decimal("0"))
        item_currencies = [item.get("currency_symbol") for item in items]
    else:
        item_sum = None
        item_currencies = []
    subtotal_value = _amount(subtotal)
    tax_value = _amount(tax)
    total_value = _amount(total)
    cash_value = _amount(cash)

    return {
        "items_vs_subtotal": _check(
            item_sum, subtotal_value, _ids(*items, subtotal),
            item_currencies + [subtotal.get("currency_symbol")],
        ),
        "subtotal_plus_tax_vs_total": _check(
            subtotal_value + tax_value if subtotal_value is not None and tax_value is not None else None,
            total_value, _ids(subtotal, tax, total),
            [subtotal.get("currency_symbol"), tax.get("currency_symbol"), total.get("currency_symbol")],
        ),
        "cash_minus_total_vs_change": _check(
            cash_value - total_value if cash_value is not None and total_value is not None else None,
            _amount(change), _ids(cash, total, change),
            [cash.get("currency_symbol"), total.get("currency_symbol"), change.get("currency_symbol")],
        ),
    }
