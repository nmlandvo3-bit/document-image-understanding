"""Experiment 87: observational receipt arithmetic comparisons."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from parsing.unified_structured_document_builder import build_unified_structured_document
from services.receipt_consistency import assess_receipt_consistency

ROOT = Path(__file__).resolve().parent.parent / "results/gui_v4/single"


def build(name: str) -> dict:
    path = ROOT / name / "unified_raw_ocr.json"
    return build_unified_structured_document(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    for name in ("receipt", "supermarket"):
        if not (ROOT / name / "unified_raw_ocr.json").is_file():
            print(f"[MISSING] {ROOT / name / 'unified_raw_ocr.json'}")
            return 1
    campus = build("receipt")["receipt_semantics"]
    market = build("supermarket")["receipt_semantics"]
    campus_checks = campus.get("consistency_checks", {})
    market_checks = market.get("consistency_checks", {})
    mismatch = deepcopy(campus)
    mismatch["fields"]["subtotal"]["normalized_value"] = "294"
    before = deepcopy(mismatch)
    differing = assess_receipt_consistency(mismatch)
    ambiguous = deepcopy(campus)
    ambiguous["fields"]["total"]["status"] = "ambiguous"
    skipped = assess_receipt_consistency(ambiguous)
    mixed = deepcopy(campus)
    mixed["items"][0]["currency_symbol"] = "$"
    mixed_result = assess_receipt_consistency(mixed)

    def check(receipt_checks: dict, key: str, computed: str, observed: str) -> bool:
        entry = receipt_checks.get(key, {})
        return (entry.get("status") == "matches"
                and entry.get("computed_value") == computed
                and entry.get("observed_value") == observed)

    checks = [
        ("campus items sum to subtotal", check(campus_checks, "items_vs_subtotal", "295", "295")),
        ("campus subtotal plus tax equals total", check(campus_checks, "subtotal_plus_tax_vs_total", "310", "310")),
        ("campus cash minus total equals change", check(campus_checks, "cash_minus_total_vs_change", "190", "190")),
        ("supermarket items sum to subtotal", check(market_checks, "items_vs_subtotal", "18.95", "18.95")),
        ("supermarket subtotal plus tax equals total", check(market_checks, "subtotal_plus_tax_vs_total", "20.09", "20.09")),
        ("debit receipt cash comparison skipped", market_checks.get("cash_minus_total_vs_change", {}).get("status") == "not_evaluated"),
        ("comparison evidence linked", all(len(campus_checks.get(key, {}).get("evidence_item_ids", [])) >= 3 for key in campus_checks)),
        ("checks are not semantic verification", all(campus_checks.get(key, {}).get("semantic_verified") is False for key in campus_checks)),
        ("synthetic subtotal difference reported", differing.get("items_vs_subtotal", {}).get("status") == "differs"),
        ("synthetic subtotal difference does not alter source", mismatch == before),
        ("ambiguous total prevents dependent checks", all(skipped.get(key, {}).get("status") == "not_evaluated" for key in ("subtotal_plus_tax_vs_total", "cash_minus_total_vs_change"))),
        ("mixed currency prevents item comparison", mixed_result.get("items_vs_subtotal", {}).get("reason") == "currency_mismatch"),
        ("receipt candidates remain unverified", campus.get("semantic_verified") is False and market.get("semantic_verified") is False),
    ]
    print("EXPERIMENT 87: RECEIPT CONSISTENCY")
    print("Campus:", campus_checks)
    print("Supermarket:", market_checks)
    for i, (label, ok) in enumerate(checks, 1):
        print(f"[{'PASS' if ok else 'FAIL'}] {i:02d}. {label}")
    passed = sum(ok for _, ok in checks)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
