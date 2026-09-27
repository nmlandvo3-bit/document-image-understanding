"""Experiment 86: standalone date, time, and receipt number on receipts."""
from __future__ import annotations

import json
from pathlib import Path

from parsing.unified_structured_document_builder import build_unified_structured_document
from services.receipt_standalone_metadata import standalone_receipt_metadata

ROOT = Path(__file__).resolve().parent.parent / "results/gui_v4/single"


def build(name: str) -> tuple[dict, dict]:
    raw = json.loads((ROOT / name / "unified_raw_ocr.json").read_text(encoding="utf-8"))
    return raw, build_unified_structured_document(raw)


def main() -> int:
    for name in ("receipt", "supermarket", "pic13"):
        if not (ROOT / name / "unified_raw_ocr.json").is_file():
            print(f"[MISSING] {ROOT / name / 'unified_raw_ocr.json'}")
            return 1
    campus_raw, campus = build("receipt")
    _, supermarket = build("supermarket")
    _, form = build("pic13")
    fields = (campus.get("receipt_semantics") or {}).get("fields", {})
    other = (supermarket.get("receipt_semantics") or {}).get("fields", {})
    duplicate = standalone_receipt_metadata({
        "document_id": "synthetic",
        "evidence": [
            {"evidence_type": "ocr_region", "region_id": 1, "text": "2026-09-14"},
            {"evidence_type": "ocr_region", "region_id": 2, "text": "2026-09-15"},
            {"evidence_type": "ocr_region", "region_id": 3, "text": "Receipt No. 0007"},
            {"evidence_type": "ocr_region", "region_id": 4, "text": "not a time 10:32"},
        ],
    })
    checks = [
        ("campus remains receipt", (campus.get("receipt_semantics") or {}).get("status") == "receipt_candidate"),
        ("campus date observed exactly", fields.get("date", {}).get("observed_value") == "2026-09-14"),
        ("campus date normalized", fields.get("date", {}).get("normalized_value") == "2026-09-14"),
        ("campus time observed and normalized", fields.get("time", {}).get("observed_value") == "10:32" and fields.get("time", {}).get("normalized_value") == "10:32"),
        ("receipt number preserves leading zeros", fields.get("receipt_number", {}).get("normalized_value") == "004512"),
        ("standalone fields cite one region each", all(len(fields.get(k, {}).get("evidence_item_ids", [])) == 1 for k in ("date", "time", "receipt_number"))),
        ("standalone relation explicit", all(fields.get(k, {}).get("relationship") == "standalone_ocr_pattern" for k in ("date", "time", "receipt_number"))),
        ("campus total remains 310", fields.get("total", {}).get("normalized_value") == "310"),
        ("five campus items retained", len((campus.get("receipt_semantics") or {}).get("items", [])) == 5),
        ("supermarket inline date retained", other.get("date", {}).get("observed_value") == "08/24/2023"),
        ("supermarket transaction number retained", other.get("transaction_number", {}).get("normalized_value") == "9887"),
        ("form remains nonreceipt", form.get("receipt_semantics") is None),
        ("raw OCR evidence unchanged", campus.get("evidence") == campus_raw.get("evidence")),
        ("multiple standalone dates retained as candidates", len(duplicate.get("date", [])) == 2),
        ("embedded time is not extracted", "time" not in duplicate),
        ("synthetic receipt number exact", duplicate.get("receipt_number", [{}])[0].get("observed_value") == "0007"),
    ]
    print("EXPERIMENT 86: STANDALONE RECEIPT METADATA")
    print("Campus metadata:", {k: fields.get(k, {}).get("observed_value") for k in ("date", "time", "receipt_number")})
    for i, (name, ok) in enumerate(checks, 1):
        print(f"[{'PASS' if ok else 'FAIL'}] {i:02d}. {name}")
    passed = sum(ok for _, ok in checks)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
