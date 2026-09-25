"""Experiment 74: production structured-Q&A integration regression.

No Groq/network calls are made. _call_groq is mocked only at the service boundary.
"""
from __future__ import annotations

import json
from pathlib import Path

from services.document_qa_service import DocumentQAService

CASES = {
    "receipt": Path("results/gui_v4/single/receipt/unified_raw_ocr.json"),
    "pic13": Path("results/gui_v4/single/pic13/unified_raw_ocr.json"),
    "nhi": Path(
        "results/gui_v4/single/WhatsApp Image 2026-09-22 at 22.21.24/unified_raw_ocr.json"
    ),
    "synthetic": Path(
        "results/gui_v4/single/ChatGPT Image Sep 24, 2026, 06_13_58 PM/unified_raw_ocr.json"
    ),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_service(document: dict):
    service = DocumentQAService.__new__(DocumentQAService)
    service.model = "mock-model"
    service.documents = [document]
    service.corpus = []
    service.api_calls = 0

    def mock_call(packet):
        service.api_calls += 1
        candidate = packet.get("structured_candidate")
        if candidate:
            return {
                "answer": str(candidate.get("value", "")),
                "evidence_item_ids": list(candidate.get("evidence_item_ids", [])),
                "uncertainty": None,
                "usage": {},
            }

        evidence = packet.get("evidence", [])
        if not evidence:
            return {
                "answer": "",
                "evidence_item_ids": [],
                "uncertainty": "insufficient_document_evidence",
                "usage": {},
            }

        item = evidence[-1]
        return {
            "answer": str((item.get("content") or {}).get("text", "")),
            "evidence_item_ids": [item["evidence_item_id"]],
            "uncertainty": None,
            "usage": {},
        }

    service._call_groq = mock_call
    return service


def main() -> int:
    print("=" * 112)
    print("EXPERIMENT 74: PRODUCTION STRUCTURED-Q&A INTEGRATION")
    print("=" * 112)

    docs = {}
    for name, path in CASES.items():
        if not path.is_file():
            print(f"[MISSING] {name}: {path}")
            return 1
        docs[name] = load(path)

    checks = []

    receipt_service = make_service(docs["receipt"])
    receipt_packet = receipt_service.build_llm_packet("What is the total?")
    checks += [
        ("receipt total uses structured packet", receipt_packet.get("answer_mode") == "structured_answer_candidate"),
        ("receipt structured value is 310", (receipt_packet.get("structured_candidate") or {}).get("value") == "310"),
    ]

    nhi_service = make_service(docs["nhi"])
    nhi_packet = nhi_service.build_llm_packet("What is the payment date?")
    checks += [
        ("NHI payment date uses structured packet", nhi_packet.get("answer_mode") == "structured_answer_candidate"),
        ("NHI structured date is correct", (nhi_packet.get("structured_candidate") or {}).get("value") == "115/09/21"),
    ]

    synthetic_service = make_service(docs["synthetic"])
    synthetic_packet = synthetic_service.build_llm_packet("What is the total?")
    checks += [
        ("synthetic total is structured ambiguity packet", synthetic_packet.get("answer_mode") == "structured_ambiguous_candidates"),
        ("synthetic ambiguity is preserved", bool(synthetic_packet.get("ambiguities"))),
    ]

    fallback_service = make_service(docs["receipt"])
    fallback_packet = fallback_service.build_llm_packet("What is the shop name?")
    checks.append((
        "non-structured query preserves raw OCR fallback",
        fallback_packet.get("answer_mode") in {"raw_ocr_spatial_candidates", "no_answer"},
    ))

    service = make_service(docs["receipt"])
    result = service.answer("What is the total?")
    checks += [
        ("structured total is grounded", result.get("grounded") is True),
        ("structured total answer is 310", result.get("answer") == "310"),
        ("structured consistency passes", result.get("answer_consistency", {}).get("passed") is True),
        ("structured consistency reason", result.get("answer_consistency", {}).get("reason") == "literal_structured_candidate_value"),
        ("structured answer calls mocked API once", service.api_calls == 1),
        ("structured answer type is money", result.get("answer_type") == "money"),
        ("structured confidence is numeric", isinstance(result.get("confidence_score"), (int, float))),
        ("confidence is not a calibrated probability", result.get("confidence_is_calibrated_probability") is False),
    ]

    ambiguous_service = make_service(docs["synthetic"])
    ambiguous_result = ambiguous_service.answer("What is the total?")
    checks += [
        ("structured ambiguity is ungrounded", ambiguous_result.get("grounded") is False),
        ("structured ambiguity blocks before API", ambiguous_service.api_calls == 0),
        ("structured ambiguity confidence is low", ambiguous_result.get("confidence_level") == "low"),
        ("structured ambiguity score is 0.49", ambiguous_result.get("confidence_score") == 0.49),
    ]

    absent_service = make_service(docs["receipt"])
    absent_result = absent_service.answer("What is the date of birth?")
    checks += [
        ("absent DOB remains ungrounded", absent_result.get("grounded") is False),
        ("absent DOB makes no API call", absent_service.api_calls == 0),
        ("absent DOB confidence is low", absent_result.get("confidence_level") == "low"),
    ]

    print()
    print("-" * 112)
    print("PRODUCTION INTEGRATION CHECKS")
    print("-" * 112)

    passed = 0
    for index, (label, ok) in enumerate(checks, start=1):
        print(f"[{'PASS' if ok else 'FAIL'}] {index:02d}. {label}")
        if ok:
            passed += 1

    print()
    print("=" * 112)
    print(f"RESULT: {passed}/{len(checks)} production integration checks passed")
    print("=" * 112)
    print("No Groq requests were made.")

    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
