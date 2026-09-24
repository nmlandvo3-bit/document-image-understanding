from __future__ import annotations

from services.document_qa_service import DocumentQAService


def main() -> int:
    print("=" * 92)
    print("EXPERIMENT 65: SERVICE TYPE + CONFIDENCE INTEGRATION")
    print("=" * 92)

    service = DocumentQAService.__new__(DocumentQAService)

    payment_packet = {
        "question": "What is the payment date?",
        "answer_mode": "raw_ocr_spatial_candidates",
        "evidence": [
            {
                "evidence_item_id": "doc:ocr_region:1",
                "document_id": "doc",
                "item_type": "ocr_region",
                "content": {
                    "region_id": 1,
                    "text": "Payment Date",
                    "confidence": 0.99,
                },
            },
            {
                "evidence_item_id": "doc:ocr_region:2",
                "document_id": "doc",
                "item_type": "ocr_region",
                "content": {
                    "region_id": 2,
                    "text": "115/09/21",
                    "confidence": 0.98,
                },
            },
        ],
        "spatial_candidates": [
            {
                "document_id": "doc",
                "label_region_id": 1,
                "candidate_region_id": 2,
                "spatial_direction": "table_column_below",
                "relationship": "spatial_candidate_not_verified",
            }
        ],
    }

    payment_model_result = {
        "answer": "115/09/21",
        "evidence_item_ids": ["doc:ocr_region:2"],
        "uncertainty": None,
    }

    payment = service._raw_ocr_confidence_metadata(
        payment_packet,
        payment_model_result,
    )

    checks = [
        ("payment answer type", payment["answer_type"] == "date"),
        ("payment confidence level", payment["confidence_level"] == "high"),
        (
            "payment score is numeric",
            isinstance(payment["confidence_score"], float),
        ),
        (
            "payment score not probability flag",
            payment["confidence_is_calibrated_probability"] is False,
        ),
    ]

    ambiguous = service._empty_confidence(
        "What is the total?",
        "ambiguous_document_evidence",
        score=0.49,
    )

    checks.extend(
        [
            ("ambiguous answer type", ambiguous["answer_type"] == "money"),
            ("ambiguous confidence level", ambiguous["confidence_level"] == "low"),
            ("ambiguous confidence score", ambiguous["confidence_score"] == 0.49),
        ]
    )

    no_evidence = service._empty_confidence(
        "What is the customer name?",
        "insufficient_document_evidence",
        score=0.0,
    )

    checks.extend(
        [
            ("no-evidence answer type", no_evidence["answer_type"] == "name"),
            ("no-evidence confidence level", no_evidence["confidence_level"] == "low"),
            ("no-evidence confidence score", no_evidence["confidence_score"] == 0.0),
        ]
    )

    passed = 0

    for i, (name, ok) in enumerate(checks, start=1):
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {i:02d}. {name}")
        if ok:
            passed += 1

    print()
    print("PAYMENT METADATA:", payment)
    print("AMBIGUOUS METADATA:", ambiguous)
    print("NO-EVIDENCE METADATA:", no_evidence)
    print()
    print("=" * 92)
    print(f"RESULT: {passed}/{len(checks)} tests passed")
    print("=" * 92)

    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
