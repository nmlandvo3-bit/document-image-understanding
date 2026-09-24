"""Experiment 66: local regression gate for typed answer + confidence branch.

Run from the project root:
    python -m experiments.experiment_66_typed_confidence_regression

This invokes the already-validated Experiment 60, 62, 63, 64, and 65
suites, and exercises DocumentQAService.answer() with simulated Groq
responses. No OCR or network requests are made.
"""

from __future__ import annotations

from importlib import import_module


SUITES = (
    ("Experiment 60: retrieval regression", "experiments.experiment_60_regression_suite", "main"),
    ("Experiment 62: answer type", "experiments.experiment_62_answer_type_inference", "run_validation"),
    ("Experiment 63: candidate validation", "experiments.experiment_63_candidate_type_validation", "run_validation"),
    ("Experiment 64: confidence scoring", "experiments.experiment_64_confidence_scoring", "run_validation"),
    ("Experiment 65: service metadata", "experiments.experiment_65_service_confidence_integration", "main"),
)


def _region(region_id: int, text: str, confidence: float = 0.98) -> dict:
    return {
        "evidence_item_id": f"test_doc:ocr_region:{region_id}",
        "document_id": "test_doc",
        "item_type": "ocr_region",
        "content": {
            "region_id": region_id,
            "text": text,
            "confidence": confidence,
        },
    }


def _payment_packet() -> dict:
    return {
        "question": "What is the payment date?",
        "answer_mode": "raw_ocr_spatial_candidates",
        "evidence": [
            _region(1, "Payment Date"),
            _region(2, "115/09/21"),
        ],
        "spatial_candidates": [
            {
                "document_id": "test_doc",
                "label_region_id": 1,
                "candidate_region_id": 2,
                "spatial_direction": "table_column_below",
                "relationship": "spatial_candidate_not_verified",
            }
        ],
        "ambiguities": [],
    }


def _ambiguous_packet() -> dict:
    return {
        "question": "What is the total?",
        "answer_mode": "raw_ocr_spatial_candidates",
        "evidence": [
            _region(3, "TOTAL"),
            _region(4, "310"),
            _region(5, "500"),
        ],
        "spatial_candidates": [
            {
                "document_id": "test_doc",
                "label_region_id": 3,
                "candidate_region_id": candidate_id,
                "spatial_direction": "same_row_right",
                "relationship": "spatial_candidate_not_verified",
            }
            for candidate_id in (4, 5)
        ],
        "ambiguities": [
            {
                "document_id": "test_doc",
                "label_region_id": 3,
                "candidate_region_ids": [4, 5],
                "candidate_count": 2,
                "reason": "multiple_plausible_spatial_values",
            }
        ],
    }


def run_service_boundary_checks() -> bool:
    """Exercise answer() without constructing a Groq client or sending data."""
    from services.document_qa_service import DocumentQAService

    service = DocumentQAService.__new__(DocumentQAService)
    service.model = "simulated-model"
    api_calls = []

    def fake_api(packet):
        api_calls.append(packet["question"])
        return {
            "answer": "115/09/21",
            "evidence_item_ids": ["test_doc:ocr_region:1", "test_doc:ocr_region:2"],
            "uncertainty": None,
            "usage": {},
        }

    service._call_groq = fake_api

    checks = []

    service.build_llm_packet = lambda question: _ambiguous_packet()
    blocked = service.answer("What is the total?")
    checks.extend(
        [
            ("ambiguity remains ungrounded", blocked.get("grounded") is False),
            ("ambiguity retains low confidence", blocked.get("confidence_level") == "low"),
            ("ambiguity is blocked before API", len(api_calls) == 0),
        ]
    )

    service.build_llm_packet = lambda question: _payment_packet()
    grounded = service.answer("What is the payment date?")
    checks.extend(
        [
            ("date answer is grounded", grounded.get("grounded") is True),
            ("date type is preserved", grounded.get("answer_type") == "date"),
            ("date has numeric heuristic score", isinstance(grounded.get("confidence_score"), (float, int))),
            ("date invokes simulated API exactly once", len(api_calls) == 1),
            ("date cites supplied value", any(
                item.get("evidence_item_id") == "test_doc:ocr_region:2"
                for item in grounded.get("evidence", [])
            )),
        ]
    )

    service.build_llm_packet = lambda question: {
        "question": question,
        "answer_mode": "no_answer",
        "evidence": [],
        "spatial_candidates": [],
        "ambiguities": [],
    }
    unanswered = service.answer("What is the customer name?")
    checks.extend(
        [
            ("no evidence remains ungrounded", unanswered.get("grounded") is False),
            ("no evidence has low confidence", unanswered.get("confidence_level") == "low"),
            ("no evidence makes no additional API request", len(api_calls) == 1),
        ]
    )

    print("\n" + "=" * 88)
    print("EXPERIMENT 66: MOCKED SERVICE BOUNDARY CHECKS (NO NETWORK)")
    print("=" * 88)
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"RESULT: {sum(bool(value) for _, value in checks)}/{len(checks)} checks passed")
    return all(passed for _, passed in checks)


def main() -> int:
    suite_results = []
    print("=" * 88)
    print("EXPERIMENT 66: FULL LOCAL TYPED-CONFIDENCE REGRESSION")
    print("=" * 88)

    for title, module_name, function_name in SUITES:
        print(f"\n--- {title} ---")
        try:
            module = import_module(module_name)
            status = getattr(module, function_name)()
            ok = status == 0
        except Exception as exc:
            ok = False
            print(f"[ERROR] {type(exc).__name__}: {exc}")
        suite_results.append((title, ok))

    try:
        boundary_ok = run_service_boundary_checks()
    except Exception as exc:
        boundary_ok = False
        print(f"\n[ERROR] Service boundary checks: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 88)
    print("EXPERIMENT 66 SUMMARY")
    print("=" * 88)
    for title, passed in suite_results:
        print(f"[{'PASS' if passed else 'FAIL'}] {title}")
    print(f"[{'PASS' if boundary_ok else 'FAIL'}] Mocked service boundary checks")

    count = sum(passed for _, passed in suite_results) + int(boundary_ok)
    total = len(suite_results) + 1
    print(f"OVERALL: {count}/{total} suites passed")
    print("No Groq requests were made by this script.")
    return 0 if count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
