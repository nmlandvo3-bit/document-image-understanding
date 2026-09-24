"""
Experiment 53: Verify spatial evidence-pair completeness.

This experiment simulates an answer that cites only a value region.
It checks whether the original retrieval packet identifies the
associated label as additional spatial context.

No OCR, Groq calls, or production-file changes.
"""

import json
from pathlib import Path

from services.layout_aware_ocr_retrieval import build_raw_ocr_packet


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS = [
    {
        "name": "Form date of birth",
        "path": (
            PROJECT_ROOT
            / "results/gui_v4/single/pic13/unified_raw_ocr.json"
        ),
        "question": "What is the date of birth?",
        "model_cited_region": 15,
        "expected_label_region": 13,
    },
    {
        "name": "Receipt total",
        "path": (
            PROJECT_ROOT
            / "results/gui_v4/single/receipt/unified_raw_ocr.json"
        ),
        "question": "What is the total amount?",
        "model_cited_region": 22,
        "expected_label_region": 21,
    },
    {
        "name": "Receipt tax amount",
        "path": (
            PROJECT_ROOT
            / "results/gui_v4/single/receipt/unified_raw_ocr.json"
        ),
        "question": "What is the tax amount?",
        "model_cited_region": 19,
        "expected_label_region": 20,
    },
]


def inspect_case(case):
    document = json.loads(
        case["path"].read_text(encoding="utf-8")
    )

    packet = build_raw_ocr_packet(
        case["question"],
        [document],
    )

    document_id = document["document_id"]

    cited_id = (
        f"{document_id}:ocr_region:"
        f"{case['model_cited_region']}"
    )

    expected_label_id = (
        f"{document_id}:ocr_region:"
        f"{case['expected_label_region']}"
    )

    evidence_by_id = {
        item["evidence_item_id"]: item
        for item in packet["evidence"]
    }

    # Simulate a model response that cites the value but not its label.
    model_cited_ids = {cited_id}

    context_ids = set()

    for relation in packet["spatial_candidates"]:
        value_id = (
            f"{relation['document_id']}:ocr_region:"
            f"{relation['candidate_region_id']}"
        )

        label_id = (
            f"{relation['document_id']}:ocr_region:"
            f"{relation['label_region_id']}"
        )

        if (
            value_id in model_cited_ids
            and label_id in evidence_by_id
        ):
            context_ids.add(label_id)

    passed = (
        cited_id in evidence_by_id
        and expected_label_id in context_ids
    )

    print(f"\nTEST: {case['name']}")
    print("Question:", case["question"])
    print("Model-cited value:", cited_id)

    if cited_id in evidence_by_id:
        print(
            "Value text:",
            evidence_by_id[cited_id]["content"]["text"],
        )

    print("Additional spatial context:")

    for context_id in sorted(context_ids):
        print(
            " ",
            context_id,
            "|",
            evidence_by_id[context_id]["content"]["text"],
        )

    print("Result:", "PASS" if passed else "FAIL")

    return passed


def main():
    print("=" * 70)
    print("EXPERIMENT 53: EVIDENCE COMPLETENESS")
    print("=" * 70)

    results = [inspect_case(case) for case in DOCUMENTS]

    print("\n" + "=" * 70)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print("=" * 70)


if __name__ == "__main__":
    main()