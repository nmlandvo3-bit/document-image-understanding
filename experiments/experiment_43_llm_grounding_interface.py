"""
Experiment 43: LLM Grounding Interface

Purpose
-------
Create provider-neutral grounding packets for future LLM/agent use.

The packet contains:
- user query
- retrieval intent
- retrieved structured evidence
- provenance
- review flags
- deterministic grounding instructions

This experiment does NOT:
- rerun OCR
- rerun parsing
- call an LLM
- alter source evidence

Outputs
-------
results/unified_documents/llm_grounding/
    grounding_packet_tests.json
    grounding_packet_report.txt
"""

import json
from pathlib import Path

from experiments.experiment_41_generic_cross_document_retrieval import (
    PIC1_PATH,
    PIC13_PATH,
    build_search_corpus,
    infer_query_intent,
    load_json,
    search,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "unified_documents"
    / "llm_grounding"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "grounding_packet_tests.json"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "grounding_packet_report.txt"
)


SYSTEM_GROUNDING_RULES = [
    (
        "Answer only from the supplied evidence."
    ),
    (
        "Do not invent missing facts, dates, names, values, "
        "checkbox states, or relationships."
    ),
    (
        "Preserve uncertainty and review flags when they exist."
    ),
    (
        "If the supplied evidence is insufficient, say that the "
        "answer cannot be determined from the document evidence."
    ),
    (
        "Do not silently correct OCR text."
    ),
    (
        "Use provenance IDs when explaining where an answer came from."
    ),
]


def compact_record_payload(payload):
    """
    Convert a unified table record into a compact LLM-ready structure.
    """

    return {
        "record_id": payload.get(
            "record_id"
        ),
        "category": payload.get(
            "category"
        ),
        "event": payload.get(
            "event"
        ),
        "time": payload.get(
            "time"
        ),
        "detail": payload.get(
            "detail"
        ),
        "explanation": payload.get(
            "explanation"
        ),
        "review_flags": payload.get(
            "review_flags",
            [],
        ),
        "source_cells": payload.get(
            "source_cells",
            {},
        ),
    }


def compact_field_payload(payload):
    return {
        "field_id": payload.get(
            "field_id"
        ),
        "label": payload.get(
            "label"
        ),
        "value": payload.get(
            "value"
        ),
        "label_region_id": payload.get(
            "label_region_id"
        ),
        "value_region_id": payload.get(
            "value_region_id"
        ),
        "label_confidence": payload.get(
            "label_confidence"
        ),
        "value_confidence": payload.get(
            "value_confidence"
        ),
    }


def compact_control_payload(payload):
    return {
        "control_id": payload.get(
            "control_id"
        ),
        "control_type": payload.get(
            "control_type"
        ),
        "label": payload.get(
            "label"
        ),
        "checked": bool(
            payload.get(
                "checked",
                False,
            )
        ),
        "state": payload.get(
            "state"
        ),
        "label_region_id": payload.get(
            "label_region_id"
        ),
        "bbox": payload.get(
            "bbox"
        ),
        "interior_dark_ratio": payload.get(
            "interior_dark_ratio"
        ),
    }


def compact_free_text_payload(payload):
    return {
        "source_control_id": payload.get(
            "source_control_id"
        ),
        "source_control_label": payload.get(
            "source_control_label"
        ),
        "text_region_id": payload.get(
            "text_region_id"
        ),
        "text": payload.get(
            "text"
        ),
    }


def compact_payload(result):
    item_type = result[
        "item_type"
    ]

    payload = result[
        "payload"
    ]

    if item_type == "record":
        return compact_record_payload(
            payload
        )

    if item_type == "field":
        return compact_field_payload(
            payload
        )

    if item_type == "control":
        return compact_control_payload(
            payload
        )

    if item_type == "free_text":
        return compact_free_text_payload(
            payload
        )

    return payload


def has_real_evidence(result):
    """
    Intent bonuses should never create grounding evidence alone.
    """

    return bool(
        result.get(
            "matched_tokens"
        )
    )


def filter_grounding_results(
    results,
):
    return [
        result
        for result in results
        if has_real_evidence(
            result
        )
    ]


def build_evidence_item(
    result,
):
    return {
        "item_id": result[
            "item_id"
        ],
        "document_id": result[
            "document_id"
        ],
        "document_type": result[
            "document_type"
        ],
        "item_type": result[
            "item_type"
        ],
        "score": result[
            "score"
        ],
        "coverage": result[
            "coverage"
        ],
        "matched_tokens": result[
            "matched_tokens"
        ],
        "provenance": result[
            "provenance"
        ],
        "content": compact_payload(
            result
        ),
    }


def build_grounding_packet(
    query,
    corpus,
    top_k=8,
):
    retrieval_results = search(
        corpus,
        query,
        top_k=top_k,
    )

    grounded_results = (
        filter_grounding_results(
            retrieval_results
        )
    )

    intents = sorted(
        infer_query_intent(
            query
        )
    )

    evidence = [
        build_evidence_item(
            result
        )
        for result in grounded_results
    ]

    document_ids = sorted(
        {
            item[
                "document_id"
            ]
            for item in evidence
        }
    )

    item_types = sorted(
        {
            item[
                "item_type"
            ]
            for item in evidence
        }
    )

    packet = {
        "interface_version": (
            "1.0-llm-grounding-experimental"
        ),
        "query": query,
        "query_intents": intents,
        "grounding_policy": {
            "rules": (
                SYSTEM_GROUNDING_RULES
            ),
            "allow_external_knowledge": False,
            "allow_ocr_autocorrection": False,
            "require_evidence": True,
            "insufficient_evidence_behavior": (
                "state_insufficient_document_evidence"
            ),
        },
        "retrieval_summary": {
            "retrieved_items": (
                len(retrieval_results)
            ),
            "grounded_items": (
                len(evidence)
            ),
            "document_ids": (
                document_ids
            ),
            "item_types": (
                item_types
            ),
        },
        "evidence": evidence,
        "recommended_output": {
            "answer": (
                "Natural-language answer grounded only "
                "in the supplied evidence."
            ),
            "evidence_item_ids": (
                "List the item IDs actually used."
            ),
            "uncertainty": (
                "Mention review flags or insufficient evidence."
            ),
        },
    }

    return packet


def validate_packet(
    packet,
):
    errors = []

    required_keys = {
        "interface_version",
        "query",
        "query_intents",
        "grounding_policy",
        "retrieval_summary",
        "evidence",
        "recommended_output",
    }

    missing = (
        required_keys
        - set(
            packet.keys()
        )
    )

    if missing:
        errors.append(
            "Missing top-level keys: "
            + ", ".join(
                sorted(missing)
            )
        )

    policy = packet.get(
        "grounding_policy",
        {},
    )

    if policy.get(
        "allow_external_knowledge"
    ) is not False:
        errors.append(
            "External knowledge must be disabled."
        )

    if policy.get(
        "require_evidence"
    ) is not True:
        errors.append(
            "Evidence must be required."
        )

    for item in packet.get(
        "evidence",
        [],
    ):
        for key in (
            "item_id",
            "document_id",
            "document_type",
            "item_type",
            "provenance",
            "content",
        ):
            if key not in item:
                errors.append(
                    f"Evidence item missing {key}."
                )

    return errors


def run_tests(
    corpus,
):
    tests = [
        {
            "name": (
                "pic1 dormitory opening packet"
            ),
            "query": (
                "When does the dormitory open?"
            ),
            "expected_first": (
                "pic1:record:row_18"
            ),
            "expected_type": (
                "record"
            ),
        },
        {
            "name": (
                "pic1 payment packet"
            ),
            "query": (
                "Who needs to pay $2000?"
            ),
            "expected_first": (
                "pic1:record:row_03"
            ),
            "expected_type": (
                "record"
            ),
        },
        {
            "name": (
                "pic13 date of birth packet"
            ),
            "query": (
                "What is the date of birth?"
            ),
            "expected_first": (
                "pic13:field:2"
            ),
            "expected_type": (
                "field"
            ),
        },
        {
            "name": (
                "pic13 claims checkbox packet"
            ),
            "query": (
                "Is Medicare claims selected?"
            ),
            "expected_first": (
                "pic13:control:4"
            ),
            "expected_type": (
                "control"
            ),
        },
        {
            "name": (
                "pic13 Other free-text packet"
            ),
            "query": (
                "What information was written under Other?"
            ),
            "expected_first": (
                "pic13:free_text:1"
            ),
            "expected_type": (
                "free_text"
            ),
        },
    ]

    outputs = []

    for test in tests:
        packet = build_grounding_packet(
            test["query"],
            corpus,
            top_k=8,
        )

        validation_errors = (
            validate_packet(
                packet
            )
        )

        evidence = packet[
            "evidence"
        ]

        first_id = (
            evidence[0][
                "item_id"
            ]
            if evidence
            else None
        )

        first_type = (
            evidence[0][
                "item_type"
            ]
            if evidence
            else None
        )

        first_pass = (
            first_id
            == test[
                "expected_first"
            ]
        )

        type_pass = (
            first_type
            == test[
                "expected_type"
            ]
        )

        structure_pass = (
            len(
                validation_errors
            )
            == 0
        )

        passed = (
            first_pass
            and type_pass
            and structure_pass
        )

        outputs.append(
            {
                "name": (
                    test["name"]
                ),
                "query": (
                    test["query"]
                ),
                "expected_first": (
                    test[
                        "expected_first"
                    ]
                ),
                "expected_type": (
                    test[
                        "expected_type"
                    ]
                ),
                "first_pass": (
                    first_pass
                ),
                "type_pass": (
                    type_pass
                ),
                "structure_pass": (
                    structure_pass
                ),
                "validation_errors": (
                    validation_errors
                ),
                "passed": passed,
                "packet": packet,
            }
        )

    return outputs


def make_report(
    tests,
):
    passed_count = sum(
        1
        for test in tests
        if test[
            "passed"
        ]
    )

    lines = [
        "=" * 78,
        "EXPERIMENT 43: LLM GROUNDING INTERFACE",
        "=" * 78,
        (
            f"Tests: "
            f"{len(tests)}"
        ),
        (
            f"Passed: "
            f"{passed_count}"
        ),
        "",
    ]

    for index, test in enumerate(
        tests,
        start=1,
    ):
        packet = test[
            "packet"
        ]

        evidence = packet[
            "evidence"
        ]

        lines.extend(
            [
                "=" * 78,
                (
                    f"TEST {index}: "
                    f"{test['name']}"
                ),
                "-" * 78,
                (
                    f"Query: "
                    f"{test['query']}"
                ),
                (
                    f"Status: "
                    f"{'PASS' if test['passed'] else 'FAIL'}"
                ),
                (
                    f"Structure: "
                    f"{'PASS' if test['structure_pass'] else 'FAIL'}"
                ),
                (
                    f"Expected first: "
                    f"{test['expected_first']}"
                ),
                (
                    f"Expected type: "
                    f"{test['expected_type']}"
                ),
                (
                    "Retrieved items: "
                    f"{packet['retrieval_summary']['retrieved_items']}"
                ),
                (
                    "Grounded items: "
                    f"{packet['retrieval_summary']['grounded_items']}"
                ),
                (
                    "Documents: "
                    f"{packet['retrieval_summary']['document_ids']}"
                ),
                (
                    "Item types: "
                    f"{packet['retrieval_summary']['item_types']}"
                ),
                "",
                "Evidence:",
            ]
        )

        if not evidence:
            lines.append(
                "  None"
            )

        for item in evidence:
            lines.append(
                (
                    f"  - "
                    f"{item['item_id']} "
                    f"type={item['item_type']} "
                    f"score={item['score']} "
                    f"coverage={item['coverage']}"
                )
            )

        if test[
            "validation_errors"
        ]:
            lines.append("")
            lines.append(
                "Validation errors:"
            )

            for error in test[
                "validation_errors"
            ]:
                lines.append(
                    f"  - {error}"
                )

        lines.append("")

    lines.extend(
        [
            "=" * 78,
            "LLM GROUNDING INTERFACE SUMMARY",
            "=" * 78,
            (
                f"Tests passed: "
                f"{passed_count}/{len(tests)}"
            ),
        ]
    )

    if passed_count == len(
        tests
    ):
        lines.append(
            (
                "STATUS: LLM GROUNDING "
                "INTERFACE CHECKPOINT PASSED"
            )
        )
    else:
        lines.append(
            (
                "STATUS: LLM GROUNDING "
                "INTERFACE NEEDS REFINEMENT"
            )
        )

    lines.append(
        "=" * 78
    )

    return "\n".join(
        lines
    ) + "\n"


def main():
    print("=" * 72)
    print(
        "EXPERIMENT 43: "
        "LLM GROUNDING INTERFACE"
    )
    print("=" * 72)

    pic1 = load_json(
        PIC1_PATH
    )

    pic13 = load_json(
        PIC13_PATH
    )

    corpus = build_search_corpus(
        [
            pic1,
            pic13,
        ]
    )

    tests = run_tests(
        corpus
    )

    passed_count = sum(
        1
        for test in tests
        if test[
            "passed"
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": 43,
        "schema_version": (
            "1.0-llm-grounding-interface"
        ),
        "corpus_items": (
            len(corpus)
        ),
        "tests_passed": (
            passed_count
        ),
        "tests_total": (
            len(tests)
        ),
        "tests": tests,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    report = make_report(
        tests
    )

    OUTPUT_REPORT.write_text(
        report,
        encoding="utf-8",
    )

    for test in tests:
        print(
            (
                f"[{'PASS' if test['passed'] else 'FAIL'}] "
                f"{test['name']}"
            )
        )

        evidence = test[
            "packet"
        ][
            "evidence"
        ]

        if evidence:
            print(
                (
                    "       first: "
                    f"{evidence[0]['item_id']} "
                    f"type="
                    f"{evidence[0]['item_type']}"
                )
            )

    print()

    print(
        f"Tests passed: "
        f"{passed_count}/{len(tests)}"
    )

    print()

    print(
        f"JSON:   {OUTPUT_JSON}"
    )

    print(
        f"Report: {OUTPUT_REPORT}"
    )

    print("=" * 72)

    if passed_count == len(
        tests
    ):
        print(
            "EXPERIMENT 43 CHECKPOINT PASSED"
        )
    else:
        print(
            "EXPERIMENT 43 NEEDS REFINEMENT"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()