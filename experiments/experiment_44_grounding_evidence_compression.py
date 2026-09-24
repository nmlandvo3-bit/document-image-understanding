"""
Experiment 44: Grounding Evidence Compression

Purpose
-------
Reduce generic retrieval output into the smallest sufficient
grounding context before sending evidence to an LLM.

This experiment builds on:
- Experiment 40 unified documents
- Experiment 41 generic retrieval
- Experiment 42 grounded-answer selection behavior
- Experiment 43 provider-neutral grounding interface

Goals
-----
- Preserve relevant multi-record evidence.
- Preserve relevant multi-control evidence.
- Remove unrelated cross-document retrieval noise.
- Prevent intent bonuses alone from creating evidence.
- Produce compact provider-neutral LLM packets.
- Use no LLM.

Outputs
-------
results/unified_documents/llm_grounding_compressed/
    compressed_grounding_tests.json
    compressed_grounding_report.txt
"""

import json
from pathlib import Path

from experiments.experiment_41_generic_cross_document_retrieval import (
    PIC1_PATH,
    PIC13_PATH,
    build_search_corpus,
    infer_query_intent,
    load_json,
    normalize_text,
    search,
)

from experiments.experiment_42_generic_grounded_answers import (
    infer_answer_mode,
    select_evidence,
)

from experiments.experiment_43_llm_grounding_interface import (
    SYSTEM_GROUNDING_RULES,
    build_evidence_item,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "unified_documents"
    / "llm_grounding_compressed"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "compressed_grounding_tests.json"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "compressed_grounding_report.txt"
)


def has_real_evidence(result):
    """
    A retrieval result must contain an actual lexical match.
    Intent bonuses alone are not enough.
    """

    return bool(
        result.get(
            "matched_tokens"
        )
    )


def remove_non_evidence(results):
    return [
        result
        for result in results
        if has_real_evidence(
            result
        )
    ]


def compress_results(
    query,
    retrieval_results,
):
    """
    Compress retrieval results according to the same semantic
    answer modes validated in Experiment 42.

    This keeps only evidence likely to be needed by an LLM.
    """

    usable_results = (
        remove_non_evidence(
            retrieval_results
        )
    )

    if not usable_results:
        return (
            "no_answer",
            [],
        )

    mode = infer_answer_mode(
        query,
        usable_results,
    )

    selected = select_evidence(
        query,
        usable_results,
        mode,
    )

    return (
        mode,
        selected,
    )


def build_compressed_packet(
    query,
    corpus,
    top_k=15,
):
    retrieval_results = search(
        corpus,
        query,
        top_k=top_k,
    )

    (
        answer_mode,
        selected_results,
    ) = compress_results(
        query,
        retrieval_results,
    )

    evidence = [
        build_evidence_item(
            result
        )
        for result
        in selected_results
    ]

    query_intents = sorted(
        infer_query_intent(
            query
        )
    )

    documents = sorted(
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

    compression_ratio = 0.0

    if retrieval_results:
        compression_ratio = (
            len(evidence)
            / len(
                retrieval_results
            )
        )

    return {
        "interface_version": (
            "1.1-llm-grounding-compressed"
        ),
        "query": query,
        "query_intents": (
            query_intents
        ),
        "answer_mode": (
            answer_mode
        ),
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
            "raw_retrieved_items": (
                len(
                    retrieval_results
                )
            ),
            "compressed_items": (
                len(evidence)
            ),
            "compression_ratio": round(
                compression_ratio,
                4,
            ),
            "document_ids": (
                documents
            ),
            "item_types": (
                item_types
            ),
        },
        "evidence": evidence,
        "recommended_output": {
            "answer": (
                "Answer naturally using only "
                "the supplied evidence."
            ),
            "evidence_item_ids": (
                "Return the IDs of evidence "
                "actually used."
            ),
            "uncertainty": (
                "State uncertainty or review "
                "flags when relevant."
            ),
        },
    }


def evidence_ids(packet):
    return [
        item[
            "item_id"
        ]
        for item in packet[
            "evidence"
        ]
    ]


def validate_packet_structure(
    packet,
):
    errors = []

    required = {
        "interface_version",
        "query",
        "query_intents",
        "answer_mode",
        "grounding_policy",
        "retrieval_summary",
        "evidence",
        "recommended_output",
    }

    missing = (
        required
        - set(
            packet.keys()
        )
    )

    if missing:
        errors.append(
            "Missing keys: "
            + ", ".join(
                sorted(missing)
            )
        )

    policy = packet.get(
        "grounding_policy",
        {},
    )

    if (
        policy.get(
            "allow_external_knowledge"
        )
        is not False
    ):
        errors.append(
            "External knowledge must "
            "remain disabled."
        )

    if (
        policy.get(
            "require_evidence"
        )
        is not True
    ):
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
                    (
                        "Evidence item missing "
                        f"{key}."
                    )
                )

    return errors


def validate_exact_ids(
    packet,
    expected,
):
    return (
        evidence_ids(
            packet
        )
        == expected
    )


def validate_id_set(
    packet,
    expected,
):
    return (
        set(
            evidence_ids(
                packet
            )
        )
        == set(expected)
    )


def run_tests(
    corpus,
):
    tests = [
        {
            "name": (
                "pic1 dormitory opening"
            ),
            "query": (
                "When does the dormitory open?"
            ),
            "expected_mode": (
                "single_record"
            ),
            "validation": "exact",
            "expected_evidence": [
                "pic1:record:row_18",
            ],
        },
        {
            "name": (
                "pic1 2000 payment groups"
            ),
            "query": (
                "Who needs to pay $2000?"
            ),
            "expected_mode": (
                "multi_record"
            ),
            "validation": "set",
            "expected_evidence": [
                "pic1:record:row_03",
                "pic1:record:row_06",
                "pic1:record:row_09",
                "pic1:record:row_13",
                "pic1:record:row_16",
            ],
        },
        {
            "name": (
                "pic13 date of birth"
            ),
            "query": (
                "What is the date of birth?"
            ),
            "expected_mode": (
                "single_field"
            ),
            "validation": "exact",
            "expected_evidence": [
                "pic13:field:2",
            ],
        },
        {
            "name": (
                "pic13 Medicare ID"
            ),
            "query": (
                "What is the Medicare "
                "identification number?"
            ),
            "expected_mode": (
                "single_field"
            ),
            "validation": "exact",
            "expected_evidence": [
                "pic13:field:3",
            ],
        },
        {
            "name": (
                "pic13 claims checkbox"
            ),
            "query": (
                "Is Medicare claims selected?"
            ),
            "expected_mode": (
                "single_control"
            ),
            "validation": "exact",
            "expected_evidence": [
                "pic13:control:4",
            ],
        },
        {
            "name": (
                "pic13 Other response"
            ),
            "query": (
                "What information was "
                "written under Other?"
            ),
            "expected_mode": (
                "single_free_text"
            ),
            "validation": "exact",
            "expected_evidence": [
                "pic13:free_text:1",
            ],
        },
        {
            "name": (
                "pic13 checked options"
            ),
            "query": (
                "Which options are checked?"
            ),
            "expected_mode": (
                "multi_control"
            ),
            "validation": "set",
            "expected_evidence": [
                "pic13:control:1",
                "pic13:control:4",
                "pic13:control:7",
            ],
        },
    ]

    outputs = []

    for test in tests:
        packet = (
            build_compressed_packet(
                test["query"],
                corpus,
                top_k=15,
            )
        )

        structure_errors = (
            validate_packet_structure(
                packet
            )
        )

        structure_pass = (
            len(
                structure_errors
            )
            == 0
        )

        mode_pass = (
            packet[
                "answer_mode"
            ]
            == test[
                "expected_mode"
            ]
        )

        if (
            test[
                "validation"
            ]
            == "exact"
        ):
            evidence_pass = (
                validate_exact_ids(
                    packet,
                    test[
                        "expected_evidence"
                    ],
                )
            )

        elif (
            test[
                "validation"
            ]
            == "set"
        ):
            evidence_pass = (
                validate_id_set(
                    packet,
                    test[
                        "expected_evidence"
                    ],
                )
            )

        else:
            raise ValueError(
                "Unknown validation type."
            )

        passed = (
            structure_pass
            and mode_pass
            and evidence_pass
        )

        outputs.append(
            {
                "name": (
                    test["name"]
                ),
                "query": (
                    test["query"]
                ),
                "expected_mode": (
                    test[
                        "expected_mode"
                    ]
                ),
                "expected_evidence": (
                    test[
                        "expected_evidence"
                    ]
                ),
                "structure_pass": (
                    structure_pass
                ),
                "mode_pass": (
                    mode_pass
                ),
                "evidence_pass": (
                    evidence_pass
                ),
                "structure_errors": (
                    structure_errors
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
        if test["passed"]
    )

    lines = [
        "=" * 78,
        (
            "EXPERIMENT 44: "
            "GROUNDING EVIDENCE COMPRESSION"
        ),
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

        summary = packet[
            "retrieval_summary"
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
                    f"Answer mode: "
                    f"{packet['answer_mode']}"
                ),
                (
                    "Raw retrieval items: "
                    f"{summary['raw_retrieved_items']}"
                ),
                (
                    "Compressed evidence items: "
                    f"{summary['compressed_items']}"
                ),
                (
                    "Compression ratio: "
                    f"{summary['compression_ratio']}"
                ),
                (
                    "Documents: "
                    f"{summary['document_ids']}"
                ),
                (
                    "Item types: "
                    f"{summary['item_types']}"
                ),
                "",
                "Evidence:",
            ]
        )

        evidence = packet[
            "evidence"
        ]

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
            "structure_errors"
        ]:
            lines.append("")
            lines.append(
                "Structure errors:"
            )

            for error in test[
                "structure_errors"
            ]:
                lines.append(
                    f"  - {error}"
                )

        lines.append("")

    lines.extend(
        [
            "=" * 78,
            "GROUNDING COMPRESSION SUMMARY",
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
                "STATUS: GROUNDING EVIDENCE "
                "COMPRESSION CHECKPOINT PASSED"
            )
        )
    else:
        lines.append(
            (
                "STATUS: GROUNDING EVIDENCE "
                "COMPRESSION NEEDS REFINEMENT"
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
        "EXPERIMENT 44: "
        "GROUNDING EVIDENCE COMPRESSION"
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
        if test["passed"]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": 44,
        "schema_version": (
            "1.1-llm-grounding-compressed"
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
        packet = test[
            "packet"
        ]

        summary = packet[
            "retrieval_summary"
        ]

        print(
            (
                f"[{'PASS' if test['passed'] else 'FAIL'}] "
                f"{test['name']}"
            )
        )

        print(
            (
                f"       mode: "
                f"{packet['answer_mode']}"
            )
        )

        print(
            (
                "       evidence: "
                + ", ".join(
                    evidence_ids(
                        packet
                    )
                )
            )
        )

        print(
            (
                "       compression: "
                f"{summary['raw_retrieved_items']}"
                " -> "
                f"{summary['compressed_items']}"
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
            "EXPERIMENT 44 CHECKPOINT PASSED"
        )
    else:
        print(
            "EXPERIMENT 44 NEEDS REFINEMENT"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()