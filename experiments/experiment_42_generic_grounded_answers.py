"""
Experiment 42: Generic Grounded Answer Generation

Purpose
-------
Generate deterministic grounded answers from the generic retrieval layer
created in Experiment 41.

Supported semantic object types:
- record
- field
- control
- free_text

Goals
-----
1. Do not rerun OCR.
2. Do not rerun parsing.
3. Reuse the unified documents.
4. Reuse the generic retrieval engine from Experiment 41.
5. Produce answers only from retrieved structured evidence.
6. Preserve source item IDs and provenance.
7. Support both single-result and multi-result answers.
8. Use no LLM.

Outputs
-------
results/unified_documents/grounded_answers/
    grounded_answer_tests.json
    grounded_answer_report.txt
"""

import json
from pathlib import Path

from experiments.experiment_41_generic_cross_document_retrieval import (
    PIC1_PATH,
    PIC13_PATH,
    build_search_corpus,
    load_json,
    normalize_text,
    search,
    tokenize,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "unified_documents"
    / "grounded_answers"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "grounded_answer_tests.json"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "grounded_answer_report.txt"
)


def flatten_display(value):
    """
    Convert structured values into readable text.

    This is for deterministic answer presentation only.
    Source JSON remains unchanged.
    """

    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        return str(value)

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts = [
            flatten_display(item)
            for item in value
        ]

        parts = [
            item
            for item in parts
            if item
        ]

        return "; ".join(parts)

    if isinstance(value, dict):
        # Prefer human-readable normalized/string fields
        # if they exist.
        preferred_keys = (
            "display",
            "text",
            "normalized_text",
            "normalized",
            "value",
            "raw_text",
        )

        for key in preferred_keys:
            if key in value:
                result = flatten_display(
                    value[key]
                )

                if result:
                    return result

        # Otherwise preserve meaningful dictionary content.
        parts = []

        for key, item in value.items():
            rendered = flatten_display(
                item
            )

            if rendered:
                parts.append(
                    f"{key}: {rendered}"
                )

        return "; ".join(parts)

    return str(value)


def useful_record_parts(
    record,
):
    """
    Convert a unified table record into readable components.
    """

    category = flatten_display(
        record.get(
            "category"
        )
    )

    event = flatten_display(
        record.get(
            "event"
        )
    )

    time = flatten_display(
        record.get(
            "time"
        )
    )

    detail = flatten_display(
        record.get(
            "detail"
        )
    )

    explanation = flatten_display(
        record.get(
            "explanation"
        )
    )

    return {
        "category": category,
        "event": event,
        "time": time,
        "detail": detail,
        "explanation": explanation,
    }


def infer_answer_mode(
    query,
    results,
):
    """
    Infer answer style from the query and retrieved item types.

    This does not route by document type.
    """

    normalized = normalize_text(
        query
    )

    tokens = set(
        tokenize(
            query
        )
    )

    if not results:
        return "no_answer"

    top_type = results[0][
        "item_type"
    ]

    if (
        "which" in normalized
        and (
            "checked" in normalized
            or "selected" in normalized
        )
    ):
        return "multi_control"

    if (
        normalized.startswith(
            "who "
        )
        or "who need" in normalized
    ):
        return "multi_record"

    if (
        "what happens" in normalized
        or "what happen" in normalized
    ):
        return "multi_record"

    if top_type == "field":
        return "single_field"

    if top_type == "control":
        return "single_control"

    if top_type == "free_text":
        return "single_free_text"

    if top_type == "record":
        return "single_record"

    return "single_generic"


def result_has_real_evidence(
    result,
):
    """
    Prevent intent bonuses alone from creating answer evidence.

    Experiment 41 intentionally allowed intent bonuses to rerank items.
    Here, an item must also contain an actual lexical match.
    """

    return bool(
        result.get(
            "matched_tokens"
        )
    )


def select_single_result(
    results,
):
    for result in results:
        if result_has_real_evidence(
            result
        ):
            return [result]

    return []


def select_multi_record_results(
    query,
    results,
):
    """
    Select a grounded set of records.

    For payment/who queries, equal high-quality matches should all survive.
    For category-summary queries, relevant same-category records can survive.
    """

    candidates = [
        result
        for result in results
        if (
            result[
                "item_type"
            ] == "record"
            and result_has_real_evidence(
                result
            )
        )
    ]

    if not candidates:
        return []

    top_score = candidates[0][
        "score"
    ]

    normalized = normalize_text(
        query
    )

    if normalized.startswith(
        "who "
    ) or "who need" in normalized:
        # Keep strong tied results.
        selected = [
            item
            for item in candidates
            if item["score"] == top_score
        ]

        return selected

    # Category-summary behavior:
    # preserve high-coverage records from the same query.
    selected = []

    for item in candidates:
        if item[
            "coverage"
        ] >= 0.50:
            selected.append(
                item
            )

    if selected:
        return selected

    return [
        candidates[0]
    ]


def select_multi_control_results(
    results,
):
    """
    For questions asking which options are checked,
    return only grounded checked controls.

    This also fixes the small Experiment 41 quality issue where
    unchecked controls received an intent bonus despite having no
    lexical evidence.
    """

    selected = []

    for result in results:
        if result[
            "item_type"
        ] != "control":
            continue

        if not result_has_real_evidence(
            result
        ):
            continue

        payload = result[
            "payload"
        ]

        if bool(
            payload.get(
                "checked",
                False,
            )
        ):
            selected.append(
                result
            )

    return selected


def select_evidence(
    query,
    results,
    mode,
):
    if mode in {
        "single_field",
        "single_control",
        "single_free_text",
        "single_record",
        "single_generic",
    }:
        return select_single_result(
            results
        )

    if mode == "multi_record":
        return select_multi_record_results(
            query,
            results,
        )

    if mode == "multi_control":
        return select_multi_control_results(
            results
        )

    return []


def answer_field(
    result,
):
    payload = result[
        "payload"
    ]

    label = flatten_display(
        payload.get(
            "label"
        )
    )

    value = flatten_display(
        payload.get(
            "value"
        )
    )

    return (
        f"{label}: {value}"
    )


def answer_control(
    result,
):
    payload = result[
        "payload"
    ]

    label = flatten_display(
        payload.get(
            "label"
        )
    )

    checked = bool(
        payload.get(
            "checked",
            False,
        )
    )

    state = (
        "checked"
        if checked
        else "unchecked"
    )

    return (
        f"{label} is {state}."
    )


def answer_free_text(
    result,
):
    payload = result[
        "payload"
    ]

    source_label = flatten_display(
        payload.get(
            "source_control_label"
        )
    )

    text = flatten_display(
        payload.get(
            "text"
        )
    )

    if source_label:
        return (
            f"Under {source_label}: {text}"
        )

    return text


def answer_record(
    result,
):
    record = result[
        "payload"
    ]

    parts = useful_record_parts(
        record
    )

    components = []

    if parts["category"]:
        components.append(
            parts["category"]
        )

    if parts["event"]:
        components.append(
            parts["event"]
        )

    heading = " — ".join(
        components
    )

    details = []

    if parts["time"]:
        details.append(
            f"Time: {parts['time']}"
        )

    if parts["detail"]:
        details.append(
            f"Detail: {parts['detail']}"
        )

    if parts["explanation"]:
        details.append(
            f"Explanation: {parts['explanation']}"
        )

    if heading and details:
        return (
            heading
            + ". "
            + " ".join(details)
        )

    if heading:
        return heading

    return " ".join(
        details
    )


def answer_multi_records(
    results,
):
    lines = []

    for result in results:
        record = result[
            "payload"
        ]

        parts = useful_record_parts(
            record
        )

        label_parts = []

        if parts["category"]:
            label_parts.append(
                parts["category"]
            )

        if parts["event"]:
            label_parts.append(
                parts["event"]
            )

        label = " — ".join(
            label_parts
        )

        if not label:
            label = result[
                "item_id"
            ]

        suffix_parts = []

        if parts["time"]:
            suffix_parts.append(
                parts["time"]
            )

        if parts["detail"]:
            suffix_parts.append(
                parts["detail"]
            )

        if suffix_parts:
            lines.append(
                (
                    f"{label}: "
                    + " | ".join(
                        suffix_parts
                    )
                )
            )
        else:
            lines.append(
                label
            )

    return "\n".join(
        lines
    )


def answer_multi_controls(
    results,
):
    labels = []

    for result in results:
        payload = result[
            "payload"
        ]

        label = flatten_display(
            payload.get(
                "label"
            )
        )

        if label:
            labels.append(
                label
            )

    if not labels:
        return (
            "No checked options were found "
            "in the retrieved evidence."
        )

    return (
        "Checked options: "
        + "; ".join(labels)
        + "."
    )


def generate_answer(
    query,
    results,
):
    mode = infer_answer_mode(
        query,
        results,
    )

    evidence = select_evidence(
        query,
        results,
        mode,
    )

    if not evidence:
        return {
            "query": query,
            "mode": mode,
            "answer": (
                "No sufficiently grounded answer "
                "was found in the retrieved evidence."
            ),
            "grounded": False,
            "evidence": [],
        }

    if mode == "single_field":
        answer = answer_field(
            evidence[0]
        )

    elif mode == "single_control":
        answer = answer_control(
            evidence[0]
        )

    elif mode == "single_free_text":
        answer = answer_free_text(
            evidence[0]
        )

    elif mode == "single_record":
        answer = answer_record(
            evidence[0]
        )

    elif mode == "multi_record":
        answer = answer_multi_records(
            evidence
        )

    elif mode == "multi_control":
        answer = answer_multi_controls(
            evidence
        )

    else:
        top = evidence[0]

        item_type = top[
            "item_type"
        ]

        if item_type == "field":
            answer = answer_field(
                top
            )

        elif item_type == "control":
            answer = answer_control(
                top
            )

        elif item_type == "free_text":
            answer = answer_free_text(
                top
            )

        elif item_type == "record":
            answer = answer_record(
                top
            )

        else:
            answer = flatten_display(
                top.get(
                    "payload"
                )
            )

    grounded_evidence = []

    for item in evidence:
        grounded_evidence.append(
            {
                "item_id": (
                    item[
                        "item_id"
                    ]
                ),
                "document_id": (
                    item[
                        "document_id"
                    ]
                ),
                "document_type": (
                    item[
                        "document_type"
                    ]
                ),
                "item_type": (
                    item[
                        "item_type"
                    ]
                ),
                "score": (
                    item[
                        "score"
                    ]
                ),
                "coverage": (
                    item[
                        "coverage"
                    ]
                ),
                "matched_tokens": (
                    item[
                        "matched_tokens"
                    ]
                ),
                "provenance": (
                    item[
                        "provenance"
                    ]
                ),
            }
        )

    return {
        "query": query,
        "mode": mode,
        "answer": answer,
        "grounded": True,
        "evidence": (
            grounded_evidence
        ),
    }


def evidence_ids(
    answer_result,
):
    return [
        item[
            "item_id"
        ]
        for item in answer_result[
            "evidence"
        ]
    ]


def validate_exact_evidence(
    answer_result,
    expected_ids,
):
    return (
        evidence_ids(
            answer_result
        )
        == expected_ids
    )


def validate_evidence_set(
    answer_result,
    expected_ids,
):
    return (
        set(
            evidence_ids(
                answer_result
            )
        )
        == set(
            expected_ids
        )
    )


def validate_answer_fragments(
    answer_result,
    required_fragments,
):
    answer = normalize_text(
        answer_result[
            "answer"
        ]
    )

    return all(
        normalize_text(
            fragment
        )
        in answer
        for fragment
        in required_fragments
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
            "top_k": 10,
            "expected_mode": (
                "single_record"
            ),
            "validation": (
                "exact"
            ),
            "expected_evidence": [
                "pic1:record:row_18",
            ],
            "answer_fragments": [
                "Dormitory Opening",
            ],
        },
        {
            "name": (
                "pic1 3+2 deposit"
            ),
            "query": (
                "3+2 Program pay deposit"
            ),
            "top_k": 10,
            "expected_mode": (
                "single_record"
            ),
            "validation": (
                "exact"
            ),
            "expected_evidence": [
                "pic1:record:row_16",
            ],
            "answer_fragments": [
                "3+2 Program",
                "deposit",
            ],
        },
        {
            "name": (
                "pic1 all 2000 payment groups"
            ),
            "query": (
                "Who needs to pay $2000?"
            ),
            "top_k": 15,
            "expected_mode": (
                "multi_record"
            ),
            "validation": (
                "set"
            ),
            "expected_evidence": [
                "pic1:record:row_03",
                "pic1:record:row_06",
                "pic1:record:row_09",
                "pic1:record:row_13",
                "pic1:record:row_16",
            ],
            "answer_fragments": [
                "Graduate",
                "Undergraduate",
                "Waiting list 1",
                "Waiting list II",
                "3+2 Program",
            ],
        },
        {
            "name": (
                "pic13 date of birth"
            ),
            "query": (
                "What is the date of birth?"
            ),
            "top_k": 10,
            "expected_mode": (
                "single_field"
            ),
            "validation": (
                "exact"
            ),
            "expected_evidence": [
                "pic13:field:2",
            ],
            "answer_fragments": [
                "Date of Birth",
                "03/14/1992",
            ],
        },
        {
            "name": (
                "pic13 Medicare identification"
            ),
            "query": (
                "What is the Medicare "
                "identification number?"
            ),
            "top_k": 10,
            "expected_mode": (
                "single_field"
            ),
            "validation": (
                "exact"
            ),
            "expected_evidence": [
                "pic13:field:3",
            ],
            "answer_fragments": [
                "Medicare Identification Number",
                "1EG4-TE5-MK72",
            ],
        },
        {
            "name": (
                "pic13 Medicare claims selection"
            ),
            "query": (
                "Is Medicare claims selected?"
            ),
            "top_k": 10,
            "expected_mode": (
                "single_control"
            ),
            "validation": (
                "exact"
            ),
            "expected_evidence": [
                "pic13:control:4",
            ],
            "answer_fragments": [
                "Medicare claims",
                "checked",
            ],
        },
        {
            "name": (
                "pic13 Other written response"
            ),
            "query": (
                "What information was written "
                "under Other?"
            ),
            "top_k": 10,
            "expected_mode": (
                "single_free_text"
            ),
            "validation": (
                "exact"
            ),
            "expected_evidence": [
                "pic13:free_text:1",
            ],
            "answer_fragments": [
                "Coverage questions",
                "claims history",
            ],
        },
        {
            "name": (
                "pic13 checked options"
            ),
            "query": (
                "Which options are checked?"
            ),
            "top_k": 10,
            "expected_mode": (
                "multi_control"
            ),
            "validation": (
                "set"
            ),
            "expected_evidence": [
                "pic13:control:1",
                "pic13:control:4",
                "pic13:control:7",
            ],
            "answer_fragments": [
                "Limited Information",
                "Medicare claims",
                "Other",
            ],
        },
    ]

    outputs = []

    for test in tests:
        retrieval_results = search(
            corpus,
            test["query"],
            top_k=test[
                "top_k"
            ],
        )

        answer_result = (
            generate_answer(
                test["query"],
                retrieval_results,
            )
        )

        if (
            test[
                "validation"
            ]
            == "exact"
        ):
            evidence_pass = (
                validate_exact_evidence(
                    answer_result,
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
                validate_evidence_set(
                    answer_result,
                    test[
                        "expected_evidence"
                    ],
                )
            )

        else:
            raise ValueError(
                "Unknown validation type."
            )

        mode_pass = (
            answer_result[
                "mode"
            ]
            == test[
                "expected_mode"
            ]
        )

        answer_pass = (
            validate_answer_fragments(
                answer_result,
                test[
                    "answer_fragments"
                ],
            )
        )

        grounded_pass = bool(
            answer_result[
                "grounded"
            ]
        )

        passed = (
            evidence_pass
            and mode_pass
            and answer_pass
            and grounded_pass
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
                "mode_pass": (
                    mode_pass
                ),
                "evidence_pass": (
                    evidence_pass
                ),
                "answer_pass": (
                    answer_pass
                ),
                "grounded_pass": (
                    grounded_pass
                ),
                "passed": passed,
                "answer_result": (
                    answer_result
                ),
                "retrieval_results": (
                    retrieval_results
                ),
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
        "EXPERIMENT 42: GENERIC GROUNDED ANSWER GENERATION",
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
        answer_result = test[
            "answer_result"
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
                    f"Mode: "
                    f"{answer_result['mode']}"
                ),
                (
                    f"Grounded: "
                    f"{answer_result['grounded']}"
                ),
                (
                    f"Mode validation: "
                    f"{'PASS' if test['mode_pass'] else 'FAIL'}"
                ),
                (
                    f"Evidence validation: "
                    f"{'PASS' if test['evidence_pass'] else 'FAIL'}"
                ),
                (
                    f"Answer validation: "
                    f"{'PASS' if test['answer_pass'] else 'FAIL'}"
                ),
                "",
                "Answer:",
                answer_result[
                    "answer"
                ],
                "",
                "Evidence:",
            ]
        )

        for evidence in answer_result[
            "evidence"
        ]:
            lines.append(
                (
                    f"  - "
                    f"{evidence['item_id']} "
                    f"score={evidence['score']} "
                    f"coverage="
                    f"{evidence['coverage']}"
                )
            )

        lines.append("")

    lines.extend(
        [
            "=" * 78,
            "GROUNDED ANSWER VALIDATION SUMMARY",
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
                "STATUS: GENERIC GROUNDED "
                "ANSWER CHECKPOINT PASSED"
            )
        )

    else:
        lines.append(
            (
                "STATUS: GENERIC GROUNDED "
                "ANSWER GENERATION NEEDS "
                "REFINEMENT"
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
        "EXPERIMENT 42: "
        "GENERIC GROUNDED ANSWER GENERATION"
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
        "experiment": 42,
        "schema_version": (
            "1.0-grounded-answer-experimental"
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

        print(
            (
                f"       mode: "
                f"{test['answer_result']['mode']}"
            )
        )

        print(
            (
                "       evidence: "
                + ", ".join(
                    evidence_ids(
                        test[
                            "answer_result"
                        ]
                    )
                )
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
            "EXPERIMENT 42 CHECKPOINT PASSED"
        )
    else:
        print(
            "EXPERIMENT 42 NEEDS REFINEMENT"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()