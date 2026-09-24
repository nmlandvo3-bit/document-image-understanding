"""
Experiment 46: Production Query Service Validation

Validates the reusable DocumentQAService created after
Experiments 40-45.

This test makes real Groq API calls.
"""

import json
import sys
import time
from pathlib import Path

from services.document_qa_service import (
    DocumentQAService,
)


if hasattr(
    sys.stdout,
    "reconfigure",
):
    try:
        sys.stdout.reconfigure(
            encoding="utf-8"
        )
    except Exception:
        pass


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "unified_documents"
    / "production_service"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "production_service_tests.json"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "production_service_report.txt"
)


REQUEST_DELAY_SECONDS = 10


def normalized(
    value,
):
    return str(
        value
    ).lower()


def run_tests(
    service,
):
    tests = [
        {
            "name": (
                "table question"
            ),
            "question": (
                "When does the "
                "dormitory open?"
            ),
            "expected_evidence": {
                "pic1:record:row_18",
            },
            "answer_contains": [
                "mid-september",
                "2026",
            ],
        },
        {
            "name": (
                "form field question"
            ),
            "question": (
                "What is the date "
                "of birth?"
            ),
            "expected_evidence": {
                "pic13:field:2",
            },
            "answer_contains": [
                "03/14/1992",
            ],
        },
        {
            "name": (
                "checkbox question"
            ),
            "question": (
                "Which options "
                "are checked?"
            ),
            "expected_evidence": {
                "pic13:control:1",
                "pic13:control:4",
                "pic13:control:7",
            },
            "answer_contains": [
                "limited information",
                "medicare claims",
                "other",
            ],
        },
    ]

    outputs = []

    for index, test in enumerate(
        tests,
        start=1,
    ):
        print(
            (
                f"[{index}/{len(tests)}] "
                f"{test['name']}"
            )
        )

        try:
            result = (
                service.answer(
                    test[
                        "question"
                    ]
                )
            )

            actual_evidence = {
                item[
                    "evidence_item_id"
                ]
                for item in result[
                    "evidence"
                ]
            }

            evidence_pass = (
                actual_evidence
                == test[
                    "expected_evidence"
                ]
            )

            answer_text = normalized(
                result[
                    "answer"
                ]
            )

            answer_pass = all(
                fragment
                in answer_text
                for fragment
                in test[
                    "answer_contains"
                ]
            )

            grounded_pass = (
                result[
                    "grounded"
                ]
                is True
            )

            provenance_pass = (
                result[
                    "evidence_validation"
                ][
                    "valid"
                ]
                is True
            )

            passed = all(
                (
                    evidence_pass,
                    answer_pass,
                    grounded_pass,
                    provenance_pass,
                )
            )

            error = None

        except Exception as exc:
            result = None
            evidence_pass = False
            answer_pass = False
            grounded_pass = False
            provenance_pass = False
            passed = False

            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        outputs.append(
            {
                "name": (
                    test["name"]
                ),
                "question": (
                    test[
                        "question"
                    ]
                ),
                "expected_evidence": (
                    sorted(
                        test[
                            "expected_evidence"
                        ]
                    )
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
                "provenance_pass": (
                    provenance_pass
                ),
                "passed": passed,
                "error": error,
                "result": result,
            }
        )

        print(
            (
                "       "
                + (
                    "PASS"
                    if passed
                    else "FAIL"
                )
            )
        )

        if error:
            print(
                f"       {error}"
            )

        if (
            index
            < len(tests)
        ):
            print(
                (
                    f"       waiting "
                    f"{REQUEST_DELAY_SECONDS}s..."
                )
            )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    return outputs


def make_report(
    service,
    tests,
):
    passed = sum(
        1
        for test in tests
        if test[
            "passed"
        ]
    )

    lines = [
        "=" * 78,
        (
            "EXPERIMENT 46: "
            "PRODUCTION QUERY SERVICE"
        ),
        "=" * 78,
        (
            f"Provider: groq"
        ),
        (
            f"Model: "
            f"{service.model}"
        ),
        (
            f"Indexed corpus items: "
            f"{len(service.corpus)}"
        ),
        (
            f"Tests passed: "
            f"{passed}/{len(tests)}"
        ),
        "",
    ]

    for index, test in enumerate(
        tests,
        start=1,
    ):
        lines.extend(
            [
                "=" * 78,
                (
                    f"TEST {index}: "
                    f"{test['name']}"
                ),
                "-" * 78,
                (
                    f"Question: "
                    f"{test['question']}"
                ),
                (
                    f"Status: "
                    f"{'PASS' if test['passed'] else 'FAIL'}"
                ),
                (
                    "Evidence validation: "
                    f"{test['evidence_pass']}"
                ),
                (
                    "Answer validation: "
                    f"{test['answer_pass']}"
                ),
                (
                    "Grounded validation: "
                    f"{test['grounded_pass']}"
                ),
                (
                    "Provenance validation: "
                    f"{test['provenance_pass']}"
                ),
            ]
        )

        if test[
            "error"
        ]:
            lines.extend(
                [
                    "",
                    "Error:",
                    test[
                        "error"
                    ],
                ]
            )

        elif test[
            "result"
        ]:
            result = test[
                "result"
            ]

            lines.extend(
                [
                    "",
                    "Answer:",
                    result[
                        "answer"
                    ],
                    "",
                    "Evidence:",
                ]
            )

            for item in result[
                "evidence"
            ]:
                lines.append(
                    (
                        "  - "
                        f"{item['evidence_item_id']}"
                    )
                )

            lines.append(
                (
                    "Usage: "
                    f"{result['usage']}"
                )
            )

        lines.append("")

    lines.extend(
        [
            "=" * 78,
            "PRODUCTION SERVICE SUMMARY",
            "=" * 78,
            (
                f"Tests passed: "
                f"{passed}/{len(tests)}"
            ),
        ]
    )

    if passed == len(
        tests
    ):
        lines.append(
            (
                "STATUS: PRODUCTION "
                "QUERY SERVICE "
                "CHECKPOINT PASSED"
            )
        )
    else:
        lines.append(
            (
                "STATUS: PRODUCTION "
                "QUERY SERVICE "
                "NEEDS REFINEMENT"
            )
        )

    lines.append(
        "=" * 78
    )

    return "\n".join(
        lines
    ) + "\n"


def main():
    print(
        "=" * 72
    )

    print(
        "EXPERIMENT 46: "
        "PRODUCTION QUERY SERVICE"
    )

    print(
        "=" * 72
    )

    service = (
        DocumentQAService()
    )

    print(
        "Indexed documents:"
    )

    for document in (
        service
        .get_document_summary()
    ):
        print(
            (
                f"  {document['document_id']} "
                f"type={document['document_type']}"
            )
        )

    print(
        f"Corpus items: "
        f"{len(service.corpus)}"
    )

    print()

    tests = run_tests(
        service
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    passed = sum(
        1
        for test in tests
        if test[
            "passed"
        ]
    )

    output = {
        "experiment": 46,
        "schema_version": (
            "1.0-production-query-service"
        ),
        "provider": "groq",
        "model": (
            service.model
        ),
        "documents": (
            service
            .get_document_summary()
        ),
        "corpus_items": (
            len(service.corpus)
        ),
        "tests_passed": (
            passed
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
        service,
        tests,
    )

    OUTPUT_REPORT.write_text(
        report,
        encoding="utf-8",
    )

    print()

    print(
        f"Tests passed: "
        f"{passed}/{len(tests)}"
    )

    print(
        f"JSON: {OUTPUT_JSON}"
    )

    print(
        f"Report: {OUTPUT_REPORT}"
    )

    print(
        "=" * 72
    )

    if passed == len(
        tests
    ):
        print(
            "EXPERIMENT 46 "
            "CHECKPOINT PASSED"
        )
    else:
        print(
            "EXPERIMENT 46 "
            "NEEDS REFINEMENT"
        )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()