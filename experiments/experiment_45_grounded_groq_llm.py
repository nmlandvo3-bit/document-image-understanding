"""
Experiment 45: Grounded Groq LLM Answer Generation

Refined version:
- Uses compact evidence packets.
- Adds delay between Groq calls for free-tier TPM limits.
- Uses semantic checkbox validation.
- Does not rerun OCR or parsing.
"""

import json
import os
import time
from pathlib import Path

from groq import Groq

from experiments.experiment_41_generic_cross_document_retrieval import (
    PIC1_PATH,
    PIC13_PATH,
    build_search_corpus,
    load_json,
    normalize_text,
)

from experiments.experiment_44_grounding_evidence_compression import (
    build_compressed_packet,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "unified_documents"
    / "groq_llm"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "groq_grounded_answer_tests.json"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "groq_grounded_answer_report.txt"
)


MODEL_NAME = "openai/gpt-oss-20b"

# Keeps repeated test calls friendlier to the free-tier rolling TPM limit.
REQUEST_DELAY_SECONDS = 10


SYSTEM_PROMPT = """
You answer questions about documents.

Rules:
- Use only the supplied evidence.
- Do not use outside knowledge.
- Do not invent missing facts.
- Do not silently correct OCR.
- If evidence is insufficient, say so.
- Use only supplied evidence_item_ids.
- Keep the answer concise.

Return JSON only with:
{
  "answer": "natural-language answer",
  "evidence_item_ids": ["ids actually used"],
  "uncertainty": null
}
""".strip()


def get_client():
    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set."
        )

    return Groq(
        api_key=api_key
    )


def compact_content(
    item,
):
    """
    Strip retrieval/debug metadata before sending evidence
    to the LLM.

    The deterministic system already selected the evidence.
    """

    item_type = item[
        "item_type"
    ]

    content = item[
        "content"
    ]

    if item_type == "record":
        return {
            "record_id": content.get(
                "record_id"
            ),
            "category": content.get(
                "category"
            ),
            "event": content.get(
                "event"
            ),
            "time": content.get(
                "time"
            ),
            "detail": content.get(
                "detail"
            ),
            "explanation": content.get(
                "explanation"
            ),
            "review_flags": content.get(
                "review_flags",
                [],
            ),
        }

    if item_type == "field":
        return {
            "label": content.get(
                "label"
            ),
            "value": content.get(
                "value"
            ),
        }

    if item_type == "control":
        return {
            "label": content.get(
                "label"
            ),
            "checked": bool(
                content.get(
                    "checked",
                    False,
                )
            ),
            "state": content.get(
                "state"
            ),
        }

    if item_type == "free_text":
        return {
            "source_control_label": (
                content.get(
                    "source_control_label"
                )
            ),
            "text": content.get(
                "text"
            ),
        }

    return content


def make_minimal_packet(
    packet,
):
    """
    Produce the smallest sufficient packet for the LLM.
    """

    evidence = []

    for item in packet[
        "evidence"
    ]:
        evidence.append(
            {
                "evidence_item_id": (
                    item[
                        "item_id"
                    ]
                ),
                "item_type": (
                    item[
                        "item_type"
                    ]
                ),
                "content": (
                    compact_content(
                        item
                    )
                ),
            }
        )

    return {
        "question": packet[
            "query"
        ],
        "answer_mode": packet[
            "answer_mode"
        ],
        "evidence": evidence,
    }


def build_user_message(
    packet,
):
    minimal = make_minimal_packet(
        packet
    )

    return json.dumps(
        minimal,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )


def parse_json_response(
    content,
):
    """
    Parse normal JSON output.

    Includes a defensive cleanup in case a model surrounds
    the response with markdown fences.
    """

    content = content.strip()

    if content.startswith(
        "```"
    ):
        lines = content.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip().startswith(
                "```"
            )
        ):
            lines = lines[:-1]

        content = "\n".join(
            lines
        ).strip()

    return json.loads(
        content
    )


def call_groq(
    client,
    packet,
):
    response = (
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        build_user_message(
                            packet
                        )
                    ),
                },
            ],
            temperature=0.0,
            reasoning_effort="low",
            include_reasoning=False,
            response_format={
                "type": "json_object",
            },
            max_completion_tokens=400,
            stream=False,
        )
    )

    message = (
        response
        .choices[0]
        .message
    )

    content = message.content

    if not content:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    parsed = parse_json_response(
        content
    )

    for required in (
        "answer",
        "evidence_item_ids",
        "uncertainty",
    ):
        if required not in parsed:
            raise RuntimeError(
                "Groq JSON response is missing "
                f"{required!r}."
            )

    usage = {}

    if getattr(
        response,
        "usage",
        None,
    ):
        usage = {
            "prompt_tokens": getattr(
                response.usage,
                "prompt_tokens",
                None,
            ),
            "completion_tokens": getattr(
                response.usage,
                "completion_tokens",
                None,
            ),
            "total_tokens": getattr(
                response.usage,
                "total_tokens",
                None,
            ),
        }

    return {
        "model": getattr(
            response,
            "model",
            MODEL_NAME,
        ),
        "answer": str(
            parsed[
                "answer"
            ]
        ),
        "evidence_item_ids": list(
            parsed[
                "evidence_item_ids"
            ]
        ),
        "uncertainty": parsed[
            "uncertainty"
        ],
        "usage": usage,
    }


def packet_evidence_ids(
    packet,
):
    return [
        item[
            "item_id"
        ]
        for item in packet[
            "evidence"
        ]
    ]


def validate_evidence_ids(
    packet,
    llm_result,
    expected_ids,
):
    supplied = set(
        packet_evidence_ids(
            packet
        )
    )

    used = set(
        llm_result[
            "evidence_item_ids"
        ]
    )

    expected = set(
        expected_ids
    )

    no_hallucinated_ids = (
        used.issubset(
            supplied
        )
    )

    expected_used = (
        used == expected
    )

    return (
        no_hallucinated_ids,
        expected_used,
    )


def validate_answer_fragments(
    llm_result,
    fragments,
):
    answer = normalize_text(
        llm_result[
            "answer"
        ]
    )

    return all(
        normalize_text(
            fragment
        )
        in answer
        for fragment in fragments
    )


def validate_checked_answer(
    llm_result,
    label_fragment,
):
    """
    Accept equivalent affirmative answers for a checkbox
    that is known from the supplied evidence to be checked.

    Valid examples:
    - "Yes"
    - "Yes, it is selected."
    - "Medicare claims is checked."
    - "Selected"
    """

    answer = normalize_text(
        llm_result[
            "answer"
        ]
    )

    label = normalize_text(
        label_fragment
    )

    # Direct affirmative response.
    affirmative_only = {
        "yes",
        "yes.",
        "selected",
        "checked",
        "true",
    }

    if answer in affirmative_only:
        return True

    affirmative_terms = (
        "yes",
        "checked",
        "selected",
        "is marked",
        "marked",
        "true",
    )

    negative_terms = (
        "not checked",
        "not selected",
        "unchecked",
        "unselected",
        "false",
        "no",
    )

    has_affirmative = any(
        term in answer
        for term in affirmative_terms
    )

    has_negative = any(
        term in answer
        for term in negative_terms
    )

    # If the answer mentions the label, require a positive state.
    if label in answer:
        return (
            has_affirmative
            and not has_negative
        )

    # A short affirmative answer is also valid because the
    # question already identifies the control.
    if (
        has_affirmative
        and not has_negative
        and len(
            answer.split()
        ) <= 8
    ):
        return True

    return False

def run_tests(
    client,
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
            "expected_evidence": [
                "pic1:record:row_18",
            ],
            "validation_type": (
                "fragments"
            ),
            "answer_fragments": [
                "Mid-September",
                "2026",
            ],
        },
        {
            "name": (
                "pic1 2000 payment groups"
            ),
            "query": (
                "Who needs to pay $2000?"
            ),
            "expected_evidence": [
                "pic1:record:row_03",
                "pic1:record:row_06",
                "pic1:record:row_09",
                "pic1:record:row_13",
                "pic1:record:row_16",
            ],
            "validation_type": (
                "fragments"
            ),
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
            "expected_evidence": [
                "pic13:field:2",
            ],
            "validation_type": (
                "fragments"
            ),
            "answer_fragments": [
                "03/14/1992",
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
            "expected_evidence": [
                "pic13:field:3",
            ],
            "validation_type": (
                "fragments"
            ),
            "answer_fragments": [
                "1EG4-TE5-MK72",
            ],
        },
        {
            "name": (
                "pic13 claims checkbox"
            ),
            "query": (
                "Is Medicare claims selected?"
            ),
            "expected_evidence": [
                "pic13:control:4",
            ],
            "validation_type": (
                "checked"
            ),
            "label_fragment": (
                "Medicare claims"
            ),
        },
        {
            "name": (
                "pic13 Other response"
            ),
            "query": (
                "What information was "
                "written under Other?"
            ),
            "expected_evidence": [
                "pic13:free_text:1",
            ],
            "validation_type": (
                "fragments"
            ),
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
            "expected_evidence": [
                "pic13:control:1",
                "pic13:control:4",
                "pic13:control:7",
            ],
            "validation_type": (
                "fragments"
            ),
            "answer_fragments": [
                "Limited Information",
                "Medicare claims",
                "Other",
            ],
        },
    ]

    outputs = []

    for index, test in enumerate(
        tests,
        start=1,
    ):
        print(
            f"[{index}/{len(tests)}] "
            f"{test['name']}"
        )

        packet = (
            build_compressed_packet(
                test["query"],
                corpus,
                top_k=15,
            )
        )

        try:
            llm_result = call_groq(
                client,
                packet,
            )

            (
                no_hallucinated_ids,
                evidence_pass,
            ) = validate_evidence_ids(
                packet,
                llm_result,
                test[
                    "expected_evidence"
                ],
            )

            if (
                test[
                    "validation_type"
                ]
                == "checked"
            ):
                answer_pass = (
                    validate_checked_answer(
                        llm_result,
                        test[
                            "label_fragment"
                        ],
                    )
                )

            else:
                answer_pass = (
                    validate_answer_fragments(
                        llm_result,
                        test[
                            "answer_fragments"
                        ],
                    )
                )

            passed = (
                no_hallucinated_ids
                and evidence_pass
                and answer_pass
            )

            error = None

        except Exception as exc:
            llm_result = None

            no_hallucinated_ids = False
            evidence_pass = False
            answer_pass = False
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
                "query": (
                    test["query"]
                ),
                "expected_evidence": (
                    test[
                        "expected_evidence"
                    ]
                ),
                "packet_evidence": (
                    packet_evidence_ids(
                        packet
                    )
                ),
                "no_hallucinated_ids": (
                    no_hallucinated_ids
                ),
                "evidence_pass": (
                    evidence_pass
                ),
                "answer_pass": (
                    answer_pass
                ),
                "passed": passed,
                "error": error,
                "llm_result": (
                    llm_result
                ),
            }
        )

        if passed:
            print(
                "       PASS"
            )
        else:
            print(
                "       FAIL"
            )

            if error:
                print(
                    f"       {error}"
                )

        # Avoid hammering the Groq free-tier TPM window.
        if index < len(tests):
            print(
                f"       waiting "
                f"{REQUEST_DELAY_SECONDS}s..."
            )

            time.sleep(
                REQUEST_DELAY_SECONDS
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
        (
            "EXPERIMENT 45: "
            "GROQ GROUNDED LLM ANSWERS"
        ),
        "=" * 78,
        (
            f"Model: {MODEL_NAME}"
        ),
        (
            f"Tests: {len(tests)}"
        ),
        (
            f"Passed: {passed_count}"
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
                    f"Query: "
                    f"{test['query']}"
                ),
                (
                    f"Status: "
                    f"{'PASS' if test['passed'] else 'FAIL'}"
                ),
                (
                    "Packet evidence: "
                    f"{test['packet_evidence']}"
                ),
                (
                    "Expected evidence: "
                    f"{test['expected_evidence']}"
                ),
                (
                    "No hallucinated evidence IDs: "
                    f"{test['no_hallucinated_ids']}"
                ),
                (
                    "Evidence validation: "
                    f"{test['evidence_pass']}"
                ),
                (
                    "Answer validation: "
                    f"{test['answer_pass']}"
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
            "llm_result"
        ]:
            result = test[
                "llm_result"
            ]

            lines.extend(
                [
                    "",
                    "LLM Answer:",
                    result[
                        "answer"
                    ],
                    "",
                    (
                        "Evidence IDs used: "
                        f"{result['evidence_item_ids']}"
                    ),
                    (
                        "Uncertainty: "
                        f"{result['uncertainty']}"
                    ),
                    (
                        "Usage: "
                        f"{result['usage']}"
                    ),
                ]
            )

        lines.append("")

    lines.extend(
        [
            "=" * 78,
            "GROQ LLM VALIDATION SUMMARY",
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
                "STATUS: GROQ GROUNDED "
                "LLM CHECKPOINT PASSED"
            )
        )

    else:
        lines.append(
            (
                "STATUS: GROQ GROUNDED "
                "LLM NEEDS REFINEMENT"
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
        "EXPERIMENT 45: "
        "GROQ GROUNDED LLM ANSWERS"
    )
    print("=" * 72)

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        "GROQ_API_KEY loaded:",
        bool(
            os.getenv(
                "GROQ_API_KEY"
            )
        ),
    )

    print(
        "Request delay:",
        REQUEST_DELAY_SECONDS,
        "seconds",
    )

    print()

    client = get_client()

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
        client,
        corpus,
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
        "experiment": 45,
        "schema_version": (
            "1.1-groq-grounded-llm"
        ),
        "provider": "groq",
        "model": MODEL_NAME,
        "request_delay_seconds": (
            REQUEST_DELAY_SECONDS
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
            "EXPERIMENT 45 CHECKPOINT PASSED"
        )

    else:
        print(
            "EXPERIMENT 45 NEEDS REFINEMENT"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()