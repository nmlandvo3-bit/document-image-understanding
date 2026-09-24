"""
Experiment 41: Generic Cross-Document Retrieval

Purpose
-------
Build one deterministic retrieval engine over the unified document schema
created by Experiment 40.

The retrieval engine searches semantic objects from BOTH document types:

pic1:
    records

pic13:
    fields
    controls
    free_text

The search algorithm does not need to know whether a document is a table
or form before retrieval.

No OCR is rerun.
No parsing is rerun.
No LLM is used.

Outputs
-------
results/unified_documents/retrieval/
    retrieval_test_results.json
    retrieval_test_report.txt
"""

import json
import re
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UNIFIED_DIR = (
    PROJECT_ROOT
    / "results"
    / "unified_documents"
)

PIC1_PATH = (
    UNIFIED_DIR
    / "pic1_unified_document.json"
)

PIC13_PATH = (
    UNIFIED_DIR
    / "pic13_unified_document.json"
)

OUTPUT_DIR = (
    UNIFIED_DIR
    / "retrieval"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "retrieval_test_results.json"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "retrieval_test_report.txt"
)


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "do",
    "for",
    "from",
    "give",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "please",
    "show",
    "tell",
    "the",
    "their",
    "there",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


TOKEN_ALIASES = {
    "selected": {
        "selected",
        "checked",
    },
    "select": {
        "select",
        "selected",
        "checked",
    },
    "checked": {
        "checked",
        "selected",
    },
    "unselected": {
        "unselected",
        "unchecked",
    },
    "unchecked": {
        "unchecked",
        "unselected",
    },
    "dob": {
        "dob",
        "date",
        "birth",
    },
    "id": {
        "id",
        "identification",
    },
}


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def flatten_text(value):
    """
    Convert arbitrary JSON-compatible content into searchable text.

    Handles:
    - strings
    - numbers
    - dictionaries
    - lists
    - booleans
    - None
    """

    if value is None:
        return ""

    if isinstance(
        value,
        bool,
    ):
        return (
            "true"
            if value
            else "false"
        )

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        return str(value)

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        list,
    ):
        return " ".join(
            flatten_text(item)
            for item in value
        )

    if isinstance(
        value,
        dict,
    ):
        return " ".join(
            flatten_text(item)
            for item in value.values()
        )

    return str(value)


def normalize_text(text):
    """
    Conservative searchable normalization.

    Raw source text is never modified.
    """

    text = flatten_text(text)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.lower()

    # Convert $2,000 -> $2000
    text = re.sub(
        r"(?<=\d),(?=\d)",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_token(token):
    """
    Small morphology normalization for retrieval only.
    """

    token = token.lower()

    if len(token) > 5:
        if token.endswith(
            "ing"
        ):
            candidate = token[:-3]

            if len(candidate) >= 4:
                token = candidate

    if len(token) > 4:
        if token.endswith(
            "ies"
        ):
            token = (
                token[:-3]
                + "y"
            )

        elif token.endswith(
            "s"
        ) and not token.endswith(
            "ss"
        ):
            token = token[:-1]

    return token


def tokenize(text):
    text = normalize_text(
        text
    )

    # Keeps expressions such as 3+2 together.
    tokens = re.findall(
        r"[a-z0-9]+(?:\+[a-z0-9]+)?",
        text,
    )

    result = []

    for token in tokens:
        normalized = normalize_token(
            token
        )

        if normalized in STOP_WORDS:
            continue

        if len(normalized) <= 1:
            continue

        result.append(
            normalized
        )

    return result


def expand_query_tokens(tokens):
    expanded = set(
        tokens
    )

    for token in list(tokens):
        aliases = TOKEN_ALIASES.get(
            token,
            set(),
        )

        for alias in aliases:
            expanded.add(
                normalize_token(
                    alias
                )
            )

    return expanded


def make_facet(
    name,
    value,
    weight,
):
    raw_text = flatten_text(
        value
    )

    return {
        "name": name,
        "raw_text": raw_text,
        "normalized_text": (
            normalize_text(
                raw_text
            )
        ),
        "tokens": tokenize(
            raw_text
        ),
        "weight": weight,
    }


def build_record_entry(
    document,
    record,
):
    """
    Convert a unified record into a generic searchable item.
    """

    facets = [
        make_facet(
            "category",
            record.get(
                "category"
            ),
            7,
        ),
        make_facet(
            "event",
            record.get(
                "event"
            ),
            8,
        ),
        make_facet(
            "time",
            record.get(
                "time"
            ),
            4,
        ),
        make_facet(
            "detail",
            record.get(
                "detail"
            ),
            6,
        ),
        make_facet(
            "explanation",
            record.get(
                "explanation"
            ),
            3,
        ),
    ]

    record_id = record.get(
        "record_id"
    )

    return {
        "document_id": document[
            "document_id"
        ],
        "document_type": document[
            "document_type"
        ],
        "item_type": "record",
        "item_id": (
            f"{document['document_id']}:"
            f"record:{record_id}"
        ),
        "local_id": record_id,
        "facets": facets,
        "payload": record,
        "provenance": {
            "source_row": (
                record.get(
                    "source_row"
                )
            ),
            "source_cells": (
                record.get(
                    "source_cells",
                    {},
                )
            ),
        },
    }


def build_field_entry(
    document,
    field,
):
    facets = [
        make_facet(
            "label",
            field.get(
                "label"
            ),
            9,
        ),
        make_facet(
            "value",
            field.get(
                "value"
            ),
            6,
        ),
    ]

    field_id = field.get(
        "field_id"
    )

    return {
        "document_id": document[
            "document_id"
        ],
        "document_type": document[
            "document_type"
        ],
        "item_type": "field",
        "item_id": (
            f"{document['document_id']}:"
            f"field:{field_id}"
        ),
        "local_id": field_id,
        "facets": facets,
        "payload": field,
        "provenance": {
            "label_region_id": (
                field.get(
                    "label_region_id"
                )
            ),
            "value_region_id": (
                field.get(
                    "value_region_id"
                )
            ),
        },
    }


def build_control_entry(
    document,
    control,
):
    checked = bool(
        control.get(
            "checked",
            False,
        )
    )

    if checked:
        state_search_text = (
            "checked selected yes"
        )
    else:
        state_search_text = (
            "unchecked unselected no"
        )

    facets = [
        make_facet(
            "label",
            control.get(
                "label"
            ),
            9,
        ),
        make_facet(
            "state",
            state_search_text,
            10,
        ),
    ]

    control_id = control.get(
        "control_id"
    )

    return {
        "document_id": document[
            "document_id"
        ],
        "document_type": document[
            "document_type"
        ],
        "item_type": "control",
        "item_id": (
            f"{document['document_id']}:"
            f"control:{control_id}"
        ),
        "local_id": control_id,
        "facets": facets,
        "payload": control,
        "provenance": {
            "label_region_id": (
                control.get(
                    "label_region_id"
                )
            ),
            "bbox": (
                control.get(
                    "bbox"
                )
            ),
        },
    }


def build_free_text_entry(
    document,
    item,
    index,
):
    facets = [
        make_facet(
            "source_control_label",
            item.get(
                "source_control_label"
            ),
            9,
        ),
        make_facet(
            "text",
            item.get(
                "text"
            ),
            7,
        ),
    ]

    return {
        "document_id": document[
            "document_id"
        ],
        "document_type": document[
            "document_type"
        ],
        "item_type": "free_text",
        "item_id": (
            f"{document['document_id']}:"
            f"free_text:{index}"
        ),
        "local_id": index,
        "facets": facets,
        "payload": item,
        "provenance": {
            "source_control_id": (
                item.get(
                    "source_control_id"
                )
            ),
            "text_region_id": (
                item.get(
                    "text_region_id"
                )
            ),
        },
    }


def build_search_corpus(
    documents,
):
    """
    Build one corpus from all unified documents.

    Notice that retrieval itself does not have separate
    table/form search algorithms.
    """

    corpus = []

    for document in documents:
        for record in document.get(
            "records",
            [],
        ):
            corpus.append(
                build_record_entry(
                    document,
                    record,
                )
            )

        for field in document.get(
            "fields",
            [],
        ):
            corpus.append(
                build_field_entry(
                    document,
                    field,
                )
            )

        for control in document.get(
            "controls",
            [],
        ):
            corpus.append(
                build_control_entry(
                    document,
                    control,
                )
            )

        for index, item in enumerate(
            document.get(
                "free_text",
                [],
            ),
            start=1,
        ):
            corpus.append(
                build_free_text_entry(
                    document,
                    item,
                    index,
                )
            )

    return corpus


def score_facet(
    facet,
    query_tokens,
    query_normalized,
):
    facet_tokens = set(
        facet["tokens"]
    )

    weight = facet[
        "weight"
    ]

    matched_tokens = (
        query_tokens
        & facet_tokens
    )

    score = 0

    # Token evidence.
    score += (
        len(
            matched_tokens
        )
        * weight
        * 5
    )

    # Full query phrase inside one facet.
    if (
        query_normalized
        and len(
            query_normalized
        ) >= 4
        and query_normalized
        in facet[
            "normalized_text"
        ]
    ):
        score += (
            weight
            * 8
        )

    return (
        score,
        sorted(
            matched_tokens
        ),
    )

def infer_query_intent(query):
    """
    Infer lightweight retrieval intent from query wording.

    This does not identify document type.
    It only helps distinguish semantic object types such as
    free-text responses versus checkbox controls.
    """

    normalized = normalize_text(
        query
    )

    tokens = set(
        tokenize(
            query
        )
    )

    free_text_hints = {
        "written",
        "write",
        "entered",
        "enter",
        "provided",
        "provide",
        "under",
        "response",
        "text",
    }

    control_hints = {
        "checked",
        "selected",
        "unchecked",
        "unselected",
        "option",
        "options",
    }

    field_hints = {
        "name",
        "birth",
        "date",
        "identification",
        "number",
        "address",
        "city",
        "state",
        "zip",
    }

    intents = set()

    if (
        tokens
        & free_text_hints
    ):
        intents.add(
            "free_text"
        )

    if (
        tokens
        & control_hints
    ):
        intents.add(
            "control"
        )

    if (
        tokens
        & field_hints
    ):
        intents.add(
            "field"
        )

    # Phrase-level cue.
    if "written under" in normalized:
        intents.add(
            "free_text"
        )

    return intents

def search(
    corpus,
    query,
    top_k=10,
):
    query_normalized = (
        normalize_text(
            query
        )
    )

    basic_tokens = tokenize(
        query
    )

    query_tokens = (
        expand_query_tokens(
            basic_tokens
        )
    )

    query_intents = (
        infer_query_intent(
            query
        )
    )

    results = []

    for entry in corpus:
        total_score = 0

        facet_matches = []

        all_entry_tokens = set()

        for facet in entry[
            "facets"
        ]:
            all_entry_tokens.update(
                facet["tokens"]
            )

            (
                facet_score,
                matched_tokens,
            ) = score_facet(
                facet,
                query_tokens,
                query_normalized,
            )

            if facet_score > 0:
                facet_matches.append(
                    {
                        "facet": (
                            facet["name"]
                        ),
                        "score": (
                            facet_score
                        ),
                        "matched_tokens": (
                            matched_tokens
                        ),
                    }
                )

                total_score += (
                    facet_score
                )

        matched_global = (
            query_tokens
            & all_entry_tokens
        )

        if query_tokens:
            coverage = (
                len(
                    matched_global
                )
                / len(
                    query_tokens
                )
            )
        else:
            coverage = 0.0

        total_score += int(
            coverage * 20
        )

        basic_token_set = set(
            basic_tokens
        )

        if (
            basic_token_set
            and basic_token_set.issubset(
                all_entry_tokens
            )
        ):
            total_score += 25

        intent_bonus = 0

        if (
            "free_text"
            in query_intents
            and entry[
                "item_type"
            ] == "free_text"
        ):
            intent_bonus += 30

        if (
            "control"
            in query_intents
            and entry[
                "item_type"
            ] == "control"
        ):
            intent_bonus += 20

        if (
            "field"
            in query_intents
            and entry[
                "item_type"
            ] == "field"
        ):
            intent_bonus += 12

        total_score += (
            intent_bonus
        )

        if total_score <= 0:
            continue

        results.append(
            {
                "document_id": (
                    entry[
                        "document_id"
                    ]
                ),
                "document_type": (
                    entry[
                        "document_type"
                    ]
                ),
                "item_type": (
                    entry[
                        "item_type"
                    ]
                ),
                "item_id": (
                    entry[
                        "item_id"
                    ]
                ),
                "local_id": (
                    entry[
                        "local_id"
                    ]
                ),
                "score": (
                    total_score
                ),
                "coverage": round(
                    coverage,
                    4,
                ),
                "matched_tokens": (
                    sorted(
                        matched_global
                    )
                ),
                "query_intents": (
                    sorted(
                        query_intents
                    )
                ),
                "intent_bonus": (
                    intent_bonus
                ),
                "facet_matches": (
                    facet_matches
                ),
                "payload": (
                    entry[
                        "payload"
                    ]
                ),
                "provenance": (
                    entry[
                        "provenance"
                    ]
                ),
            }
        )

    results.sort(
        key=lambda item: (
            -item["score"],
            -item["coverage"],
            item["document_id"],
            item["item_type"],
            str(
                item["local_id"]
            ),
        )
    )

    return results[
        :top_k
    ]

def result_ids(results):
    return [
        item["item_id"]
        for item in results
    ]


def validate_first(
    results,
    expected_item_id,
):
    if not results:
        return False

    return (
        results[0][
            "item_id"
        ]
        == expected_item_id
    )


def validate_contains(
    results,
    expected_ids,
    limit=None,
):
    selected = results

    if limit is not None:
        selected = results[
            :limit
        ]

    actual = {
        item["item_id"]
        for item in selected
    }

    return set(
        expected_ids
    ).issubset(
        actual
    )


def validate_top_set(
    results,
    expected_ids,
):
    count = len(
        expected_ids
    )

    actual = {
        item["item_id"]
        for item in results[
            :count
        ]
    }

    return (
        actual
        == set(
            expected_ids
        )
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
            "top_k": 5,
            "validation_type": (
                "first"
            ),
            "expected": (
                "pic1:record:row_18"
            ),
        },
        {
            "name": (
                "pic1 3+2 deposit"
            ),
            "query": (
                "3+2 Program pay deposit"
            ),
            "top_k": 5,
            "validation_type": (
                "first"
            ),
            "expected": (
                "pic1:record:row_16"
            ),
        },
        {
            "name": (
                "pic1 all 2000 payment records"
            ),
            "query": (
                "Who needs to pay $2000?"
            ),
            "top_k": 10,
            "validation_type": (
                "contains"
            ),
            "expected": [
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
            "top_k": 5,
            "validation_type": (
                "first"
            ),
            "expected": (
                "pic13:field:2"
            ),
        },
        {
            "name": (
                "pic13 Medicare identification"
            ),
            "query": (
                "What is the Medicare "
                "identification number?"
            ),
            "top_k": 5,
            "validation_type": (
                "first"
            ),
            "expected": (
                "pic13:field:3"
            ),
        },
        {
            "name": (
                "pic13 Medicare claims selection"
            ),
            "query": (
                "Is Medicare claims selected?"
            ),
            "top_k": 5,
            "validation_type": (
                "first"
            ),
            "expected": (
                "pic13:control:4"
            ),
        },
        {
            "name": (
                "pic13 Other free text"
            ),
            "query": (
                "What information was written "
                "under Other?"
            ),
            "top_k": 5,
            "validation_type": (
                "first"
            ),
            "expected": (
                "pic13:free_text:1"
            ),
        },
        {
            "name": (
                "pic13 checked controls"
            ),
            "query": (
                "Which options are checked?"
            ),
            "top_k": 7,
            "validation_type": (
                "top_set"
            ),
            "expected": [
                "pic13:control:1",
                "pic13:control:4",
                "pic13:control:7",
            ],
        },
    ]

    outputs = []

    for test in tests:
        results = search(
            corpus,
            test["query"],
            top_k=test[
                "top_k"
            ],
        )

        validation_type = test[
            "validation_type"
        ]

        if validation_type == "first":
            passed = validate_first(
                results,
                test[
                    "expected"
                ],
            )

        elif validation_type == "contains":
            passed = validate_contains(
                results,
                test[
                    "expected"
                ],
            )

        elif validation_type == "top_set":
            passed = validate_top_set(
                results,
                test[
                    "expected"
                ],
            )

        else:
            raise ValueError(
                "Unknown validation type: "
                f"{validation_type}"
            )

        outputs.append(
            {
                "name": (
                    test["name"]
                ),
                "query": (
                    test["query"]
                ),
                "validation_type": (
                    validation_type
                ),
                "expected": (
                    test["expected"]
                ),
                "passed": (
                    passed
                ),
                "results": (
                    results
                ),
            }
        )

    return outputs


def make_report(
    corpus,
    tests,
):
    passed_count = sum(
        1
        for test in tests
        if test["passed"]
    )

    document_counts = {}

    item_type_counts = {}

    for item in corpus:
        document_id = item[
            "document_id"
        ]

        item_type = item[
            "item_type"
        ]

        document_counts[
            document_id
        ] = (
            document_counts.get(
                document_id,
                0,
            )
            + 1
        )

        item_type_counts[
            item_type
        ] = (
            item_type_counts.get(
                item_type,
                0,
            )
            + 1
        )

    lines = [
        "=" * 78,
        "EXPERIMENT 41: GENERIC CROSS-DOCUMENT RETRIEVAL",
        "=" * 78,
        (
            f"Search corpus items: "
            f"{len(corpus)}"
        ),
        (
            f"Documents indexed: "
            f"{len(document_counts)}"
        ),
        "",
        "CORPUS BY DOCUMENT",
        "-" * 78,
    ]

    for key in sorted(
        document_counts
    ):
        lines.append(
            (
                f"{key}: "
                f"{document_counts[key]}"
            )
        )

    lines.extend(
        [
            "",
            "CORPUS BY ITEM TYPE",
            "-" * 78,
        ]
    )

    for key in sorted(
        item_type_counts
    ):
        lines.append(
            (
                f"{key}: "
                f"{item_type_counts[key]}"
            )
        )

    lines.extend(
        [
            "",
            "RETRIEVAL TESTS",
            "=" * 78,
        ]
    )

    for index, test in enumerate(
        tests,
        start=1,
    ):
        lines.extend(
            [
                "",
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
                    f"Expected: "
                    f"{test['expected']}"
                ),
                "",
                "Top results:",
            ]
        )

        if not test[
            "results"
        ]:
            lines.append(
                "  No results"
            )

        for rank, result in enumerate(
            test["results"],
            start=1,
        ):
            lines.append(
                (
                    f"  {rank}. "
                    f"{result['item_id']} "
                    f"score={result['score']} "
                    f"coverage="
                    f"{result['coverage']:.3f} "
                    f"matched="
                    f"{result['matched_tokens']}"
                )
            )

    lines.extend(
        [
            "",
            "=" * 78,
            "RETRIEVAL VALIDATION SUMMARY",
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
                "STATUS: GENERIC "
                "CROSS-DOCUMENT RETRIEVAL "
                "CHECKPOINT PASSED"
            )
        )

    else:
        lines.append(
            (
                "STATUS: GENERIC "
                "CROSS-DOCUMENT RETRIEVAL "
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
    print("=" * 72)
    print(
        "EXPERIMENT 41: "
        "GENERIC CROSS-DOCUMENT RETRIEVAL"
    )
    print("=" * 72)

    pic1 = load_json(
        PIC1_PATH
    )

    pic13 = load_json(
        PIC13_PATH
    )

    documents = [
        pic1,
        pic13,
    ]

    corpus = build_search_corpus(
        documents
    )

    tests = run_tests(
        corpus
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {
        "experiment": 41,
        "schema_version": (
            "1.0-retrieval-experimental"
        ),
        "documents": [
            {
                "document_id": (
                    document[
                        "document_id"
                    ]
                ),
                "document_type": (
                    document[
                        "document_type"
                    ]
                ),
            }
            for document
            in documents
        ],
        "corpus_statistics": {
            "total_items": (
                len(corpus)
            ),
            "records": sum(
                1
                for item in corpus
                if (
                    item[
                        "item_type"
                    ]
                    == "record"
                )
            ),
            "fields": sum(
                1
                for item in corpus
                if (
                    item[
                        "item_type"
                    ]
                    == "field"
                )
            ),
            "controls": sum(
                1
                for item in corpus
                if (
                    item[
                        "item_type"
                    ]
                    == "control"
                )
            ),
            "free_text": sum(
                1
                for item in corpus
                if (
                    item[
                        "item_type"
                    ]
                    == "free_text"
                )
            ),
        },
        "tests": tests,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    report = make_report(
        corpus,
        tests,
    )

    OUTPUT_REPORT.write_text(
        report,
        encoding="utf-8",
    )

    passed_count = sum(
        1
        for test in tests
        if test["passed"]
    )

    print(
        f"Corpus items: "
        f"{len(corpus)}"
    )

    print(
        "Records:",
        output_data[
            "corpus_statistics"
        ][
            "records"
        ],
    )

    print(
        "Fields:",
        output_data[
            "corpus_statistics"
        ][
            "fields"
        ],
    )

    print(
        "Controls:",
        output_data[
            "corpus_statistics"
        ][
            "controls"
        ],
    )

    print(
        "Free text:",
        output_data[
            "corpus_statistics"
        ][
            "free_text"
        ],
    )

    print()

    for test in tests:
        print(
            (
                f"[{'PASS' if test['passed'] else 'FAIL'}] "
                f"{test['name']}"
            )
        )

        if test["results"]:
            first = test[
                "results"
            ][0]

            print(
                (
                    "       top: "
                    f"{first['item_id']} "
                    f"score={first['score']}"
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
            "EXPERIMENT 41 CHECKPOINT PASSED"
        )

    else:
        print(
            "EXPERIMENT 41 NEEDS REFINEMENT"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()