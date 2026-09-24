"""
Experiment 31: Queryable Timetable Engine

Creates a deterministic search layer over the normalized
timetable records from Experiment 30.

Purpose:
- Test whether the structured document can answer factual
  queries without an LLM.
- Search category, event, detail, explanation, and time.
- Return source row and source cell references.
- Preserve uncertainty and review flags.

This experiment does not use an LLM and does not modify
previous results.
"""

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "normalized_records"
    / "normalized_timetable_records.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "query_engine"
)


def load_document() -> dict:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_for_search(text: str | None) -> str:
    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9+:/.-]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def record_search_text(record: dict) -> str:
    parts = []

    for field_name in (
        "category",
        "event",
        "detail",
        "explanation",
    ):
        field = record.get(field_name, {})

        parts.append(
            field.get("text") or ""
        )

    normalized_time = record.get(
        "normalized_time",
        {},
    )

    parts.append(
        normalized_time.get(
            "raw_text",
            "",
        )
    )

    for value in normalized_time.get(
        "values",
        [],
    ):
        if value.get("date"):
            parts.append(value["date"])

        if value.get("time"):
            parts.append(value["time"])

    if normalized_time.get(
        "text_expression"
    ):
        parts.append(
            normalized_time[
                "text_expression"
            ]
        )

    return normalize_for_search(
        " ".join(parts)
    )


def tokenize_query(query: str) -> list[str]:
    normalized = normalize_for_search(
        query
    )

    return [
        token
        for token in normalized.split()
        if len(token) >= 2
    ]


def score_record(
    query: str,
    record: dict,
) -> tuple[int, list[str]]:
    """
    Simple deterministic scoring.

    Stronger fields:
        category = 5
        event = 5
        detail = 3
        explanation = 2
        time = 3

    Every matched query token contributes to the score.
    """

    tokens = tokenize_query(query)

    if not tokens:
        return 0, []

    fields = {
        "category": normalize_for_search(
            record.get(
                "category",
                {},
            ).get("text")
        ),
        "event": normalize_for_search(
            record.get(
                "event",
                {},
            ).get("text")
        ),
        "detail": normalize_for_search(
            record.get(
                "detail",
                {},
            ).get("text")
        ),
        "explanation": normalize_for_search(
            record.get(
                "explanation",
                {},
            ).get("text")
        ),
        "time": normalize_for_search(
            record.get(
                "normalized_time",
                {},
            ).get(
                "raw_text",
                "",
            )
        ),
    }

    weights = {
        "category": 5,
        "event": 5,
        "detail": 3,
        "explanation": 2,
        "time": 3,
    }

    score = 0
    matched_fields = set()

    for token in tokens:
        for field_name, text in fields.items():
            if token in text:
                score += weights[field_name]
                matched_fields.add(
                    field_name
                )

    return (
        score,
        sorted(matched_fields),
    )


def format_time(record: dict) -> str:
    normalized = record[
        "normalized_time"
    ]

    if (
        normalized["type"]
        == "explicit_datetime_sequence"
    ):
        parts = []

        for value in normalized["values"]:
            parts.append(
                f"{value['date']} "
                f"{value['time']}"
            )

        return " -> ".join(parts)

    if normalized.get(
        "text_expression"
    ):
        return normalized[
            "text_expression"
        ]

    return normalized.get(
        "raw_text",
        "",
    )


def source_cell_id(
    record: dict,
    field_name: str,
):
    source = (
        record.get(
            field_name,
            {},
        ).get(
            "source_cell"
        )
    )

    if source is None:
        return None

    return source.get(
        "cell_id"
    )


def make_result(
    record: dict,
    score: int,
    matched_fields: list[str],
) -> dict:
    return {
        "record_id": record["record_id"],
        "source_row": record[
            "source_row"
        ],
        "score": score,
        "matched_fields": matched_fields,
        "category": record[
            "category"
        ]["text"],
        "event": record[
            "event"
        ]["text"],
        "time": format_time(record),
        "detail": record[
            "detail"
        ]["text"],
        "explanation": record[
            "explanation"
        ]["text"],
        "review_flags": record.get(
            "review_flags",
            [],
        ),
        "source_cells": {
            "category": source_cell_id(
                record,
                "category",
            ),
            "event": source_cell_id(
                record,
                "event",
            ),
            "time": source_cell_id(
                record,
                "time",
            ),
            "detail": source_cell_id(
                record,
                "detail",
            ),
            "explanation": source_cell_id(
                record,
                "explanation",
            ),
        },
    }


def search_records(
    records: list[dict],
    query: str,
    limit: int = 5,
) -> list[dict]:
    scored = []

    for record in records:
        score, matched_fields = (
            score_record(
                query,
                record,
            )
        )

        if score > 0:
            scored.append(
                (
                    score,
                    record[
                        "source_row"
                    ],
                    matched_fields,
                    record,
                )
            )

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    return [
        make_result(
            record=item[3],
            score=item[0],
            matched_fields=item[2],
        )
        for item in scored[:limit]
    ]


def print_result(
    query: str,
    results: list[dict],
) -> None:
    print()
    print("=" * 78)
    print(f"QUERY: {query}")
    print("=" * 78)

    if not results:
        print("No matching timetable records.")
        return

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(
            f"[{index}] "
            f"{result['record_id']} "
            f"(score={result['score']})"
        )

        print(
            "Category:   ",
            result["category"],
        )

        print(
            "Event:      ",
            result["event"],
        )

        print(
            "Time:       ",
            result["time"],
        )

        print(
            "Detail:     ",
            result["detail"],
        )

        explanation = (
            result["explanation"]
            or ""
        )

        if len(explanation) > 180:
            explanation = (
                explanation[:177]
                + "..."
            )

        print(
            "Explanation:",
            explanation,
        )

        print(
            "Matched:    ",
            ", ".join(
                result[
                    "matched_fields"
                ]
            ),
        )

        print(
            "Source row: ",
            result[
                "source_row"
            ],
        )

        print(
            "Source cells:",
            result[
                "source_cells"
            ],
        )

        print(
            "Review:     ",
            result[
                "review_flags"
            ],
        )


def run_test_queries(
    records: list[dict],
) -> list[dict]:
    queries = [
        "graduate students room selection",
        "female undergraduate room selection",
        "male undergraduate room selection",
        "undergraduate pay deposit",
        "waiting list 1",
        "waiting list II room announcement",
        "3+2 program room announcement",
        "3+2 program pay deposit",
        "dormitory opening",
        "2000 deposit",
    ]

    all_results = []

    for query in queries:
        results = search_records(
            records,
            query,
            limit=5,
        )

        print_result(
            query,
            results,
        )

        all_results.append(
            {
                "query": query,
                "results": results,
            }
        )

    return all_results


def interactive_mode(
    records: list[dict],
) -> None:
    print()
    print("=" * 78)
    print("INTERACTIVE TIMETABLE SEARCH")
    print("=" * 78)
    print(
        "Enter a question or search phrase."
    )
    print(
        "Type 'exit' to stop."
    )

    while True:
        print()

        query = input(
            "Search> "
        ).strip()

        if query.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        if not query:
            continue

        results = search_records(
            records,
            query,
            limit=5,
        )

        print_result(
            query,
            results,
        )


def main() -> None:
    document = load_document()

    records = document.get(
        "records",
        [],
    )

    if not records:
        raise ValueError(
            "No normalized timetable "
            "records found."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "EXPERIMENT 31: QUERYABLE TIMETABLE ENGINE"
    )
    print("=" * 78)
    print(
        f"Loaded records: {len(records)}"
    )

    test_results = run_test_queries(
        records
    )

    output_path = (
        OUTPUT_DIR
        / "query_test_results.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "experiment": 31,
                "method": (
                    "deterministic_weighted_keyword_search"
                ),
                "source": str(
                    INPUT_PATH
                ),
                "test_queries": (
                    test_results
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 78)
    print(
        f"Saved test results: "
        f"{output_path}"
    )
    print("=" * 78)

    interactive_mode(
        records
    )


if __name__ == "__main__":
    main()