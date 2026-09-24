"""
Experiment 34: Multi-Record Grounded Answer Generation

Extends Experiment 33 so that broad questions can answer
from multiple retrieved records.

Examples:
- "When do graduate students select rooms?"
    -> one best record

- "Who needs to pay $2000?"
    -> all equally strong matching deposit records

- "What happens for the 3+2 Program?"
    -> all records belonging to the retrieved category

No LLM is used.
"""

import json
from pathlib import Path

from experiments.experiment_32_improved_query_engine import (
    search_records,
)


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
    / "grounded_answers_v2"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "multi_record_answer_tests.json"
)


def load_document():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def format_time(record):
    normalized = record[
        "normalized_time"
    ]

    if (
        normalized["type"]
        == "explicit_datetime_sequence"
    ):
        values = normalized["values"]

        if len(values) == 1:
            value = values[0]

            return (
                f"{value['date']} "
                f"{value['time']}"
            )

        return (
            f"{values[0]['date']} "
            f"{values[0]['time']} to "
            f"{values[-1]['date']} "
            f"{values[-1]['time']}"
        )

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


def get_record_by_id(
    records,
    record_id,
):
    for record in records:
        if record["record_id"] == record_id:
            return record

    raise ValueError(
        f"Record not found: {record_id}"
    )


def source_cell_id(
    record,
    field_name,
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


def make_provenance(record):
    return {
        "record_id": record[
            "record_id"
        ],
        "source_row": record[
            "source_row"
        ],
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
        "review_flags": record.get(
            "review_flags",
            [],
        ),
    }


def classify_question(query):
    q = query.lower().strip()

    if q.startswith("when"):
        return "single_when"

    if q.startswith("who"):
        return "multi_who"

    if (
        q.startswith("what happens")
        or q.startswith("what happen")
    ):
        return "category_summary"

    return "single_general"


def answer_single_when(record):
    category = (
        record["category"]["text"]
        or ""
    )

    event = (
        record["event"]["text"]
        or "Event"
    )

    time_text = format_time(record)

    if category:
        return (
            f"{category}: {event} is scheduled "
            f"for {time_text}."
        )

    return (
        f"{event} is scheduled for "
        f"{time_text}."
    )


def answer_single_general(record):
    category = record[
        "category"
    ]["text"]

    event = record[
        "event"
    ]["text"]

    detail = record[
        "detail"
    ]["text"]

    explanation = record[
        "explanation"
    ]["text"]

    parts = []

    if category:
        parts.append(
            f"For {category},"
        )

    if event:
        parts.append(event)

    if detail:
        parts.append(detail)

    if explanation:
        parts.append(explanation)

    return " ".join(parts)


def answer_multi_who(
    matched_records,
):
    lines = []

    for record in matched_records:
        category = record[
            "category"
        ]["text"]

        event = record[
            "event"
        ]["text"]

        detail = record[
            "detail"
        ]["text"]

        label = (
            category
            if category
            else event
        )

        lines.append(
            f"- {label}: {detail}"
        )

    return (
        "The timetable shows these matching groups:\n"
        + "\n".join(lines)
    )


def answer_category_summary(
    category,
    matched_records,
):
    lines = []

    for record in matched_records:
        event = (
            record["event"]["text"]
            or "Event"
        )

        time_text = format_time(
            record
        )

        detail = (
            record["detail"]["text"]
            or ""
        )

        if detail:
            lines.append(
                f"- {event}: {time_text}. "
                f"{detail}"
            )
        else:
            lines.append(
                f"- {event}: {time_text}."
            )

    return (
        f"For {category}, the timetable contains:\n"
        + "\n".join(lines)
    )


def answer_query(
    records,
    query,
):
    mode = classify_question(
        query
    )

    search_results = search_records(
        records,
        query,
        limit=len(records),
    )

    if not search_results:
        return {
            "query": query,
            "mode": mode,
            "answer": (
                "No matching timetable "
                "record was found."
            ),
            "records_used": [],
        }

    top_result = search_results[0]

    top_record = get_record_by_id(
        records,
        top_result["record_id"],
    )

    if mode == "single_when":
        selected = [
            top_record
        ]

        answer = answer_single_when(
            top_record
        )

    elif mode == "multi_who":
        top_score = top_result[
            "score"
        ]

        # For "who" questions, preserve all records
        # tied with the strongest retrieval score.
        tied_results = [
            result
            for result in search_results
            if result["score"] == top_score
        ]

        selected = [
            get_record_by_id(
                records,
                result["record_id"],
            )
            for result
            in tied_results
        ]

        answer = answer_multi_who(
            selected
        )

    elif mode == "category_summary":
        category = top_record[
            "category"
        ]["text"]

        if category:
            selected = [
                record
                for record in records
                if (
                    record["category"]["text"]
                    == category
                )
            ]

            selected.sort(
                key=lambda record:
                record["source_row"]
            )

            answer = answer_category_summary(
                category,
                selected,
            )

        else:
            selected = [
                top_record
            ]

            answer = answer_single_general(
                top_record
            )

    else:
        selected = [
            top_record
        ]

        answer = answer_single_general(
            top_record
        )

    return {
        "query": query,
        "mode": mode,
        "answer": answer,
        "records_used": [
            make_provenance(record)
            for record in selected
        ],
    }


def print_result(result):
    print()
    print("=" * 78)
    print(
        f"QUESTION: {result['query']}"
    )
    print(
        f"MODE: {result['mode']}"
    )
    print("=" * 78)

    print(
        result["answer"]
    )

    print()

    print(
        "Records used:",
        [
            item["record_id"]
            for item
            in result[
                "records_used"
            ]
        ],
    )

    for provenance in result[
        "records_used"
    ]:
        print(
            f"  {provenance['record_id']} "
            f"| row={provenance['source_row']} "
            f"| cells={provenance['source_cells']} "
            f"| flags={provenance['review_flags']}"
        )


def run_tests(records):
    questions = [
        (
            "When do graduate students "
            "select rooms?"
        ),
        (
            "When do male undergraduate "
            "students select rooms?"
        ),
        (
            "When does Waiting list II "
            "pay the deposit?"
        ),
        (
            "When does the dormitory open?"
        ),
        (
            "Who needs to pay $2000?"
        ),
        (
            "What happens for the "
            "3+2 Program?"
        ),
    ]

    results = []

    for query in questions:
        result = answer_query(
            records,
            query,
        )

        print_result(
            result
        )

        results.append(
            result
        )

    return results


def interactive_mode(records):
    print()
    print("=" * 78)
    print(
        "MULTI-RECORD GROUNDED QA"
    )
    print("=" * 78)

    print(
        "Ask a timetable question."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:
        print()

        query = input(
            "Question> "
        ).strip()

        if query.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        if not query:
            continue

        result = answer_query(
            records,
            query,
        )

        print_result(
            result
        )


def main():
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
        "EXPERIMENT 34: MULTI-RECORD GROUNDED ANSWERS"
    )
    print("=" * 78)

    print(
        f"Loaded records: {len(records)}"
    )

    test_results = run_tests(
        records
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "experiment": 34,
                "method": (
                    "intent_aware_multi_record_"
                    "grounded_answering"
                ),
                "source": str(
                    INPUT_PATH
                ),
                "results": test_results,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 78)
    print(
        f"Saved: {OUTPUT_JSON}"
    )
    print("=" * 78)

    interactive_mode(
        records
    )


if __name__ == "__main__":
    main()