"""
Experiment 33: Grounded Answer Generation

Uses the improved retrieval logic from Experiment 32 and
generates concise deterministic answers from the top result.

No LLM is used.

Principles:
- Answer only from retrieved structured records.
- Preserve review flags.
- Preserve source row and source cell references.
- Do not invent missing information.
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
    / "grounded_answers"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "grounded_answer_tests.json"
)


STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does",
    "for", "happen", "happens", "how", "i", "in",
    "is", "it", "me", "of", "on", "the", "to",
    "what", "when", "where", "which", "who", "with",
}

FIELD_WEIGHTS = {
    "category": 8,
    "event": 8,
    "detail": 3,
    "explanation": 2,
    "time": 4,
}


def load_document():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def raw_tokens(text):
    if not text:
        return []

    return re.findall(
        r"[a-z0-9]+(?:\+[a-z0-9]+)*",
        text.lower(),
    )


def normalize_token(token):
    protected = {
        "students",
        "services",
    }

    if token in protected:
        return token

    if (
        len(token) > 4
        and token.endswith("s")
        and not token.endswith("ss")
    ):
        return token[:-1]

    return token


def searchable_tokens(text):
    return [
        normalize_token(token)
        for token in raw_tokens(text)
    ]


def query_tokens(query):
    result = []

    for token in raw_tokens(query):
        if token in STOP_WORDS:
            continue

        result.append(
            normalize_token(token)
        )

    return result


def make_field_data(text):
    tokens = searchable_tokens(text)

    return {
        "tokens": tokens,
        "token_set": set(tokens),
    }


def record_fields(record):
    normalized_time = record.get(
        "normalized_time",
        {},
    )

    time_parts = [
        normalized_time.get(
            "raw_text",
            "",
        )
    ]

    for value in normalized_time.get(
        "values",
        [],
    ):
        if value.get("date"):
            time_parts.append(value["date"])

        if value.get("time"):
            time_parts.append(value["time"])

    if normalized_time.get(
        "text_expression"
    ):
        time_parts.append(
            normalized_time[
                "text_expression"
            ]
        )

    return {
        "category": make_field_data(
            record.get(
                "category",
                {},
            ).get("text")
        ),
        "event": make_field_data(
            record.get(
                "event",
                {},
            ).get("text")
        ),
        "detail": make_field_data(
            record.get(
                "detail",
                {},
            ).get("text")
        ),
        "explanation": make_field_data(
            record.get(
                "explanation",
                {},
            ).get("text")
        ),
        "time": make_field_data(
            " ".join(time_parts)
        ),
    }


def contains_phrase(
    phrase_tokens,
    field_tokens,
):
    if not phrase_tokens:
        return False

    width = len(phrase_tokens)

    for index in range(
        len(field_tokens) - width + 1
    ):
        if (
            field_tokens[
                index:index + width
            ]
            == phrase_tokens
        ):
            return True

    return False


def distinctive_phrases(query):
    tokens = query_tokens(query)

    phrases = []

    for width in (4, 3, 2):
        for index in range(
            len(tokens) - width + 1
        ):
            phrases.append(
                tokens[
                    index:index + width
                ]
            )

    return phrases


def score_record(query, record):
    q_tokens = query_tokens(query)

    if not q_tokens:
        return 0

    fields = record_fields(record)

    score = 0
    matched_fields = set()

    for field_name, field in fields.items():
        matches = (
            set(q_tokens)
            & field["token_set"]
        )

        if matches:
            matched_fields.add(field_name)

            score += (
                len(matches)
                * FIELD_WEIGHTS[field_name]
            )

    for phrase in distinctive_phrases(query):
        if contains_phrase(
            phrase,
            fields["category"]["tokens"],
        ):
            score += len(phrase) * 6

        if contains_phrase(
            phrase,
            fields["event"]["tokens"],
        ):
            score += len(phrase) * 6

    combined_tokens = set()

    for field in fields.values():
        combined_tokens.update(
            field["token_set"]
        )

    if set(q_tokens).issubset(
        combined_tokens
    ):
        score += 15

    if (
        "category" in matched_fields
        or "event" in matched_fields
    ):
        score += 5

    return score


def retrieve_top_record(
    records,
    query,
):
    scored = []

    for record in records:
        score = score_record(
            query,
            record,
        )

        if score > 0:
            scored.append(
                (
                    score,
                    record[
                        "source_row"
                    ],
                    record,
                )
            )

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    if not scored:
        return None, 0

    return (
        scored[0][2],
        scored[0][0],
    )


def format_time(record):
    normalized = record[
        "normalized_time"
    ]

    if (
        normalized["type"]
        == "explicit_datetime_sequence"
    ):
        values = normalized[
            "values"
        ]

        if len(values) == 1:
            value = values[0]

            return (
                f"{value['date']} "
                f"{value['time']}"
            )

        return (
            f"{values[0]['date']} "
            f"{values[0]['time']}"
            " to "
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


def answer_when(record):
    category = (
        record["category"]["text"]
        or ""
    )

    event = (
        record["event"]["text"]
        or "event"
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


def answer_who(record):
    category = (
        record["category"]["text"]
    )

    event = (
        record["event"]["text"]
    )

    detail = (
        record["detail"]["text"]
    )

    if category:
        return (
            f"{category}. "
            f"{event}. "
            f"{detail}"
        )

    return (
        f"{event}. {detail}"
    )


def answer_what(record):
    category = (
        record["category"]["text"]
    )

    event = (
        record["event"]["text"]
    )

    detail = (
        record["detail"]["text"]
    )

    explanation = (
        record[
            "explanation"
        ]["text"]
    )

    parts = []

    if category:
        parts.append(
            f"For {category},"
        )

    parts.append(
        event
    )

    if detail:
        parts.append(
            detail
        )

    if explanation:
        parts.append(
            explanation
        )

    return " ".join(parts)


def generate_answer(
    query,
    record,
):
    q = query.lower().strip()

    if q.startswith("when"):
        return answer_when(
            record
        )

    if q.startswith("who"):
        return answer_who(
            record
        )

    return answer_what(
        record
    )


def make_grounded_result(
    query,
    record,
    score,
):
    if record is None:
        return {
            "query": query,
            "answer": (
                "No matching timetable "
                "record was found."
            ),
            "retrieval_score": 0,
            "source_row": None,
            "source_cells": None,
            "review_flags": [],
        }

    return {
        "query": query,
        "answer": generate_answer(
            query,
            record,
        ),
        "retrieval_score": score,
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


def print_result(result):
    print()
    print("=" * 78)
    print(
        f"QUESTION: "
        f"{result['query']}"
    )
    print("=" * 78)

    print(
        "ANSWER:",
        result["answer"],
    )

    print(
        "Retrieval score:",
        result[
            "retrieval_score"
        ],
    )

    print(
        "Source row:",
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
        "Review flags:",
        result[
            "review_flags"
        ],
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
            "When do female undergraduate "
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

    for question in questions:
        record, score = retrieve_top_record(
            records,
            question,
        )

        result = make_grounded_result(
            question,
            record,
            score,
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
        "GROUNDED TIMETABLE QUESTION ANSWERING"
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

        record, score = retrieve_top_record(
            records,
            query,
        )

        result = make_grounded_result(
            query,
            record,
            score,
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
            "No normalized records found."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "EXPERIMENT 33: GROUNDED ANSWER GENERATION"
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
                "experiment": 33,
                "method": (
                    "deterministic_retrieval_"
                    "plus_template_generation"
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