"""
Experiment 32: Improved Deterministic Query Engine

Improves Experiment 31 by using:
- exact token matching instead of substring matching
- stop-word removal
- conservative singular normalization
- category/event phrase boosts
- transparent scoring
- source traceability

No LLM is used.
Previous results are not modified.
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
    / "query_engine_v2"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "query_test_results_v2.json"
)


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "happen",
    "happens",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


FIELD_WEIGHTS = {
    "category": 8,
    "event": 8,
    "detail": 3,
    "explanation": 2,
    "time": 4,
}


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


def raw_tokens(text: str | None) -> list[str]:
    """
    Extract real tokens.

    Important:
        'male' and 'female' become separate tokens.
        'graduate' and 'undergraduate' remain separate.
        '3+2' stays usable as a token.
    """

    if not text:
        return []

    return re.findall(
        r"[a-z0-9]+(?:\+[a-z0-9]+)*",
        text.lower(),
    )


def normalize_token(token: str) -> str:
    """
    Apply only conservative singular normalization.

    This is intentionally not a general stemmer.
    """

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


def searchable_tokens(
    text: str | None,
) -> list[str]:
    return [
        normalize_token(token)
        for token in raw_tokens(text)
    ]


def query_tokens(
    query: str,
) -> list[str]:
    result = []

    for token in raw_tokens(query):
        if token in STOP_WORDS:
            continue

        normalized = normalize_token(
            token
        )

        if normalized:
            result.append(normalized)

    return result


def normalized_phrase(
    text: str | None,
) -> str:
    return " ".join(
        searchable_tokens(text)
    )


def make_field_data(
    text: str | None,
) -> dict:
    tokens = searchable_tokens(
        text
    )

    return {
        "text": text or "",
        "tokens": tokens,
        "token_set": set(tokens),
        "phrase": " ".join(tokens),
    }


def record_fields(
    record: dict,
) -> dict:
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
            time_parts.append(
                value["date"]
            )

        if value.get("time"):
            time_parts.append(
                value["time"]
            )

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
    phrase_tokens: list[str],
    field_tokens: list[str],
) -> bool:
    if not phrase_tokens:
        return False

    if len(phrase_tokens) > len(
        field_tokens
    ):
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


def distinctive_phrases(
    query: str,
) -> list[list[str]]:
    """
    Extract useful multi-token phrases that deserve
    additional weight.
    """

    tokens = query_tokens(query)

    phrases = []

    # Longest useful sequences first.
    for width in (4, 3, 2):
        for index in range(
            len(tokens) - width + 1
        ):
            phrase = tokens[
                index:index + width
            ]

            phrases.append(
                phrase
            )

    return phrases


def score_record(
    query: str,
    record: dict,
) -> dict:
    q_tokens = query_tokens(
        query
    )

    if not q_tokens:
        return {
            "score": 0,
            "matched_fields": [],
            "matched_tokens": {},
            "phrase_matches": [],
        }

    fields = record_fields(
        record
    )

    score = 0

    matched_fields = set()
    matched_tokens = {}

    # Exact token matches.
    for field_name, field in fields.items():
        matches = sorted(
            set(q_tokens)
            & field["token_set"]
        )

        if not matches:
            continue

        matched_fields.add(
            field_name
        )

        matched_tokens[
            field_name
        ] = matches

        score += (
            len(matches)
            * FIELD_WEIGHTS[field_name]
        )

    phrase_matches = []

    # Phrase boosts are intentionally focused on
    # category and event fields.
    for phrase in distinctive_phrases(
        query
    ):
        phrase_text = " ".join(
            phrase
        )

        if contains_phrase(
            phrase,
            fields[
                "category"
            ]["tokens"],
        ):
            score += (
                len(phrase) * 6
            )

            phrase_matches.append(
                {
                    "field": "category",
                    "phrase": phrase_text,
                }
            )

        if contains_phrase(
            phrase,
            fields[
                "event"
            ]["tokens"],
        ):
            score += (
                len(phrase) * 6
            )

            phrase_matches.append(
                {
                    "field": "event",
                    "phrase": phrase_text,
                }
            )

    # Strong bonus when every meaningful query token
    # appears somewhere in the record.
    combined_tokens = set()

    for field in fields.values():
        combined_tokens.update(
            field["token_set"]
        )

    if set(q_tokens).issubset(
        combined_tokens
    ):
        score += 15

    # Prefer category/event evidence over matches that
    # occur only in long explanatory text.
    if (
        "category" in matched_fields
        or "event" in matched_fields
    ):
        score += 5

    return {
        "score": score,
        "matched_fields": sorted(
            matched_fields
        ),
        "matched_tokens": matched_tokens,
        "phrase_matches": phrase_matches,
    }


def format_time(
    record: dict,
) -> str:
    normalized = record[
        "normalized_time"
    ]

    if (
        normalized["type"]
        == "explicit_datetime_sequence"
    ):
        return " -> ".join(
            f"{value['date']} "
            f"{value['time']}"
            for value
            in normalized[
                "values"
            ]
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
    scoring: dict,
) -> dict:
    return {
        "record_id": record[
            "record_id"
        ],
        "source_row": record[
            "source_row"
        ],
        "score": scoring[
            "score"
        ],
        "matched_fields": scoring[
            "matched_fields"
        ],
        "matched_tokens": scoring[
            "matched_tokens"
        ],
        "phrase_matches": scoring[
            "phrase_matches"
        ],
        "category": record[
            "category"
        ]["text"],
        "event": record[
            "event"
        ]["text"],
        "time": format_time(
            record
        ),
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
        scoring = score_record(
            query,
            record,
        )

        if scoring["score"] <= 0:
            continue

        scored.append(
            (
                scoring[
                    "score"
                ],
                record[
                    "source_row"
                ],
                scoring,
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
            scoring=item[2],
        )
        for item
        in scored[:limit]
    ]


def print_results(
    query: str,
    results: list[dict],
) -> None:
    print()
    print("=" * 78)
    print(
        f"QUERY: {query}"
    )
    print(
        "Keywords:",
        query_tokens(query),
    )
    print("=" * 78)

    if not results:
        print(
            "No matching timetable records."
        )

        return

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(
            f"[{rank}] "
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
            "Matched:    ",
            result[
                "matched_tokens"
            ],
        )

        print(
            "Phrases:    ",
            result[
                "phrase_matches"
            ],
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
        "When do graduate students select rooms?",
        "female undergraduate room selection",
        "male undergraduate room selection",
        "undergraduate pay deposit",
        "waiting list 1",
        "waiting list II room announcement",
        "3+2 program room announcement",
        "3+2 program pay deposit",
        "When does the dormitory open?",
        "Who needs to pay $2000?",
    ]

    outputs = []

    for query in queries:
        results = search_records(
            records,
            query,
            limit=5,
        )

        print_results(
            query,
            results,
        )

        outputs.append(
            {
                "query": query,
                "keywords": query_tokens(
                    query
                ),
                "results": results,
            }
        )

    return outputs


def interactive_mode(
    records: list[dict],
) -> None:
    print()
    print("=" * 78)
    print(
        "INTERACTIVE TIMETABLE SEARCH V2"
    )
    print("=" * 78)

    print(
        "Type a question or search phrase."
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

        print_results(
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
        "EXPERIMENT 32: IMPROVED QUERY ENGINE"
    )
    print("=" * 78)

    print(
        f"Loaded records: {len(records)}"
    )

    test_results = run_test_queries(
        records
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "experiment": 32,
                "method": (
                    "exact_token_phrase_weighted_search"
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
        f"Saved: {OUTPUT_JSON}"
    )
    print("=" * 78)

    interactive_mode(
        records
    )


if __name__ == "__main__":
    main()