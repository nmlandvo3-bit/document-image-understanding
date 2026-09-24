"""
Experiment 30: Conservative Time Normalization

Converts OCR timetable time fields into machine-readable
date/time expressions without changing the original OCR text.

Principles:
- Preserve raw OCR text.
- Split joined date/time strings such as:
      2026/03/1612:30
- Parse already separated date/time strings.
- Preserve 24:00 exactly and flag it for review.
- Preserve expressions such as "Mid-August, 2026" as text.
- Do not guess missing dates, deadlines, or meanings.

Input:
    results/parsing/pic1/structured_records/
        timetable_records.json

Outputs:
    results/parsing/pic1/normalized_records/
        normalized_timetable_records.json
        time_normalization_report.txt
"""

import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "structured_records"
    / "timetable_records.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "normalized_records"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "normalized_timetable_records.json"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "time_normalization_report.txt"
)


# Matches both:
#
#   2026/03/16 12:30
#
# and:
#
#   2026/03/1612:30
#
DATE_TIME_PATTERN = re.compile(
    r"(?P<year>\d{4})/"
    r"(?P<month>\d{2})/"
    r"(?P<day>\d{2})"
    r"\s*"
    r"(?P<hour>\d{2}):"
    r"(?P<minute>\d{2})"
)


DATE_ONLY_PATTERN = re.compile(
    r"(?P<year>\d{4})/"
    r"(?P<month>\d{2})/"
    r"(?P<day>\d{2})"
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_calendar_date(
    year: int,
    month: int,
    day: int,
) -> bool:
    try:
        datetime(
            year=year,
            month=month,
            day=day,
        )

        return True

    except ValueError:
        return False


def parse_datetime_matches(
    raw_text: str,
) -> list[dict]:
    """
    Extract all explicit numeric date/time pairs.

    Does not reinterpret 24:00 as the following day.
    """

    results = []

    for index, match in enumerate(
        DATE_TIME_PATTERN.finditer(raw_text),
        start=1,
    ):
        year = int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day"))
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))

        date_valid = validate_calendar_date(
            year,
            month,
            day,
        )

        time_valid = (
            0 <= hour <= 23
            and 0 <= minute <= 59
        )

        is_24_00 = (
            hour == 24
            and minute == 0
        )

        flags = []

        if not date_valid:
            flags.append(
                "invalid_calendar_date"
            )

        if is_24_00:
            flags.append(
                "contains_24_00_time_verify_interpretation"
            )

        elif not time_valid:
            flags.append(
                "invalid_clock_time"
            )

        preceding_character = (
            raw_text[match.start() - 1]
            if match.start() > 0
            else ""
        )

        matched_text = match.group(0)

        # Detect whether OCR originally joined the date
        # directly to the time.
        joined_match = re.search(
            r"\d{4}/\d{2}/\d{2}\d{2}:\d{2}",
            matched_text,
        )

        if joined_match:
            flags.append(
                "date_and_time_joined_in_ocr"
            )

        normalized_date = (
            f"{year:04d}-{month:02d}-{day:02d}"
            if date_valid
            else None
        )

        normalized_time = (
            f"{hour:02d}:{minute:02d}"
        )

        results.append(
            {
                "sequence": index,
                "source_text": matched_text,
                "date": normalized_date,
                "time": normalized_time,
                "date_valid": date_valid,
                "time_valid": (
                    time_valid or is_24_00
                ),
                "special_24_00": is_24_00,
                "review_flags": flags,
            }
        )

    return results


def classify_time_field(
    raw_text: str,
) -> dict:
    """
    Classify a timetable Time cell conservatively.
    """

    text = raw_text.strip()

    datetime_values = parse_datetime_matches(
        text
    )

    if datetime_values:
        matched_ranges = [
            (
                match.start(),
                match.end(),
            )
            for match in DATE_TIME_PATTERN.finditer(
                text
            )
        ]

        leftover_parts = []

        previous_end = 0

        for start, end in matched_ranges:
            leftover = text[
                previous_end:start
            ].strip()

            if leftover:
                leftover_parts.append(leftover)

            previous_end = end

        final_leftover = text[
            previous_end:
        ].strip()

        if final_leftover:
            leftover_parts.append(
                final_leftover
            )

        review_flags = []

        for value in datetime_values:
            review_flags.extend(
                value["review_flags"]
            )

        if leftover_parts:
            review_flags.append(
                "unparsed_time_text_remains"
            )

        return {
            "type": "explicit_datetime_sequence",
            "raw_text": raw_text,
            "values": datetime_values,
            "text_expression": None,
            "unparsed_text": leftover_parts,
            "review_flags": sorted(
                set(review_flags)
            ),
        }

    date_only_matches = list(
        DATE_ONLY_PATTERN.finditer(text)
    )

    if date_only_matches:
        values = []

        for index, match in enumerate(
            date_only_matches,
            start=1,
        ):
            year = int(match.group("year"))
            month = int(match.group("month"))
            day = int(match.group("day"))

            valid = validate_calendar_date(
                year,
                month,
                day,
            )

            values.append(
                {
                    "sequence": index,
                    "source_text": match.group(0),
                    "date": (
                        f"{year:04d}-"
                        f"{month:02d}-"
                        f"{day:02d}"
                        if valid
                        else None
                    ),
                    "date_valid": valid,
                }
            )

        return {
            "type": "explicit_date_sequence",
            "raw_text": raw_text,
            "values": values,
            "text_expression": None,
            "unparsed_text": [],
            "review_flags": [],
        }

    # Examples:
    #
    #   Mid-August, 2026
    #   Mid-September, 2026
    #
    # These must remain textual because the source does
    # not provide an exact calendar day.
    if text:
        return {
            "type": "textual_time_expression",
            "raw_text": raw_text,
            "values": [],
            "text_expression": text,
            "unparsed_text": [],
            "review_flags": [
                "textual_time_expression_not_converted"
            ],
        }

    return {
        "type": "empty",
        "raw_text": raw_text,
        "values": [],
        "text_expression": None,
        "unparsed_text": [],
        "review_flags": [
            "empty_time_field"
        ],
    }


def create_normalized_record(
    original_record: dict,
) -> dict:
    record = deepcopy(original_record)

    raw_time = (
        original_record
        .get("time", {})
        .get("raw_text")
    )

    if raw_time is None:
        raw_time = (
            original_record
            .get("time", {})
            .get("text", "")
        )

    normalized_time = classify_time_field(
        raw_time
    )

    record["normalized_time"] = (
        normalized_time
    )

    combined_flags = set(
        record.get(
            "review_flags",
            [],
        )
    )

    combined_flags.update(
        normalized_time[
            "review_flags"
        ]
    )

    record[
        "review_flags"
    ] = sorted(combined_flags)

    return record


def make_report(
    records: list[dict],
) -> str:
    lines = [
        "=" * 78,
        "EXPERIMENT 30: TIME NORMALIZATION",
        "=" * 78,
        f"Records processed: {len(records)}",
        "=" * 78,
    ]

    for record in records:
        normalized = record[
            "normalized_time"
        ]

        lines.extend(
            [
                "",
                f"RECORD: {record['record_id']}",
                (
                    "Original: "
                    f"{normalized['raw_text']}"
                ),
                (
                    "Type:     "
                    f"{normalized['type']}"
                ),
            ]
        )

        if normalized["values"]:
            for value in normalized["values"]:
                if (
                    normalized["type"]
                    == "explicit_datetime_sequence"
                ):
                    lines.append(
                        "  "
                        f"{value['sequence']}. "
                        f"date={value['date']} "
                        f"time={value['time']} "
                        f"24:00={value['special_24_00']}"
                    )

                else:
                    lines.append(
                        "  "
                        f"{value['sequence']}. "
                        f"date={value['date']}"
                    )

        if normalized["text_expression"]:
            lines.append(
                "  Text expression: "
                + normalized[
                    "text_expression"
                ]
            )

        if normalized["unparsed_text"]:
            lines.append(
                "  Unparsed: "
                + repr(
                    normalized[
                        "unparsed_text"
                    ]
                )
            )

        lines.append(
            "Review flags: "
            + (
                ", ".join(
                    record[
                        "review_flags"
                    ]
                )
                if record[
                    "review_flags"
                ]
                else "none"
            )
        )

        lines.append(
            "-" * 78
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    document = load_json(
        INPUT_PATH
    )

    original_records = document.get(
        "records",
        [],
    )

    if not original_records:
        raise ValueError(
            "No structured timetable "
            "records were found."
        )

    normalized_records = [
        create_normalized_record(record)
        for record in original_records
    ]

    type_counts = {}

    for record in normalized_records:
        time_type = record[
            "normalized_time"
        ]["type"]

        type_counts[time_type] = (
            type_counts.get(
                time_type,
                0,
            )
            + 1
        )

    result = deepcopy(document)

    result[
        "schema_version"
    ] = "0.6-experimental"

    result[
        "document_type"
    ] = "normalized_timetable_records"

    result[
        "source_structured_records"
    ] = str(INPUT_PATH)

    result[
        "records"
    ] = normalized_records

    result[
        "time_normalization"
    ] = {
        "method": (
            "conservative_regex_extraction"
        ),
        "type_counts": type_counts,
        "rules": [
            (
                "Explicit YYYY/MM/DD HH:MM "
                "pairs are extracted."
            ),
            (
                "Joined OCR date/time strings "
                "are separated structurally."
            ),
            (
                "24:00 is preserved exactly "
                "and flagged."
            ),
            (
                "Textual expressions such as "
                "Mid-August are not converted "
                "to invented dates."
            ),
            (
                "Original OCR time text remains "
                "available unchanged."
            ),
        ],
    }

    result[
        "statistics"
    ][
        "records_with_review_flags_after_time_normalization"
    ] = sum(
        bool(
            record[
                "review_flags"
            ]
        )
        for record in normalized_records
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    OUTPUT_REPORT.write_text(
        make_report(
            normalized_records
        ),
        encoding="utf-8",
    )

    print("=" * 65)
    print(
        "EXPERIMENT 30: CONSERVATIVE TIME NORMALIZATION"
    )
    print("=" * 65)
    print(
        "Records processed:",
        len(normalized_records),
    )

    for time_type, count in sorted(
        type_counts.items()
    ):
        print(
            f"{time_type}: {count}"
        )

    print(
        "Records with review flags:",
        result["statistics"][
            "records_with_review_flags_after_time_normalization"
        ],
    )

    print(
        f"JSON:   {OUTPUT_JSON}"
    )

    print(
        f"Report: {OUTPUT_REPORT}"
    )

    print("=" * 65)
    print("EXPERIMENT 30 COMPLETE")


if __name__ == "__main__":
    main()