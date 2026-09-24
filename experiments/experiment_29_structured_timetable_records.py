
"""
Experiment 29: Structured Timetable Records

Converts the logical table from Experiment 28 into one searchable
record per body row.

Handles:
- Merged category cells spanning multiple rows
- Event labels spanning the first two columns
- Source-cell references and bounding boxes
- Conservative text normalization
- Date/time strings requiring later verification

Does NOT:
- Rerun OCR
- Invent missing student categories
- Interpret ambiguous dates or times
- Modify earlier experiment outputs or production code
"""

import json
import re
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "logical_table"
    / "logical_table.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "structured_records"
)

OUTPUT_JSON = OUTPUT_DIR / "timetable_records.json"
OUTPUT_REPORT = OUTPUT_DIR / "timetable_records.txt"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(raw_text: str) -> str:
    """
    Make cell text easier to search without altering the
    original OCR text stored in the source document.

    The only word-specific repair below is the line-wrapped
    'Undergradu' + 'ate' observed in this timetable.
    """

    text = raw_text.strip()

    text = re.sub(
        r"\bUndergradu[ \t]*\n[ \t]*ate\b",
        "Undergraduate",
        text,
        flags=re.IGNORECASE,
    )

    # Join other visual line wraps with spaces.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def make_cell_reference(cell: dict) -> dict:
    """
    Retain enough source information to locate and inspect
    the original image region.
    """

    position = cell["logical_position"]

    return {
        "cell_id": cell["cell_id"],
        "bounding_box": deepcopy(
            cell["bounding_box"]
        ),
        "logical_position": deepcopy(position),
        "text_source": cell.get(
            "text_source",
            "original_full_page_ocr",
        ),
        "ocr_region_ids": [
            region["region_id"]
            for region in cell.get("ocr_regions", [])
        ],
    }


def make_field(cell: dict | None) -> dict:
    """
    Store both the unchanged cell text and a separate
    searchable version. None means that this field has no
    corresponding source cell.
    """

    if cell is None:
        return {
            "text": None,
            "raw_text": None,
            "source_cell": None,
        }

    raw_text = cell.get("text", "")

    return {
        "text": normalize_text(raw_text),
        "raw_text": raw_text,
        "source_cell": make_cell_reference(cell),
    }


def build_cell_lookup(
    document: dict,
) -> tuple[dict[int, dict], list[list[int]]]:
    """
    Validate and index the logical table.

    The occupancy grid already accounts for merged cells:
    a single merged cell ID appears in every position it
    occupies.
    """

    geometry = document["table_geometry"]

    row_count = geometry["row_count"]
    column_count = geometry["column_count"]

    if column_count != 5:
        raise ValueError(
            "Experiment 29 expects the verified five-column "
            f"timetable, but found {column_count} columns."
        )

    if row_count < 2:
        raise ValueError(
            "The table must contain a header and at "
            "least one body row."
        )

    cells_by_id = {
        cell["cell_id"]: cell
        for cell in document["cells"]
    }

    if len(cells_by_id) != len(document["cells"]):
        raise ValueError(
            "Duplicate cell IDs found."
        )

    occupancy = document["logical_occupancy"]

    if len(occupancy) != row_count:
        raise ValueError(
            "Occupancy row count does not match table geometry."
        )

    for row_index, row in enumerate(occupancy):
        if len(row) != column_count:
            raise ValueError(
                f"Row {row_index} has an unexpected "
                "number of columns."
            )

        for cell_id in row:
            if cell_id not in cells_by_id:
                raise ValueError(
                    f"Occupancy references unknown cell "
                    f"ID {cell_id}."
                )

    validation = document.get("validation", {})

    if validation.get("collisions"):
        raise ValueError(
            "Logical table contains overlapping positions."
        )

    if validation.get("missing_positions"):
        raise ValueError(
            "Logical table contains missing positions."
        )

    return cells_by_id, occupancy


def cell_at(
    cells_by_id: dict[int, dict],
    occupancy: list[list[int]],
    row: int,
    column: int,
) -> dict:
    cell_id = occupancy[row][column]
    return cells_by_id[cell_id]


def inspect_time_text(text: str) -> list[str]:
    """
    Flag time formats that deserve manual review.

    We keep their original wording rather than guessing
    how to interpret them.
    """

    flags = []

    # Example: 2026/03/1612:30
    if re.search(
        r"\d{4}/\d{2}/\d{2}\d{2}:\d{2}",
        text,
    ):
        flags.append(
            "date_and_time_joined_in_ocr"
        )

    if "24:00" in text:
        flags.append(
            "contains_24_00_time_verify_interpretation"
        )

    return flags


def build_record(
    row_index: int,
    cells_by_id: dict[int, dict],
    occupancy: list[list[int]],
) -> dict:
    """
    Convert one body row into a structured record.

    Column 0: Category, or first half of a merged event cell
    Column 1: Event
    Column 2: Time
    Column 3: Detail
    Column 4: Explanation
    """

    first_cell = cell_at(
        cells_by_id,
        occupancy,
        row_index,
        0,
    )

    second_cell = cell_at(
        cells_by_id,
        occupancy,
        row_index,
        1,
    )

    time_cell = cell_at(
        cells_by_id,
        occupancy,
        row_index,
        2,
    )

    detail_cell = cell_at(
        cells_by_id,
        occupancy,
        row_index,
        3,
    )

    explanation_cell = cell_at(
        cells_by_id,
        occupancy,
        row_index,
        4,
    )

    first_position = first_cell["logical_position"]

    if first_cell["cell_id"] == second_cell["cell_id"]:
        # A cell spanning columns 0 and 1 is an event label,
        # not a student category.
        category_cell = None
        event_cell = first_cell

    else:
        category_cell = first_cell
        event_cell = second_cell

        # The source category may begin in an earlier row.
        category_start = first_position["row"]
        category_end = (
            category_start + first_position["row_span"]
        )

        if not (
            category_start <= row_index < category_end
        ):
            raise ValueError(
                f"Category cell {first_cell['cell_id']} "
                f"does not cover row {row_index}."
            )

    category = make_field(category_cell)
    event = make_field(event_cell)
    time = make_field(time_cell)
    detail = make_field(detail_cell)
    explanation = make_field(explanation_cell)

    review_flags = inspect_time_text(
        time["raw_text"] or ""
    )

    if not detail["text"]:
        review_flags.append(
            "empty_detail_cell"
        )

    # A standalone numeral at the end of an explanation
    # could be a line number or a grid-detection artifact.
    # Flag it rather than silently deleting it.
    explanation_lines = (
        explanation["raw_text"] or ""
    ).splitlines()

    if (
        explanation_lines
        and explanation_lines[-1].strip().isdigit()
    ):
        review_flags.append(
            "explanation_ends_with_standalone_number"
        )

    return {
        "record_id": f"row_{row_index:02d}",
        "source_row": row_index,
        "category": category,
        "event": event,
        "time": time,
        "detail": detail,
        "explanation": explanation,
        "review_flags": review_flags,
    }


def validate_records(
    records: list[dict],
    expected_count: int,
) -> None:
    if len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} body records, "
            f"but created {len(records)}."
        )

    record_ids = [
        record["record_id"]
        for record in records
    ]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError(
            "Duplicate record IDs found."
        )

    for record in records:
        if not record["event"]["source_cell"]:
            raise ValueError(
                f"{record['record_id']} has no event cell."
            )

        for field_name in (
            "time",
            "detail",
            "explanation",
        ):
            if not record[field_name]["source_cell"]:
                raise ValueError(
                    f"{record['record_id']} has no "
                    f"{field_name} source cell."
                )


def make_report(records: list[dict]) -> str:
    lines = [
        "=" * 78,
        "EXPERIMENT 29: STRUCTURED TIMETABLE RECORDS",
        "=" * 78,
        f"Total event records: {len(records)}",
        "=" * 78,
    ]

    for record in records:
        category_text = (
            record["category"]["text"]
            if record["category"]["text"] is not None
            else "[No separate category cell]"
        )

        lines.extend(
            [
                "",
                f"RECORD: {record['record_id']}",
                f"Category:    {category_text}",
                f"Event:       {record['event']['text']}",
                f"Time:        {record['time']['text']}",
                f"Detail:      {record['detail']['text']}",
                (
                    "Explanation: "
                    f"{record['explanation']['text']}"
                ),
                (
                    "Source cells: "
                    + ", ".join(
                        f"{name}="
                        + (
                            str(
                                record[name][
                                    "source_cell"
                                ]["cell_id"]
                            )
                            if record[name]["source_cell"]
                            else "none"
                        )
                        for name in (
                            "category",
                            "event",
                            "time",
                            "detail",
                            "explanation",
                        )
                    )
                ),
                (
                    "Review flags: "
                    + (
                        ", ".join(
                            record["review_flags"]
                        )
                        if record["review_flags"]
                        else "none"
                    )
                ),
                "-" * 78,
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    document = load_json(INPUT_PATH)

    cells_by_id, occupancy = build_cell_lookup(
        document
    )

    row_count = document[
        "table_geometry"
    ]["row_count"]

    # Row 0 is the header, not an event.
    records = [
        build_record(
            row_index,
            cells_by_id,
            occupancy,
        )
        for row_index in range(1, row_count)
    ]

    validate_records(
        records,
        expected_count=row_count - 1,
    )

    headers = {
        str(column): make_field(
            cell_at(
                cells_by_id,
                occupancy,
                0,
                column,
            )
        )
        for column in range(5)
    }

    result = {
        "schema_version": "0.5-experimental",
        "document_type": "timetable_records",
        "source_logical_table": str(INPUT_PATH),
        "source_image": document["source"][
            "perspective_image"
        ],
        "row_indexing": "zero_based",
        "header_row": 0,
        "headers_by_column": headers,
        "statistics": {
            "logical_rows": row_count,
            "logical_columns": 5,
            "event_records": len(records),
            "records_with_review_flags": sum(
                bool(record["review_flags"])
                for record in records
            ),
        },
        "records": records,
        "notes": [
            "Each body row becomes one event record.",
            "A merged category cell is referenced by each "
            "row it spans.",
            "A label spanning the first two columns is "
            "treated as an event, with no separate category.",
            "Original OCR text is retained as raw_text.",
            "Normalized text is provided for searching.",
            "Date and time strings have not been converted "
            "to timestamps or interpreted as deadlines.",
            "Source cell references retain image coordinates.",
            "Records require comparison with the source "
            "image before use in reliable question answering.",
        ],
    }

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
        make_report(records),
        encoding="utf-8",
    )

    print("=" * 65)
    print("EXPERIMENT 29: STRUCTURED TIMETABLE RECORDS")
    print("=" * 65)
    print(
        "Event records:",
        result["statistics"]["event_records"],
    )
    print(
        "Records with review flags:",
        result["statistics"][
            "records_with_review_flags"
        ],
    )
    print(f"JSON:   {OUTPUT_JSON}")
    print(f"Report: {OUTPUT_REPORT}")
    print("=" * 65)
    print("EXPERIMENT 29 COMPLETE")


if __name__ == "__main__":
    main()