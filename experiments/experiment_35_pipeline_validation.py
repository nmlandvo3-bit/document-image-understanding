"""
Experiment 35: End-to-End Pipeline Validation

Checks that the major artifacts produced by Experiments 22-34
exist and are internally consistent.

This experiment does not rerun OCR or modify previous outputs.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
)


FILES = {
    "structured_ocr": (
        BASE / "structured_document.json"
    ),

    "cell_reconstruction": (
        BASE
        / "cell_reconstruction"
        / "candidate_table_cells.json"
    ),

    "header_recovery": (
        BASE
        / "header_recovery"
        / "candidate_table_cells_with_header.json"
    ),

    "cell_ocr_recovery": (
        BASE
        / "cell_ocr_recovery"
        / "cell_ocr_diagnostics.json"
    ),

    "cell_24_recovery": (
        BASE
        / "cell_24_recovery"
        / "cell_24_variant_results.json"
    ),

    "consolidated_table": (
        BASE
        / "consolidated_table"
        / "consolidated_table_cells.json"
    ),

    "logical_table": (
        BASE
        / "logical_table"
        / "logical_table.json"
    ),

    "structured_records": (
        BASE
        / "structured_records"
        / "timetable_records.json"
    ),

    "normalized_records": (
        BASE
        / "normalized_records"
        / "normalized_timetable_records.json"
    ),

    "query_v1": (
        BASE
        / "query_engine"
        / "query_test_results.json"
    ),

    "query_v2": (
        BASE
        / "query_engine_v2"
        / "query_test_results_v2.json"
    ),

    "grounded_answers": (
        BASE
        / "grounded_answers"
        / "grounded_answer_tests.json"
    ),

    "grounded_answers_v2": (
        BASE
        / "grounded_answers_v2"
        / "multi_record_answer_tests.json"
    ),
}


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def check_files():
    print("=" * 72)
    print("ARTIFACT CHECK")
    print("=" * 72)

    missing = []

    for name, path in FILES.items():
        exists = path.is_file()

        print(
            f"{'[OK]' if exists else '[MISSING]':<10}"
            f"{name:<24}"
            f"{path}"
        )

        if not exists:
            missing.append(name)

    return missing


def validate_logical_table():
    document = load_json(
        FILES["logical_table"]
    )

    geometry = document[
        "table_geometry"
    ]

    statistics = document[
        "statistics"
    ]

    errors = []

    if geometry["row_count"] != 19:
        errors.append(
            "Logical row count is not 19."
        )

    if geometry["column_count"] != 5:
        errors.append(
            "Logical column count is not 5."
        )

    if len(document["cells"]) != 83:
        errors.append(
            "Logical table does not contain 83 cells."
        )

    if (
        statistics[
            "overlapping_positions"
        ]
        != 0
    ):
        errors.append(
            "Logical table contains overlapping positions."
        )

    if (
        statistics[
            "missing_positions"
        ]
        != 0
    ):
        errors.append(
            "Logical table contains missing positions."
        )

    return errors


def validate_records():
    document = load_json(
        FILES["normalized_records"]
    )

    records = document[
        "records"
    ]

    errors = []

    if len(records) != 18:
        errors.append(
            "Expected 18 structured timetable records."
        )

    ids = [
        record["record_id"]
        for record in records
    ]

    if len(ids) != len(set(ids)):
        errors.append(
            "Duplicate record IDs found."
        )

    expected_ids = {
        f"row_{index:02d}"
        for index in range(1, 19)
    }

    if set(ids) != expected_ids:
        errors.append(
            "Record IDs do not match rows 01-18."
        )

    return errors


def validate_recovered_cells():
    document = load_json(
        FILES["consolidated_table"]
    )

    cells = {
        cell["cell_id"]: cell
        for cell in document["cells"]
    }

    errors = []

    expected = {
        18: (
            "Dormitory\n"
            "application online\n"
            "room selection"
        ),

        19: (
            "Female\n"
            "Undergradu\n"
            "ate students"
        ),

        23: (
            "Male\n"
            "Undergradu\n"
            "ate students"
        ),

        24: (
            "Dormitory\n"
            "application online\n"
            "room selection"
        ),
    }

    for cell_id, text in expected.items():
        if cell_id not in cells:
            errors.append(
                f"Recovered cell {cell_id} missing."
            )

            continue

        if cells[cell_id]["text"] != text:
            errors.append(
                f"Recovered cell {cell_id} "
                "does not contain expected text."
            )

    return errors


def validate_time_normalization():
    document = load_json(
        FILES["normalized_records"]
    )

    records = {
        record["record_id"]: record
        for record in document["records"]
    }

    errors = []

    checks = {
        "row_02": (
            "2026-03-16",
            "12:30",
            "2026-03-18",
            "12:00",
        ),

        "row_05": (
            "2026-03-24",
            "12:30",
            "2026-03-26",
            "12:00",
        ),

        "row_13": (
            "2026-06-03",
            "12:00",
            "2026-06-11",
            "24:00",
        ),
    }

    for record_id, expected in checks.items():
        record = records[
            record_id
        ]

        values = record[
            "normalized_time"
        ]["values"]

        actual = (
            values[0]["date"],
            values[0]["time"],
            values[-1]["date"],
            values[-1]["time"],
        )

        if actual != expected:
            errors.append(
                f"{record_id} normalized "
                f"time mismatch: {actual}"
            )

    return errors


def validate_query_v2():
    document = load_json(
        FILES["query_v2"]
    )

    errors = []

    expected_top = {
        (
            "When do graduate students "
            "select rooms?"
        ): "row_02",

        (
            "female undergraduate "
            "room selection"
        ): "row_04",

        (
            "male undergraduate "
            "room selection"
        ): "row_05",

        (
            "waiting list II "
            "room announcement"
        ): "row_12",

        (
            "3+2 program "
            "room announcement"
        ): "row_15",

        (
            "3+2 program "
            "pay deposit"
        ): "row_16",

        (
          "When does the dormitory open?"
        ): "row_18",

         
    }

    available = {
        item["query"]: item
        for item in document[
            "test_queries"
        ]
    }

    for query, expected_id in (
        expected_top.items()
    ):
        if query not in available:
            errors.append(
                f"Missing query test: {query}"
            )

            continue

        results = available[
            query
        ]["results"]

        if not results:
            errors.append(
                f"No results for query: {query}"
            )

            continue

        actual_id = results[0][
            "record_id"
        ]

        if actual_id != expected_id:
            errors.append(
                f"Query '{query}' returned "
                f"{actual_id}, expected "
                f"{expected_id}."
            )

    return errors


def validate_multi_record_answers():
    document = load_json(
        FILES["grounded_answers_v2"]
    )

    results = {
        item["query"]: item
        for item in document["results"]
    }

    errors = []

    who_query = (
        "Who needs to pay $2000?"
    )

    expected_who = {
        "row_03",
        "row_06",
        "row_09",
        "row_13",
        "row_16",
    }

    if who_query not in results:
        errors.append(
            "Missing $2000 multi-record test."
        )

    else:
        actual = {
            item["record_id"]
            for item
            in results[
                who_query
            ]["records_used"]
        }

        if actual != expected_who:
            errors.append(
                "$2000 multi-record answer "
                "uses unexpected records."
            )

    program_query = (
        "What happens for the "
        "3+2 Program?"
    )

    expected_program = {
        "row_14",
        "row_15",
        "row_16",
    }

    if program_query not in results:
        errors.append(
            "Missing 3+2 summary test."
        )

    else:
        actual = {
            item["record_id"]
            for item
            in results[
                program_query
            ]["records_used"]
        }

        if actual != expected_program:
            errors.append(
                "3+2 category summary "
                "uses unexpected records."
            )

    return errors


def print_validation(
    title,
    errors,
):
    print()
    print("-" * 72)
    print(title)
    print("-" * 72)

    if not errors:
        print("[PASS]")

        return True

    print("[FAIL]")

    for error in errors:
        print(
            f"  - {error}"
        )

    return False


def main():
    print("=" * 72)
    print(
        "EXPERIMENT 35: "
        "END-TO-END PIPELINE VALIDATION"
    )
    print("=" * 72)

    missing = check_files()

    if missing:
        print()
        print(
            "Cannot continue because "
            "required artifacts are missing:"
        )

        for name in missing:
            print(
                f"  - {name}"
            )

        raise SystemExit(1)

    checks = [
        (
            "Logical table",
            validate_logical_table(),
        ),
        (
            "Structured records",
            validate_records(),
        ),
        (
            "Recovered OCR cells",
            validate_recovered_cells(),
        ),
        (
            "Time normalization",
            validate_time_normalization(),
        ),
        (
            "Improved retrieval",
            validate_query_v2(),
        ),
        (
            "Multi-record grounded answers",
            validate_multi_record_answers(),
        ),
    ]

    passed = 0

    for title, errors in checks:
        if print_validation(
            title,
            errors,
        ):
            passed += 1

    print()
    print("=" * 72)
    print("PIPELINE SUMMARY")
    print("=" * 72)

    print(
        f"Validation groups passed: "
        f"{passed}/{len(checks)}"
    )

    if passed == len(checks):
        print(
            "STATUS: PIPELINE CHECKPOINT PASSED"
        )
    else:
        print(
            "STATUS: PIPELINE CHECKPOINT FAILED"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()