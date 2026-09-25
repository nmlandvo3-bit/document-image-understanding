"""Experiment 69: automatic table reconstruction, revision 3."""

from __future__ import annotations

import json
from pathlib import Path

from parsing.table_reconstructor import reconstruct_candidate_tables


CASES = {
    "receipt": Path(
        "results/gui_v4/single/receipt/unified_raw_ocr.json"
    ),
    "nhi": Path(
        "results/gui_v4/single/"
        "WhatsApp Image 2026-09-22 at 22.21.24/"
        "unified_raw_ocr.json"
    ),
    "synthetic": Path(
        "results/gui_v4/single/"
        "ChatGPT Image Sep 24, 2026, 06_13_58 PM/"
        "unified_raw_ocr.json"
    ),
}


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def same_row_pair(
    result: dict,
    left_text: str,
    right_text: str,
) -> bool:
    left = left_text.casefold()
    right = right_text.casefold()

    for table in result.get("tables", []):
        for row in table.get("rows", []):
            cells = [
                str(
                    cell.get("text", "")
                ).casefold()
                for cell in row.get(
                    "cells",
                    [],
                )
            ]

            if (
                any(left in cell for cell in cells)
                and any(
                    right in cell
                    for cell in cells
                )
            ):
                return True

    return False


def same_column_pair(
    result: dict,
    upper_text: str,
    lower_text: str,
) -> bool:
    upper = upper_text.casefold()
    lower = lower_text.casefold()

    for table in result.get("tables", []):
        columns = {}

        for row in table.get("rows", []):
            for cell in row.get(
                "cells",
                [],
            ):
                text = str(
                    cell.get(
                        "text",
                        "",
                    )
                ).casefold()

                if text:
                    columns.setdefault(
                        int(
                            cell[
                                "column_index"
                            ]
                        ),
                        [],
                    ).append(text)

        for texts in columns.values():
            if (
                any(
                    upper in text
                    for text in texts
                )
                and any(
                    lower in text
                    for text in texts
                )
            ):
                return True

    return False


def table_has_rows(
    result: dict,
    required_rows: list[int],
) -> bool:
    wanted = set(required_rows)

    return any(
        wanted.issubset(
            set(
                table.get(
                    "row_ids",
                    [],
                )
            )
        )
        for table in result.get(
            "tables",
            [],
        )
    )


def print_result(
    name: str,
    result: dict,
) -> None:
    print()
    print("=" * 100)
    print(f"DOCUMENT: {name}")
    print(
        f"regions={result['region_count']} | "
        f"rows={result['row_count']} | "
        f"candidate_tables={result['candidate_table_count']}"
    )
    print("=" * 100)

    for table in result.get(
        "tables",
        [],
    ):
        print()
        print(
            f"{table['table_id']} | "
            f"seed={table.get('seed_row_ids')} | "
            f"rows={table['row_ids']} | "
            f"columns={table['column_count']} | "
            f"centers={table['column_positions']}"
        )

        for row in table.get(
            "rows",
            [],
        ):
            texts = [
                cell.get(
                    "text",
                    "",
                )
                for cell in row.get(
                    "cells",
                    [],
                )
            ]

            print(
                f"  row "
                f"{row['row_id']:>2}: "
                f"{texts}"
            )


def main() -> int:
    print("=" * 100)
    print(
        "EXPERIMENT 69: AUTOMATIC TABLE "
        "RECONSTRUCTION - REVISION 3"
    )
    print("=" * 100)

    results = {}

    for name, path in CASES.items():
        if not path.is_file():
            print(
                f"[MISSING] {name}: {path}"
            )
            return 1

        result = reconstruct_candidate_tables(
            load_json(path)
        )

        results[name] = result
        print_result(
            name,
            result,
        )

    checks = [
        (
            "receipt has candidate table",
            results[
                "receipt"
            ][
                "candidate_table_count"
            ] >= 1,
        ),
        (
            "receipt contains TOTAL / 310",
            same_row_pair(
                results["receipt"],
                "TOTAL",
                "310",
            ),
        ),
        (
            "receipt contains CASH / 500",
            same_row_pair(
                results["receipt"],
                "CASH",
                "500",
            ),
        ),
        (
            "synthetic has candidate table",
            results[
                "synthetic"
            ][
                "candidate_table_count"
            ] >= 1,
        ),
        (
            "synthetic contains Name / Alex Carter",
            same_row_pair(
                results["synthetic"],
                "Name",
                "Alex Carter",
            ),
        ),
        (
            "synthetic contains Date of Birth / 03/14/1992",
            same_row_pair(
                results["synthetic"],
                "Date of Birth",
                "03/14/1992",
            ),
        ),
        (
            "NHI has candidate table",
            results[
                "nhi"
            ][
                "candidate_table_count"
            ] >= 1,
        ),
        (
            "NHI candidate includes rows 7, 8, 9",
            table_has_rows(
                results["nhi"],
                [7, 8, 9],
            ),
        ),
        (
            "NHI keeps Payment Date above 115/09/21",
            same_column_pair(
                results["nhi"],
                "Payment Date",
                "115/09/21",
            ),
        ),
        (
            "NHI keeps Card Pick-up above 0055",
            same_column_pair(
                results["nhi"],
                "Card Pick-up",
                "0055",
            ),
        ),
        (
            "synthetic amount table does not absorb customer section",
            not any(
                13 in table.get(
                    "row_ids",
                    [],
                )
                and 8 in table.get(
                    "row_ids",
                    [],
                )
                for table in results[
                    "synthetic"
                ].get(
                    "tables",
                    [],
                )
            ),
        ),
    ]

    print()
    print("-" * 100)
    print(
        "TABLE RECONSTRUCTION CHECKS"
    )
    print("-" * 100)

    passed = 0

    for index, (
        label,
        ok,
    ) in enumerate(
        checks,
        start=1,
    ):
        print(
            f"[{'PASS' if ok else 'FAIL'}] "
            f"{index:02d}. {label}"
        )

        if ok:
            passed += 1

    print()
    print("=" * 100)
    print(
        f"RESULT: {passed}/{len(checks)} "
        "table reconstruction checks passed"
    )
    print("=" * 100)

    return (
        0
        if passed == len(checks)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
