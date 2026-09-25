"""Experiment 68: refined row segmentation comparison.

Compares Experiment 67 row grouping with a stricter representative-based
segmentation. No semantic table/form reconstruction is performed yet.
"""

from __future__ import annotations

import json
from pathlib import Path

from parsing.layout_analyzer import analyze_document_layout
from parsing.layout_analyzer_refined import analyze_document_layout_refined


CASES = {
    "receipt": Path(
        "results/gui_v4/single/receipt/unified_raw_ocr.json"
    ),
    "pic13": Path(
        "results/gui_v4/single/pic13/unified_raw_ocr.json"
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


def row_lookup(layout: dict) -> dict:
    return {
        region["region_id"]: region["row_id"]
        for region in layout.get(
            "regions",
            [],
        )
    }


def same_row(
    lookup: dict,
    *region_ids,
) -> bool:
    rows = {
        lookup.get(region_id)
        for region_id in region_ids
    }

    return (
        None not in rows
        and len(rows) == 1
    )


def different_rows(
    lookup: dict,
    *region_ids,
) -> bool:
    rows = [
        lookup.get(region_id)
        for region_id in region_ids
    ]

    return (
        None not in rows
        and len(set(rows)) == len(rows)
    )


def main() -> int:
    print(
        "=" * 100
    )
    print(
        "EXPERIMENT 68: REFINED ROW SEGMENTATION"
    )
    print(
        "=" * 100
    )

    layouts = {}

    for name, path in CASES.items():
        if not path.is_file():
            print(
                f"[MISSING] {name}: {path}"
            )

            return 1

        document = load_json(path)

        old = analyze_document_layout(
            document
        )

        new = analyze_document_layout_refined(
            document
        )

        layouts[name] = new

        print()
        print(
            f"{name.upper():10s} | "
            f"regions={new['region_count']:>3} | "
            f"rows old={old['row_count']:>2} "
            f"-> refined={new['row_count']:>2} | "
            f"blocks old={old['block_count']:>2} "
            f"-> refined={new['block_count']:>2}"
        )

    checks = []

    receipt_rows = row_lookup(
        layouts["receipt"]
    )

    # Clean receipt label/value pairs should remain visual rows.
    for label, ids in (
        (
            "receipt date/time",
            (4, 5),
        ),
        (
            "receipt item Black Tea",
            (7, 8),
        ),
        (
            "receipt subtotal",
            (17, 18),
        ),
        (
            "receipt total",
            (21, 22),
        ),
        (
            "receipt cash",
            (24, 23),
        ),
        (
            "receipt change",
            (26, 25),
        ),
    ):
        checks.append(
            (
                label,
                same_row(
                    receipt_rows,
                    *ids,
                ),
            )
        )

    synthetic_rows = row_lookup(
        layouts["synthetic"]
    )

    for label, ids in (
        (
            "synthetic payment date",
            (3, 2),
        ),
        (
            "synthetic alternative date",
            (5, 6),
        ),
        (
            "synthetic reference first value",
            (8, 7),
        ),
        (
            "synthetic total ambiguity",
            (15, 16, 17),
        ),
        (
            "synthetic customer name",
            (23, 24),
        ),
        (
            "synthetic date of birth",
            (25, 26),
        ),
        (
            "synthetic card pickup",
            (27, 28),
        ),
    ):
        checks.append(
            (
                label,
                same_row(
                    synthetic_rows,
                    *ids,
                ),
            )
        )

    pic13_rows = row_lookup(
        layouts["pic13"]
    )

    # These are separate header lines in the form and should not collapse
    # into one row after refinement.
    checks.append(
        (
            "pic13 department header lines separated",
            different_rows(
                pic13_rows,
                3,
                4,
            ),
        )
    )

    print()
    print(
        "-" * 100
    )
    print(
        "ROW INVARIANT CHECKS"
    )
    print(
        "-" * 100
    )

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
    print(
        "=" * 100
    )
    print(
        f"RESULT: {passed}/{len(checks)} "
        "row invariants passed"
    )
    print(
        "=" * 100
    )

    # Diagnostic output for pic13's early header rows.
    print()
    print(
        "PIC13 EARLY HEADER ROWS:"
    )

    for region in layouts[
        "pic13"
    ][
        "regions"
    ]:
        if region[
            "region_id"
        ] in {
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        }:
            print(
                f"  ID={region['region_id']:>2} "
                f"row={region['row_id']:>2} "
                f"text={region['text']!r}"
            )

    return (
        0
        if passed == len(checks)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
