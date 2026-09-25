"""Experiment 69 diagnostic: inspect NHI rows immediately after candidate table 1.

No files are modified. No OCR, LLM, or network calls.
"""

from __future__ import annotations

import json
from pathlib import Path

from parsing.layout_analyzer_refined import analyze_document_layout_refined


PATH = Path(
    "results/gui_v4/single/"
    "WhatsApp Image 2026-09-22 at 22.21.24/"
    "unified_raw_ocr.json"
)

CENTERS = [0.131, 0.324, 0.518, 0.760, 0.912]
MAX_DISTANCE = 0.12


def nearest(position: float):
    ranked = sorted(
        (abs(position - center), index, center)
        for index, center in enumerate(CENTERS)
    )
    return ranked[0]


def main() -> int:
    document = json.loads(PATH.read_text(encoding="utf-8"))
    layout = analyze_document_layout_refined(document)

    rows = {}
    for region in layout["regions"]:
        rows.setdefault(region["row_id"], []).append(region)

    print("=" * 100)
    print("EXPERIMENT 69 DIAGNOSTIC: NHI REFINED ROWS 5-12")
    print("=" * 100)
    print(f"Candidate centers: {CENTERS}")
    print(f"Max assignment distance: {MAX_DISTANCE}")

    for row_id in range(5, 13):
        items = sorted(
            rows.get(row_id, []),
            key=lambda r: r["column_position"],
        )

        print()
        print(f"ROW {row_id}")

        aligned_columns = set()

        for region in items:
            position = float(region["column_position"])
            distance, column_index, center = nearest(position)
            aligned = distance <= MAX_DISTANCE

            if aligned:
                aligned_columns.add(column_index)

            print(
                f"  id={region['region_id']:>3} "
                f"col={position:.3f} "
                f"nearest=C{column_index}@{center:.3f} "
                f"distance={distance:.3f} "
                f"aligned={aligned} "
                f"text={region['text']!r}"
            )

        print(
            f"  aligned_column_count={len(aligned_columns)} "
            f"aligned_columns={sorted(aligned_columns)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
