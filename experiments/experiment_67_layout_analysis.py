"""Experiment 67: Generic document layout analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parsing.layout_analyzer import analyze_document_layout


DEFAULT_CASES = {
    "receipt": Path("results/gui_v4/single/receipt/unified_raw_ocr.json"),
    "pic13": Path("results/gui_v4/single/pic13/unified_raw_ocr.json"),
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
    return json.loads(path.read_text(encoding="utf-8"))


def print_layout(name: str, path: Path) -> bool:
    if not path.is_file():
        print(f"[MISSING] {name}: {path}")
        return False

    document = load_json(path)
    layout = analyze_document_layout(document)

    print()
    print("=" * 100)
    print(f"DOCUMENT: {name}")
    print(f"PATH: {path}")
    print(f"DOCUMENT ID: {layout.get('document_id')}")
    print(
        f"REGIONS: {layout['region_count']} | "
        f"ROWS: {layout['row_count']} | "
        f"BLOCKS: {layout['block_count']}"
    )
    print("=" * 100)

    for region in layout["regions"]:
        print(
            f"ID={str(region['region_id']):>3} "
            f"row={region['row_id']:>2} "
            f"block={region['block_id']:>2} "
            f"col={region['column_position']:.3f} "
            f"L={region['neighbors_left']} "
            f"R={region['neighbors_right']} "
            f"A={region['neighbors_above']} "
            f"B={region['neighbors_below']} "
            f"text={region['text']!r}"
        )

    return True


def run_default_suite() -> int:
    print("=" * 100)
    print("EXPERIMENT 67: GENERIC DOCUMENT LAYOUT ANALYSIS")
    print("=" * 100)
    print(
        "Goal: inspect spatial rows, blocks, column positions, and neighbors "
        "without semantic table/form classification."
    )

    passed = 0
    for name, path in DEFAULT_CASES.items():
        if print_layout(name, path):
            passed += 1

    print()
    print("=" * 100)
    print(f"RESULT: {passed}/{len(DEFAULT_CASES)} documents analyzed")
    print("=" * 100)
    return 0 if passed == len(DEFAULT_CASES) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Experiment 67: generic OCR layout analysis."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional unified_raw_ocr.json to analyze instead of the default corpus.",
    )
    args = parser.parse_args()

    if args.input_json is not None:
        return 0 if print_layout("custom", args.input_json) else 1

    return run_default_suite()


if __name__ == "__main__":
    raise SystemExit(main())
