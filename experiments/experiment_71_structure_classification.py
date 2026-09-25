"""Experiment 71: mixed document structure classification, revision 4."""

from __future__ import annotations

import json
from pathlib import Path

from parsing.structure_classifier import classify_document_structure


CASES = {
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


def has_type(result: dict, structure_type: str) -> bool:
    return structure_type in result.get("structure_types_present", [])


def region_type(result: dict, region_id: int) -> str | None:
    for region in result.get("regions", []):
        if region.get("region_id") == region_id:
            return region.get("structure_type")
    return None


def print_result(name: str, result: dict) -> None:
    print()
    print("=" * 105)
    print(f"DOCUMENT: {name}")
    print(
        f"regions={result['region_count']} | "
        f"rows={result['row_count']} | "
        f"tables={result['candidate_table_count']} | "
        f"fields={result['field_candidate_count']} | "
        f"sections={result['section_count']}"
    )
    print(
        f"types={result['structure_types_present']} | "
        f"counts={result['structure_counts']}"
    )
    print("=" * 105)

    for section in result.get("sections", []):
        print(
            f"{section['section_id']} | "
            f"type={section['structure_type']:<10} | "
            f"rows={section['row_start']}-{section['row_end']}"
        )


def main() -> int:
    print("=" * 105)
    print("EXPERIMENT 71: MIXED DOCUMENT STRUCTURE CLASSIFICATION - REVISION 4")
    print("=" * 105)

    results = {}

    for name, path in CASES.items():
        if not path.is_file():
            print(f"[MISSING] {name}: {path}")
            return 1

        result = classify_document_structure(load_json(path))
        results[name] = result
        print_result(name, result)

    checks = [
        ("receipt contains table structure", has_type(results["receipt"], "table")),
        ("receipt title is heading", region_type(results["receipt"], 1) == "heading"),
        ("pic13 contains form structure", has_type(results["pic13"], "form")),
        ("pic13 contains paragraph structure", has_type(results["pic13"], "paragraph")),
        ("pic13 authorization title is heading", region_type(results["pic13"], 5) == "heading"),
        ("NHI contains table structure", has_type(results["nhi"], "table")),
        (
            "NHI also contains paragraph/free-text outside table",
            has_type(results["nhi"], "paragraph")
            or has_type(results["nhi"], "free_text"),
        ),
        ("synthetic contains table structure", has_type(results["synthetic"], "table")),
        ("synthetic contains form structure", has_type(results["synthetic"], "form")),
        ("synthetic payment title is heading", region_type(results["synthetic"], 1) == "heading"),
        ("synthetic customer name is form", region_type(results["synthetic"], 24) == "form"),
        (
            "mixed structure does not collapse synthetic to one type",
            len(results["synthetic"]["structure_types_present"]) >= 3,
        ),
        ("synthetic payment title is not form", region_type(results["synthetic"], 1) != "form"),
        ("receipt total remains table", region_type(results["receipt"], 21) == "table"),
        ("receipt total value remains table", region_type(results["receipt"], 22) == "table"),
        ("receipt cash remains table", region_type(results["receipt"], 24) == "table"),
        ("receipt change remains table", region_type(results["receipt"], 26) == "table"),
    ]

    print()
    print("-" * 105)
    print("STRUCTURE CLASSIFICATION CHECKS")
    print("-" * 105)

    passed = 0
    for index, (label, ok) in enumerate(checks, start=1):
        print(f"[{'PASS' if ok else 'FAIL'}] {index:02d}. {label}")
        if ok:
            passed += 1

    print()
    print("=" * 105)
    print(
        f"RESULT: {passed}/{len(checks)} "
        "structure classification checks passed"
    )
    print("=" * 105)

    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
