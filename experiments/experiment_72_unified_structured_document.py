"""Experiment 72: unified structured-document builder.

Builds structured documents in memory from the four validated OCR documents.
No source JSON files are overwritten and no network calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

from parsing.unified_structured_document_builder import (
    build_unified_structured_document,
)


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
    return json.loads(path.read_text(encoding="utf-8"))


def has_field(
    result: dict,
    label_fragment: str,
    value_fragment: str,
) -> bool:
    label_fragment = label_fragment.casefold()
    value_fragment = value_fragment.casefold()

    return any(
        label_fragment in str(field.get("label", "")).casefold()
        and value_fragment in str(field.get("value", "")).casefold()
        for field in result.get("fields", [])
    )


def has_record_row_texts(
    result: dict,
    *fragments: str,
) -> bool:
    wanted = [fragment.casefold() for fragment in fragments]

    for record in result.get("records", []):
        texts = [
            str(cell.get("text", "")).casefold()
            for cell in record.get("cells", [])
        ]

        if all(
            any(fragment in text for text in texts)
            for fragment in wanted
        ):
            return True

    return False


def same_table_column(
    result: dict,
    upper_fragment: str,
    lower_fragment: str,
) -> bool:
    upper_fragment = upper_fragment.casefold()
    lower_fragment = lower_fragment.casefold()

    by_table: dict[str, dict[int, list[str]]] = {}

    for record in result.get("records", []):
        table_id = str(record.get("table_id"))
        table_columns = by_table.setdefault(table_id, {})

        for cell in record.get("cells", []):
            text = str(cell.get("text", "")).casefold()
            if not text:
                continue

            table_columns.setdefault(
                int(cell["column_index"]),
                [],
            ).append(text)

    for columns in by_table.values():
        for texts in columns.values():
            if (
                any(upper_fragment in text for text in texts)
                and any(lower_fragment in text for text in texts)
            ):
                return True

    return False


def print_summary(name: str, result: dict) -> None:
    print()
    print("=" * 110)
    print(f"DOCUMENT: {name}")
    print(
        f"type={result['document_type']} | "
        f"regions={result['region_count']} | "
        f"rows={result['row_count']} | "
        f"fields={len(result['fields'])} | "
        f"records={len(result['records'])} | "
        f"controls={len(result['controls'])} | "
        f"sections={len(result['sections'])} | "
        f"free_text={len(result['free_text'])} | "
        f"evidence={len(result['evidence'])}"
    )
    print(
        f"structures={result['structure_types_present']} | "
        f"metadata={result['parsing_metadata']}"
    )
    print("=" * 110)


def main() -> int:
    print("=" * 110)
    print("EXPERIMENT 72: UNIFIED STRUCTURED-DOCUMENT BUILDER")
    print("=" * 110)
    print(
        "Goal: combine tables, form fields, sections, controls, free text, and original OCR evidence "
        "without losing provenance or resolving ambiguity."
    )

    source_docs = {}
    results = {}

    for name, path in CASES.items():
        if not path.is_file():
            print(f"[MISSING] {name}: {path}")
            return 1

        source = load_json(path)
        source_docs[name] = source

        result = build_unified_structured_document(source)
        results[name] = result
        print_summary(name, result)

    checks = [
        (
            "all documents use mixed_structured_ocr",
            all(
                result.get("document_type") == "mixed_structured_ocr"
                for result in results.values()
            ),
        ),
        (
            "receipt preserves original OCR evidence count",
            len(results["receipt"]["evidence"])
            == len(source_docs["receipt"].get("evidence", [])),
        ),
        (
            "receipt table row preserves TOTAL / 310",
            has_record_row_texts(results["receipt"], "TOTAL", "310"),
        ),
        (
            "pic13 Name -> Alex Carter field preserved",
            has_field(results["pic13"], "Name", "Alex Carter"),
        ),
        (
            "pic13 Medicare ID field preserved",
            has_field(
                results["pic13"],
                "Medicare Identification Number",
                "1EG4-TE5-MK72",
            ),
        ),
        (
            "pic13 control evidence preserved",
            len(results["pic13"]["controls"]) >= 1,
        ),
        (
            "NHI Payment Date / date remain same reconstructed column",
            same_table_column(
                results["nhi"],
                "Payment Date",
                "115/09/21",
            ),
        ),
        (
            "NHI Card Pick-up / 0055 remain same reconstructed column",
            same_table_column(
                results["nhi"],
                "Card Pick-up",
                "0055",
            ),
        ),
        (
            "synthetic Name -> Alex Carter field preserved",
            has_field(results["synthetic"], "Name", "Alex Carter"),
        ),
        (
            "synthetic DOB field preserved",
            has_field(
                results["synthetic"],
                "Date of Birth",
                "03/14/1992",
            ),
        ),
        (
            "synthetic TOTAL row preserves both 310 and 500 candidates",
            has_record_row_texts(
                results["synthetic"],
                "TOTAL",
                "310",
                "500",
            ),
        ),
        (
            "synthetic ambiguity policy does not claim semantic resolution",
            results["synthetic"]["parsing_metadata"]["ambiguity_policy"]
            == "preserve_observed_candidates_without_semantic_resolution",
        ),
        (
            "all generated fields retain evidence references",
            all(
                field.get("evidence_item_ids")
                for result in results.values()
                for field in result.get("fields", [])
            ),
        ),
        (
            "all table cells with source regions retain evidence references",
            all(
                bool(cell.get("evidence_item_ids"))
                for result in results.values()
                for record in result.get("records", [])
                for cell in record.get("cells", [])
                if cell.get("region_ids")
            ),
        ),
    ]

    print()
    print("-" * 110)
    print("UNIFIED DOCUMENT CHECKS")
    print("-" * 110)

    passed = 0

    for index, (label, ok) in enumerate(checks, start=1):
        print(f"[{'PASS' if ok else 'FAIL'}] {index:02d}. {label}")
        if ok:
            passed += 1

    print()
    print("=" * 110)
    print(
        f"RESULT: {passed}/{len(checks)} "
        "unified-document checks passed"
    )
    print("=" * 110)

    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
