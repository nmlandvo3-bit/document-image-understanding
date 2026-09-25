"""Experiment 70: automatic form reconstruction, revision 4."""

from __future__ import annotations

import json
from pathlib import Path

from parsing.form_reconstructor import reconstruct_form_fields


CASES = {
    "pic13": Path("results/gui_v4/single/pic13/unified_raw_ocr.json"),
    "synthetic": Path(
        "results/gui_v4/single/"
        "ChatGPT Image Sep 24, 2026, 06_13_58 PM/"
        "unified_raw_ocr.json"
    ),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def has_pair(result: dict, label_fragment: str, value_fragment: str) -> bool:
    label_fragment = label_fragment.casefold()
    value_fragment = value_fragment.casefold()

    return any(
        label_fragment in str(field.get("label", "")).casefold()
        and value_fragment in str(field.get("value", "")).casefold()
        for field in result.get("fields", [])
    )


def print_result(name: str, result: dict) -> None:
    print()
    print("=" * 100)
    print(f"DOCUMENT: {name}")
    print(
        f"regions={result['region_count']} | "
        f"rows={result['row_count']} | "
        f"field_candidates={result['field_candidate_count']}"
    )
    print("=" * 100)

    for field in result.get("fields", []):
        print(
            f"{field['field_id']} | "
            f"{field['relationship']:<16} | "
            f"score={field['geometry_score']:>8} | "
            f"label={field['label']!r} -> "
            f"value={field['value']!r} | "
            f"regions={field['label_region_id']}->{field['value_region_id']}"
        )


def main() -> int:
    print("=" * 100)
    print("EXPERIMENT 70: AUTOMATIC FORM RECONSTRUCTION - REVISION 4")
    print("=" * 100)

    results = {}

    for name, path in CASES.items():
        if not path.is_file():
            print(f"[MISSING] {name}: {path}")
            return 1

        result = reconstruct_form_fields(load_json(path))
        results[name] = result
        print_result(name, result)

    checks = [
        ("pic13 has field candidates", results["pic13"]["field_candidate_count"] >= 1),
        ("pic13 Name -> Alex Carter", has_pair(results["pic13"], "Name", "Alex Carter")),
        ("pic13 Date of Birth -> 03/14/1992", has_pair(results["pic13"], "Date of Birth", "03/14/1992")),
        ("pic13 Medicare ID -> 1EG4-TE5-MK72", has_pair(results["pic13"], "Medicare Identification Number", "1EG4-TE5-MK72")),
        ("pic13 Street Address -> 101 Sample Ridge Drive", has_pair(results["pic13"], "Street Address", "101 Sample Ridge Drive")),
        ("pic13 City -> Austin", has_pair(results["pic13"], "City", "Austin")),
        ("pic13 State -> TX", has_pair(results["pic13"], "State", "TX")),
        ("pic13 Zip code -> 78701", has_pair(results["pic13"], "Zip code", "78701")),
        ("synthetic Name -> Alex Carter", has_pair(results["synthetic"], "Name", "Alex Carter")),
        ("synthetic Date of Birth -> 03/14/1992", has_pair(results["synthetic"], "Date of Birth", "03/14/1992")),
        ("synthetic Card Pick-up Number -> 0055", has_pair(results["synthetic"], "Card Pick-up Number", "0055")),
        ("synthetic TOTAL does not map to CASH", not has_pair(results["synthetic"], "TOTAL", "CASH")),
        ("synthetic Name does not map to Date of Birth", not has_pair(results["synthetic"], "Name", "Date of Birth")),
        ("time region does not become a form label", not any(
            str(field.get("label", "")).strip() == "15:01:23"
            for field in results["synthetic"].get("fields", [])
        )),
        ("address value does not become a form label", not any(
            str(field.get("label", "")).startswith("101 Sample Ridge Drive")
            for field in results["pic13"].get("fields", [])
        )),
    ]

    print()
    print("-" * 100)
    print("FORM RECONSTRUCTION CHECKS")
    print("-" * 100)

    passed = 0
    for index, (label, ok) in enumerate(checks, start=1):
        print(f"[{'PASS' if ok else 'FAIL'}] {index:02d}. {label}")
        if ok:
            passed += 1

    print()
    print("=" * 100)
    print(
        f"RESULT: {passed}/{len(checks)} "
        "form reconstruction checks passed"
    )
    print("=" * 100)

    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
