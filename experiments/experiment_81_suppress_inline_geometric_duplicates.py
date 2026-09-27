"""Experiment 81: inline fields must not spawn geometric field candidates.

Uses synthetic layout regions; no OCR, Groq, GUI, or network access.
"""
from __future__ import annotations

from unittest.mock import patch

from parsing.form_reconstructor import reconstruct_form_fields


def region(number: int, row: int, text: str, x: int, y: int, width: int = 100) -> dict:
    return {
        "region_id": number,
        "row_id": row,
        "text": text,
        "bbox": {"x": x, "y": y, "width": width, "height": 20},
    }


def main() -> int:
    layout = {
        "document_id": "exp81_synthetic",
        "region_count": 6,
        "row_count": 5,
        "regions": [
            region(1, 1, "DATE: 08/24/2023", 10, 10),
            region(2, 2, "14:48", 10, 40),
            region(3, 3, "NAME", 10, 70, 60),
            region(4, 3, "Jane Smith", 95, 70),
            region(5, 4, "ADDRESS", 10, 100),
            region(6, 5, "101 Pine Lane", 10, 130),
        ],
    }
    with patch("parsing.form_reconstructor.analyze_document_layout_refined", return_value=layout):
        fields = reconstruct_form_fields({"document_id": "exp81_synthetic"})["fields"]

    for field in fields:
        print(f"{field['relationship']}: {field['label']} -> {field['value']}")

    checks = [
        ("inline date retained once", sum(f["label"] == "DATE" and f["value"] == "08/24/2023" for f in fields) == 1),
        ("inline region has no geometric candidate", not any(f["label"].startswith("DATE:") for f in fields)),
        ("standalone name field retained", any(f["label"] == "NAME" and f["value"] == "Jane Smith" and f["relationship"] == "same_row_right" for f in fields)),
        ("stacked address retained", any(f["label"] == "ADDRESS" and f["value"] == "101 Pine Lane" and f["relationship"] == "next_row_below" for f in fields)),
        ("all expected fields accounted for", len(fields) == 3),
    ]
    for label, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
    passed = sum(ok for _, ok in checks)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
