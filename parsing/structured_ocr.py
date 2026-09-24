
"""
Structured OCR Parser
Intelligent Document Parsing Engine

Converts the existing OCR results.csv output into structured JSON.

Input CSV columns:
    region, text, confidence, meaningful, x, y, w, h

This module does not run OCR or modify the original document.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = {
    "region",
    "text",
    "confidence",
    "meaningful",
    "x",
    "y",
    "w",
    "h",
}


def parse_boolean(value: str) -> bool:
    """Convert a CSV boolean value to a Python boolean."""

    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(f"Invalid meaningful value: {value!r}")


def parse_region(row: dict, csv_row_number: int) -> dict:
    """Convert one OCR CSV row into a structured region."""

    try:
        region_id = int(row["region"])
        text = row["text"]
        confidence = float(row["confidence"])
        meaningful = parse_boolean(row["meaningful"])

        x = int(row["x"])
        y = int(row["y"])
        width = int(row["w"])
        height = int(row["h"])

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid data in CSV row {csv_row_number}: {exc}"
        ) from exc

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"CSV row {csv_row_number}: "
            f"confidence must be between 0 and 1."
        )

    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"CSV row {csv_row_number}: invalid bounding box."
        )

    return {
        "region_id": region_id,
        "source_order": csv_row_number - 1,
        "text": text,
        "confidence": confidence,
        "meaningful": meaningful,
        "bounding_box": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "x2": x + width,
            "y2": y + height,
        },
    }


def parse_ocr_csv(input_csv: Path) -> dict:
    """Read an existing OCR CSV and return structured document data."""

    if not input_csv.is_file():
        raise FileNotFoundError(
            f"OCR CSV file does not exist: {input_csv}"
        )

    regions = []

    with input_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("The OCR CSV is empty or has no header.")

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                "Missing required CSV columns: "
                + ", ".join(sorted(missing_columns))
            )

        for csv_row_number, row in enumerate(reader, start=1):
            region = parse_region(row, csv_row_number)
            regions.append(region)

    meaningful_regions = [
        region
        for region in regions
        if region["meaningful"]
    ]

    document = {
        "schema_version": "1.0",
        "document_type": "unclassified",
        "source": {
            "ocr_csv": str(input_csv.resolve()),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "statistics": {
            "total_regions": len(regions),
            "meaningful_regions": len(meaningful_regions),
            "non_meaningful_regions": (
                len(regions) - len(meaningful_regions)
            ),
        },
        "regions": regions,
        "meaningful_text": "\n".join(
            region["text"]
            for region in meaningful_regions
        ),
    }

    return document


def save_document_json(document: dict, output_json: Path) -> None:
    """Save the structured document as a UTF-8 JSON file."""

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(
            document,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an existing OCR CSV into structured JSON."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the existing OCR results.csv file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where structured JSON will be saved.",
    )

    args = parser.parse_args()

    document = parse_ocr_csv(args.input)

    save_document_json(
        document=document,
        output_json=args.output,
    )

    print("=" * 60)
    print("STRUCTURED OCR PARSING COMPLETE")
    print("=" * 60)

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    print(
        "Total OCR regions: "
        f"{document['statistics']['total_regions']}"
    )

    print(
        "Meaningful regions: "
        f"{document['statistics']['meaningful_regions']}"
    )

    print(
        "Non-meaningful regions: "
        f"{document['statistics']['non_meaningful_regions']}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()