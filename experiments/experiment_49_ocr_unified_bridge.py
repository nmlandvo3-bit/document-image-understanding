"""Experiment 49: convert existing OCR CSV into a conservative unified document.

No OCR is run. Text regions are NOT interpreted as table cells, form fields,
checkboxes, or semantic records. This is a raw-OCR Q&A bridge only.

Run from project root:
    python -m experiments.experiment_49_ocr_unified_bridge --input-csv ... --output-json ...
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.structured_ocr import parse_ocr_csv


def make_unified_ocr_document(
    csv_path: Path,
    *,
    document_id: str,
    source_image: str | None = None,
) -> dict:
    """Preserve OCR text and bounding boxes without inventing semantics."""
    document_id = document_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", document_id):
        raise ValueError("document_id must be 1-120 letters, digits, _, . or - (start alphanumeric).")

    parsed = parse_ocr_csv(Path(csv_path))
    regions = parsed["regions"]
    if not regions:
        raise ValueError("The OCR CSV contains no regions.")

    seen = set()
    evidence = []
    free_text = []
    for region in regions:
        rid = region["region_id"]
        if rid in seen:
            raise ValueError(f"Duplicate OCR region ID: {rid}")
        seen.add(rid)
        bbox = region["bounding_box"]
        item = {
            "evidence_type": "ocr_region",
            "region_id": rid,
            "text": region["text"],
            "bbox": bbox,
            "confidence": region["confidence"],
            "meaningful": region["meaningful"],
            "source_order": region["source_order"],
        }
        evidence.append(item)
        if region["meaningful"] and region["text"].strip():
            free_text.append({
                "source_control_id": None,
                "source_control_label": f"OCR region {rid}",
                "text_region_id": rid,
                "text": region["text"],
                "bbox": bbox,
                "confidence": region["confidence"],
            })

    if not free_text:
        raise ValueError("No meaningful OCR text; cannot create a searchable document.")

    return {
        "document_id": document_id,
        "schema_version": "1.0-unified-experimental",
        "document_type": "unclassified",
        "source": {
            "image": source_image,
            "ocr_csv": str(Path(csv_path).resolve()),
        },
        "sections": [],
        "fields": [],
        "records": [],
        "controls": [],
        "free_text": free_text,
        "evidence": evidence,
        "review_flags": [
            {
                "scope": "document",
                "flag": "RAW_OCR_ONLY: region text is not verified table/form structure; positional relationships are not inferred.",
            }
        ],
        "metadata": {
            "semantic_type": "raw_ocr_regions",
            "structure_recognition": "not_performed",
            "ocr_region_count": len(evidence),
            "searchable_region_count": len(free_text),
            "limitations": [
                "Field/value associations and table rows have not been reconstructed.",
                "Checkbox states cannot be inferred from OCR text alone.",
                "OCR errors and source text are preserved without correction.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--source-image", default=None)
    args = parser.parse_args()

    if args.input_csv.resolve() == args.output_json.resolve():
        parser.error("Input CSV and output JSON cannot be the same path.")

    doc = make_unified_ocr_document(
        args.input_csv,
        document_id=args.document_id,
        source_image=args.source_image,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print("EXPERIMENT 49: OCR -> UNIFIED DOCUMENT (RAW REGIONS)")
    print(f"Document ID: {doc['document_id']}")
    print(f"OCR evidence regions: {len(doc['evidence'])}")
    print(f"Searchable OCR regions: {len(doc['free_text'])}")
    print("Structure recognition: NOT PERFORMED")
    print(f"Saved: {args.output_json.resolve()}")
    print("EXPERIMENT 49 CONVERSION COMPLETE")


if __name__ == "__main__":
    main()
