"""
Experiment 50: Layout-aware OCR evidence expansion.

Reads existing unified OCR JSON.
Does not run OCR, call Groq, or modify production files.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "unified_documents"
    / "pic13_raw_ocr_unified.json"
)


def horizontal_overlap_fraction(a, b):
    left = max(a["x"], b["x"])
    right = min(a["x2"], b["x2"])

    overlap = max(0, right - left)

    smaller_width = min(
        a["width"],
        b["width"],
    )

    if smaller_width <= 0:
        return 0.0

    return overlap / smaller_width


def find_neighboring_regions(
    anchor,
    regions,
    min_overlap=0.50,
    max_vertical_distance=1.60,
):
    anchor_box = anchor["bbox"]
    candidates = []

    for region in regions:
        if region["region_id"] == anchor["region_id"]:
            continue

        candidate_box = region["bbox"]

        overlap = horizontal_overlap_fraction(
            anchor_box,
            candidate_box,
        )

        top_distance = (
            candidate_box["y"]
            - anchor_box["y"]
        )

        max_distance = (
            max_vertical_distance
            * anchor_box["height"]
        )

        if overlap < min_overlap:
            continue

        if not 0 <= top_distance <= max_distance:
            continue

        candidates.append(
            {
                "region_id": region["region_id"],
                "text": region["text"],
                "horizontal_overlap": round(
                    overlap,
                    3,
                ),
                "vertical_top_distance": top_distance,
                "bbox": candidate_box,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["vertical_top_distance"],
            -item["horizontal_overlap"],
        )
    )

    return candidates


def main():
    document = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    regions = document["evidence"]

    test_labels = [
        "Name (First, Middle, Last, Suffix)",
        "Date of Birth",
        "Medicare Identification Number",
        "Street Address",
    ]

    print("=" * 72)
    print("EXPERIMENT 50: LAYOUT-AWARE OCR EVIDENCE")
    print("=" * 72)

    for label in test_labels:
        anchors = [
            region
            for region in regions
            if label.lower() in region["text"].lower()
        ]

        print()
        print("TARGET LABEL:", label)

        if not anchors:
            print("  Label not found.")
            continue

        for anchor in anchors:
            print(
                "  Anchor:",
                anchor["region_id"],
                repr(anchor["text"]),
            )

            candidates = find_neighboring_regions(
                anchor,
                regions,
            )

            if not candidates:
                print("  No neighboring candidates.")
                continue

            for candidate in candidates:
                print(
                    "  Candidate:",
                    candidate["region_id"],
                    repr(candidate["text"]),
                    "| overlap:",
                    candidate["horizontal_overlap"],
                    "| vertical distance:",
                    candidate["vertical_top_distance"],
                )

    print()
    print("=" * 72)
    print("EXPERIMENT 50 INSPECTION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()