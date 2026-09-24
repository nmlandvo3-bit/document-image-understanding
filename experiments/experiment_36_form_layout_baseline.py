"""
Experiment 36: Unseen Form Layout Baseline

Purpose:
- Test the existing document pipeline on a non-table form.
- Reuse existing OCR; do NOT rerun PaddleOCR.
- Preserve all OCR text/confidence/bounding boxes.
- Detect checkbox-like square controls using computer vision.
- Associate checkbox candidates with OCR text to their right.
- Generate generic vertical key/value pairing candidates.

Input:
    results/gui_v4/single/pic13/scan/scan.png
    results/gui_v4/single/pic13/ocr/results.csv

Outputs:
    results/parsing/pic13/form_layout_baseline/
        structured_form_baseline.json
        checkbox_overlay.png
        layout_overlay.png
        form_layout_report.txt
"""

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "results"
    / "gui_v4"
    / "single"
    / "pic13"
    / "scan"
    / "scan.png"
)

OCR_CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "gui_v4"
    / "single"
    / "pic13"
    / "ocr"
    / "results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic13"
    / "form_layout_baseline"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "structured_form_baseline.json"
)

CHECKBOX_OVERLAY = (
    OUTPUT_DIR
    / "checkbox_overlay.png"
)

LAYOUT_OVERLAY = (
    OUTPUT_DIR
    / "layout_overlay.png"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "form_layout_report.txt"
)


def load_ocr_regions():
    regions = []

    with OCR_CSV_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            region_id = int(row["region"])

            x = int(row["x"])
            y = int(row["y"])
            w = int(row["w"])
            h = int(row["h"])

            region = {
                "region_id": region_id,
                "text": row["text"],
                "confidence": float(
                    row["confidence"]
                ),
                "meaningful": (
                    str(row["meaningful"])
                    .strip()
                    .lower()
                    == "true"
                ),
                "bbox": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "x2": x + w,
                    "y2": y + h,
                },
                "center": {
                    "x": x + (w / 2.0),
                    "y": y + (h / 2.0),
                },
            }

            regions.append(region)

    return regions


def detect_checkbox_candidates(image):
    """
    Detect small square outlines.

    This is intentionally geometric.
    It does not rely on OCR checkbox characters.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )[1]

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    image_height, image_width = (
        gray.shape
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Checkbox size relative to this form.
        if w < 12 or h < 12:
            continue

        if w > 32 or h > 32:
            continue

        aspect = w / float(h)

        if not 0.70 <= aspect <= 1.30:
            continue

        area = cv2.contourArea(
            contour
        )

        box_area = w * h

        if box_area <= 0:
            continue

        contour_fill = (
            area / box_area
        )

        # Remove tiny glyph fragments.
        if contour_fill < 0.08:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        approx = cv2.approxPolyDP(
            contour,
            0.08 * perimeter,
            True,
        )

        if len(approx) < 4:
            continue

        # Ignore page edges.
        if (
            x < 5
            or y < 5
            or x + w
            >= image_width - 5
            or y + h
            >= image_height - 5
        ):
            continue

        # Inspect the checkbox interior.
        margin_x = max(
            2,
            int(w * 0.22),
        )

        margin_y = max(
            2,
            int(h * 0.22),
        )

        inner = binary[
            y + margin_y:
            y + h - margin_y,
            x + margin_x:
            x + w - margin_x,
        ]

        if inner.size == 0:
            continue

        interior_dark_ratio = (
            np.count_nonzero(inner)
            / inner.size
        )

        # Conservative classification.
        #
        # We preserve the measurement regardless of
        # the final state prediction.
        if interior_dark_ratio >= 0.16:
            predicted_state = "checked_candidate"
        else:
            predicted_state = "unchecked_candidate"

        candidates.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "x2": x + w,
                "y2": y + h,
                "center_x": x + w / 2.0,
                "center_y": y + h / 2.0,
                "aspect_ratio": aspect,
                "contour_fill_ratio": (
                    contour_fill
                ),
                "interior_dark_ratio": (
                    interior_dark_ratio
                ),
                "predicted_state": (
                    predicted_state
                ),
            }
        )

    # Remove near-duplicate nested contours.
    candidates.sort(
        key=lambda item: (
            item["y"],
            item["x"],
            item["w"] * item["h"],
        )
    )

    deduplicated = []

    for candidate in candidates:
        duplicate = False

        for existing in deduplicated:
            center_distance = math.hypot(
                candidate["center_x"]
                - existing["center_x"],
                candidate["center_y"]
                - existing["center_y"],
            )

            if center_distance <= 6:
                duplicate = True
                break

        if not duplicate:
            deduplicated.append(
                candidate
            )

    for index, item in enumerate(
        deduplicated,
        start=1,
    ):
        item[
            "checkbox_candidate_id"
        ] = index

    return binary, deduplicated


def vertical_overlap_ratio(
    box_a,
    box_b,
):
    top = max(
        box_a["y"],
        box_b["y"],
    )

    bottom = min(
        box_a["y2"],
        box_b["y2"],
    )

    overlap = max(
        0,
        bottom - top,
    )

    minimum_height = min(
        box_a["h"],
        box_b["h"],
    )

    if minimum_height <= 0:
        return 0.0

    return (
        overlap / minimum_height
    )


def associate_checkbox_labels(
    checkboxes,
    regions,
):
    """
    Associate a detected checkbox with nearby OCR text
    positioned to its right.

    No option names are hard-coded.
    """

    associations = []

    for checkbox in checkboxes:
        possible = []

        checkbox_box = {
            "x": checkbox["x"],
            "y": checkbox["y"],
            "w": checkbox["w"],
            "h": checkbox["h"],
            "x2": checkbox["x2"],
            "y2": checkbox["y2"],
        }

        for region in regions:
            bbox = region["bbox"]

            # Label should start to the right.
            horizontal_gap = (
                bbox["x"]
                - checkbox["x2"]
            )

            if horizontal_gap < -5:
                continue

            if horizontal_gap > 90:
                continue

            vertical_distance = abs(
                region["center"]["y"]
                - checkbox["center_y"]
            )

            if vertical_distance > 25:
                continue

            overlap = vertical_overlap_ratio(
                checkbox_box,
                bbox,
            )

            score = (
                vertical_distance
                + max(
                    horizontal_gap,
                    0,
                )
                - (overlap * 20)
            )

            possible.append(
                (
                    score,
                    region,
                    horizontal_gap,
                    vertical_distance,
                    overlap,
                )
            )

        possible.sort(
            key=lambda item: item[0]
        )

        if possible:
            (
                score,
                region,
                gap,
                vertical_distance,
                overlap,
            ) = possible[0]

            association = {
                "checkbox_candidate_id": (
                    checkbox[
                        "checkbox_candidate_id"
                    ]
                ),
                "predicted_state": (
                    checkbox[
                        "predicted_state"
                    ]
                ),
                "interior_dark_ratio": (
                    checkbox[
                        "interior_dark_ratio"
                    ]
                ),
                "label_region_id": (
                    region["region_id"]
                ),
                "label_text": (
                    region["text"]
                ),
                "horizontal_gap": gap,
                "vertical_center_distance": (
                    vertical_distance
                ),
                "vertical_overlap_ratio": (
                    overlap
                ),
                "association_score": (
                    score
                ),
            }

        else:
            association = {
                "checkbox_candidate_id": (
                    checkbox[
                        "checkbox_candidate_id"
                    ]
                ),
                "predicted_state": (
                    checkbox[
                        "predicted_state"
                    ]
                ),
                "interior_dark_ratio": (
                    checkbox[
                        "interior_dark_ratio"
                    ]
                ),
                "label_region_id": None,
                "label_text": None,
                "horizontal_gap": None,
                "vertical_center_distance": None,
                "vertical_overlap_ratio": None,
                "association_score": None,
            }

        associations.append(
            association
        )

    return associations


def generate_vertical_pair_candidates(
    regions,
):
    """
    Produce generic label/value candidates.

    This intentionally does NOT declare that the pairs
    are correct form fields yet.

    It looks for OCR regions directly below another
    region with similar horizontal alignment.
    """

    candidates = []

    for upper in regions:
        upper_box = upper["bbox"]

        possible = []

        for lower in regions:
            if (
                lower["region_id"]
                == upper["region_id"]
            ):
                continue

            lower_box = lower["bbox"]

            vertical_gap = (
                lower_box["y"]
                - upper_box["y2"]
            )

            if vertical_gap < -5:
                continue

            if vertical_gap > 60:
                continue

            left_difference = abs(
                lower_box["x"]
                - upper_box["x"]
            )

            if left_difference > 45:
                continue

            # Prevent enormous paragraph blocks from
            # dominating field-pair detection.
            if len(upper["text"]) > 80:
                continue

            if len(lower["text"]) > 120:
                continue

            score = (
                vertical_gap
                + left_difference
            )

            possible.append(
                (
                    score,
                    lower,
                    vertical_gap,
                    left_difference,
                )
            )

        possible.sort(
            key=lambda item: item[0]
        )

        if possible:
            (
                score,
                lower,
                vertical_gap,
                left_difference,
            ) = possible[0]

            candidates.append(
                {
                    "upper_region_id": (
                        upper["region_id"]
                    ),
                    "upper_text": (
                        upper["text"]
                    ),
                    "lower_region_id": (
                        lower["region_id"]
                    ),
                    "lower_text": (
                        lower["text"]
                    ),
                    "vertical_gap": (
                        vertical_gap
                    ),
                    "left_alignment_difference": (
                        left_difference
                    ),
                    "pair_score": score,
                }
            )

    return candidates


def draw_checkbox_overlay(
    image,
    checkboxes,
    associations,
):
    overlay = image.copy()

    association_by_id = {
        item["checkbox_candidate_id"]:
        item
        for item in associations
    }

    for checkbox in checkboxes:
        candidate_id = checkbox[
            "checkbox_candidate_id"
        ]

        association = (
            association_by_id[
                candidate_id
            ]
        )

        x = checkbox["x"]
        y = checkbox["y"]
        x2 = checkbox["x2"]
        y2 = checkbox["y2"]

        cv2.rectangle(
            overlay,
            (x, y),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        label = (
            f"B{candidate_id} "
            f"{checkbox['predicted_state']}"
        )

        if association[
            "label_region_id"
        ] is not None:
            label += (
                f" -> R"
                f"{association['label_region_id']}"
            )

        cv2.putText(
            overlay,
            label,
            (
                max(0, x - 5),
                max(15, y - 6),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return overlay


def draw_layout_overlay(
    image,
    regions,
):
    overlay = image.copy()

    for region in regions:
        bbox = region["bbox"]

        cv2.rectangle(
            overlay,
            (
                bbox["x"],
                bbox["y"],
            ),
            (
                bbox["x2"],
                bbox["y2"],
            ),
            (255, 0, 0),
            1,
        )

        cv2.putText(
            overlay,
            f"R{region['region_id']}",
            (
                bbox["x"],
                max(
                    12,
                    bbox["y"] - 4,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return overlay


def make_report(
    image,
    regions,
    checkboxes,
    associations,
    pair_candidates,
):
    height, width = image.shape[:2]

    lines = [
        "=" * 78,
        "EXPERIMENT 36: UNSEEN FORM LAYOUT BASELINE",
        "=" * 78,
        f"Image width:  {width}",
        f"Image height: {height}",
        f"OCR regions:  {len(regions)}",
        f"Checkbox candidates: {len(checkboxes)}",
        f"Vertical pair candidates: {len(pair_candidates)}",
        "=" * 78,
        "",
        "CHECKBOX CANDIDATES",
        "-" * 78,
    ]

    association_by_id = {
        item["checkbox_candidate_id"]:
        item
        for item in associations
    }

    for checkbox in checkboxes:
        candidate_id = checkbox[
            "checkbox_candidate_id"
        ]

        association = (
            association_by_id[
                candidate_id
            ]
        )

        lines.append(
            (
                f"B{candidate_id:02d} "
                f"bbox=({checkbox['x']}, "
                f"{checkbox['y']}, "
                f"{checkbox['w']}, "
                f"{checkbox['h']}) "
                f"state={checkbox['predicted_state']} "
                f"interior="
                f"{checkbox['interior_dark_ratio']:.3f} "
                f"label={association['label_text']!r}"
            )
        )

    lines.extend(
        [
            "",
            "VERTICAL PAIR CANDIDATES",
            "-" * 78,
        ]
    )

    for pair in pair_candidates:
        lines.append(
            (
                f"R{pair['upper_region_id']:02d} "
                f"{pair['upper_text']!r}"
                "  ->  "
                f"R{pair['lower_region_id']:02d} "
                f"{pair['lower_text']!r} "
                f"[gap={pair['vertical_gap']}, "
                f"dx={pair['left_alignment_difference']}, "
                f"score={pair['pair_score']}]"
            )
        )

    return "\n".join(
        lines
    ) + "\n"


def main():
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            IMAGE_PATH
        )

    if not OCR_CSV_PATH.is_file():
        raise FileNotFoundError(
            OCR_CSV_PATH
        )

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise RuntimeError(
            f"Could not load image: "
            f"{IMAGE_PATH}"
        )

    regions = load_ocr_regions()

    (
        binary,
        checkboxes,
    ) = detect_checkbox_candidates(
        image
    )

    associations = (
        associate_checkbox_labels(
            checkboxes,
            regions,
        )
    )

    pair_candidates = (
        generate_vertical_pair_candidates(
            regions
        )
    )

    height, width = (
        image.shape[:2]
    )

    result = {
        "experiment": 36,
        "schema_version": (
            "0.1-form-experimental"
        ),
        "document_type": (
            "unknown_form_candidate"
        ),
        "source": {
            "image": str(
                IMAGE_PATH
            ),
            "ocr_csv": str(
                OCR_CSV_PATH
            ),
        },
        "image": {
            "width": width,
            "height": height,
        },
        "statistics": {
            "ocr_regions": (
                len(regions)
            ),
            "checkbox_candidates": (
                len(checkboxes)
            ),
            "vertical_pair_candidates": (
                len(pair_candidates)
            ),
        },
        "ocr_regions": regions,
        "checkbox_candidates": (
            checkboxes
        ),
        "checkbox_associations": (
            associations
        ),
        "vertical_pair_candidates": (
            pair_candidates
        ),
        "notes": [
            (
                "Existing production OCR was reused; "
                "OCR was not rerun."
            ),
            (
                "Checkbox states are geometric "
                "predictions and require visual "
                "validation."
            ),
            (
                "Vertical pairs are candidates only; "
                "they are not yet confirmed "
                "key/value fields."
            ),
            (
                "No form-specific names or values "
                "were hard-coded."
            ),
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    checkbox_overlay = (
        draw_checkbox_overlay(
            image,
            checkboxes,
            associations,
        )
    )

    layout_overlay = (
        draw_layout_overlay(
            image,
            regions,
        )
    )

    cv2.imwrite(
        str(CHECKBOX_OVERLAY),
        checkbox_overlay,
    )

    cv2.imwrite(
        str(LAYOUT_OVERLAY),
        layout_overlay,
    )

    report = make_report(
        image,
        regions,
        checkboxes,
        associations,
        pair_candidates,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "EXPERIMENT 36: "
        "UNSEEN FORM LAYOUT BASELINE"
    )
    print("=" * 72)

    print(
        f"Image: {width}x{height}"
    )

    print(
        f"OCR regions: "
        f"{len(regions)}"
    )

    print(
        f"Checkbox candidates: "
        f"{len(checkboxes)}"
    )

    print(
        f"Vertical pair candidates: "
        f"{len(pair_candidates)}"
    )

    print()
    print(
        f"JSON:    {OUTPUT_JSON}"
    )

    print(
        f"Checkbox overlay: "
        f"{CHECKBOX_OVERLAY}"
    )

    print(
        f"Layout overlay:   "
        f"{LAYOUT_OVERLAY}"
    )

    print(
        f"Report:  {REPORT_PATH}"
    )

    print("=" * 72)
    print(
        "EXPERIMENT 36 COMPLETE"
    )


if __name__ == "__main__":
    main()