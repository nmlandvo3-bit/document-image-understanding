
"""
Experiment 23: Table Cell Reconstruction
Intelligent Document Parsing Engine

Inputs:
    - Perspective-corrected document image
    - Binary grid mask from Experiment 22
    - Structured OCR JSON from Milestone 1

Outputs:
    - Numbered table-cell overlay
    - Cell-label image
    - Candidate table-cell JSON
    - Summary printed in the terminal

This is an experimental parser. It does not modify the original
scanner, OCR engine, GUI, or existing processing results.
"""

import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIC1_DIR = (
    PROJECT_ROOT
    / "results"
    / "gui_v4"
    / "single"
    / "pic1"
)

PARSING_DIR = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
)

PERSPECTIVE_PATH = PIC1_DIR / "perspective.png"

GRID_PATH = (
    PARSING_DIR
    / "grid_detection"
    / "04_combined_grid.png"
)

OCR_JSON_PATH = (
    PARSING_DIR
    / "structured_document.json"
)

OUTPUT_DIR = (
    PARSING_DIR
    / "cell_reconstruction"
)

# Small gaps in grid lines can cause multiple cells to merge.
# Start with a conservative closing kernel.
GRID_CLOSING_KERNEL = 3

# Filters for obviously invalid candidate cell areas.
MIN_CELL_WIDTH = 18
MIN_CELL_HEIGHT = 12
MIN_CELL_AREA = 200

# Reject a single candidate covering an implausibly large
# portion of the document. Such a candidate may indicate
# a broken or missing grid border.
MAX_CELL_IMAGE_FRACTION = 0.35

# Assignment thresholds:
# At least 60% of an OCR box must overlap the selected cell.
MIN_OCR_OVERLAP = 0.60

# If the two best cells have similar overlap scores,
# flag the assignment as ambiguous.
MIN_OVERLAP_DIFFERENCE = 0.15


def load_inputs():
    """Load and validate the three experiment inputs."""

    image = cv2.imread(
        str(PERSPECTIVE_PATH),
        cv2.IMREAD_COLOR,
    )

    grid = cv2.imread(
        str(GRID_PATH),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            f"Perspective image not found: {PERSPECTIVE_PATH}"
        )

    if grid is None:
        raise FileNotFoundError(
            f"Grid image not found: {GRID_PATH}"
        )

    if image.shape[:2] != grid.shape[:2]:
        raise ValueError(
            "Perspective image and grid mask have "
            "different dimensions."
        )

    with OCR_JSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        document = json.load(file)

    if "regions" not in document:
        raise ValueError(
            "Structured OCR JSON does not contain 'regions'."
        )

    return image, grid, document


def detect_candidate_cells(grid):
    """
    Find enclosed regions between the detected table lines.

    The grid is white on black. Inverting it produces white
    areas inside the table cells. Connected components can
    then identify candidate cell interiors.
    """

    height, width = grid.shape

    _, binary_grid = cv2.threshold(
        grid,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            GRID_CLOSING_KERNEL,
            GRID_CLOSING_KERNEL,
        ),
    )

    closed_grid = cv2.morphologyEx(
        binary_grid,
        cv2.MORPH_CLOSE,
        kernel,
    )

    cell_interiors = cv2.bitwise_not(closed_grid)

    number_of_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            cell_interiors,
            connectivity=8,
        )
    )

    image_area = height * width

    candidates = []
    rejected = []

    for label_id in range(1, number_of_labels):
        x, y, cell_width, cell_height, area = (
            int(value)
            for value in stats[label_id]
        )

        touches_image_border = (
            x <= 0
            or y <= 0
            or x + cell_width >= width
            or y + cell_height >= height
        )

        rejection_reason = None

        if touches_image_border:
            rejection_reason = "touches_image_border"

        elif cell_width < MIN_CELL_WIDTH:
            rejection_reason = "too_narrow"

        elif cell_height < MIN_CELL_HEIGHT:
            rejection_reason = "too_short"

        elif area < MIN_CELL_AREA:
            rejection_reason = "too_small"

        elif area / image_area > MAX_CELL_IMAGE_FRACTION:
            rejection_reason = "unusually_large"

        candidate = {
            "component_label": label_id,
            "bounding_box": {
                "x": x,
                "y": y,
                "width": cell_width,
                "height": cell_height,
                "x2": x + cell_width,
                "y2": y + cell_height,
            },
            "interior_area_pixels": area,
        }

        if rejection_reason is not None:
            candidate["rejection_reason"] = rejection_reason
            rejected.append(candidate)
        else:
            candidates.append(candidate)

    # Provide a stable, easy-to-inspect order.
    # This is a spatial sorting heuristic, not a verified
    # logical table reading order.
    candidates.sort(
        key=lambda cell: (
            cell["bounding_box"]["y"],
            cell["bounding_box"]["x"],
        )
    )

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        candidate["cell_id"] = index
        candidate["ocr_regions"] = []
        candidate["text"] = ""

    return (
        candidates,
        rejected,
        labels,
        closed_grid,
    )


def intersection_area(box_a, box_b):
    """Calculate the rectangular overlap of two bounding boxes."""

    left = max(box_a["x"], box_b["x"])
    top = max(box_a["y"], box_b["y"])

    right = min(box_a["x2"], box_b["x2"])
    bottom = min(box_a["y2"], box_b["y2"])

    overlap_width = max(0, right - left)
    overlap_height = max(0, bottom - top)

    return overlap_width * overlap_height


def assign_ocr_regions(cells, regions):
    """
    Match OCR boxes to candidate cells using overlap.

    Ambiguous or insufficient-overlap detections remain
    unassigned for later inspection.
    """

    unassigned = []

    for region in regions:
        region_box = region["bounding_box"]

        region_area = (
            region_box["width"]
            * region_box["height"]
        )

        if region_area <= 0:
            unassigned.append({
                "region_id": region["region_id"],
                "text": region["text"],
                "reason": "invalid_region_area",
            })
            continue

        matches = []

        for cell in cells:
            overlap_area = intersection_area(
                region_box,
                cell["bounding_box"],
            )

            overlap_fraction = (
                overlap_area / region_area
            )

            if overlap_fraction > 0:
                matches.append(
                    (
                        overlap_fraction,
                        cell,
                    )
                )

        matches.sort(
            key=lambda match: match[0],
            reverse=True,
        )

        if not matches:
            unassigned.append({
                "region_id": region["region_id"],
                "text": region["text"],
                "reason": "no_cell_overlap",
            })
            continue

        best_score, best_cell = matches[0]

        second_score = (
            matches[1][0]
            if len(matches) > 1
            else 0.0
        )

        if best_score < MIN_OCR_OVERLAP:
            unassigned.append({
                "region_id": region["region_id"],
                "text": region["text"],
                "reason": "insufficient_cell_overlap",
                "best_overlap": round(best_score, 4),
            })
            continue

        if (
            best_score - second_score
            < MIN_OVERLAP_DIFFERENCE
        ):
            unassigned.append({
                "region_id": region["region_id"],
                "text": region["text"],
                "reason": "ambiguous_cell_overlap",
                "best_overlap": round(best_score, 4),
                "second_overlap": round(second_score, 4),
            })
            continue

        best_cell["ocr_regions"].append({
            "region_id": region["region_id"],
            "text": region["text"],
            "confidence": region["confidence"],
            "meaningful": region["meaningful"],
            "source_order": region["source_order"],
            "bounding_box": region_box,
            "cell_overlap_fraction": round(
                best_score,
                4,
            ),
        })

    for cell in cells:
        cell["ocr_regions"].sort(
            key=lambda region: (
                region["bounding_box"]["y"],
                region["bounding_box"]["x"],
            )
        )

        cell["text"] = "\n".join(
            region["text"]
            for region in cell["ocr_regions"]
            if region["meaningful"]
        )

    return unassigned


def create_visualizations(
    image,
    labels,
    closed_grid,
    cells,
):
    """Save a numbered overlay and a candidate-cell label image."""

    overlay = image.copy()

    label_image = np.zeros_like(image)

    for cell in cells:
        box = cell["bounding_box"]

        x1 = box["x"]
        y1 = box["y"]
        x2 = box["x2"]
        y2 = box["y2"]

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            1,
        )

        cv2.putText(
            overlay,
            str(cell["cell_id"]),
            (
                x1 + 3,
                min(y2 - 3, y1 + 14),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        component_mask = (
            labels == cell["component_label"]
        )

        label_image[component_mask] = (
            180,
            180,
            180,
        )

        cv2.putText(
            label_image,
            str(cell["cell_id"]),
            (
                x1 + 3,
                min(y2 - 3, y1 + 14),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    output_images = {
        "01_closed_grid.png": closed_grid,
        "02_candidate_cell_labels.png": label_image,
        "03_numbered_cell_overlay.png": overlay,
    }

    for filename, output_image in output_images.items():
        output_path = OUTPUT_DIR / filename

        if not cv2.imwrite(
            str(output_path),
            output_image,
        ):
            raise RuntimeError(
                f"Could not save image: {output_path}"
            )

        print(f"[OK] {filename}")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image, grid, document = load_inputs()

    print("=" * 65)
    print("EXPERIMENT 23: TABLE CELL RECONSTRUCTION")
    print("=" * 65)

    cells, rejected, labels, closed_grid = (
        detect_candidate_cells(grid)
    )

    unassigned = assign_ocr_regions(
        cells,
        document["regions"],
    )

    assigned_count = sum(
        len(cell["ocr_regions"])
        for cell in cells
    )

    empty_cells = sum(
        len(cell["ocr_regions"]) == 0
        for cell in cells
    )

    result = {
        "schema_version": "0.1-experimental",
        "document_type": "table_candidate",
        "source": {
            "perspective_image": str(PERSPECTIVE_PATH),
            "grid_mask": str(GRID_PATH),
            "structured_ocr_json": str(OCR_JSON_PATH),
        },
        "image_dimensions": {
            "width": image.shape[1],
            "height": image.shape[0],
        },
        "statistics": {
            "candidate_cells": len(cells),
            "rejected_components": len(rejected),
            "empty_candidate_cells": empty_cells,
            "total_ocr_regions": len(document["regions"]),
            "assigned_ocr_regions": assigned_count,
            "unassigned_ocr_regions": len(unassigned),
        },
        "cells": cells,
        "unassigned_regions": unassigned,
        "rejected_components": rejected,
        "limitations": [
            "Detected cells are experimental candidates.",
            "Cell and text reading order has not been verified.",
            "Merged-cell row spans and column spans are not yet inferred.",
            "OCR assignment uses rectangular overlap, not semantic understanding.",
            "Broken grid lines may merge neighboring cells.",
        ],
    }

    output_json = (
        OUTPUT_DIR
        / "candidate_table_cells.json"
    )

    with output_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    create_visualizations(
        image,
        labels,
        closed_grid,
        cells,
    )

    print("-" * 65)
    print(f"Candidate cells:         {len(cells)}")
    print(f"Rejected components:     {len(rejected)}")
    print(f"Empty candidate cells:   {empty_cells}")
    print(f"Total OCR regions:       {len(document['regions'])}")
    print(f"Assigned OCR regions:    {assigned_count}")
    print(f"Unassigned OCR regions:  {len(unassigned)}")
    print(f"JSON output:             {output_json}")
    print("=" * 65)
    print("EXPERIMENT 23 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()