from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from document_scanner import (
    detect_document,
    enhance_document,
    four_point_transform,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_COMPONENT_AREA_FRACTION = 0.00015
MAX_COMPONENT_AREA_FRACTION = 0.25

MIN_REGION_WIDTH_FRACTION = 0.015
MIN_REGION_HEIGHT_FRACTION = 0.008

MAX_REGION_HEIGHT_FRACTION = 0.40


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 11 - detect text and content "
            "regions inside a cleaned document."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original document image.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for Experiment 11 outputs.",
    )

    return parser.parse_args()


# =============================================================================
# LINE DETECTION
# =============================================================================

def detect_document_lines(
    binary_scan: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    binary_scan:
        white background, black foreground.

    Returns:
        horizontal_lines
        vertical_lines
        combined_lines
    """

    foreground = cv2.bitwise_not(
        binary_scan
    )

    height, width = (
        foreground.shape[:2]
    )

    horizontal_length = max(
        25,
        width // 20,
    )

    vertical_length = max(
        25,
        height // 20,
    )

    horizontal_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                horizontal_length,
                1,
            ),
        )
    )

    vertical_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                vertical_length,
            ),
        )
    )

    horizontal_lines = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1,
    )

    vertical_lines = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1,
    )

    combined_lines = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines,
    )

    return (
        horizontal_lines,
        vertical_lines,
        combined_lines,
    )


# =============================================================================
# TEXT MASK
# =============================================================================

def remove_document_lines(
    binary_scan: np.ndarray,
    line_mask: np.ndarray,
) -> np.ndarray:
    foreground = cv2.bitwise_not(
        binary_scan
    )

    text_only = cv2.subtract(
        foreground,
        line_mask,
    )

    cleanup_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (2, 2),
        )
    )

    text_only = cv2.morphologyEx(
        text_only,
        cv2.MORPH_OPEN,
        cleanup_kernel,
        iterations=1,
    )

    return text_only


# =============================================================================
# TEXT GROUPING
# =============================================================================

def group_text_regions(
    text_mask: np.ndarray,
) -> np.ndarray:
    height, width = (
        text_mask.shape[:2]
    )

    horizontal_size = max(
        9,
        width // 80,
    )

    vertical_size = max(
        3,
        height // 250,
    )

    grouping_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                horizontal_size,
                vertical_size,
            ),
        )
    )

    grouped = cv2.dilate(
        text_mask,
        grouping_kernel,
        iterations=2,
    )

    close_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 3),
        )
    )

    grouped = cv2.morphologyEx(
        grouped,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    return grouped


# =============================================================================
# REGION EXTRACTION
# =============================================================================

def find_text_regions(
    grouped_mask: np.ndarray,
) -> list[
    dict[str, float | int]
]:
    contours, _ = cv2.findContours(
        grouped_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = (
        grouped_mask.shape[:2]
    )

    image_area = float(
        height
        * width
    )

    regions = []

    for contour in contours:
        x, y, w, h = (
            cv2.boundingRect(
                contour
            )
        )

        area = float(
            w * h
        )

        area_fraction = (
            area
            / image_area
        )

        width_fraction = (
            w
            / float(width)
        )

        height_fraction = (
            h
            / float(height)
        )

        if (
            area_fraction
            < MIN_COMPONENT_AREA_FRACTION
        ):
            continue

        if (
            area_fraction
            > MAX_COMPONENT_AREA_FRACTION
        ):
            continue

        if (
            width_fraction
            < MIN_REGION_WIDTH_FRACTION
        ):
            continue

        if (
            height_fraction
            < MIN_REGION_HEIGHT_FRACTION
        ):
            continue

        if (
            height_fraction
            > MAX_REGION_HEIGHT_FRACTION
        ):
            continue

        regions.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area_fraction":
                    area_fraction,
            }
        )

    regions.sort(
        key=lambda region: (
            int(region["y"]),
            int(region["x"]),
        )
    )

    return regions


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_regions(
    image: np.ndarray,
    regions: list[
        dict[str, float | int]
    ],
) -> np.ndarray:
    preview = (
        image.copy()
    )

    for index, region in enumerate(
        regions,
        start=1,
    ):
        x = int(
            region["x"]
        )

        y = int(
            region["y"]
        )

        w = int(
            region["w"]
        )

        h = int(
            region["h"]
        )

        cv2.rectangle(
            preview,
            (x, y),
            (
                x + w,
                y + h,
            ),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            preview,
            f"R{index}",
            (
                x,
                max(
                    18,
                    y - 6,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return preview


def create_region_mask(
    shape: tuple[int, int],
    regions: list[
        dict[str, float | int]
    ],
) -> np.ndarray:
    mask = np.zeros(
        shape,
        dtype=np.uint8,
    )

    for region in regions:
        x = int(
            region["x"]
        )

        y = int(
            region["y"]
        )

        w = int(
            region["w"]
        )

        h = int(
            region["h"]
        )

        cv2.rectangle(
            mask,
            (x, y),
            (
                x + w,
                y + h,
            ),
            255,
            -1,
        )

    return mask


# =============================================================================
# REGION CROPS
# =============================================================================

def save_region_crops(
    image: np.ndarray,
    regions: list[
        dict[str, float | int]
    ],
    output_dir: Path,
) -> None:
    crop_dir = (
        output_dir
        / "regions"
    )

    crop_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, region in enumerate(
        regions,
        start=1,
    ):
        x = int(
            region["x"]
        )

        y = int(
            region["y"]
        )

        w = int(
            region["w"]
        )

        h = int(
            region["h"]
        )

        crop = image[
            y:y + h,
            x:x + w,
        ]

        if crop.size == 0:
            continue

        cv2.imwrite(
            str(
                crop_dir
                / f"region_{index:03d}.png"
            ),
            crop,
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    input_path = Path(
        args.input
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image not found:\n"
            f"{input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(
        str(
            input_path
        )
    )

    if image is None:
        raise RuntimeError(
            f"Could not load image:\n"
            f"{input_path}"
        )

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 11 — "
        "TEXT REGION / DOCUMENT STRUCTURE DETECTION"
    )

    print(
        "=" * 100
    )

    print(
        f"Input      : {input_path}"
    )

    print(
        f"Image size : "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    print()

    # -------------------------------------------------------------------------
    # Frozen production pipeline
    # -------------------------------------------------------------------------

    print(
        "Running frozen document detector..."
    )

    (
        document_points,
        _edge_map,
        route,
    ) = detect_document(
        image
    )

    print(
        f"Detector route: {route}"
    )

    perspective = (
        four_point_transform(
            image,
            document_points,
        )
    )

    enhanced = (
        enhance_document(
            perspective
        )
    )

    scan = enhanced[
        "scan"
    ]

    corrected = enhanced[
        "contrast"
    ]

    # -------------------------------------------------------------------------
    # Detect long structural lines
    # -------------------------------------------------------------------------

    (
        horizontal_lines,
        vertical_lines,
        line_mask,
    ) = detect_document_lines(
        scan
    )

    # -------------------------------------------------------------------------
    # Remove lines
    # -------------------------------------------------------------------------

    text_mask = (
        remove_document_lines(
            scan,
            line_mask,
        )
    )

    # -------------------------------------------------------------------------
    # Group nearby text
    # -------------------------------------------------------------------------

    grouped_mask = (
        group_text_regions(
            text_mask
        )
    )

    # -------------------------------------------------------------------------
    # Region proposals
    # -------------------------------------------------------------------------

    regions = (
        find_text_regions(
            grouped_mask
        )
    )

    region_preview = (
        draw_regions(
            perspective,
            regions,
        )
    )

    corrected_preview = (
        cv2.cvtColor(
            corrected,
            cv2.COLOR_GRAY2BGR,
        )
    )

    corrected_preview = (
        draw_regions(
            corrected_preview,
            regions,
        )
    )

    region_mask = (
        create_region_mask(
            scan.shape,
            regions,
        )
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    outputs = {
        "01_perspective":
            perspective,

        "02_corrected":
            corrected,

        "03_scan":
            scan,

        "04_horizontal_lines":
            horizontal_lines,

        "05_vertical_lines":
            vertical_lines,

        "06_combined_lines":
            line_mask,

        "07_text_only_mask":
            text_mask,

        "08_grouped_text":
            grouped_mask,

        "09_region_mask":
            region_mask,

        "10_regions_on_original":
            region_preview,

        "11_regions_on_corrected":
            corrected_preview,
    }

    for name, output in outputs.items():
        cv2.imwrite(
            str(
                output_dir
                / f"{name}.png"
            ),
            output,
        )

    save_region_crops(
        corrected,
        regions,
        output_dir,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    total_region_area = sum(
        float(
            region[
                "area_fraction"
            ]
        )
        for region in regions
    )

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 11 SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Detector route       : "
        f"{route}"
    )

    print(
        f"Document size        : "
        f"{perspective.shape[1]} x "
        f"{perspective.shape[0]}"
    )

    print(
        f"Regions detected     : "
        f"{len(regions)}"
    )

    print(
        f"Total region coverage: "
        f"{total_region_area:.4f}"
    )

    print()

    if regions:
        print(
            "REGIONS"
        )

        print(
            "-" * 100
        )

        for index, region in enumerate(
            regions,
            start=1,
        ):
            print(
                f"R{index:02d} "
                f"| x={int(region['x']):4d} "
                f"| y={int(region['y']):4d} "
                f"| w={int(region['w']):4d} "
                f"| h={int(region['h']):4d} "
                f"| area="
                f"{float(region['area_fraction']):.4f}"
            )

    print()

    print(
        f"Outputs:\n"
        f"{output_dir}"
    )

    print()

    print(
        "Most important files:"
    )

    print(
        "  06_combined_lines.png"
    )

    print(
        "  07_text_only_mask.png"
    )

    print(
        "  08_grouped_text.png"
    )

    print(
        "  10_regions_on_original.png"
    )

    print(
        "  11_regions_on_corrected.png"
    )


if __name__ == "__main__":
    main()