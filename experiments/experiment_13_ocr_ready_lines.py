from __future__ import annotations

import argparse
import csv
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

BORDER_MARGIN_FRACTION = 0.015

MIN_RAW_COMPONENT_AREA = 6

# Candidate line geometry.
MIN_LINE_WIDTH_FRACTION = 0.018
MAX_LINE_WIDTH_FRACTION = 0.90

MIN_LINE_HEIGHT_FRACTION = 0.010
MAX_LINE_HEIGHT_FRACTION = 0.10

MIN_LINE_AREA_FRACTION = 0.00025
MAX_LINE_AREA_FRACTION = 0.08

MIN_ASPECT_RATIO = 1.0

# Ink-density validation.
MIN_INK_DENSITY = 0.035
MAX_INK_DENSITY = 0.75

# Reject boxes with too little real foreground content.
MIN_FOREGROUND_PIXELS = 15

# Padding when saving OCR crops.
CROP_PADDING_X = 5
CROP_PADDING_Y = 4


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 13 - detect and validate "
            "OCR-ready text lines."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to original document photograph.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for Experiment 13 outputs.",
    )

    return parser.parse_args()


# =============================================================================
# STRUCTURAL LINE DETECTION
# =============================================================================

def detect_structural_lines(
    binary_scan: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    foreground = cv2.bitwise_not(
        binary_scan
    )

    height, width = (
        foreground.shape[:2]
    )

    horizontal_length = max(
        25,
        width // 18,
    )

    vertical_length = max(
        25,
        height // 18,
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

def remove_structural_lines(
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

    return text_only


# =============================================================================
# BORDER SUPPRESSION
# =============================================================================

def suppress_page_border(
    mask: np.ndarray,
) -> np.ndarray:
    result = mask.copy()

    height, width = (
        result.shape[:2]
    )

    margin_x = max(
        2,
        int(
            width
            * BORDER_MARGIN_FRACTION
        ),
    )

    margin_y = max(
        2,
        int(
            height
            * BORDER_MARGIN_FRACTION
        ),
    )

    result[
        :margin_y,
        :
    ] = 0

    result[
        height - margin_y:,
        :
    ] = 0

    result[
        :,
        :margin_x
    ] = 0

    result[
        :,
        width - margin_x:
    ] = 0

    return result


# =============================================================================
# COMPONENT CLEANUP
# =============================================================================

def remove_tiny_components(
    mask: np.ndarray,
) -> np.ndarray:
    (
        component_count,
        labels,
        stats,
        _centroids,
    ) = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(
        mask
    )

    for component_id in range(
        1,
        component_count,
    ):
        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < MIN_RAW_COMPONENT_AREA:
            continue

        cleaned[
            labels == component_id
        ] = 255

    return cleaned


# =============================================================================
# HORIZONTAL TEXT GROUPING
# =============================================================================

def group_characters_into_lines(
    text_mask: np.ndarray,
) -> np.ndarray:
    height, width = (
        text_mask.shape[:2]
    )

    horizontal_width = max(
        9,
        width // 55,
    )

    vertical_height = max(
        2,
        height // 350,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            horizontal_width,
            vertical_height,
        ),
    )

    grouped = cv2.dilate(
        text_mask,
        kernel,
        iterations=1,
    )

    close_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 2),
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
# INK DENSITY
# =============================================================================

def compute_ink_density(
    foreground_mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> tuple[
    float,
    int,
]:
    crop = foreground_mask[
        y:y + h,
        x:x + w,
    ]

    if crop.size == 0:
        return (
            0.0,
            0,
        )

    foreground_pixels = int(
        np.count_nonzero(
            crop
        )
    )

    total_pixels = int(
        crop.size
    )

    if total_pixels <= 0:
        return (
            0.0,
            foreground_pixels,
        )

    density = (
        foreground_pixels
        / float(
            total_pixels
        )
    )

    return (
        density,
        foreground_pixels,
    )


# =============================================================================
# LINE EXTRACTION + VALIDATION
# =============================================================================

def find_valid_text_lines(
    grouped_mask: np.ndarray,
    foreground_mask: np.ndarray,
) -> tuple[
    list[
        dict[str, int | float]
    ],
    list[
        dict[str, int | float | str]
    ],
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

    accepted = []
    rejected = []

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

        aspect_ratio = (
            w
            / float(
                max(
                    h,
                    1,
                )
            )
        )

        (
            ink_density,
            foreground_pixels,
        ) = compute_ink_density(
            foreground_mask,
            x,
            y,
            w,
            h,
        )

        reason = None

        if (
            area_fraction
            < MIN_LINE_AREA_FRACTION
        ):
            reason = "area_too_small"

        elif (
            area_fraction
            > MAX_LINE_AREA_FRACTION
        ):
            reason = "area_too_large"

        elif (
            width_fraction
            < MIN_LINE_WIDTH_FRACTION
        ):
            reason = "width_too_small"

        elif (
            width_fraction
            > MAX_LINE_WIDTH_FRACTION
        ):
            reason = "width_too_large"

        elif (
            height_fraction
            < MIN_LINE_HEIGHT_FRACTION
        ):
            reason = "height_too_small"

        elif (
            height_fraction
            > MAX_LINE_HEIGHT_FRACTION
        ):
            reason = "height_too_large"

        elif (
            aspect_ratio
            < MIN_ASPECT_RATIO
        ):
            reason = "aspect_ratio"

        elif (
            foreground_pixels
            < MIN_FOREGROUND_PIXELS
        ):
            reason = "too_few_pixels"

        elif (
            ink_density
            < MIN_INK_DENSITY
        ):
            reason = "ink_too_low"

        elif (
            ink_density
            > MAX_INK_DENSITY
        ):
            reason = "ink_too_high"

        entry = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area_fraction":
                area_fraction,
            "aspect_ratio":
                aspect_ratio,
            "ink_density":
                ink_density,
            "foreground_pixels":
                foreground_pixels,
        }

        if reason is None:
            accepted.append(
                entry
            )

        else:
            rejected.append(
                {
                    **entry,
                    "reason":
                        reason,
                }
            )

    accepted.sort(
        key=lambda item: (
            int(
                item["y"]
            ),
            int(
                item["x"]
            ),
        )
    )

    rejected.sort(
        key=lambda item: (
            int(
                item["y"]
            ),
            int(
                item["x"]
            ),
        )
    )

    return (
        accepted,
        rejected,
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_lines(
    image: np.ndarray,
    lines: list[
        dict[str, int | float]
    ],
) -> np.ndarray:
    preview = image.copy()

    for index, line in enumerate(
        lines,
        start=1,
    ):
        x = int(
            line["x"]
        )

        y = int(
            line["y"]
        )

        w = int(
            line["w"]
        )

        h = int(
            line["h"]
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
            f"L{index}",
            (
                x,
                max(
                    18,
                    y - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return preview


def draw_rejected(
    image: np.ndarray,
    rejected: list[
        dict[
            str,
            int | float | str
        ]
    ],
) -> np.ndarray:
    preview = image.copy()

    for index, line in enumerate(
        rejected,
        start=1,
    ):
        x = int(
            line["x"]
        )

        y = int(
            line["y"]
        )

        w = int(
            line["w"]
        )

        h = int(
            line["h"]
        )

        cv2.rectangle(
            preview,
            (x, y),
            (
                x + w,
                y + h,
            ),
            (0, 0, 255),
            1,
        )

        cv2.putText(
            preview,
            f"R{index}",
            (
                x,
                max(
                    12,
                    y - 3,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return preview


# =============================================================================
# OCR CROP SAVING
# =============================================================================

def save_line_crops(
    corrected: np.ndarray,
    binary_scan: np.ndarray,
    lines: list[
        dict[str, int | float]
    ],
    output_dir: Path,
) -> None:
    gray_dir = (
        output_dir
        / "ocr_lines_gray"
    )

    binary_dir = (
        output_dir
        / "ocr_lines_binary"
    )

    gray_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    binary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    height, width = (
        corrected.shape[:2]
    )

    for index, line in enumerate(
        lines,
        start=1,
    ):
        x = int(
            line["x"]
        )

        y = int(
            line["y"]
        )

        w = int(
            line["w"]
        )

        h = int(
            line["h"]
        )

        x1 = max(
            0,
            x - CROP_PADDING_X,
        )

        y1 = max(
            0,
            y - CROP_PADDING_Y,
        )

        x2 = min(
            width,
            x + w + CROP_PADDING_X,
        )

        y2 = min(
            height,
            y + h + CROP_PADDING_Y,
        )

        gray_crop = corrected[
            y1:y2,
            x1:x2,
        ]

        binary_crop = binary_scan[
            y1:y2,
            x1:x2,
        ]

        if (
            gray_crop.size == 0
            or binary_crop.size == 0
        ):
            continue

        cv2.imwrite(
            str(
                gray_dir
                / f"line_{index:03d}.png"
            ),
            gray_crop,
        )

        cv2.imwrite(
            str(
                binary_dir
                / f"line_{index:03d}.png"
            ),
            binary_crop,
        )


# =============================================================================
# CSV
# =============================================================================

def save_csv(
    lines: list[
        dict[str, int | float]
    ],
    output_dir: Path,
) -> None:
    csv_path = (
        output_dir
        / "text_lines.csv"
    )

    fieldnames = [
        "line",
        "x",
        "y",
        "w",
        "h",
        "area_fraction",
        "aspect_ratio",
        "ink_density",
        "foreground_pixels",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index, line in enumerate(
            lines,
            start=1,
        ):
            writer.writerow(
                {
                    "line":
                        index,

                    "x":
                        int(
                            line["x"]
                        ),

                    "y":
                        int(
                            line["y"]
                        ),

                    "w":
                        int(
                            line["w"]
                        ),

                    "h":
                        int(
                            line["h"]
                        ),

                    "area_fraction":
                        f"{float(line['area_fraction']):.8f}",

                    "aspect_ratio":
                        f"{float(line['aspect_ratio']):.4f}",

                    "ink_density":
                        f"{float(line['ink_density']):.4f}",

                    "foreground_pixels":
                        int(
                            line[
                                "foreground_pixels"
                            ]
                        ),
                }
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
            f"Could not load:\n"
            f"{input_path}"
        )

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 13 — "
        "OCR-READY TEXT LINE DETECTION"
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

    # -------------------------------------------------------------------------
    # Frozen Production v2
    # -------------------------------------------------------------------------

    (
        document_points,
        _edges,
        route,
    ) = detect_document(
        image
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

    corrected = enhanced[
        "contrast"
    ]

    scan = enhanced[
        "scan"
    ]

    # -------------------------------------------------------------------------
    # Structural lines
    # -------------------------------------------------------------------------

    (
        horizontal_lines,
        vertical_lines,
        structural_lines,
    ) = detect_structural_lines(
        scan
    )

    text_without_lines = (
        remove_structural_lines(
            scan,
            structural_lines,
        )
    )

    # -------------------------------------------------------------------------
    # Border suppression
    # -------------------------------------------------------------------------

    border_suppressed = (
        suppress_page_border(
            text_without_lines
        )
    )

    # -------------------------------------------------------------------------
    # Tiny component cleanup
    # -------------------------------------------------------------------------

    component_cleaned = (
        remove_tiny_components(
            border_suppressed
        )
    )

    # -------------------------------------------------------------------------
    # Character-to-line grouping
    # -------------------------------------------------------------------------

    grouped_lines = (
        group_characters_into_lines(
            component_cleaned
        )
    )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    (
        valid_lines,
        rejected_lines,
    ) = find_valid_text_lines(
        grouped_lines,
        component_cleaned,
    )

    corrected_bgr = cv2.cvtColor(
        corrected,
        cv2.COLOR_GRAY2BGR,
    )

    accepted_preview = draw_lines(
        corrected_bgr,
        valid_lines,
    )

    rejected_preview = draw_rejected(
        corrected_bgr,
        rejected_lines,
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

        "04_structural_lines":
            structural_lines,

        "05_text_without_lines":
            text_without_lines,

        "06_border_suppressed":
            border_suppressed,

        "07_component_cleaned":
            component_cleaned,

        "08_grouped_lines":
            grouped_lines,

        "09_valid_lines":
            accepted_preview,

        "10_rejected_candidates":
            rejected_preview,
    }

    for name, output in outputs.items():
        cv2.imwrite(
            str(
                output_dir
                / f"{name}.png"
            ),
            output,
        )

    save_line_crops(
        corrected,
        scan,
        valid_lines,
        output_dir,
    )

    save_csv(
        valid_lines,
        output_dir,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 13 SUMMARY"
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
        f"Accepted text lines  : "
        f"{len(valid_lines)}"
    )

    print(
        f"Rejected candidates  : "
        f"{len(rejected_lines)}"
    )

    print()

    if valid_lines:
        print(
            "ACCEPTED LINES"
        )

        print(
            "-" * 100
        )

        for index, line in enumerate(
            valid_lines,
            start=1,
        ):
            print(
                f"L{index:02d} "
                f"| x={int(line['x']):4d} "
                f"| y={int(line['y']):4d} "
                f"| w={int(line['w']):4d} "
                f"| h={int(line['h']):4d} "
                f"| AR={float(line['aspect_ratio']):6.2f} "
                f"| ink={float(line['ink_density']):.4f}"
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
        "  08_grouped_lines.png"
    )

    print(
        "  09_valid_lines.png"
    )

    print(
        "  10_rejected_candidates.png"
    )

    print(
        "  text_lines.csv"
    )

    print(
        "  ocr_lines_gray\\"
    )

    print(
        "  ocr_lines_binary\\"
    )


if __name__ == "__main__":
    main()