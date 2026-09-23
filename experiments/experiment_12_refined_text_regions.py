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

# Ignore artifacts very close to the rectified page border.
BORDER_MARGIN_FRACTION = 0.015

# Candidate line geometry.
MIN_LINE_WIDTH_FRACTION = 0.025
MAX_LINE_WIDTH_FRACTION = 0.95

MIN_LINE_HEIGHT_FRACTION = 0.008
MAX_LINE_HEIGHT_FRACTION = 0.12

MIN_LINE_AREA_FRACTION = 0.00025

# Reject extremely thin components.
MIN_ASPECT_RATIO = 0.75

# Merge nearby text lines into larger blocks.
BLOCK_VERTICAL_GAP_FACTOR = 1.8
BLOCK_HORIZONTAL_OVERLAP = 0.20

# Remove very tiny connected components before grouping.
MIN_RAW_COMPONENT_AREA = 6


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 12 - refined text-line and "
            "text-block detection."
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
        help="Directory for Experiment 12 outputs.",
    )

    return parser.parse_args()


# =============================================================================
# STRUCTURAL LINE REMOVAL
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

    combined = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines,
    )

    return (
        horizontal_lines,
        vertical_lines,
        combined,
    )


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
# SMALL COMPONENT CLEANUP
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
# TEXT-LINE GROUPING
# =============================================================================

def group_characters_into_lines(
    text_mask: np.ndarray,
) -> np.ndarray:
    height, width = (
        text_mask.shape[:2]
    )

    horizontal_kernel_width = max(
        11,
        width // 45,
    )

    vertical_kernel_height = max(
        2,
        height // 300,
    )

    horizontal_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                horizontal_kernel_width,
                vertical_kernel_height,
            ),
        )
    )

    grouped = cv2.dilate(
        text_mask,
        horizontal_kernel,
        iterations=1,
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
# TEXT-LINE EXTRACTION
# =============================================================================

def find_text_lines(
    grouped_mask: np.ndarray,
) -> list[
    dict[str, int | float]
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

    lines = []

    for contour in contours:
        x, y, w, h = (
            cv2.boundingRect(
                contour
            )
        )

        area_fraction = (
            float(
                w * h
            )
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

        if (
            area_fraction
            < MIN_LINE_AREA_FRACTION
        ):
            continue

        if (
            width_fraction
            < MIN_LINE_WIDTH_FRACTION
        ):
            continue

        if (
            width_fraction
            > MAX_LINE_WIDTH_FRACTION
        ):
            continue

        if (
            height_fraction
            < MIN_LINE_HEIGHT_FRACTION
        ):
            continue

        if (
            height_fraction
            > MAX_LINE_HEIGHT_FRACTION
        ):
            continue

        if (
            aspect_ratio
            < MIN_ASPECT_RATIO
        ):
            continue

        lines.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area_fraction":
                    area_fraction,
                "aspect_ratio":
                    aspect_ratio,
            }
        )

    lines.sort(
        key=lambda item: (
            int(
                item["y"]
            ),
            int(
                item["x"]
            ),
        )
    )

    return lines


# =============================================================================
# BLOCK MERGING
# =============================================================================

def horizontal_overlap_ratio(
    a: dict[str, int | float],
    b: dict[str, int | float],
) -> float:
    a_left = int(
        a["x"]
    )

    a_right = (
        a_left
        + int(
            a["w"]
        )
    )

    b_left = int(
        b["x"]
    )

    b_right = (
        b_left
        + int(
            b["w"]
        )
    )

    overlap = max(
        0,
        min(
            a_right,
            b_right,
        )
        - max(
            a_left,
            b_left,
        ),
    )

    minimum_width = min(
        int(
            a["w"]
        ),
        int(
            b["w"]
        ),
    )

    if minimum_width <= 0:
        return 0.0

    return (
        overlap
        / float(
            minimum_width
        )
    )


def merge_two_regions(
    a: dict[str, int | float],
    b: dict[str, int | float],
) -> dict[str, int | float]:
    left = min(
        int(
            a["x"]
        ),
        int(
            b["x"]
        ),
    )

    top = min(
        int(
            a["y"]
        ),
        int(
            b["y"]
        ),
    )

    right = max(
        int(
            a["x"]
        )
        + int(
            a["w"]
        ),
        int(
            b["x"]
        )
        + int(
            b["w"]
        ),
    )

    bottom = max(
        int(
            a["y"]
        )
        + int(
            a["h"]
        ),
        int(
            b["y"]
        )
        + int(
            b["h"]
        ),
    )

    return {
        "x": left,
        "y": top,
        "w": right - left,
        "h": bottom - top,
    }


def should_merge_lines(
    current: dict[str, int | float],
    candidate: dict[str, int | float],
) -> bool:
    current_bottom = (
        int(
            current["y"]
        )
        + int(
            current["h"]
        )
    )

    candidate_top = int(
        candidate["y"]
    )

    vertical_gap = (
        candidate_top
        - current_bottom
    )

    average_height = (
        int(
            current["h"]
        )
        + int(
            candidate["h"]
        )
    ) / 2.0

    maximum_gap = (
        average_height
        * BLOCK_VERTICAL_GAP_FACTOR
    )

    if vertical_gap < 0:
        vertical_gap = 0

    if vertical_gap > maximum_gap:
        return False

    overlap = (
        horizontal_overlap_ratio(
            current,
            candidate,
        )
    )

    return (
        overlap
        >= BLOCK_HORIZONTAL_OVERLAP
    )


def merge_lines_into_blocks(
    lines: list[
        dict[str, int | float]
    ],
) -> list[
    dict[str, int | float]
]:
    if not lines:
        return []

    blocks = []

    used = [
        False
        for _ in lines
    ]

    for index, line in enumerate(
        lines
    ):
        if used[index]:
            continue

        current = {
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
        }

        used[index] = True

        changed = True

        while changed:
            changed = False

            for other_index, other in enumerate(
                lines
            ):
                if used[
                    other_index
                ]:
                    continue

                if should_merge_lines(
                    current,
                    other,
                ):
                    current = (
                        merge_two_regions(
                            current,
                            other,
                        )
                    )

                    used[
                        other_index
                    ] = True

                    changed = True

        blocks.append(
            current
        )

    blocks.sort(
        key=lambda item: (
            int(
                item["y"]
            ),
            int(
                item["x"]
            ),
        )
    )

    return blocks


# =============================================================================
# BLOCK FILTERING
# =============================================================================

def filter_blocks(
    blocks: list[
        dict[str, int | float]
    ],
    image_shape: tuple[
        int,
        int,
    ],
) -> list[
    dict[str, int | float]
]:
    height, width = (
        image_shape
    )

    image_area = float(
        height
        * width
    )

    filtered = []

    for block in blocks:
        x = int(
            block["x"]
        )

        y = int(
            block["y"]
        )

        w = int(
            block["w"]
        )

        h = int(
            block["h"]
        )

        area_fraction = (
            w
            * h
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

        if area_fraction < 0.0007:
            continue

        if width_fraction < 0.035:
            continue

        if height_fraction < 0.012:
            continue

        if height_fraction > 0.50:
            continue

        filtered.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area_fraction":
                    area_fraction,
            }
        )

    return filtered


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_regions(
    image: np.ndarray,
    regions: list[
        dict[str, int | float]
    ],
    prefix: str,
    color: tuple[
        int,
        int,
        int,
    ],
) -> np.ndarray:
    preview = image.copy()

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
            color,
            2,
        )

        cv2.putText(
            preview,
            f"{prefix}{index}",
            (
                x,
                max(
                    18,
                    y - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return preview


def create_region_mask(
    shape: tuple[
        int,
        int,
    ],
    regions: list[
        dict[str, int | float]
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
# CROPS
# =============================================================================

def save_block_crops(
    image: np.ndarray,
    blocks: list[
        dict[str, int | float]
    ],
    output_dir: Path,
) -> None:
    crop_dir = (
        output_dir
        / "blocks"
    )

    crop_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, block in enumerate(
        blocks,
        start=1,
    ):
        x = int(
            block["x"]
        )

        y = int(
            block["y"]
        )

        w = int(
            block["w"]
        )

        h = int(
            block["h"]
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
                / f"block_{index:03d}.png"
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
            f"Input not found:\n"
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
        "EXPERIMENT 12 — "
        "REFINED TEXT REGION DETECTION"
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
    # Remove structural lines
    # -------------------------------------------------------------------------

    (
        horizontal_lines,
        vertical_lines,
        line_mask,
    ) = detect_structural_lines(
        scan
    )

    text_mask = (
        remove_structural_lines(
            scan,
            line_mask,
        )
    )

    # -------------------------------------------------------------------------
    # Suppress page-border artifacts
    # -------------------------------------------------------------------------

    border_cleaned = (
        suppress_page_border(
            text_mask
        )
    )

    # -------------------------------------------------------------------------
    # Remove tiny connected-component noise
    # -------------------------------------------------------------------------

    component_cleaned = (
        remove_tiny_components(
            border_cleaned
        )
    )

    # -------------------------------------------------------------------------
    # Group into lines
    # -------------------------------------------------------------------------

    grouped_lines = (
        group_characters_into_lines(
            component_cleaned
        )
    )

    lines = (
        find_text_lines(
            grouped_lines
        )
    )

    # -------------------------------------------------------------------------
    # Merge lines into blocks
    # -------------------------------------------------------------------------

    raw_blocks = (
        merge_lines_into_blocks(
            lines
        )
    )

    blocks = filter_blocks(
        raw_blocks,
        scan.shape,
    )

    # -------------------------------------------------------------------------
    # Visuals
    # -------------------------------------------------------------------------

    corrected_bgr = (
        cv2.cvtColor(
            corrected,
            cv2.COLOR_GRAY2BGR,
        )
    )

    line_preview = (
        draw_regions(
            corrected_bgr,
            lines,
            "L",
            (255, 0, 0),
        )
    )

    block_preview = (
        draw_regions(
            corrected_bgr,
            blocks,
            "B",
            (0, 255, 0),
        )
    )

    block_mask = (
        create_region_mask(
            scan.shape,
            blocks,
        )
    )

    # -------------------------------------------------------------------------
    # Save
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

        "06_structural_lines":
            line_mask,

        "07_text_without_lines":
            text_mask,

        "08_border_suppressed":
            border_cleaned,

        "09_component_cleaned":
            component_cleaned,

        "10_grouped_lines":
            grouped_lines,

        "11_line_regions":
            line_preview,

        "12_block_mask":
            block_mask,

        "13_final_blocks":
            block_preview,
    }

    for name, output in outputs.items():
        cv2.imwrite(
            str(
                output_dir
                / f"{name}.png"
            ),
            output,
        )

    save_block_crops(
        corrected,
        blocks,
        output_dir,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    block_coverage = sum(
        float(
            block[
                "area_fraction"
            ]
        )
        for block in blocks
    )

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 12 SUMMARY"
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
        f"Text lines detected  : "
        f"{len(lines)}"
    )

    print(
        f"Raw merged blocks    : "
        f"{len(raw_blocks)}"
    )

    print(
        f"Final blocks         : "
        f"{len(blocks)}"
    )

    print(
        f"Final block coverage : "
        f"{block_coverage:.4f}"
    )

    print()

    if blocks:
        print(
            "FINAL BLOCKS"
        )

        print(
            "-" * 100
        )

        for index, block in enumerate(
            blocks,
            start=1,
        ):
            print(
                f"B{index:02d} "
                f"| x={int(block['x']):4d} "
                f"| y={int(block['y']):4d} "
                f"| w={int(block['w']):4d} "
                f"| h={int(block['h']):4d} "
                f"| area="
                f"{float(block['area_fraction']):.4f}"
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
        "  08_border_suppressed.png"
    )

    print(
        "  10_grouped_lines.png"
    )

    print(
        "  11_line_regions.png"
    )

    print(
        "  13_final_blocks.png"
    )


if __name__ == "__main__":
    main()