from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from document_scanner import (
    detect_document,
    four_point_transform,
)


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 9 - compare document "
            "illumination and shadow correction methods."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original document photograph.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for Experiment 9 outputs.",
    )

    return parser.parse_args()


# =============================================================================
# BASIC PREPROCESSING
# =============================================================================

def to_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def clahe_enhancement(
    gray: np.ndarray,
) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(
        gray
    )


# =============================================================================
# BACKGROUND / ILLUMINATION ESTIMATION
# =============================================================================

def estimate_background(
    gray: np.ndarray,
) -> np.ndarray:
    height, width = (
        gray.shape[:2]
    )

    minimum_dimension = min(
        height,
        width,
    )

    kernel_size = int(
        minimum_dimension
        * 0.08
    )

    kernel_size = max(
        31,
        kernel_size,
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    background = cv2.GaussianBlur(
        gray,
        (
            kernel_size,
            kernel_size,
        ),
        0,
    )

    return background


# =============================================================================
# DIVISION NORMALIZATION
# =============================================================================

def division_normalization(
    gray: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    gray_float = (
        gray.astype(
            np.float32
        )
    )

    background_float = (
        background.astype(
            np.float32
        )
    )

    background_float = np.maximum(
        background_float,
        1.0,
    )

    normalized = (
        gray_float
        / background_float
    )

    normalized *= 220.0

    normalized = np.clip(
        normalized,
        0,
        255,
    )

    return normalized.astype(
        np.uint8
    )


# =============================================================================
# MORPHOLOGICAL SHADOW CORRECTION
# =============================================================================

def morphological_shadow_correction(
    gray: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    kernel_size = 31

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                kernel_size,
                kernel_size,
            ),
        )
    )

    background = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        kernel,
    )

    corrected = cv2.divide(
        gray,
        background,
        scale=255,
    )

    corrected = cv2.normalize(
        corrected,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    return (
        background,
        corrected,
    )


# =============================================================================
# BINARY OUTPUTS
# =============================================================================

def adaptive_binary(
    gray: np.ndarray,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )


def otsu_binary(
    gray: np.ndarray,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        (
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        ),
    )

    return binary


# =============================================================================
# QUALITY STATISTICS
# =============================================================================

def image_statistics(
    image: np.ndarray,
) -> dict[str, float]:
    return {
        "mean":
            float(
                np.mean(
                    image
                )
            ),

        "std":
            float(
                np.std(
                    image
                )
            ),

        "min":
            float(
                np.min(
                    image
                )
            ),

        "max":
            float(
                np.max(
                    image
                )
            ),
    }


def print_statistics(
    name: str,
    image: np.ndarray,
) -> None:
    stats = image_statistics(
        image
    )

    print(
        f"{name:<28} "
        f"| mean={stats['mean']:7.2f} "
        f"| std={stats['std']:7.2f} "
        f"| min={stats['min']:6.1f} "
        f"| max={stats['max']:6.1f}"
    )


# =============================================================================
# CONTACT SHEET
# =============================================================================

def label_image(
    image: np.ndarray,
    label: str,
) -> np.ndarray:
    if len(
        image.shape
    ) == 2:
        preview = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    else:
        preview = image.copy()

    cv2.rectangle(
        preview,
        (0, 0),
        (
            preview.shape[1],
            45,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        preview,
        label,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return preview


def resize_for_contact_sheet(
    image: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    return cv2.resize(
        image,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_AREA,
    )


def make_contact_sheet(
    images: list[
        tuple[
            str,
            np.ndarray,
        ]
    ],
) -> np.ndarray:
    tile_width = 600
    tile_height = 800

    tiles = []

    for label, image in images:
        labeled = label_image(
            image,
            label,
        )

        resized = (
            resize_for_contact_sheet(
                labeled,
                tile_width,
                tile_height,
            )
        )

        tiles.append(
            resized
        )

    rows = []

    for index in range(
        0,
        len(tiles),
        2,
    ):
        first = tiles[
            index
        ]

        if (
            index + 1
            < len(tiles)
        ):
            second = tiles[
                index + 1
            ]

        else:
            second = np.zeros_like(
                first
            )

        row = np.hstack(
            [
                first,
                second,
            ]
        )

        rows.append(
            row
        )

    return np.vstack(
        rows
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
        "EXPERIMENT 9 — "
        "ILLUMINATION AND SHADOW CORRECTION"
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
    # Frozen detector
    # -------------------------------------------------------------------------

    print(
        "Running frozen production detector..."
    )

    (
        document_points,
        _edges,
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

    # -------------------------------------------------------------------------
    # Grayscale
    # -------------------------------------------------------------------------

    gray = to_grayscale(
        perspective
    )

    # -------------------------------------------------------------------------
    # Existing CLAHE baseline
    # -------------------------------------------------------------------------

    clahe = clahe_enhancement(
        gray
    )

    # -------------------------------------------------------------------------
    # Smooth illumination estimation
    # -------------------------------------------------------------------------

    illumination = (
        estimate_background(
            gray
        )
    )

    division = (
        division_normalization(
            gray,
            illumination,
        )
    )

    division_clahe = (
        clahe_enhancement(
            division
        )
    )

    # -------------------------------------------------------------------------
    # Morphological illumination correction
    # -------------------------------------------------------------------------

    (
        morph_background,
        morph_corrected,
    ) = (
        morphological_shadow_correction(
            gray
        )
    )

    morph_clahe = (
        clahe_enhancement(
            morph_corrected
        )
    )

    # -------------------------------------------------------------------------
    # Binary versions
    # -------------------------------------------------------------------------

    baseline_binary = (
        adaptive_binary(
            clahe
        )
    )

    division_binary = (
        adaptive_binary(
            division_clahe
        )
    )

    morph_binary = (
        adaptive_binary(
            morph_clahe
        )
    )

    division_otsu = (
        otsu_binary(
            division_clahe
        )
    )

    morph_otsu = (
        otsu_binary(
            morph_clahe
        )
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    outputs = {
        "01_perspective":
            perspective,

        "02_gray":
            gray,

        "03_clahe_baseline":
            clahe,

        "04_estimated_illumination":
            illumination,

        "05_division_normalized":
            division,

        "06_division_clahe":
            division_clahe,

        "07_morph_background":
            morph_background,

        "08_morph_corrected":
            morph_corrected,

        "09_morph_clahe":
            morph_clahe,

        "10_baseline_adaptive":
            baseline_binary,

        "11_division_adaptive":
            division_binary,

        "12_morph_adaptive":
            morph_binary,

        "13_division_otsu":
            division_otsu,

        "14_morph_otsu":
            morph_otsu,
    }

    for name, output in outputs.items():
        cv2.imwrite(
            str(
                output_dir
                / f"{name}.png"
            ),
            output,
        )

    # -------------------------------------------------------------------------
    # Contact sheet
    # -------------------------------------------------------------------------

    contact_sheet = (
        make_contact_sheet(
            [
                (
                    "Original perspective",
                    perspective,
                ),
                (
                    "Baseline CLAHE",
                    clahe,
                ),
                (
                    "Estimated illumination",
                    illumination,
                ),
                (
                    "Division normalized",
                    division,
                ),
                (
                    "Division + CLAHE",
                    division_clahe,
                ),
                (
                    "Morph corrected",
                    morph_corrected,
                ),
                (
                    "Baseline adaptive",
                    baseline_binary,
                ),
                (
                    "Division adaptive",
                    division_binary,
                ),
                (
                    "Morph adaptive",
                    morph_binary,
                ),
                (
                    "Division Otsu",
                    division_otsu,
                ),
                (
                    "Morph Otsu",
                    morph_otsu,
                ),
            ]
        )
    )

    cv2.imwrite(
        str(
            output_dir
            / "15_comparison_sheet.png"
        ),
        contact_sheet,
    )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "IMAGE STATISTICS"
    )

    print(
        "=" * 100
    )

    print_statistics(
        "Original gray",
        gray,
    )

    print_statistics(
        "Baseline CLAHE",
        clahe,
    )

    print_statistics(
        "Division normalized",
        division,
    )

    print_statistics(
        "Division + CLAHE",
        division_clahe,
    )

    print_statistics(
        "Morph corrected",
        morph_corrected,
    )

    print_statistics(
        "Morph + CLAHE",
        morph_clahe,
    )

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 9 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"Detector route : {route}"
    )

    print(
        f"Outputs        : "
        f"{output_dir}"
    )

    print()

    print(
        "Most important comparison:"
    )

    print(
        "  03_clahe_baseline.png"
    )

    print(
        "  06_division_clahe.png"
    )

    print(
        "  09_morph_clahe.png"
    )

    print(
        "  10_baseline_adaptive.png"
    )

    print(
        "  11_division_adaptive.png"
    )

    print(
        "  12_morph_adaptive.png"
    )

    print(
        "  15_comparison_sheet.png"
    )


if __name__ == "__main__":
    main()