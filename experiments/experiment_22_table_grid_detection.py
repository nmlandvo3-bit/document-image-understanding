
"""
Experiment 22: Table Grid Detection

Detect likely horizontal and vertical table borders from an existing
perspective-corrected document image.

This is a diagnostic experiment. It does not reconstruct cells or
modify the existing OCR pipeline.
"""

from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_IMAGE = (
    PROJECT_ROOT
    / "results"
    / "gui_v4"
    / "single"
    / "pic1"
    / "perspective.png"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "grid_detection"
)


def save_image(filename: str, image: np.ndarray) -> None:
    output_path = OUTPUT_DIR / filename

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not save image: {output_path}")

    print(f"[OK] {output_path.name}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Could not read perspective image: {INPUT_IMAGE}"
        )

    height, width = image.shape[:2]

    print("=" * 65)
    print("EXPERIMENT 22: TABLE GRID DETECTION")
    print("=" * 65)
    print(f"Input image: {INPUT_IMAGE}")
    print(f"Image dimensions: {width} x {height}")

    # 1. Convert the document into grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Convert dark printed content into white foreground pixels.
    # Adaptive thresholding helps with uneven document illumination.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    # 3. Extract long horizontal structures.
    horizontal_kernel_width = max(20, width // 30)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_kernel_width, 1),
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    # 4. Extract long vertical structures.
    vertical_kernel_height = max(20, height // 30)

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_kernel_height),
    )

    vertical_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    # 5. Combine both sets of possible table borders.
    grid_mask = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines,
    )

    # 6. Draw the detections over the original corrected document.
    overlay = image.copy()

    # Horizontal lines are shown in green.
    overlay[horizontal_lines > 0] = (0, 255, 0)

    # Vertical lines are shown in red.
    overlay[vertical_lines > 0] = (0, 0, 255)

    # Intersections are shown in yellow.
    intersections = cv2.bitwise_and(
        horizontal_lines,
        vertical_lines,
    )

    overlay[intersections > 0] = (0, 255, 255)

    # 7. Save diagnostic images.
    save_image("01_binary.png", binary)
    save_image("02_horizontal_lines.png", horizontal_lines)
    save_image("03_vertical_lines.png", vertical_lines)
    save_image("04_combined_grid.png", grid_mask)
    save_image("05_grid_overlay.png", overlay)

    print("-" * 65)
    print(
        "Horizontal kernel width: "
        f"{horizontal_kernel_width} pixels"
    )
    print(
        "Vertical kernel height: "
        f"{vertical_kernel_height} pixels"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 65)
    print("GRID DETECTION EXPERIMENT COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()