from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from document_scanner import (
    detect_document,
    four_point_transform,
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SMARTDOC_ROOT = (
    PROJECT_ROOT
    / "data"
    / "smartdoc_frames"
)

GT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "smartdoc"
    / "sampleDataset"
    / "input_sample_groundtruth"
    / "background00_gt"
)

CLASSROOM_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "document1.jpeg"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiment_10_batch_illumination_stress_test"
)

SUMMARY_CSV = (
    OUTPUT_ROOT
    / "experiment_10_summary.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MORPH_KERNEL_SIZE = 31

ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 15

SMALL_COMPONENT_AREA = 12


# =============================================================================
# SMARTDOC GT VALIDITY
# =============================================================================

def load_valid_frames(
    xml_path: Path,
) -> set[int]:
    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()

    segmentation = root.find(
        "segmentation_results"
    )

    if segmentation is None:
        raise RuntimeError(
            f"No segmentation_results in:\n"
            f"{xml_path}"
        )

    valid_frames = set()

    for frame in segmentation.findall(
        "frame"
    ):
        rejected = (
            frame.attrib.get(
                "rejected",
                "false",
            ).lower()
            == "true"
        )

        if rejected:
            continue

        frame_index = int(
            frame.attrib["index"]
        )

        valid_frames.add(
            frame_index
        )

    return valid_frames


def extract_frame_index(
    path: Path,
) -> int:
    match = re.search(
        r"_frame_(\d+)$",
        path.stem,
    )

    if match is None:
        raise RuntimeError(
            f"Cannot parse frame index: "
            f"{path.name}"
        )

    # SmartDoc XML is 1-based.
    return int(
        match.group(1)
    ) + 1


# =============================================================================
# BASELINE PIPELINE
# =============================================================================

def grayscale(
    image: np.ndarray,
) -> np.ndarray:
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def apply_clahe(
    gray: np.ndarray,
) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(
        gray
    )


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
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )


# =============================================================================
# MORPHOLOGICAL ILLUMINATION CORRECTION
# =============================================================================

def morphological_correction(
    gray: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE,
        ),
    )

    background = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        kernel,
    )

    safe_background = np.maximum(
        background,
        1,
    )

    corrected = cv2.divide(
        gray,
        safe_background,
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
# METRICS
# =============================================================================

def background_uniformity(
    gray: np.ndarray,
) -> float:
    """
    Estimate background uniformity using the brighter
    portion of the image.

    Lower standard deviation = more uniform background.
    """

    threshold = float(
        np.percentile(
            gray,
            70,
        )
    )

    background_pixels = gray[
        gray >= threshold
    ]

    if (
        background_pixels.size
        < 100
    ):
        return float(
            np.std(
                gray
            )
        )

    return float(
        np.std(
            background_pixels
        )
    )


def foreground_ratio(
    binary: np.ndarray,
) -> float:
    """
    Fraction of pixels classified as black foreground.
    """

    black_pixels = np.count_nonzero(
        binary == 0
    )

    total_pixels = (
        binary.shape[0]
        * binary.shape[1]
    )

    if total_pixels <= 0:
        return 0.0

    return (
        black_pixels
        / total_pixels
    )


def small_component_noise_ratio(
    binary: np.ndarray,
) -> float:
    """
    Estimate noise by looking at tiny connected black components.

    Returns:
        small foreground component pixels /
        total foreground pixels
    """

    foreground = cv2.bitwise_not(
        binary
    )

    (
        component_count,
        _labels,
        stats,
        _centroids,
    ) = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )

    total_foreground = int(
        np.count_nonzero(
            foreground
        )
    )

    if total_foreground <= 0:
        return 0.0

    small_pixels = 0

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

        if area <= SMALL_COMPONENT_AREA:
            small_pixels += area

    return (
        small_pixels
        / total_foreground
    )


def binary_difference_ratio(
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> float:
    difference = cv2.absdiff(
        baseline,
        candidate,
    )

    changed = np.count_nonzero(
        difference
    )

    total = (
        difference.shape[0]
        * difference.shape[1]
    )

    if total <= 0:
        return 0.0

    return changed / total


# =============================================================================
# VISUALIZATION
# =============================================================================

def label_image(
    image: np.ndarray,
    text: str,
) -> np.ndarray:
    if image.ndim == 2:
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
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return preview


def make_comparison(
    perspective: np.ndarray,
    baseline_gray: np.ndarray,
    corrected_gray: np.ndarray,
    baseline_binary: np.ndarray,
    candidate_binary: np.ndarray,
) -> np.ndarray:
    target_width = 500
    target_height = 650

    images = [
        (
            "Perspective",
            perspective,
        ),
        (
            "Baseline CLAHE",
            baseline_gray,
        ),
        (
            "Morph corrected + CLAHE",
            corrected_gray,
        ),
        (
            "Baseline adaptive",
            baseline_binary,
        ),
        (
            "Morph adaptive",
            candidate_binary,
        ),
    ]

    tiles = []

    for label, image in images:
        labeled = label_image(
            image,
            label,
        )

        resized = cv2.resize(
            labeled,
            (
                target_width,
                target_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        tiles.append(
            resized
        )

    blank = np.zeros_like(
        tiles[0]
    )

    row1 = np.hstack(
        [
            tiles[0],
            tiles[1],
        ]
    )

    row2 = np.hstack(
        [
            tiles[2],
            tiles[3],
        ]
    )

    row3 = np.hstack(
        [
            tiles[4],
            blank,
        ]
    )

    return np.vstack(
        [
            row1,
            row2,
            row3,
        ]
    )


# =============================================================================
# SINGLE IMAGE PROCESSING
# =============================================================================

def process_image(
    image_path: Path,
    output_dir: Path,
    sample_name: str,
) -> dict[str, object]:
    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:
        raise RuntimeError(
            f"Could not load:\n"
            f"{image_path}"
        )

    (
        document_points,
        _edges,
        route,
    ) = detect_document(
        image
    )

    perspective = four_point_transform(
        image,
        document_points,
    )

    gray = grayscale(
        perspective
    )

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------

    baseline_clahe = apply_clahe(
        gray
    )

    baseline_binary = adaptive_binary(
        baseline_clahe
    )

    # -------------------------------------------------------------------------
    # Candidate
    # -------------------------------------------------------------------------

    (
        morph_background,
        morph_corrected,
    ) = morphological_correction(
        gray
    )

    morph_clahe = apply_clahe(
        morph_corrected
    )

    morph_binary = adaptive_binary(
        morph_clahe
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    baseline_uniformity = (
        background_uniformity(
            baseline_clahe
        )
    )

    morph_uniformity = (
        background_uniformity(
            morph_clahe
        )
    )

    baseline_foreground = (
        foreground_ratio(
            baseline_binary
        )
    )

    morph_foreground = (
        foreground_ratio(
            morph_binary
        )
    )

    baseline_noise = (
        small_component_noise_ratio(
            baseline_binary
        )
    )

    morph_noise = (
        small_component_noise_ratio(
            morph_binary
        )
    )

    difference_ratio = (
        binary_difference_ratio(
            baseline_binary,
            morph_binary,
        )
    )

    uniformity_delta = (
        morph_uniformity
        - baseline_uniformity
    )

    foreground_delta = (
        morph_foreground
        - baseline_foreground
    )

    noise_delta = (
        morph_noise
        - baseline_noise
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            output_dir
            / "01_perspective.png"
        ),
        perspective,
    )

    cv2.imwrite(
        str(
            output_dir
            / "02_baseline_clahe.png"
        ),
        baseline_clahe,
    )

    cv2.imwrite(
        str(
            output_dir
            / "03_morph_background.png"
        ),
        morph_background,
    )

    cv2.imwrite(
        str(
            output_dir
            / "04_morph_corrected.png"
        ),
        morph_corrected,
    )

    cv2.imwrite(
        str(
            output_dir
            / "05_morph_clahe.png"
        ),
        morph_clahe,
    )

    cv2.imwrite(
        str(
            output_dir
            / "06_baseline_binary.png"
        ),
        baseline_binary,
    )

    cv2.imwrite(
        str(
            output_dir
            / "07_morph_binary.png"
        ),
        morph_binary,
    )

    comparison = make_comparison(
        perspective,
        baseline_clahe,
        morph_clahe,
        baseline_binary,
        morph_binary,
    )

    cv2.imwrite(
        str(
            output_dir
            / "08_comparison.png"
        ),
        comparison,
    )

    return {
        "sample":
            sample_name,

        "route":
            route,

        "baseline_background_std":
            baseline_uniformity,

        "morph_background_std":
            morph_uniformity,

        "background_std_delta":
            uniformity_delta,

        "baseline_foreground_ratio":
            baseline_foreground,

        "morph_foreground_ratio":
            morph_foreground,

        "foreground_ratio_delta":
            foreground_delta,

        "baseline_noise_ratio":
            baseline_noise,

        "morph_noise_ratio":
            morph_noise,

        "noise_ratio_delta":
            noise_delta,

        "binary_difference_ratio":
            difference_ratio,
    }


# =============================================================================
# SMARTDOC
# =============================================================================

def run_smartdoc() -> list[
    dict[str, object]
]:
    image_paths = sorted(
        SMARTDOC_ROOT.rglob(
            "*.png"
        )
    )

    video_names = sorted(
        {
            path.parent.name
            for path in image_paths
        }
    )

    valid_frame_cache = {}

    for video_name in video_names:
        valid_frame_cache[
            video_name
        ] = load_valid_frames(
            GT_ROOT
            / f"{video_name}.gt.xml"
        )

    rows = []

    print()
    print(
        "=" * 100
    )

    print(
        "SMARTDOC BATCH STRESS TEST"
    )

    print(
        "=" * 100
    )

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        video_name = (
            image_path.parent.name
        )

        frame_index = (
            extract_frame_index(
                image_path
            )
        )

        if (
            frame_index
            not in valid_frame_cache[
                video_name
            ]
        ):
            print(
                f"[{index:02d}] "
                f"Skipping rejected frame"
            )

            continue

        sample_name = str(
            image_path.relative_to(
                SMARTDOC_ROOT
            )
        )

        output_dir = (
            OUTPUT_ROOT
            / "smartdoc"
            / video_name
            / image_path.stem
        )

        try:
            result = process_image(
                image_path,
                output_dir,
                sample_name,
            )

        except Exception as error:
            print(
                f"[{index:02d}] FAIL "
                f"{image_path.name}: "
                f"{error}"
            )

            continue

        rows.append(
            result
        )

        print(
            f"[{index:02d}/"
            f"{len(image_paths):02d}] "
            f"{image_path.name}"
        )

        print(
            f"    route       : "
            f"{result['route']}"
        )

        print(
            f"    bg std      : "
            f"{result['baseline_background_std']:.3f}"
            f" -> "
            f"{result['morph_background_std']:.3f}"
        )

        print(
            f"    foreground  : "
            f"{result['baseline_foreground_ratio']:.4f}"
            f" -> "
            f"{result['morph_foreground_ratio']:.4f}"
        )

        print(
            f"    noise       : "
            f"{result['baseline_noise_ratio']:.4f}"
            f" -> "
            f"{result['morph_noise_ratio']:.4f}"
        )

        print(
            f"    bin changed : "
            f"{result['binary_difference_ratio']:.4f}"
        )

    return rows


# =============================================================================
# CLASSROOM
# =============================================================================

def run_classroom() -> dict[
    str,
    object,
]:
    print()
    print(
        "=" * 100
    )

    print(
        "CLASSROOM CROSS-DOMAIN TEST"
    )

    print(
        "=" * 100
    )

    result = process_image(
        CLASSROOM_IMAGE,
        OUTPUT_ROOT
        / "classroom",
        "classroom/document1.jpeg",
    )

    print(
        f"Route                  : "
        f"{result['route']}"
    )

    print(
        f"Baseline background std: "
        f"{result['baseline_background_std']:.3f}"
    )

    print(
        f"Morph background std   : "
        f"{result['morph_background_std']:.3f}"
    )

    print(
        f"Baseline foreground    : "
        f"{result['baseline_foreground_ratio']:.4f}"
    )

    print(
        f"Morph foreground       : "
        f"{result['morph_foreground_ratio']:.4f}"
    )

    print(
        f"Binary difference      : "
        f"{result['binary_difference_ratio']:.4f}"
    )

    return result


# =============================================================================
# SUMMARY
# =============================================================================

def summarize(
    rows: list[
        dict[str, object]
    ],
) -> None:
    if not rows:
        raise RuntimeError(
            "No SmartDoc samples evaluated."
        )

    baseline_background = np.array(
        [
            row[
                "baseline_background_std"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    morph_background = np.array(
        [
            row[
                "morph_background_std"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    baseline_foreground = np.array(
        [
            row[
                "baseline_foreground_ratio"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    morph_foreground = np.array(
        [
            row[
                "morph_foreground_ratio"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    baseline_noise = np.array(
        [
            row[
                "baseline_noise_ratio"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    morph_noise = np.array(
        [
            row[
                "morph_noise_ratio"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    difference = np.array(
        [
            row[
                "binary_difference_ratio"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    improved_background = int(
        np.sum(
            morph_background
            < baseline_background
        )
    )

    worse_background = int(
        np.sum(
            morph_background
            > baseline_background
        )
    )

    extreme_foreground_change = int(
        np.sum(
            np.abs(
                morph_foreground
                - baseline_foreground
            )
            > 0.10
        )
    )

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 10 SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"SmartDoc samples              : "
        f"{len(rows)}"
    )

    print()

    print(
        "BACKGROUND UNIFORMITY"
    )

    print(
        f"Baseline mean std             : "
        f"{np.mean(baseline_background):.4f}"
    )

    print(
        f"Morph mean std                : "
        f"{np.mean(morph_background):.4f}"
    )

    print(
        f"Improved samples              : "
        f"{improved_background}/"
        f"{len(rows)}"
    )

    print(
        f"Worse samples                 : "
        f"{worse_background}/"
        f"{len(rows)}"
    )

    print()

    print(
        "BINARY FOREGROUND"
    )

    print(
        f"Baseline mean foreground      : "
        f"{np.mean(baseline_foreground):.4f}"
    )

    print(
        f"Morph mean foreground         : "
        f"{np.mean(morph_foreground):.4f}"
    )

    print(
        f"Extreme foreground changes    : "
        f"{extreme_foreground_change}"
    )

    print()

    print(
        "SMALL-COMPONENT NOISE"
    )

    print(
        f"Baseline mean noise ratio     : "
        f"{np.mean(baseline_noise):.6f}"
    )

    print(
        f"Morph mean noise ratio        : "
        f"{np.mean(morph_noise):.6f}"
    )

    print()

    print(
        "BASELINE VS MORPH BINARY"
    )

    print(
        f"Mean changed-pixel ratio      : "
        f"{np.mean(difference):.4f}"
    )

    print(
        f"Maximum changed-pixel ratio   : "
        f"{np.max(difference):.4f}"
    )

    print(
        f"Minimum changed-pixel ratio   : "
        f"{np.min(difference):.4f}"
    )


# =============================================================================
# CSV
# =============================================================================

def save_summary_csv(
    rows: list[
        dict[str, object]
    ],
) -> None:
    fieldnames = [
        "sample",
        "route",
        "baseline_background_std",
        "morph_background_std",
        "background_std_delta",
        "baseline_foreground_ratio",
        "morph_foreground_ratio",
        "foreground_ratio_delta",
        "baseline_noise_ratio",
        "morph_noise_ratio",
        "noise_ratio_delta",
        "binary_difference_ratio",
    ]

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            formatted = row.copy()

            for key in fieldnames:
                if isinstance(
                    formatted.get(key),
                    float,
                ):
                    formatted[key] = (
                        f"{formatted[key]:.8f}"
                    )

            writer.writerow(
                formatted
            )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 10 — "
        "BATCH ILLUMINATION STRESS TEST"
    )

    print(
        "=" * 100
    )

    print(
        "Boundary detector is frozen."
    )

    print(
        "Testing baseline CLAHE + adaptive "
        "vs morph correction + CLAHE + adaptive."
    )

    classroom_result = (
        run_classroom()
    )

    smartdoc_rows = (
        run_smartdoc()
    )

    save_summary_csv(
        smartdoc_rows
    )

    summarize(
        smartdoc_rows
    )

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 10 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"CSV:\n{SUMMARY_CSV}"
    )

    print()

    print(
        "Classroom comparison:"
    )

    print(
        OUTPUT_ROOT
        / "classroom"
        / "08_comparison.png"
    )

    print()

    print(
        "Interpretation rule:"
    )

    print(
        "- Lower background std is better."
    )

    print(
        "- Foreground ratio should remain "
        "reasonably close to baseline."
    )

    print(
        "- Lower small-component noise is better."
    )

    print(
        "- Large binary changes require visual review."
    )

    print()

    print(
        f"Classroom route confirmed as: "
        f"{classroom_result['route']}"
    )


if __name__ == "__main__":
    main()