from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from document_scanner import (
    detect_document,
    order_points,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FRAME_ROOT = (
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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiment_7_batch_compare_detectors"
)

SUMMARY_CSV = (
    OUTPUT_ROOT
    / "experiment_7_summary.csv"
)

RESIZE_HEIGHT = 700


# =============================================================================
# SMARTDOC GROUND TRUTH
# =============================================================================

def load_ground_truth(
    xml_path: Path,
) -> dict[int, dict[str, object]]:
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

    results: dict[
        int,
        dict[str, object],
    ] = {}

    for frame in segmentation.findall(
        "frame"
    ):
        frame_index = int(
            frame.attrib["index"]
        )

        rejected = (
            frame.attrib.get(
                "rejected",
                "false",
            ).lower()
            == "true"
        )

        entry: dict[
            str,
            object,
        ] = {
            "rejected": rejected,
        }

        if not rejected:
            points = {}

            for point in frame.findall(
                "point"
            ):
                points[
                    point.attrib["name"]
                ] = np.array(
                    [
                        float(
                            point.attrib["x"]
                        ),
                        float(
                            point.attrib["y"]
                        ),
                    ],
                    dtype=np.float32,
                )

            polygon = np.array(
                [
                    points["tl"],
                    points["tr"],
                    points["br"],
                    points["bl"],
                ],
                dtype=np.float32,
            )

            entry[
                "polygon"
            ] = order_points(
                polygon
            )

        results[
            frame_index
        ] = entry

    return results


# =============================================================================
# FRAME INDEX
# =============================================================================

def extract_zero_based_frame_index(
    path: Path,
) -> int:
    match = re.search(
        r"_frame_(\d+)$",
        path.stem,
    )

    if match is None:
        raise RuntimeError(
            f"Could not parse frame index "
            f"from:\n{path.name}"
        )

    return int(
        match.group(1)
    )


# =============================================================================
# METRICS
# =============================================================================

def polygon_iou(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    hull_a = cv2.convexHull(
        a.astype(
            np.float32
        )
    )

    hull_b = cv2.convexHull(
        b.astype(
            np.float32
        )
    )

    area_a = float(
        cv2.contourArea(
            hull_a
        )
    )

    area_b = float(
        cv2.contourArea(
            hull_b
        )
    )

    if (
        area_a <= 0.0
        or area_b <= 0.0
    ):
        return 0.0

    intersection, _ = (
        cv2.intersectConvexConvex(
            hull_a,
            hull_b,
        )
    )

    intersection = float(
        intersection
    )

    if not np.isfinite(
        intersection
    ):
        return 0.0

    intersection = float(
        np.clip(
            intersection,
            0.0,
            min(
                area_a,
                area_b,
            ),
        )
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return float(
        np.clip(
            intersection
            / union,
            0.0,
            1.0,
        )
    )


def mean_corner_error(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    distances = np.linalg.norm(
        predicted - target,
        axis=1,
    )

    return float(
        np.mean(
            distances
        )
    )


# =============================================================================
# ENHANCED EDGE DETECTOR
# =============================================================================

def resize_image(
    image: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
]:
    original_height = (
        image.shape[0]
    )

    scale = (
        original_height
        / float(
            RESIZE_HEIGHT
        )
    )

    resized_width = int(
        image.shape[1]
        / scale
    )

    resized = cv2.resize(
        image,
        (
            resized_width,
            RESIZE_HEIGHT,
        ),
    )

    return (
        resized,
        scale,
    )


def enhanced_edge_map(
    image: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    lightness = lab[
        :,
        :,
        0,
    ]

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(
        lightness
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0,
    )

    grad_x = cv2.Scharr(
        blurred,
        cv2.CV_32F,
        1,
        0,
    )

    grad_y = cv2.Scharr(
        blurred,
        cv2.CV_32F,
        0,
        1,
    )

    magnitude = cv2.magnitude(
        grad_x,
        grad_y,
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(
        np.uint8
    )

    median = float(
        np.median(
            enhanced
        )
    )

    lower = int(
        max(
            20,
            0.50 * median,
        )
    )

    upper = int(
        min(
            180,
            1.20 * median,
        )
    )

    canny = cv2.Canny(
        enhanced,
        lower,
        upper,
    )

    gradient_edges = cv2.threshold(
        magnitude,
        28,
        255,
        cv2.THRESH_BINARY,
    )[1]

    combined = cv2.bitwise_or(
        canny,
        gradient_edges,
    )

    close_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (11, 11),
        )
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2,
    )

    dilate_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3),
        )
    )

    combined = cv2.dilate(
        combined,
        dilate_kernel,
        iterations=1,
    )

    return combined


def find_enhanced_candidates(
    edge_map: np.ndarray,
) -> list[
    dict[str, object]
]:
    contours, _ = cv2.findContours(
        edge_map,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )[:50]

    image_area = (
        edge_map.shape[0]
        * edge_map.shape[1]
    )

    candidates = []

    for contour in contours:
        area = float(
            cv2.contourArea(
                contour
            )
        )

        if area <= 0:
            continue

        area_fraction = (
            area
            / image_area
        )

        if area_fraction < 0.02:
            continue

        hull = cv2.convexHull(
            contour
        )

        perimeter = float(
            cv2.arcLength(
                hull,
                True,
            )
        )

        for epsilon_fraction in (
            0.01,
            0.015,
            0.02,
            0.025,
            0.03,
            0.04,
        ):
            approximation = (
                cv2.approxPolyDP(
                    hull,
                    epsilon_fraction
                    * perimeter,
                    True,
                )
            )

            if len(
                approximation
            ) != 4:
                continue

            polygon = (
                approximation
                .reshape(4, 2)
                .astype(
                    np.float32
                )
            )

            polygon_area = float(
                abs(
                    cv2.contourArea(
                        polygon
                    )
                )
            )

            if polygon_area <= 0:
                continue

            rectangularity = (
                area
                / polygon_area
            )

            rectangularity = float(
                np.clip(
                    rectangularity,
                    0.0,
                    1.0,
                )
            )

            ordered = order_points(
                polygon
            )

            width_top = np.linalg.norm(
                ordered[1]
                - ordered[0]
            )

            width_bottom = np.linalg.norm(
                ordered[2]
                - ordered[3]
            )

            height_left = np.linalg.norm(
                ordered[3]
                - ordered[0]
            )

            height_right = np.linalg.norm(
                ordered[2]
                - ordered[1]
            )

            mean_width = (
                width_top
                + width_bottom
            ) / 2.0

            mean_height = (
                height_left
                + height_right
            ) / 2.0

            if (
                mean_width <= 0
                or mean_height <= 0
            ):
                continue

            aspect_score = (
                min(
                    mean_width,
                    mean_height,
                )
                / max(
                    mean_width,
                    mean_height,
                )
            )

            score = (
                area_fraction
                * 0.60
                + rectangularity
                * 0.25
                + aspect_score
                * 0.15
            )

            candidates.append(
                {
                    "score":
                        score,

                    "area_fraction":
                        area_fraction,

                    "rectangularity":
                        rectangularity,

                    "aspect_score":
                        aspect_score,

                    "polygon":
                        ordered,
                }
            )

            break

    candidates.sort(
        key=lambda item: item[
            "score"
        ],
        reverse=True,
    )

    return candidates


def detect_document_enhanced(
    image: np.ndarray,
) -> np.ndarray:
    resized, scale = (
        resize_image(
            image
        )
    )

    edge_map = (
        enhanced_edge_map(
            resized
        )
    )

    candidates = (
        find_enhanced_candidates(
            edge_map
        )
    )

    if not candidates:
        raise RuntimeError(
            "No enhanced-edge candidate "
            "found."
        )

    polygon = (
        candidates[0][
            "polygon"
        ]
        * scale
    )

    return order_points(
        polygon
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    contour = (
        polygon
        .astype(
            np.int32
        )
        .reshape(
            (-1, 1, 2)
        )
    )

    cv2.polylines(
        image,
        [contour],
        True,
        color,
        thickness,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        FRAME_ROOT.rglob(
            "*.png"
        )
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No SmartDoc frames found:\n"
            f"{FRAME_ROOT}"
        )

    video_names = sorted(
        {
            image_path.parent.name
            for image_path
            in image_paths
        }
    )

    gt_cache = {}

    for video_name in video_names:
        gt_path = (
            GT_ROOT
            / f"{video_name}.gt.xml"
        )

        gt_cache[
            video_name
        ] = load_ground_truth(
            gt_path
        )

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 7 — OLD VS ENHANCED "
        "DETECTOR BATCH COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        f"Samples: {len(image_paths)}"
    )

    print()

    rows = []

    baseline_ious = []
    enhanced_ious = []

    baseline_errors = []
    enhanced_errors = []

    baseline_failures = 0
    enhanced_failures = 0

    improved_count = 0
    regressed_count = 0
    unchanged_count = 0

    major_regressions = 0

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        video_name = (
            image_path.parent.name
        )

        zero_based_frame = (
            extract_zero_based_frame_index(
                image_path
            )
        )

        smartdoc_frame = (
            zero_based_frame
            + 1
        )

        gt_entry = (
            gt_cache[
                video_name
            ].get(
                smartdoc_frame
            )
        )

        if gt_entry is None:
            print(
                f"[{index:02d}] "
                f"[SKIP] Missing GT"
            )

            continue

        if bool(
            gt_entry["rejected"]
        ):
            print(
                f"[{index:02d}] "
                f"[SKIP] GT rejected"
            )

            continue

        target = np.asarray(
            gt_entry["polygon"],
            dtype=np.float32,
        )

        image = cv2.imread(
            str(
                image_path
            )
        )

        if image is None:
            print(
                f"[{index:02d}] "
                f"[FAIL] Cannot load image"
            )

            continue

        baseline_iou = None
        enhanced_iou = None

        baseline_error = None
        enhanced_error = None

        baseline_pred = None
        enhanced_pred = None

        try:
            baseline_raw, _ = (
                detect_document(
                    image
                )
            )

            baseline_pred = (
                order_points(
                    baseline_raw
                )
            )

            baseline_iou = polygon_iou(
                baseline_pred,
                target,
            )

            baseline_error = (
                mean_corner_error(
                    baseline_pred,
                    target,
                )
            )

            baseline_ious.append(
                baseline_iou
            )

            baseline_errors.append(
                baseline_error
            )

        except Exception:
            baseline_failures += 1

        try:
            enhanced_pred = (
                detect_document_enhanced(
                    image
                )
            )

            enhanced_iou = (
                polygon_iou(
                    enhanced_pred,
                    target,
                )
            )

            enhanced_error = (
                mean_corner_error(
                    enhanced_pred,
                    target,
                )
            )

            enhanced_ious.append(
                enhanced_iou
            )

            enhanced_errors.append(
                enhanced_error
            )

        except Exception:
            enhanced_failures += 1

        if (
            baseline_iou is not None
            and enhanced_iou is not None
        ):
            delta = (
                enhanced_iou
                - baseline_iou
            )

            if delta > 0.01:
                improved_count += 1

            elif delta < -0.01:
                regressed_count += 1

            else:
                unchanged_count += 1

            if delta < -0.10:
                major_regressions += 1

        else:
            delta = None

        print(
            f"[{index:02d}/"
            f"{len(image_paths):02d}] "
            f"{image_path.name}"
        )

        print(
            f"    baseline : "
            f"{baseline_iou:.4f}"
            if baseline_iou
            is not None
            else
            "    baseline : FAIL"
        )

        print(
            f"    enhanced : "
            f"{enhanced_iou:.4f}"
            if enhanced_iou
            is not None
            else
            "    enhanced : FAIL"
        )

        if delta is not None:
            print(
                f"    delta    : "
                f"{delta:+.4f}"
            )

        row = {
            "sample":
                str(
                    image_path.relative_to(
                        FRAME_ROOT
                    )
                ),

            "video":
                video_name,

            "smartdoc_frame":
                smartdoc_frame,

            "baseline_iou":
                (
                    f"{baseline_iou:.8f}"
                    if baseline_iou
                    is not None
                    else ""
                ),

            "enhanced_iou":
                (
                    f"{enhanced_iou:.8f}"
                    if enhanced_iou
                    is not None
                    else ""
                ),

            "iou_delta":
                (
                    f"{delta:.8f}"
                    if delta
                    is not None
                    else ""
                ),

            "baseline_corner_error":
                (
                    f"{baseline_error:.4f}"
                    if baseline_error
                    is not None
                    else ""
                ),

            "enhanced_corner_error":
                (
                    f"{enhanced_error:.4f}"
                    if enhanced_error
                    is not None
                    else ""
                ),
        }

        rows.append(
            row
        )

        if (
            baseline_pred
            is not None
            and enhanced_pred
            is not None
        ):
            preview = (
                image.copy()
            )

            # GT = red
            draw_polygon(
                preview,
                target,
                (0, 0, 255),
                4,
            )

            # Baseline = blue
            draw_polygon(
                preview,
                baseline_pred,
                (255, 0, 0),
                3,
            )

            # Enhanced = green
            draw_polygon(
                preview,
                enhanced_pred,
                (0, 255, 0),
                3,
            )

            output_dir = (
                OUTPUT_ROOT
                / video_name
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            cv2.imwrite(
                str(
                    output_dir
                    / (
                        f"{image_path.stem}"
                        "_comparison.png"
                    )
                ),
                preview,
            )

    fieldnames = [
        "sample",
        "video",
        "smartdoc_frame",
        "baseline_iou",
        "enhanced_iou",
        "iou_delta",
        "baseline_corner_error",
        "enhanced_corner_error",
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

        writer.writerows(
            rows
        )

    baseline_array = np.asarray(
        baseline_ious,
        dtype=np.float64,
    )

    enhanced_array = np.asarray(
        enhanced_ious,
        dtype=np.float64,
    )

    baseline_error_array = (
        np.asarray(
            baseline_errors,
            dtype=np.float64,
        )
    )

    enhanced_error_array = (
        np.asarray(
            enhanced_errors,
            dtype=np.float64,
        )
    )

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 7 SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Baseline failures   : "
        f"{baseline_failures}"
    )

    print(
        f"Enhanced failures   : "
        f"{enhanced_failures}"
    )

    print()

    print(
        "BASELINE"
    )

    print(
        f"Mean IoU            : "
        f"{np.mean(baseline_array):.4f}"
    )

    print(
        f"Median IoU          : "
        f"{np.median(baseline_array):.4f}"
    )

    print(
        f"Minimum IoU         : "
        f"{np.min(baseline_array):.4f}"
    )

    print(
        f"IoU >= 0.90         : "
        f"{np.sum(baseline_array >= 0.90)}"
        f"/{len(baseline_array)}"
    )

    print(
        f"Mean corner error   : "
        f"{np.mean(baseline_error_array):.2f}px"
    )

    print()

    print(
        "ENHANCED"
    )

    print(
        f"Mean IoU            : "
        f"{np.mean(enhanced_array):.4f}"
    )

    print(
        f"Median IoU          : "
        f"{np.median(enhanced_array):.4f}"
    )

    print(
        f"Minimum IoU         : "
        f"{np.min(enhanced_array):.4f}"
    )

    print(
        f"IoU >= 0.90         : "
        f"{np.sum(enhanced_array >= 0.90)}"
        f"/{len(enhanced_array)}"
    )

    print(
        f"Mean corner error   : "
        f"{np.mean(enhanced_error_array):.2f}px"
    )

    print()

    print(
        "COMPARISON"
    )

    print(
        f"Improved > 0.01     : "
        f"{improved_count}"
    )

    print(
        f"Regressed < -0.01   : "
        f"{regressed_count}"
    )

    print(
        f"Essentially same    : "
        f"{unchanged_count}"
    )

    print(
        f"Major regressions   : "
        f"{major_regressions}"
    )

    print()

    print(
        f"Summary CSV:\n"
        f"{SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()