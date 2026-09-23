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
    / "experiment_3_groundtruth_evaluation"
)

SUMMARY_CSV = (
    OUTPUT_ROOT
    / "experiment_3_summary.csv"
)


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

    results: dict[
        int,
        dict[str, object],
    ] = {}

    segmentation = root.find(
        "segmentation_results"
    )

    if segmentation is None:
        raise RuntimeError(
            f"No segmentation_results found in:\n"
            f"{xml_path}"
        )

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
            points: dict[
                str,
                np.ndarray,
            ] = {}

            for point in frame.findall(
                "point"
            ):
                name = point.attrib[
                    "name"
                ]

                x = float(
                    point.attrib["x"]
                )

                y = float(
                    point.attrib["y"]
                )

                points[name] = np.array(
                    [x, y],
                    dtype=np.float32,
                )

            required = {
                "tl",
                "tr",
                "br",
                "bl",
            }

            missing = (
                required
                - set(
                    points.keys()
                )
            )

            if missing:
                raise RuntimeError(
                    f"Missing GT points "
                    f"{sorted(missing)} "
                    f"in frame {frame_index} "
                    f"of {xml_path.name}"
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

            entry["polygon"] = (
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
            "Could not determine frame index "
            f"from filename:\n"
            f"{path.name}"
        )

    return int(
        match.group(1)
    )


# =============================================================================
# POLYGON HELPERS
# =============================================================================

def prepare_convex_polygon(
    polygon: np.ndarray,
) -> np.ndarray:
    polygon = np.asarray(
        polygon,
        dtype=np.float32,
    ).reshape(-1, 2)

    if polygon.shape[0] < 3:
        raise ValueError(
            "Polygon must contain at least "
            "three points."
        )

    hull = cv2.convexHull(
        polygon
    )

    hull = np.asarray(
        hull,
        dtype=np.float32,
    ).reshape(-1, 2)

    return hull


def polygon_iou(
    polygon_a: np.ndarray,
    polygon_b: np.ndarray,
) -> float:
    a = prepare_convex_polygon(
        polygon_a
    )

    b = prepare_convex_polygon(
        polygon_b
    )

    area_a = float(
        cv2.contourArea(
            a
        )
    )

    area_b = float(
        cv2.contourArea(
            b
        )
    )

    if (
        not np.isfinite(area_a)
        or not np.isfinite(area_b)
        or area_a <= 0.0
        or area_b <= 0.0
    ):
        return 0.0

    intersection_result = (
        cv2.intersectConvexConvex(
            a,
            b,
        )
    )

    intersection_area = float(
        intersection_result[0]
    )

    if not np.isfinite(
        intersection_area
    ):
        return 0.0

    intersection_area = max(
        0.0,
        intersection_area,
    )

    maximum_possible_intersection = min(
        area_a,
        area_b,
    )

    intersection_area = min(
        intersection_area,
        maximum_possible_intersection,
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if (
        not np.isfinite(union_area)
        or union_area <= 0.0
    ):
        return 0.0

    iou = (
        intersection_area
        / union_area
    )

    if not np.isfinite(
        iou
    ):
        return 0.0

    return float(
        np.clip(
            iou,
            0.0,
            1.0,
        )
    )


# =============================================================================
# CORNER METRICS
# =============================================================================

def corner_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[
    float,
    float,
    float,
]:
    predicted = np.asarray(
        predicted,
        dtype=np.float32,
    )

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    distances = np.linalg.norm(
        predicted - target,
        axis=1,
    )

    mean_error = float(
        np.mean(
            distances
        )
    )

    maximum_error = float(
        np.max(
            distances
        )
    )

    diagonal = float(
        np.hypot(
            image_width,
            image_height,
        )
    )

    if diagonal <= 0.0:
        normalized_error = 0.0

    else:
        normalized_error = (
            mean_error
            / diagonal
        )

    return (
        mean_error,
        maximum_error,
        normalized_error,
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
        np.asarray(
            polygon,
            dtype=np.float32,
        )
        .astype(np.int32)
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


def draw_corner_labels(
    image: np.ndarray,
    polygon: np.ndarray,
    prefix: str,
    color: tuple[int, int, int],
) -> None:
    labels = [
        "TL",
        "TR",
        "BR",
        "BL",
    ]

    for label, point in zip(
        labels,
        polygon,
    ):
        x = int(
            round(
                float(
                    point[0]
                )
            )
        )

        y = int(
            round(
                float(
                    point[1]
                )
            )
        )

        cv2.circle(
            image,
            (x, y),
            6,
            color,
            -1,
        )

        cv2.putText(
            image,
            f"{prefix}-{label}",
            (
                x + 8,
                y - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def make_visualization(
    image: np.ndarray,
    predicted: np.ndarray,
    target: np.ndarray,
    iou: float,
    mean_error: float,
) -> np.ndarray:
    visualization = (
        image.copy()
    )

    # Ground truth = red
    draw_polygon(
        visualization,
        target,
        (0, 0, 255),
        4,
    )

    draw_corner_labels(
        visualization,
        target,
        "GT",
        (0, 0, 255),
    )

    # Prediction = green
    draw_polygon(
        visualization,
        predicted,
        (0, 255, 0),
        4,
    )

    draw_corner_labels(
        visualization,
        predicted,
        "P",
        (0, 255, 0),
    )

    cv2.rectangle(
        visualization,
        (10, 10),
        (570, 105),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        visualization,
        f"Polygon IoU: {iou:.4f}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        visualization,
        (
            "Mean corner error: "
            f"{mean_error:.2f}px"
        ),
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return visualization


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
            f"No SmartDoc images found in:\n"
            f"{FRAME_ROOT}"
        )

    videos = sorted(
        {
            image_path.parent.name
            for image_path
            in image_paths
        }
    )

    ground_truth_cache: dict[
        str,
        dict[
            int,
            dict[str, object],
        ],
    ] = {}

    for video_name in videos:
        xml_path = (
            GT_ROOT
            / f"{video_name}.gt.xml"
        )

        if not xml_path.exists():
            raise FileNotFoundError(
                f"Missing ground truth:\n"
                f"{xml_path}"
            )

        ground_truth_cache[
            video_name
        ] = load_ground_truth(
            xml_path
        )

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 3 — SMARTDOC "
        "GROUND-TRUTH BOUNDARY EVALUATION"
    )

    print(
        "=" * 100
    )

    print(
        f"Images: {len(image_paths)}"
    )

    print(
        f"Videos: {len(videos)}"
    )

    print()

    rows: list[
        dict[str, object]
    ] = []

    ious: list[
        float
    ] = []

    mean_errors: list[
        float
    ] = []

    normalized_errors: list[
        float
    ] = []

    detector_failures = 0

    gt_rejected_count = 0

    missing_gt_count = 0

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        video_name = (
            image_path.parent.name
        )

        zero_based_index = (
            extract_zero_based_frame_index(
                image_path
            )
        )

        smartdoc_frame_index = (
            zero_based_index
            + 1
        )

        print(
            f"[{index:02d}/"
            f"{len(image_paths):02d}] "
            f"{video_name} "
            f"| OpenCV={zero_based_index} "
            f"| SmartDoc="
            f"{smartdoc_frame_index}"
        )

        row: dict[
            str,
            object,
        ] = {
            "sample":
                str(
                    image_path.relative_to(
                        FRAME_ROOT
                    )
                ),

            "video":
                video_name,

            "opencv_frame_index":
                zero_based_index,

            "smartdoc_frame_index":
                smartdoc_frame_index,

            "status":
                "FAIL",

            "iou":
                "",

            "mean_corner_error_px":
                "",

            "max_corner_error_px":
                "",

            "normalized_corner_error":
                "",

            "error":
                "",
        }

        image = cv2.imread(
            str(
                image_path
            )
        )

        if image is None:
            row["error"] = (
                "Could not load image."
            )

            rows.append(
                row
            )

            print(
                "    [FAIL] "
                "Could not load image."
            )

            continue

        ground_truth_entry = (
            ground_truth_cache[
                video_name
            ].get(
                smartdoc_frame_index
            )
        )

        if ground_truth_entry is None:
            missing_gt_count += 1

            row[
                "status"
            ] = "MISSING_GT"

            row[
                "error"
            ] = (
                "Ground-truth frame "
                "not found."
            )

            rows.append(
                row
            )

            print(
                "    [SKIP] "
                "Ground-truth frame "
                "not found."
            )

            continue

        if bool(
            ground_truth_entry[
                "rejected"
            ]
        ):
            gt_rejected_count += 1

            row[
                "status"
            ] = "GT_REJECTED"

            rows.append(
                row
            )

            print(
                "    [SKIP] "
                "Ground truth marks "
                "frame rejected."
            )

            continue

        target = np.asarray(
            ground_truth_entry[
                "polygon"
            ],
            dtype=np.float32,
        )

        target = order_points(
            target
        )

        try:
            predicted_raw, _ = (
                detect_document(
                    image
                )
            )

            predicted = order_points(
                predicted_raw
            )

        except Exception as error:
            detector_failures += 1

            row[
                "status"
            ] = "DETECTOR_FAIL"

            row[
                "error"
            ] = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            rows.append(
                row
            )

            print(
                f"    [FAIL] "
                f"{row['error']}"
            )

            continue

        iou = polygon_iou(
            predicted,
            target,
        )

        (
            mean_corner_error,
            maximum_corner_error,
            normalized_corner_error,
        ) = corner_metrics(
            predicted=predicted,
            target=target,
            image_width=image.shape[1],
            image_height=image.shape[0],
        )

        row[
            "status"
        ] = "EVALUATED"

        row[
            "iou"
        ] = (
            f"{iou:.8f}"
        )

        row[
            "mean_corner_error_px"
        ] = (
            f"{mean_corner_error:.4f}"
        )

        row[
            "max_corner_error_px"
        ] = (
            f"{maximum_corner_error:.4f}"
        )

        row[
            "normalized_corner_error"
        ] = (
            f"{normalized_corner_error:.8f}"
        )

        rows.append(
            row
        )

        ious.append(
            iou
        )

        mean_errors.append(
            mean_corner_error
        )

        normalized_errors.append(
            normalized_corner_error
        )

        output_dir = (
            OUTPUT_ROOT
            / video_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        visualization = (
            make_visualization(
                image=image,
                predicted=predicted,
                target=target,
                iou=iou,
                mean_error=(
                    mean_corner_error
                ),
            )
        )

        visualization_path = (
            output_dir
            / (
                f"{image_path.stem}"
                "_comparison.png"
            )
        )

        cv2.imwrite(
            str(
                visualization_path
            ),
            visualization,
        )

        print(
            f"    IoU              : "
            f"{iou:.4f}"
        )

        print(
            f"    Mean corner error: "
            f"{mean_corner_error:.2f}px"
        )

        if iou < 0.80:
            print(
                "    [WARN] "
                "Low-IoU sample"
            )

    fieldnames = [
        "sample",
        "video",
        "opencv_frame_index",
        "smartdoc_frame_index",
        "status",
        "iou",
        "mean_corner_error_px",
        "max_corner_error_px",
        "normalized_corner_error",
        "error",
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

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 3 SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Total samples       : "
        f"{len(image_paths)}"
    )

    print(
        f"Evaluated samples   : "
        f"{len(ious)}"
    )

    print(
        f"Detector failures   : "
        f"{detector_failures}"
    )

    print(
        f"GT rejected         : "
        f"{gt_rejected_count}"
    )

    print(
        f"Missing GT          : "
        f"{missing_gt_count}"
    )

    if ious:
        iou_array = np.asarray(
            ious,
            dtype=np.float64,
        )

        error_array = np.asarray(
            mean_errors,
            dtype=np.float64,
        )

        normalized_array = (
            np.asarray(
                normalized_errors,
                dtype=np.float64,
            )
        )

        print()

        print(
            f"Mean IoU            : "
            f"{np.mean(iou_array):.4f}"
        )

        print(
            f"Median IoU          : "
            f"{np.median(iou_array):.4f}"
        )

        print(
            f"Minimum IoU         : "
            f"{np.min(iou_array):.4f}"
        )

        print(
            f"Maximum IoU         : "
            f"{np.max(iou_array):.4f}"
        )

        print()

        print(
            f"IoU >= 0.90         : "
            f"{np.sum(iou_array >= 0.90)}"
            f"/{len(iou_array)}"
        )

        print(
            f"IoU >= 0.80         : "
            f"{np.sum(iou_array >= 0.80)}"
            f"/{len(iou_array)}"
        )

        print(
            f"IoU >= 0.60         : "
            f"{np.sum(iou_array >= 0.60)}"
            f"/{len(iou_array)}"
        )

        print(
            f"IoU < 0.60          : "
            f"{np.sum(iou_array < 0.60)}"
            f"/{len(iou_array)}"
        )

        print()

        print(
            f"Mean corner error   : "
            f"{np.mean(error_array):.2f}px"
        )

        print(
            f"Median corner error : "
            f"{np.median(error_array):.2f}px"
        )

        print(
            f"Maximum mean error  : "
            f"{np.max(error_array):.2f}px"
        )

        print(
            f"Mean normalized err : "
            f"{np.mean(normalized_array):.6f}"
        )

    print()

    print(
        f"Summary CSV:\n"
        f"{SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()