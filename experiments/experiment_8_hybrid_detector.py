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
    order_points,
)


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
    / "experiment_8_hybrid_detector"
)

SUMMARY_CSV = (
    OUTPUT_ROOT
    / "experiment_8_summary.csv"
)

RESIZE_HEIGHT = 700

BORDER_MARGIN_FRACTION = 0.03
BORDER_VERTEX_LIMIT = 2


# =============================================================================
# GEOMETRY
# =============================================================================

def polygon_iou(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    hull_a = cv2.convexHull(
        a.astype(np.float32)
    )

    hull_b = cv2.convexHull(
        b.astype(np.float32)
    )

    area_a = float(
        cv2.contourArea(hull_a)
    )

    area_b = float(
        cv2.contourArea(hull_b)
    )

    if area_a <= 0 or area_b <= 0:
        return 0.0

    intersection, _ = (
        cv2.intersectConvexConvex(
            hull_a,
            hull_b,
        )
    )

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

    if union <= 0:
        return 0.0

    return float(
        np.clip(
            intersection / union,
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
        np.mean(distances)
    )


def polygon_area_fraction(
    polygon: np.ndarray,
    image: np.ndarray,
) -> float:
    area = abs(
        float(
            cv2.contourArea(
                polygon.astype(
                    np.float32
                )
            )
        )
    )

    image_area = float(
        image.shape[0]
        * image.shape[1]
    )

    if image_area <= 0:
        return 0.0

    return area / image_area


# =============================================================================
# BORDER ANALYSIS
# =============================================================================

def count_border_vertices(
    polygon: np.ndarray,
    image: np.ndarray,
) -> int:
    height, width = image.shape[:2]

    margin_x = (
        width
        * BORDER_MARGIN_FRACTION
    )

    margin_y = (
        height
        * BORDER_MARGIN_FRACTION
    )

    count = 0

    for point in polygon:
        x = float(
            point[0]
        )

        y = float(
            point[1]
        )

        near_left = (
            x <= margin_x
        )

        near_right = (
            x >= width - margin_x
        )

        near_top = (
            y <= margin_y
        )

        near_bottom = (
            y >= height - margin_y
        )

        if (
            near_left
            or near_right
            or near_top
            or near_bottom
        ):
            count += 1

    return count


def enhanced_candidate_is_suspicious(
    polygon: np.ndarray,
    image: np.ndarray,
) -> tuple[
    bool,
    int,
    float,
]:
    border_vertices = (
        count_border_vertices(
            polygon,
            image,
        )
    )

    area_fraction = (
        polygon_area_fraction(
            polygon,
            image,
        )
    )

    suspicious = (
        border_vertices
        >= BORDER_VERTEX_LIMIT
    )

    return (
        suspicious,
        border_vertices,
        area_fraction,
    )


# =============================================================================
# LEGACY DETECTOR
# =============================================================================

def detect_document_legacy(
    image: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
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

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred,
        50,
        150,
    )

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )[:10]

    document_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(
            contour,
            True,
        )

        approximation = (
            cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True,
            )
        )

        if len(
            approximation
        ) == 4:
            document_contour = (
                approximation
                .reshape(4, 2)
                .astype(
                    np.float32
                )
            )

            break

    if document_contour is None:
        raise RuntimeError(
            "Legacy detector could not "
            "find a four-point contour."
        )

    document_contour *= scale

    document_contour = order_points(
        document_contour
    )

    return (
        document_contour,
        edges,
    )


# =============================================================================
# HYBRID DETECTOR
# =============================================================================

def detect_document_hybrid(
    image: np.ndarray,
) -> tuple[
    np.ndarray,
    str,
    int,
    float,
]:
    enhanced_polygon = None
    enhanced_error = None

    try:
        enhanced_polygon, _ = (
            detect_document(
                image
            )
        )

        enhanced_polygon = order_points(
            enhanced_polygon
        )

    except Exception as error:
        enhanced_error = error

    if enhanced_polygon is not None:
        (
            suspicious,
            border_vertices,
            area_fraction,
        ) = (
            enhanced_candidate_is_suspicious(
                enhanced_polygon,
                image,
            )
        )

        if not suspicious:
            return (
                enhanced_polygon,
                "enhanced",
                border_vertices,
                area_fraction,
            )

    try:
        legacy_polygon, _ = (
            detect_document_legacy(
                image
            )
        )

        legacy_polygon = order_points(
            legacy_polygon
        )

        return (
            legacy_polygon,
            "legacy_fallback",
            (
                count_border_vertices(
                    legacy_polygon,
                    image,
                )
            ),
            (
                polygon_area_fraction(
                    legacy_polygon,
                    image,
                )
            ),
        )

    except Exception as legacy_error:
        if enhanced_polygon is not None:
            return (
                enhanced_polygon,
                "enhanced_forced",
                (
                    count_border_vertices(
                        enhanced_polygon,
                        image,
                    )
                ),
                (
                    polygon_area_fraction(
                        enhanced_polygon,
                        image,
                    )
                ),
            )

        raise RuntimeError(
            "Both detectors failed.\n"
            f"Enhanced: {enhanced_error}\n"
            f"Legacy: {legacy_error}"
        )


# =============================================================================
# SMARTDOC GROUND TRUTH
# =============================================================================

def load_ground_truth(
    xml_path: Path,
) -> dict[
    int,
    np.ndarray,
]:
    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()

    segmentation = root.find(
        "segmentation_results"
    )

    if segmentation is None:
        raise RuntimeError(
            f"No segmentation_results "
            f"in {xml_path}"
        )

    results = {}

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

        results[
            frame_index
        ] = order_points(
            polygon
        )

    return results


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

    return int(
        match.group(1)
    ) + 1


# =============================================================================
# DRAWING
# =============================================================================

def draw_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    color: tuple[
        int,
        int,
        int,
    ],
    thickness: int,
) -> None:
    contour = (
        polygon
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


# =============================================================================
# CLASSROOM TEST
# =============================================================================

def run_classroom_test() -> None:
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

    image = cv2.imread(
        str(
            CLASSROOM_IMAGE
        )
    )

    if image is None:
        raise RuntimeError(
            "Could not load classroom image."
        )

    enhanced_polygon, _ = (
        detect_document(
            image
        )
    )

    (
        hybrid_polygon,
        route,
        border_vertices,
        area_fraction,
    ) = detect_document_hybrid(
        image
    )

    print(
        f"Hybrid route       : {route}"
    )

    print(
        f"Border vertices    : "
        f"{border_vertices}"
    )

    print(
        f"Selected area frac : "
        f"{area_fraction:.4f}"
    )

    classroom_dir = (
        OUTPUT_ROOT
        / "classroom"
    )

    classroom_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = (
        image.copy()
    )

    # Enhanced = red
    draw_polygon(
        comparison,
        enhanced_polygon,
        (0, 0, 255),
        5,
    )

    # Hybrid = green
    draw_polygon(
        comparison,
        hybrid_polygon,
        (0, 255, 0),
        5,
    )

    cv2.putText(
        comparison,
        "RED = enhanced",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        comparison,
        "GREEN = hybrid selected",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(
            classroom_dir
            / "01_detector_comparison.png"
        ),
        comparison,
    )

    warped = (
        four_point_transform(
            image,
            hybrid_polygon,
        )
    )

    cv2.imwrite(
        str(
            classroom_dir
            / "02_hybrid_perspective.png"
        ),
        warped,
    )


# =============================================================================
# SMARTDOC BATCH TEST
# =============================================================================

def run_smartdoc_batch() -> None:
    image_paths = sorted(
        SMARTDOC_ROOT.rglob(
            "*.png"
        )
    )

    videos = sorted(
        {
            path.parent.name
            for path in image_paths
        }
    )

    gt_cache = {}

    for video in videos:
        gt_cache[
            video
        ] = load_ground_truth(
            GT_ROOT
            / f"{video}.gt.xml"
        )

    rows = []

    enhanced_ious = []
    hybrid_ious = []

    enhanced_errors = []
    hybrid_errors = []

    enhanced_routes = 0
    legacy_routes = 0
    forced_routes = 0

    print()
    print(
        "=" * 100
    )
    print(
        "SMARTDOC HYBRID BATCH TEST"
    )
    print(
        "=" * 100
    )

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        image = cv2.imread(
            str(
                image_path
            )
        )

        if image is None:
            continue

        video = (
            image_path.parent.name
        )

        frame_index = (
            extract_frame_index(
                image_path
            )
        )

        target = (
            gt_cache[
                video
            ].get(
                frame_index
            )
        )

        if target is None:
            continue

        enhanced_polygon, _ = (
            detect_document(
                image
            )
        )

        enhanced_polygon = (
            order_points(
                enhanced_polygon
            )
        )

        (
            hybrid_polygon,
            route,
            border_vertices,
            area_fraction,
        ) = detect_document_hybrid(
            image
        )

        enhanced_iou = polygon_iou(
            enhanced_polygon,
            target,
        )

        hybrid_iou = polygon_iou(
            hybrid_polygon,
            target,
        )

        enhanced_error = (
            mean_corner_error(
                enhanced_polygon,
                target,
            )
        )

        hybrid_error = (
            mean_corner_error(
                hybrid_polygon,
                target,
            )
        )

        enhanced_ious.append(
            enhanced_iou
        )

        hybrid_ious.append(
            hybrid_iou
        )

        enhanced_errors.append(
            enhanced_error
        )

        hybrid_errors.append(
            hybrid_error
        )

        if route == "enhanced":
            enhanced_routes += 1

        elif route == "legacy_fallback":
            legacy_routes += 1

        else:
            forced_routes += 1

        delta = (
            hybrid_iou
            - enhanced_iou
        )

        print(
            f"[{index:02d}/"
            f"{len(image_paths):02d}] "
            f"{image_path.name}"
        )

        print(
            f"    route    : {route}"
        )

        print(
            f"    enhanced : "
            f"{enhanced_iou:.4f}"
        )

        print(
            f"    hybrid   : "
            f"{hybrid_iou:.4f}"
        )

        print(
            f"    delta    : "
            f"{delta:+.4f}"
        )

        rows.append(
            {
                "sample":
                    str(
                        image_path.relative_to(
                            SMARTDOC_ROOT
                        )
                    ),

                "route":
                    route,

                "border_vertices":
                    border_vertices,

                "area_fraction":
                    f"{area_fraction:.6f}",

                "enhanced_iou":
                    f"{enhanced_iou:.8f}",

                "hybrid_iou":
                    f"{hybrid_iou:.8f}",

                "iou_delta":
                    f"{delta:.8f}",

                "enhanced_corner_error":
                    f"{enhanced_error:.4f}",

                "hybrid_corner_error":
                    f"{hybrid_error:.4f}",
            }
        )

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample",
                "route",
                "border_vertices",
                "area_fraction",
                "enhanced_iou",
                "hybrid_iou",
                "iou_delta",
                "enhanced_corner_error",
                "hybrid_corner_error",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    enhanced_array = np.asarray(
        enhanced_ious,
        dtype=np.float64,
    )

    hybrid_array = np.asarray(
        hybrid_ious,
        dtype=np.float64,
    )

    enhanced_error_array = (
        np.asarray(
            enhanced_errors,
            dtype=np.float64,
        )
    )

    hybrid_error_array = (
        np.asarray(
            hybrid_errors,
            dtype=np.float64,
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "EXPERIMENT 8 SUMMARY"
    )
    print(
        "=" * 100
    )

    print(
        f"Enhanced routes     : "
        f"{enhanced_routes}"
    )

    print(
        f"Legacy fallbacks    : "
        f"{legacy_routes}"
    )

    print(
        f"Forced enhanced     : "
        f"{forced_routes}"
    )

    print()

    print(
        "ENHANCED-ONLY"
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
        "HYBRID"
    )

    print(
        f"Mean IoU            : "
        f"{np.mean(hybrid_array):.4f}"
    )

    print(
        f"Median IoU          : "
        f"{np.median(hybrid_array):.4f}"
    )

    print(
        f"Minimum IoU         : "
        f"{np.min(hybrid_array):.4f}"
    )

    print(
        f"IoU >= 0.90         : "
        f"{np.sum(hybrid_array >= 0.90)}"
        f"/{len(hybrid_array)}"
    )

    print(
        f"Mean corner error   : "
        f"{np.mean(hybrid_error_array):.2f}px"
    )

    print()

    print(
        f"Summary CSV:\n"
        f"{SUMMARY_CSV}"
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
        "EXPERIMENT 8 — HYBRID "
        "DOCUMENT DETECTOR"
    )

    print(
        "=" * 100
    )

    print(
        f"Border margin      : "
        f"{BORDER_MARGIN_FRACTION:.1%}"
    )

    print(
        f"Fallback threshold : "
        f"{BORDER_VERTEX_LIMIT} "
        f"border vertices"
    )

    run_classroom_test()

    run_smartdoc_batch()


if __name__ == "__main__":
    main()