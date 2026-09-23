from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from document_scanner import order_points


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "smartdoc_frames"
    / "datasheet001"
    / "datasheet001_06_frame_00118.png"
)

GT_XML = (
    PROJECT_ROOT
    / "data"
    / "smartdoc"
    / "sampleDataset"
    / "input_sample_groundtruth"
    / "background00_gt"
    / "datasheet001.gt.xml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_6_enhanced_edges"
)

RESIZE_HEIGHT = 700


def extract_frame_index(
    path: Path,
) -> int:
    match = re.search(
        r"_frame_(\d+)$",
        path.stem,
    )

    if match is None:
        raise RuntimeError(
            f"Could not parse frame index: {path.name}"
        )

    return int(
        match.group(1)
    ) + 1


def load_ground_truth(
    xml_path: Path,
    frame_index: int,
) -> np.ndarray:
    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()

    segmentation = root.find(
        "segmentation_results"
    )

    if segmentation is None:
        raise RuntimeError(
            "Missing segmentation_results."
        )

    for frame in segmentation.findall(
        "frame"
    ):
        if int(
            frame.attrib["index"]
        ) != frame_index:
            continue

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

        return order_points(
            polygon
        )

    raise RuntimeError(
        f"Ground truth frame "
        f"{frame_index} not found."
    )


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
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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

    edges = cv2.Canny(
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
        edges,
        gradient_edges,
    )

    kernel_close = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (11, 11),
        )
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=2,
    )

    kernel_dilate = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3),
        )
    )

    combined = cv2.dilate(
        combined,
        kernel_dilate,
        iterations=1,
    )

    return (
        enhanced,
        magnitude,
        combined,
    )


def find_candidates(
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

            aspect_ratio = (
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
                + aspect_ratio
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

                    "aspect_ratio":
                        aspect_ratio,

                    "polygon":
                        ordered,

                    "epsilon":
                        epsilon_fraction,
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

    intersection, _ = (
        cv2.intersectConvexConvex(
            hull_a,
            hull_b,
        )
    )

    intersection = float(
        intersection
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
            intersection
            / union,
            0.0,
            1.0,
        )
    )


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


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(
        str(
            TARGET_IMAGE
        )
    )

    if image is None:
        raise RuntimeError(
            "Could not load target image."
        )

    frame_index = (
        extract_frame_index(
            TARGET_IMAGE
        )
    )

    ground_truth = (
        load_ground_truth(
            GT_XML,
            frame_index,
        )
    )

    resized, scale = (
        resize_image(
            image
        )
    )

    (
        enhanced,
        gradient,
        edges,
    ) = enhanced_edge_map(
        resized
    )

    candidates = find_candidates(
        edges
    )

    print(
        "=" * 90
    )

    print(
        "EXPERIMENT 6 — "
        "ENHANCED WEAK-EDGE DETECTOR"
    )

    print(
        "=" * 90
    )

    print(
        f"Frame             : "
        f"{frame_index}"
    )

    print(
        f"Candidates found  : "
        f"{len(candidates)}"
    )

    if not candidates:
        raise RuntimeError(
            "No document candidates found."
        )

    best_resized = candidates[
        0
    ]["polygon"]

    best = (
        best_resized
        * scale
    )

    best = order_points(
        best
    )

    iou = polygon_iou(
        best,
        ground_truth,
    )

    distances = np.linalg.norm(
        best
        - ground_truth,
        axis=1,
    )

    mean_error = float(
        np.mean(
            distances
        )
    )

    print(
        f"Best score        : "
        f"{candidates[0]['score']:.4f}"
    )

    print(
        f"Area fraction     : "
        f"{candidates[0]['area_fraction']:.4f}"
    )

    print(
        f"Rectangularity    : "
        f"{candidates[0]['rectangularity']:.4f}"
    )

    print(
        f"Aspect score      : "
        f"{candidates[0]['aspect_ratio']:.4f}"
    )

    print(
        f"Polygon IoU       : "
        f"{iou:.4f}"
    )

    print(
        f"Mean corner error : "
        f"{mean_error:.2f}px"
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "01_enhanced_lightness.png"
        ),
        enhanced,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "02_gradient_magnitude.png"
        ),
        gradient,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "03_enhanced_edges.png"
        ),
        edges,
    )

    candidate_preview = (
        resized.copy()
    )

    for index, candidate in enumerate(
        candidates[:10],
        start=1,
    ):
        polygon = candidate[
            "polygon"
        ]

        draw_polygon(
            candidate_preview,
            polygon,
            (0, 255, 255),
            2,
        )

        center = np.mean(
            polygon,
            axis=0,
        )

        cv2.putText(
            candidate_preview,
            (
                f"C{index} "
                f"{candidate['score']:.3f}"
            ),
            (
                int(
                    center[0]
                ),
                int(
                    center[1]
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "04_candidates.png"
        ),
        candidate_preview,
    )

    comparison = (
        image.copy()
    )

    # Ground truth red
    draw_polygon(
        comparison,
        ground_truth,
        (0, 0, 255),
        5,
    )

    # Prediction green
    draw_polygon(
        comparison,
        best,
        (0, 255, 0),
        5,
    )

    cv2.rectangle(
        comparison,
        (10, 10),
        (620, 105),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        comparison,
        (
            f"Enhanced-edge IoU: "
            f"{iou:.4f}"
        ),
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        comparison,
        (
            f"Mean corner error: "
            f"{mean_error:.2f}px"
        ),
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "05_prediction_vs_ground_truth.png"
        ),
        comparison,
    )

    print()

    print(
        f"Outputs:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()