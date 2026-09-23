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
    / "experiment_5_region_detector"
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
        f"Ground truth for frame "
        f"{frame_index} not found."
    )


def resize_image(
    image: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
]:
    height = image.shape[0]

    scale = (
        height
        / float(
            RESIZE_HEIGHT
        )
    )

    width = int(
        image.shape[1]
        / scale
    )

    resized = cv2.resize(
        image,
        (
            width,
            RESIZE_HEIGHT,
        ),
    )

    return (
        resized,
        scale,
    )


def create_page_mask(
    image: np.ndarray,
) -> tuple[
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

    blurred = cv2.GaussianBlur(
        lightness,
        (9, 9),
        0,
    )

    _, otsu_mask = cv2.threshold(
        blurred,
        0,
        255,
        (
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        ),
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (21, 21),
    )

    closed = cv2.morphologyEx(
        otsu_mask,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=2,
    )

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7),
    )

    cleaned = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1,
    )

    return (
        lightness,
        cleaned,
    )


def find_page_candidate(
    mask: np.ndarray,
) -> tuple[
    np.ndarray,
    list[
        tuple[
            float,
            np.ndarray,
        ]
    ],
]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError(
            "No region contours found."
        )

    image_area = (
        mask.shape[0]
        * mask.shape[1]
    )

    candidates = []

    for contour in contours:
        area = float(
            cv2.contourArea(
                contour
            )
        )

        area_fraction = (
            area
            / image_area
        )

        if area_fraction < 0.02:
            continue

        hull = cv2.convexHull(
            contour
        )

        perimeter = cv2.arcLength(
            hull,
            True,
        )

        approximation = (
            cv2.approxPolyDP(
                hull,
                0.02 * perimeter,
                True,
            )
        )

        if len(
            approximation
        ) == 4:
            polygon = (
                approximation
                .reshape(4, 2)
                .astype(
                    np.float32
                )
            )

        else:
            rectangle = (
                cv2.minAreaRect(
                    hull
                )
            )

            polygon = cv2.boxPoints(
                rectangle
            ).astype(
                np.float32
            )

        polygon_area = abs(
            cv2.contourArea(
                polygon
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

        score = (
            area_fraction
            * rectangularity
        )

        candidates.append(
            (
                score,
                polygon,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No plausible page-region "
            "candidate found."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_polygon = order_points(
        candidates[0][1]
    )

    return (
        best_polygon,
        candidates,
    )


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

    intersection = max(
        0.0,
        float(
            intersection
        ),
    )

    intersection = min(
        intersection,
        area_a,
        area_b,
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
    thickness: int = 4,
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

    lightness, mask = (
        create_page_mask(
            resized
        )
    )

    (
        predicted_resized,
        candidates,
    ) = find_page_candidate(
        mask
    )

    predicted = (
        predicted_resized
        * scale
    )

    predicted = order_points(
        predicted
    )

    iou = polygon_iou(
        predicted,
        ground_truth,
    )

    distances = np.linalg.norm(
        predicted
        - ground_truth,
        axis=1,
    )

    mean_corner_error = float(
        np.mean(
            distances
        )
    )

    print(
        "=" * 90
    )

    print(
        "EXPERIMENT 5 — "
        "PAGE REGION DETECTOR"
    )

    print(
        "=" * 90
    )

    print(
        f"Frame             : "
        f"{frame_index}"
    )

    print(
        f"Region candidates : "
        f"{len(candidates)}"
    )

    print(
        f"Polygon IoU       : "
        f"{iou:.4f}"
    )

    print(
        f"Mean corner error : "
        f"{mean_corner_error:.2f}px"
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "01_lightness.png"
        ),
        lightness,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "02_page_mask.png"
        ),
        mask,
    )

    candidate_preview = (
        resized.copy()
    )

    for index, (
        score,
        polygon,
    ) in enumerate(
        candidates[:10],
        start=1,
    ):
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
                f"{index}:"
                f"{score:.3f}"
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
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "03_region_candidates.png"
        ),
        candidate_preview,
    )

    comparison = (
        image.copy()
    )

    # Ground truth = red
    draw_polygon(
        comparison,
        ground_truth,
        (0, 0, 255),
        5,
    )

    # New region prediction = green
    draw_polygon(
        comparison,
        predicted,
        (0, 255, 0),
        5,
    )

    cv2.rectangle(
        comparison,
        (10, 10),
        (590, 105),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        comparison,
        (
            f"Region detector IoU: "
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
            f"{mean_corner_error:.2f}px"
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
            / "04_region_vs_ground_truth.png"
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