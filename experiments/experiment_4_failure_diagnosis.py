from __future__ import annotations

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

TARGET_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "smartdoc_frames"
    / "datasheet001"
    / "datasheet001_06_frame_00118.png"
)

GROUND_TRUTH_XML = (
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
    / "experiment_4_failure_diagnosis"
)

TARGET_RESIZE_HEIGHT = 700
TOP_CONTOURS = 20


def extract_zero_based_frame_index(
    path: Path,
) -> int:
    match = re.search(
        r"_frame_(\d+)$",
        path.stem,
    )

    if match is None:
        raise RuntimeError(
            f"Could not parse frame index from:\n"
            f"{path.name}"
        )

    return int(
        match.group(1)
    )


def load_ground_truth_polygon(
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
            "No segmentation_results found."
        )

    for frame in segmentation.findall(
        "frame"
    ):
        current_index = int(
            frame.attrib["index"]
        )

        if current_index != frame_index:
            continue

        rejected = (
            frame.attrib.get(
                "rejected",
                "false",
            ).lower()
            == "true"
        )

        if rejected:
            raise RuntimeError(
                f"Frame {frame_index} is rejected "
                f"in ground truth."
            )

        points = {}

        for point in frame.findall(
            "point"
        ):
            name = point.attrib["name"]

            points[name] = np.array(
                [
                    float(point.attrib["x"]),
                    float(point.attrib["y"]),
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
        f"Ground-truth frame {frame_index} "
        f"not found."
    )


def create_edge_map(
    image: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    original_height = image.shape[0]

    scale = (
        original_height
        / float(
            TARGET_RESIZE_HEIGHT
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
            TARGET_RESIZE_HEIGHT,
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

    return (
        resized,
        edges,
        scale,
    )


def find_candidate_quadrilaterals(
    edges: np.ndarray,
) -> list[
    dict[str, object]
]:
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )[:TOP_CONTOURS]

    candidates = []

    for rank, contour in enumerate(
        contours,
        start=1,
    ):
        area = float(
            cv2.contourArea(
                contour
            )
        )

        perimeter = float(
            cv2.arcLength(
                contour,
                True,
            )
        )

        approximation = (
            cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True,
            )
        )

        candidates.append(
            {
                "rank": rank,
                "area": area,
                "perimeter": perimeter,
                "vertices":
                    len(
                        approximation
                    ),
                "approximation":
                    approximation,
            }
        )

    return candidates


def draw_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    contour = (
        polygon
        .astype(np.int32)
        .reshape((-1, 1, 2))
    )

    cv2.polylines(
        image,
        [contour],
        True,
        color,
        thickness,
    )


def draw_candidate_overview(
    resized: np.ndarray,
    candidates: list[
        dict[str, object]
    ],
) -> np.ndarray:
    preview = resized.copy()

    colors = [
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (0, 255, 0),
    ]

    quad_index = 0

    for candidate in candidates:
        approximation = candidate[
            "approximation"
        ]

        if len(
            approximation
        ) != 4:
            continue

        quad_index += 1

        polygon = (
            approximation
            .reshape(4, 2)
            .astype(np.float32)
        )

        color = colors[
            (quad_index - 1)
            % len(colors)
        ]

        draw_polygon(
            preview,
            polygon,
            color,
            3,
        )

        center = np.mean(
            polygon,
            axis=0,
        )

        cv2.putText(
            preview,
            (
                f"Q{quad_index} "
                f"rank={candidate['rank']}"
            ),
            (
                int(center[0]),
                int(center[1]),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return preview


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TARGET_IMAGE.exists():
        raise FileNotFoundError(
            f"Target image not found:\n"
            f"{TARGET_IMAGE}"
        )

    if not GROUND_TRUTH_XML.exists():
        raise FileNotFoundError(
            f"Ground-truth XML not found:\n"
            f"{GROUND_TRUTH_XML}"
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

    zero_based_index = (
        extract_zero_based_frame_index(
            TARGET_IMAGE
        )
    )

    smartdoc_frame_index = (
        zero_based_index
        + 1
    )

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 4 — FAILURE DIAGNOSIS"
    )

    print(
        "=" * 100
    )

    print(
        f"Image             : "
        f"{TARGET_IMAGE}"
    )

    print(
        f"OpenCV frame      : "
        f"{zero_based_index}"
    )

    print(
        f"SmartDoc frame    : "
        f"{smartdoc_frame_index}"
    )

    print(
        f"Image size        : "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    print()

    ground_truth = (
        load_ground_truth_polygon(
            GROUND_TRUTH_XML,
            smartdoc_frame_index,
        )
    )

    predicted, detector_edges = (
        detect_document(
            image
        )
    )

    predicted = order_points(
        predicted
    )

    (
        resized,
        edges,
        scale,
    ) = create_edge_map(
        image
    )

    candidates = (
        find_candidate_quadrilaterals(
            edges
        )
    )

    print(
        f"Resize scale      : "
        f"{scale:.6f}"
    )

    print(
        f"Contours inspected: "
        f"{len(candidates)}"
    )

    print()

    quad_count = 0

    print(
        "TOP CONTOUR CANDIDATES"
    )

    print(
        "-" * 100
    )

    for candidate in candidates:
        vertices = int(
            candidate["vertices"]
        )

        if vertices == 4:
            quad_count += 1

        print(
            f"Rank {candidate['rank']:02d} "
            f"| area={candidate['area']:.2f} "
            f"| perimeter="
            f"{candidate['perimeter']:.2f} "
            f"| vertices={vertices}"
        )

    print()

    print(
        f"Quadrilateral candidates: "
        f"{quad_count}"
    )

    # -----------------------------------------------------------------
    # Original
    # -----------------------------------------------------------------

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "01_original.png"
        ),
        image,
    )

    # -----------------------------------------------------------------
    # Edge map
    # -----------------------------------------------------------------

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "02_edges.png"
        ),
        edges,
    )

    # -----------------------------------------------------------------
    # Candidate contours
    # -----------------------------------------------------------------

    candidate_preview = (
        draw_candidate_overview(
            resized,
            candidates,
        )
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "03_candidate_quadrilaterals.png"
        ),
        candidate_preview,
    )

    # -----------------------------------------------------------------
    # Prediction only
    # -----------------------------------------------------------------

    prediction_preview = (
        image.copy()
    )

    draw_polygon(
        prediction_preview,
        predicted,
        (0, 255, 0),
        5,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "04_selected_prediction.png"
        ),
        prediction_preview,
    )

    # -----------------------------------------------------------------
    # Ground truth only
    # -----------------------------------------------------------------

    ground_truth_preview = (
        image.copy()
    )

    draw_polygon(
        ground_truth_preview,
        ground_truth,
        (0, 0, 255),
        5,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "05_ground_truth.png"
        ),
        ground_truth_preview,
    )

    # -----------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------

    comparison = image.copy()

    draw_polygon(
        comparison,
        ground_truth,
        (0, 0, 255),
        5,
    )

    draw_polygon(
        comparison,
        predicted,
        (0, 255, 0),
        5,
    )

    cv2.rectangle(
        comparison,
        (10, 10),
        (620, 95),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        comparison,
        "RED = SmartDoc ground truth",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        comparison,
        "GREEN = current detector",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "06_prediction_vs_ground_truth.png"
        ),
        comparison,
    )

    print()

    print(
        "=" * 100
    )

    print(
        "DIAGNOSIS FILES SAVED"
    )

    print(
        "=" * 100
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()