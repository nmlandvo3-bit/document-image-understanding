from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


RESIZE_HEIGHT = 700

BORDER_MARGIN_FRACTION = 0.03
BORDER_VERTEX_LIMIT = 2

# A valid document should occupy a meaningful portion of the photograph.
# These checks prevent strong internal rectangles (for example, a table
# inside a form) from being mistaken for the whole document.
MIN_DOCUMENT_AREA_FRACTION = 0.25
MIN_DOCUMENT_WIDTH_FRACTION = 0.40
MIN_DOCUMENT_HEIGHT_FRACTION = 0.40

MORPH_KERNEL_SIZE = 31
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 15


# =============================================================================
# GEOMETRY
# =============================================================================

def order_points(
    points: np.ndarray,
) -> np.ndarray:
    points = np.asarray(
        points,
        dtype=np.float32,
    ).reshape(4, 2)

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    point_sum = points.sum(
        axis=1
    )

    point_diff = np.diff(
        points,
        axis=1,
    ).reshape(-1)

    ordered[0] = points[
        np.argmin(
            point_sum
        )
    ]

    ordered[2] = points[
        np.argmax(
            point_sum
        )
    ]

    ordered[1] = points[
        np.argmin(
            point_diff
        )
    ]

    ordered[3] = points[
        np.argmax(
            point_diff
        )
    ]

    return ordered


def four_point_transform(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    rect = order_points(
        points
    )

    (
        top_left,
        top_right,
        bottom_right,
        bottom_left,
    ) = rect

    width_a = np.linalg.norm(
        bottom_right
        - bottom_left
    )

    width_b = np.linalg.norm(
        top_right
        - top_left
    )

    max_width = int(
        max(
            width_a,
            width_b,
        )
    )

    height_a = np.linalg.norm(
        top_right
        - bottom_right
    )

    height_b = np.linalg.norm(
        top_left
        - bottom_left
    )

    max_height = int(
        max(
            height_a,
            height_b,
        )
    )

    if (
        max_width <= 0
        or max_height <= 0
    ):
        raise RuntimeError(
            "Invalid document dimensions."
        )

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [
                max_width - 1,
                max_height - 1,
            ],
            [
                0,
                max_height - 1,
            ],
        ],
        dtype=np.float32,
    )

    transform_matrix = (
        cv2.getPerspectiveTransform(
            rect,
            destination,
        )
    )

    warped = cv2.warpPerspective(
        image,
        transform_matrix,
        (
            max_width,
            max_height,
        ),
    )

    return warped


# =============================================================================
# DETECTION RESIZE
# =============================================================================

def resize_for_detection(
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


# =============================================================================
# ENHANCED EDGE DETECTOR
# =============================================================================

def create_enhanced_edge_map(
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

    return (
        enhanced,
        magnitude,
        combined,
    )


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

        if perimeter <= 0:
            continue

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
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    resized, scale = (
        resize_for_detection(
            image
        )
    )

    (
        _enhanced,
        _gradient,
        edge_map,
    ) = create_enhanced_edge_map(
        resized
    )

    candidates = (
        find_enhanced_candidates(
            edge_map
        )
    )

    if not candidates:
        raise RuntimeError(
            "Enhanced detector could not "
            "find a document candidate."
        )

    polygon = (
        candidates[0]["polygon"]
        * scale
    )

    polygon = order_points(
        polygon
    )

    return (
        polygon,
        edge_map,
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
    resized, scale = (
        resize_for_detection(
            image
        )
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

    # Search more than the original top 10 contours because a document
    # can contain several large internal tables/boxes.
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )[:40]

    document_contour = None

    for contour in contours:
        perimeter = float(
            cv2.arcLength(
                contour,
                True,
            )
        )

        if perimeter <= 0:
            continue

        approximation = (
            cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True,
            )
        )

        if len(
            approximation
        ) != 4:
            continue

        candidate = (
            approximation
            .reshape(4, 2)
            .astype(
                np.float32
            )
        )

        # Convert the candidate back to original-image coordinates before
        # validating its size.
        candidate *= scale

        candidate = order_points(
            candidate
        )

        (
            size_plausible,
            area_fraction,
            width_fraction,
            height_fraction,
        ) = candidate_size_is_plausible(
            candidate,
            image,
        )

        if not size_plausible:
            print(
                'Legacy candidate rejected '
                'because it is too small '
                'to represent the whole document.'
            )

            print(
                f'Area fraction   : '
                f'{area_fraction:.4f}'
            )

            print(
                f'Width fraction  : '
                f'{width_fraction:.4f}'
            )

            print(
                f'Height fraction : '
                f'{height_fraction:.4f}'
            )

            continue

        document_contour = candidate
        break

    if document_contour is None:
        raise RuntimeError(
            'Legacy detector could not find '
            'a plausible full-document contour.'
        )

    return (
        document_contour,
        edges,
    )


# =============================================================================
# HYBRID SANITY CHECK
# =============================================================================

def polygon_area_fraction(
    polygon: np.ndarray,
    image: np.ndarray,
) -> float:
    polygon_area = float(
        abs(
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

    return (
        polygon_area
        / image_area
    )


def polygon_span_fractions(
    polygon: np.ndarray,
    image: np.ndarray,
) -> tuple[
    float,
    float,
]:
    height, width = (
        image.shape[:2]
    )

    if (
        width <= 0
        or height <= 0
    ):
        return (
            0.0,
            0.0,
        )

    points = np.asarray(
        polygon,
        dtype=np.float32,
    ).reshape(
        4,
        2,
    )

    min_x = float(
        np.min(
            points[:, 0]
        )
    )

    max_x = float(
        np.max(
            points[:, 0]
        )
    )

    min_y = float(
        np.min(
            points[:, 1]
        )
    )

    max_y = float(
        np.max(
            points[:, 1]
        )
    )

    width_fraction = (
        max_x - min_x
    ) / float(
        width
    )

    height_fraction = (
        max_y - min_y
    ) / float(
        height
    )

    return (
        float(
            np.clip(
                width_fraction,
                0.0,
                1.0,
            )
        ),
        float(
            np.clip(
                height_fraction,
                0.0,
                1.0,
            )
        ),
    )


def candidate_size_is_plausible(
    polygon: np.ndarray,
    image: np.ndarray,
) -> tuple[
    bool,
    float,
    float,
    float,
]:
    area_fraction = (
        polygon_area_fraction(
            polygon,
            image,
        )
    )

    (
        width_fraction,
        height_fraction,
    ) = polygon_span_fractions(
        polygon,
        image,
    )

    plausible = (
        area_fraction
        >= MIN_DOCUMENT_AREA_FRACTION
        and width_fraction
        >= MIN_DOCUMENT_WIDTH_FRACTION
        and height_fraction
        >= MIN_DOCUMENT_HEIGHT_FRACTION
    )

    return (
        plausible,
        area_fraction,
        width_fraction,
        height_fraction,
    )


def full_frame_document_polygon(
    image: np.ndarray,
) -> np.ndarray:
    height, width = (
        image.shape[:2]
    )

    if (
        width <= 1
        or height <= 1
    ):
        raise RuntimeError(
            'Image is too small for '
            'full-frame fallback.'
        )

    return np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )


def count_border_vertices(
    polygon: np.ndarray,
    image: np.ndarray,
) -> int:
    height, width = (
        image.shape[:2]
    )

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

        near_border = (
            x <= margin_x
            or x >= width - margin_x
            or y <= margin_y
            or y >= height - margin_y
        )

        if near_border:
            count += 1

    return count


def enhanced_candidate_is_suspicious(
    polygon: np.ndarray,
    image: np.ndarray,
) -> tuple[
    bool,
    int,
    float,
    float,
    float,
]:
    border_vertices = (
        count_border_vertices(
            polygon,
            image,
        )
    )

    (
        size_plausible,
        area_fraction,
        width_fraction,
        height_fraction,
    ) = candidate_size_is_plausible(
        polygon,
        image,
    )

    suspicious = (
        border_vertices
        >= BORDER_VERTEX_LIMIT
        or not size_plausible
    )

    return (
        suspicious,
        border_vertices,
        area_fraction,
        width_fraction,
        height_fraction,
    )


# =============================================================================
# PRODUCTION HYBRID DETECTOR
# =============================================================================

def detect_document(
    image: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    str,
]:
    enhanced_polygon = None
    enhanced_edges = None
    enhanced_size_plausible = False

    try:
        (
            enhanced_polygon,
            enhanced_edges,
        ) = detect_document_enhanced(
            image
        )

        (
            suspicious,
            border_vertices,
            area_fraction,
            width_fraction,
            height_fraction,
        ) = (
            enhanced_candidate_is_suspicious(
                enhanced_polygon,
                image,
            )
        )

        enhanced_size_plausible = (
            area_fraction
            >= MIN_DOCUMENT_AREA_FRACTION
            and width_fraction
            >= MIN_DOCUMENT_WIDTH_FRACTION
            and height_fraction
            >= MIN_DOCUMENT_HEIGHT_FRACTION
        )

        if not suspicious:
            return (
                enhanced_polygon,
                enhanced_edges,
                'enhanced',
            )

        print(
            'Enhanced candidate rejected '
            'by sanity check.'
        )

        print(
            f'Border vertices: '
            f'{border_vertices}'
        )

        print(
            f'Area fraction  : '
            f'{area_fraction:.4f}'
        )

        print(
            f'Width fraction : '
            f'{width_fraction:.4f}'
        )

        print(
            f'Height fraction: '
            f'{height_fraction:.4f}'
        )

    except Exception as error:
        print(
            'Enhanced detector failed:'
        )

        print(
            f'  {error}'
        )

    try:
        (
            legacy_polygon,
            legacy_edges,
        ) = detect_document_legacy(
            image
        )

        return (
            legacy_polygon,
            legacy_edges,
            'legacy_fallback',
        )

    except Exception as error:
        print(
            'Legacy detector failed:'
        )

        print(
            f'  {error}'
        )

    # If the enhanced detector found a page-sized candidate but rejected it
    # only because it touched the image border, it is safer to use that
    # candidate than to crop to a small internal table.
    if (
        enhanced_polygon
        is not None
        and enhanced_edges
        is not None
        and enhanced_size_plausible
    ):
        print(
            'Using page-sized enhanced '
            'candidate as recovery fallback.'
        )

        return (
            enhanced_polygon,
            enhanced_edges,
            'enhanced_recovery',
        )

    # Final safety fallback:
    # preserve the entire photograph rather than incorrectly cutting away
    # most of the document. Perspective correction becomes a no-op, but all
    # page content remains available to the scan and OCR branches.
    print(
        'No reliable full-document boundary '
        'was found.'
    )

    print(
        'Using the full image as a safe '
        'content-preserving fallback.'
    )

    fallback_polygon = (
        full_frame_document_polygon(
            image
        )
    )

    if enhanced_edges is not None:
        fallback_edges = (
            enhanced_edges
        )

    else:
        resized, _ = (
            resize_for_detection(
                image
            )
        )

        gray = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2GRAY,
        )

        fallback_edges = cv2.Canny(
            gray,
            50,
            150,
        )

    return (
        fallback_polygon,
        fallback_edges,
        'full_frame_fallback',
    )


# =============================================================================
# PRODUCTION DOCUMENT ENHANCEMENT
# =============================================================================

def morphological_illumination_correction(
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


def enhance_document(
    document: np.ndarray,
) -> dict[
    str,
    np.ndarray,
]:
    gray = cv2.cvtColor(
        document,
        cv2.COLOR_BGR2GRAY,
    )

    (
        illumination_background,
        illumination_corrected,
    ) = (
        morphological_illumination_correction(
            gray
        )
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    contrast = clahe.apply(
        illumination_corrected
    )

    denoised = cv2.GaussianBlur(
        contrast,
        (3, 3),
        0,
    )

    scan = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )

    return {
        "gray":
            gray,

        "illumination_background":
            illumination_background,

        "illumination_corrected":
            illumination_corrected,

        "contrast":
            contrast,

        "scan":
            scan,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_document_boundary(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    preview = (
        image.copy()
    )

    polygon = (
        points
        .astype(np.int32)
        .reshape(
            (-1, 1, 2)
        )
    )

    cv2.polylines(
        preview,
        [polygon],
        True,
        (0, 255, 0),
        4,
    )

    return preview


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect, rectify, illumination-correct, "
            "and enhance a photographed document."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to input document image."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="data/output",
        help=(
            "Directory for generated outputs."
        ),
    )

    return parser.parse_args()


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
            f"Input file not found:\n"
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
            f"Could not load image:\n"
            f"{input_path}"
        )

    print(
        "=" * 80
    )

    print(
        "DOCUMENT IMAGE UNDERSTANDING"
    )

    print(
        "=" * 80
    )

    print(
        f"Input: {input_path}"
    )

    print(
        f"Image size: "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    print()

    print(
        "Detecting document boundary..."
    )

    (
        document_points,
        edge_preview,
        route,
    ) = detect_document(
        image
    )

    print(
        f"Detector route: {route}"
    )

    print(
        "Document boundary detected."
    )

    boundary_preview = (
        draw_document_boundary(
            image,
            document_points,
        )
    )

    print(
        "Applying perspective correction..."
    )

    perspective = (
        four_point_transform(
            image,
            document_points,
        )
    )

    print(
        "Correcting illumination and shadows..."
    )

    enhanced = (
        enhance_document(
            perspective
        )
    )

    stem = (
        input_path.stem
    )

    files = {
        "boundary":
            output_dir
            / f"{stem}_boundary.png",

        "edges":
            output_dir
            / f"{stem}_edges.png",

        "perspective":
            output_dir
            / f"{stem}_perspective.png",

        "gray":
            output_dir
            / f"{stem}_gray.png",

        "illumination":
            output_dir
            / f"{stem}_illumination.png",

        "corrected":
            output_dir
            / f"{stem}_corrected.png",

        "contrast":
            output_dir
            / f"{stem}_contrast.png",

        "scan":
            output_dir
            / f"{stem}_scan.png",
    }

    cv2.imwrite(
        str(
            files["boundary"]
        ),
        boundary_preview,
    )

    cv2.imwrite(
        str(
            files["edges"]
        ),
        edge_preview,
    )

    cv2.imwrite(
        str(
            files["perspective"]
        ),
        perspective,
    )

    cv2.imwrite(
        str(
            files["gray"]
        ),
        enhanced["gray"],
    )

    cv2.imwrite(
        str(
            files["illumination"]
        ),
        enhanced[
            "illumination_background"
        ],
    )

    cv2.imwrite(
        str(
            files["corrected"]
        ),
        enhanced[
            "illumination_corrected"
        ],
    )

    cv2.imwrite(
        str(
            files["contrast"]
        ),
        enhanced["contrast"],
    )

    cv2.imwrite(
        str(
            files["scan"]
        ),
        enhanced["scan"],
    )

    print()

    print(
        "=" * 80
    )

    print(
        "PROCESSING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Detector route: {route}"
    )

    print(
        "Enhancement pipeline:"
    )

    print(
        "  grayscale"
    )

    print(
        "  -> morphological illumination correction"
    )

    print(
        "  -> CLAHE"
    )

    print(
        "  -> adaptive threshold"
    )

    print()

    for name, path in files.items():
        print(
            f"{name:<12}: {path}"
        )


if __name__ == "__main__":
    main()