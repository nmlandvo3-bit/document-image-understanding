from __future__ import annotations

# =============================================================================
# PADDLEPADDLE WINDOWS CPU COMPATIBILITY
# =============================================================================

import os

os.environ["FLAGS_enable_pir_api"] = "0"

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

from document_scanner import (
    detect_document,
    four_point_transform,
    enhance_document,
    draw_document_boundary,
)


# =============================================================================
# OCR CONFIGURATION
# =============================================================================

MIN_CONFIDENCE = 0.25
MIN_ALPHANUMERIC_CHARACTERS = 2

# Single-character OCR tokens such as table row numbers are accepted only
# when PaddleOCR is highly confident.
SINGLE_CHARACTER_MIN_CONFIDENCE = 0.85

STRONG_MIN_REGIONS = 5
STRONG_MIN_CHARACTERS = 20
STRONG_MIN_MEAN_CONFIDENCE = 0.50

LIMITED_MIN_REGIONS = 2
LIMITED_MIN_CHARACTERS = 10


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Production Document Processor V1 - "
            "document scanning, enhancement and adaptive OCR."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input photograph containing a document.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where processed outputs will be saved.",
    )

    return parser.parse_args()


# =============================================================================
# OCR TEXT HELPERS
# =============================================================================

def alphanumeric_count(
    text: str,
) -> int:
    return sum(
        character.isalnum()
        for character in text
    )


def is_meaningful(
    text: str,
    confidence: float,
) -> bool:
    """
    Decide whether an OCR token should be preserved.

    Normal text:
        Must contain at least 2 alphanumeric characters.

    Single-character text:
        Accepted only when it is alphanumeric and PaddleOCR confidence
        is very high. This preserves valid table values such as 1-9
        while filtering low-confidence OCR noise.
    """

    text = text.strip()

    if not text:
        return False

    if confidence < MIN_CONFIDENCE:
        return False

    alnum_count = alphanumeric_count(
        text
    )

    # -------------------------------------------------------------------------
    # Normal text
    # -------------------------------------------------------------------------

    if (
        alnum_count
        >= MIN_ALPHANUMERIC_CHARACTERS
    ):
        return True

    # -------------------------------------------------------------------------
    # High-confidence single-character tokens
    # -------------------------------------------------------------------------

    if (
        len(text) == 1
        and text.isalnum()
        and confidence
        >= SINGLE_CHARACTER_MIN_CONFIDENCE
    ):
        return True

    return False


# =============================================================================
# OCR RECOVERY
# =============================================================================

def resize_bicubic(
    image: np.ndarray,
    scale: int,
) -> np.ndarray:
    height, width = image.shape[:2]

    return cv2.resize(
        image,
        (
            width * scale,
            height * scale,
        ),
        interpolation=cv2.INTER_CUBIC,
    )


def apply_unsharp(
    image: np.ndarray,
    amount: float = 1.2,
    sigma: float = 1.0,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
    )

    return cv2.addWeighted(
        image,
        1.0 + amount,
        blurred,
        -amount,
        0,
    )


def build_recovery_variant(
    perspective: np.ndarray,
) -> np.ndarray:
    upscaled = resize_bicubic(
        perspective,
        4,
    )

    recovered = apply_unsharp(
        upscaled,
        amount=1.2,
        sigma=1.0,
    )

    return recovered


# =============================================================================
# PADDLEOCR
# =============================================================================

def run_paddleocr(
    pipeline: PaddleOCR,
    image: np.ndarray,
    raw_dir: Path,
    variant_name: str,
) -> list[dict[str, object]]:
    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_objects = pipeline.predict(
        input=image,
        text_rec_score_thresh=0.0,
    )

    rows = []

    for result_index, result in enumerate(
        result_objects,
        start=1,
    ):
        json_path = (
            raw_dir
            / f"{variant_name}_{result_index:02d}.json"
        )

        result.save_to_json(
            save_path=str(
                json_path
            )
        )

        if not json_path.exists():
            continue

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(
                file
            )

        data = payload.get(
            "res",
            payload,
        )

        texts = data.get(
            "rec_texts",
            [],
        )

        scores = data.get(
            "rec_scores",
            [],
        )

        boxes = data.get(
            "rec_boxes",
            [],
        )

        count = min(
            len(texts),
            len(scores),
        )

        for index in range(
            count
        ):
            text = str(
                texts[index]
            ).strip()

            confidence = float(
                scores[index]
            )

            if not text:
                continue

            x = 0
            y = 0
            w = 0
            h = 0

            if index < len(
                boxes
            ):
                try:
                    box_array = np.asarray(
                        boxes[index],
                        dtype=np.float32,
                    )

                    if (
                        box_array.ndim == 1
                        and box_array.size >= 4
                    ):
                        x1 = int(
                            round(
                                float(
                                    box_array[0]
                                )
                            )
                        )

                        y1 = int(
                            round(
                                float(
                                    box_array[1]
                                )
                            )
                        )

                        x2 = int(
                            round(
                                float(
                                    box_array[2]
                                )
                            )
                        )

                        y2 = int(
                            round(
                                float(
                                    box_array[3]
                                )
                            )
                        )

                        x = x1
                        y = y1

                        w = max(
                            0,
                            x2 - x1,
                        )

                        h = max(
                            0,
                            y2 - y1,
                        )

                    elif (
                        box_array.ndim == 2
                        and box_array.shape[1] >= 2
                    ):
                        xs = box_array[:, 0]
                        ys = box_array[:, 1]

                        x1 = int(
                            round(
                                float(
                                    np.min(xs)
                                )
                            )
                        )

                        y1 = int(
                            round(
                                float(
                                    np.min(ys)
                                )
                            )
                        )

                        x2 = int(
                            round(
                                float(
                                    np.max(xs)
                                )
                            )
                        )

                        y2 = int(
                            round(
                                float(
                                    np.max(ys)
                                )
                            )
                        )

                        x = x1
                        y = y1

                        w = max(
                            0,
                            x2 - x1,
                        )

                        h = max(
                            0,
                            y2 - y1,
                        )

                except Exception:
                    pass

            rows.append(
                {
                    "text":
                        text,

                    "confidence":
                        confidence,

                    "meaningful":
                        is_meaningful(
                            text,
                            confidence,
                        ),

                    "x":
                        x,

                    "y":
                        y,

                    "w":
                        w,

                    "h":
                        h,
                }
            )

    rows.sort(
        key=lambda row: (
            int(
                row["y"]
            ),
            int(
                row["x"]
            ),
        )
    )

    return rows


# =============================================================================
# OCR STATISTICS
# =============================================================================

def summarize_ocr(
    rows: list[dict[str, object]],
) -> dict[str, int | float]:
    meaningful_rows = [
        row
        for row in rows
        if bool(
            row["meaningful"]
        )
    ]

    confidences = [
        float(
            row["confidence"]
        )
        for row in meaningful_rows
    ]

    if confidences:
        mean_confidence = float(
            np.mean(
                confidences
            )
        )

        median_confidence = float(
            np.median(
                confidences
            )
        )

    else:
        mean_confidence = 0.0
        median_confidence = 0.0

    words = sum(
        len(
            str(
                row["text"]
            ).split()
        )
        for row in meaningful_rows
    )

    characters = sum(
        alphanumeric_count(
            str(
                row["text"]
            )
        )
        for row in meaningful_rows
    )

    return {
        "total_regions":
            len(rows),

        "meaningful_regions":
            len(
                meaningful_rows
            ),

        "words":
            words,

        "characters":
            characters,

        "mean_confidence":
            mean_confidence,

        "median_confidence":
            median_confidence,
    }


# =============================================================================
# OCR QUALITY
# =============================================================================

def classify_quality(
    stats: dict[str, int | float],
) -> str:
    regions = int(
        stats[
            "meaningful_regions"
        ]
    )

    characters = int(
        stats[
            "characters"
        ]
    )

    confidence = float(
        stats[
            "mean_confidence"
        ]
    )

    if (
        regions >= STRONG_MIN_REGIONS
        and characters >= STRONG_MIN_CHARACTERS
        and confidence
        >= STRONG_MIN_MEAN_CONFIDENCE
    ):
        return "STRONG"

    if (
        regions >= LIMITED_MIN_REGIONS
        and characters >= LIMITED_MIN_CHARACTERS
    ):
        return "LIMITED"

    return "INSUFFICIENT"


def result_score(
    stats: dict[str, int | float],
) -> tuple:
    return (
        int(
            stats[
                "characters"
            ]
        ),
        int(
            stats[
                "meaningful_regions"
            ]
        ),
        int(
            stats[
                "words"
            ]
        ),
        float(
            stats[
                "mean_confidence"
            ]
        ),
    )


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def draw_ocr_regions(
    image: np.ndarray,
    rows: list[dict[str, object]],
) -> np.ndarray:
    preview = image.copy()

    for index, row in enumerate(
        rows,
        start=1,
    ):
        x = int(
            row["x"]
        )

        y = int(
            row["y"]
        )

        w = int(
            row["w"]
        )

        h = int(
            row["h"]
        )

        if (
            w <= 0
            or h <= 0
        ):
            continue

        if bool(
            row["meaningful"]
        ):
            color = (
                0,
                255,
                0,
            )
        else:
            color = (
                0,
                0,
                255,
            )

        cv2.rectangle(
            preview,
            (
                x,
                y,
            ),
            (
                x + w,
                y + h,
            ),
            color,
            2,
        )

        cv2.putText(
            preview,
            f"P{index}",
            (
                x,
                max(
                    15,
                    y - 4,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return preview


def save_text(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            if not bool(
                row[
                    "meaningful"
                ]
            ):
                continue

            text = str(
                row[
                    "text"
                ]
            ).strip()

            if text:
                file.write(
                    text
                )

                file.write(
                    "\n"
                )


def save_ocr_csv(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    fieldnames = [
        "region",
        "text",
        "confidence",
        "meaningful",
        "x",
        "y",
        "w",
        "h",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index, row in enumerate(
            rows,
            start=1,
        ):
            writer.writerow(
                {
                    "region":
                        index,

                    "text":
                        row[
                            "text"
                        ],

                    "confidence":
                        (
                            f"{float(row['confidence']):.6f}"
                        ),

                    "meaningful":
                        row[
                            "meaningful"
                        ],

                    "x":
                        row["x"],

                    "y":
                        row["y"],

                    "w":
                        row["w"],

                    "h":
                        row["h"],
                }
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

    scan_dir = (
        output_dir
        / "scan"
    )

    ocr_dir = (
        output_dir
        / "ocr"
    )

    raw_ocr_dir = (
        ocr_dir
        / "raw_results"
    )

    scan_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ocr_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_ocr_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load input
    # -------------------------------------------------------------------------

    image = cv2.imread(
        str(
            input_path
        )
    )

    if image is None:
        raise RuntimeError(
            f"Could not read input image:\n"
            f"{input_path}"
        )

    print(
        "=" * 100
    )

    print(
        "DOCUMENT PROCESSOR V1"
    )

    print(
        "=" * 100
    )

    print(
        f"Input: {input_path}"
    )

    # =========================================================================
    # DOCUMENT LOCALIZATION
    # =========================================================================

    print()

    print(
        "Stage 1: Document localization"
    )

    (
        document_points,
        edge_preview,
        detector_route,
    ) = detect_document(
        image
    )

    boundary = draw_document_boundary(
        image,
        document_points,
    )

    cv2.imwrite(
        str(
            output_dir
            / "boundary.png"
        ),
        boundary,
    )

    cv2.imwrite(
        str(
            output_dir
            / "edges.png"
        ),
        edge_preview,
    )

    print(
        f"Detector route : "
        f"{detector_route}"
    )

    # =========================================================================
    # PERSPECTIVE CORRECTION
    # =========================================================================

    print()

    print(
        "Stage 2: Perspective correction"
    )

    perspective = four_point_transform(
        image,
        document_points,
    )

    perspective_path = (
        output_dir
        / "perspective.png"
    )

    cv2.imwrite(
        str(
            perspective_path
        ),
        perspective,
    )

    print(
        f"Document size  : "
        f"{perspective.shape[1]} x "
        f"{perspective.shape[0]}"
    )

    # =========================================================================
    # SCAN BRANCH
    # =========================================================================

    print()

    print(
        "Stage 3: Scan enhancement"
    )

    enhanced = enhance_document(
        perspective
    )

    cv2.imwrite(
        str(
            scan_dir
            / "gray.png"
        ),
        enhanced[
            "gray"
        ],
    )

    cv2.imwrite(
        str(
            scan_dir
            / "illumination_background.png"
        ),
        enhanced[
            "illumination_background"
        ],
    )

    cv2.imwrite(
        str(
            scan_dir
            / "illumination_corrected.png"
        ),
        enhanced[
            "illumination_corrected"
        ],
    )

    cv2.imwrite(
        str(
            scan_dir
            / "contrast.png"
        ),
        enhanced[
            "contrast"
        ],
    )

    final_scan_path = (
        scan_dir
        / "scan.png"
    )

    cv2.imwrite(
        str(
            final_scan_path
        ),
        enhanced[
            "scan"
        ],
    )

    print(
        f"Final scan     : "
        f"{final_scan_path}"
    )

    # =========================================================================
    # OCR ENGINE
    # =========================================================================

    print()

    print(
        "Stage 4: Loading PaddleOCR"
    )

    print(
        "PIR workaround enabled."
    )

    print(
        "CPU oneDNN / MKL-DNN disabled."
    )

    pipeline = PaddleOCR(
        lang="en",

        use_doc_orientation_classify=False,

        use_doc_unwarping=False,

        use_textline_orientation=False,

        device="cpu",

        enable_mkldnn=False,
    )

    print(
        "PaddleOCR loaded."
    )

    # =========================================================================
    # DIRECT OCR
    # =========================================================================

    print()

    print(
        "Stage 5: Direct OCR"
    )

    direct_rows = run_paddleocr(
        pipeline,
        perspective,
        raw_ocr_dir,
        "direct",
    )

    direct_stats = summarize_ocr(
        direct_rows
    )

    direct_quality = classify_quality(
        direct_stats
    )

    print(
        f"Meaningful regions : "
        f"{direct_stats['meaningful_regions']}"
    )

    print(
        f"Words              : "
        f"{direct_stats['words']}"
    )

    print(
        f"Characters         : "
        f"{direct_stats['characters']}"
    )

    print(
        f"Mean confidence    : "
        f"{float(direct_stats['mean_confidence']):.4f}"
    )

    print(
        f"Quality            : "
        f"{direct_quality}"
    )

    # =========================================================================
    # ADAPTIVE OCR ROUTING
    # =========================================================================

    recovery_attempted = False

    chosen_rows = direct_rows
    chosen_stats = direct_stats
    chosen_quality = direct_quality

    ocr_route = "direct"

    recovery_stats = None
    recovery_quality = None

    final_ocr_image = perspective

    if direct_quality != "STRONG":
        recovery_attempted = True

        print()

        print(
            "Stage 6: OCR recovery fallback"
        )

        recovery_image = (
            build_recovery_variant(
                perspective
            )
        )

        cv2.imwrite(
            str(
                ocr_dir
                / "recovery_input.png"
            ),
            recovery_image,
        )

        recovery_rows = run_paddleocr(
            pipeline,
            recovery_image,
            raw_ocr_dir,
            "recovery",
        )

        recovery_stats = summarize_ocr(
            recovery_rows
        )

        recovery_quality = classify_quality(
            recovery_stats
        )

        print(
            f"Recovery regions    : "
            f"{recovery_stats['meaningful_regions']}"
        )

        print(
            f"Recovery words      : "
            f"{recovery_stats['words']}"
        )

        print(
            f"Recovery characters : "
            f"{recovery_stats['characters']}"
        )

        print(
            f"Recovery confidence : "
            f"{float(recovery_stats['mean_confidence']):.4f}"
        )

        print(
            f"Recovery quality    : "
            f"{recovery_quality}"
        )

        if (
            result_score(
                recovery_stats
            )
            >
            result_score(
                direct_stats
            )
        ):
            chosen_rows = recovery_rows
            chosen_stats = recovery_stats
            chosen_quality = recovery_quality

            final_ocr_image = recovery_image

            if (
                recovery_quality
                == "INSUFFICIENT"
            ):
                ocr_route = (
                    "recovery_failed"
                )

            else:
                ocr_route = (
                    "recovery_fallback"
                )

        else:
            if (
                direct_quality
                == "INSUFFICIENT"
            ):
                ocr_route = (
                    "recovery_failed"
                )

            else:
                ocr_route = (
                    "direct_retained"
                )

    else:
        print()

        print(
            "Strong direct OCR detected."
        )

        print(
            "Recovery skipped."
        )

    # =========================================================================
    # SAVE FINAL OCR
    # =========================================================================

    print()

    print(
        "Stage 7: Saving final OCR outputs"
    )

    regions_preview = draw_ocr_regions(
        final_ocr_image,
        chosen_rows,
    )

    final_regions_path = (
        ocr_dir
        / "regions.png"
    )

    cv2.imwrite(
        str(
            final_regions_path
        ),
        regions_preview,
    )

    final_text_path = (
        ocr_dir
        / "text.txt"
    )

    save_text(
        chosen_rows,
        final_text_path,
    )

    final_csv_path = (
        ocr_dir
        / "results.csv"
    )

    save_ocr_csv(
        chosen_rows,
        final_csv_path,
    )

    # =========================================================================
    # PROCESSING SUMMARY
    # =========================================================================

    summary = {
        "input_file":
            str(
                input_path
            ),

        "detector_route":
            detector_route,

        "document_width":
            int(
                perspective.shape[1]
            ),

        "document_height":
            int(
                perspective.shape[0]
            ),

        "ocr_engine":
            "PaddleOCR",

        "ocr_route":
            ocr_route,

        "ocr_quality":
            chosen_quality,

        "recovery_attempted":
            recovery_attempted,

        "meaningful_regions":
            int(
                chosen_stats[
                    "meaningful_regions"
                ]
            ),

        "words":
            int(
                chosen_stats[
                    "words"
                ]
            ),

        "characters":
            int(
                chosen_stats[
                    "characters"
                ]
            ),

        "mean_confidence":
            float(
                chosen_stats[
                    "mean_confidence"
                ]
            ),

        "direct_ocr": {
            "quality":
                direct_quality,

            "meaningful_regions":
                int(
                    direct_stats[
                        "meaningful_regions"
                    ]
                ),

            "words":
                int(
                    direct_stats[
                        "words"
                    ]
                ),

            "characters":
                int(
                    direct_stats[
                        "characters"
                    ]
                ),

            "mean_confidence":
                float(
                    direct_stats[
                        "mean_confidence"
                    ]
                ),
        },

        "recovery_ocr":
            (
                {
                    "quality":
                        recovery_quality,

                    "meaningful_regions":
                        int(
                            recovery_stats[
                                "meaningful_regions"
                            ]
                        ),

                    "words":
                        int(
                            recovery_stats[
                                "words"
                            ]
                        ),

                    "characters":
                        int(
                            recovery_stats[
                                "characters"
                            ]
                        ),

                    "mean_confidence":
                        float(
                            recovery_stats[
                                "mean_confidence"
                            ]
                        ),
                }
                if recovery_stats
                is not None
                else None
            ),

        "outputs": {
            "boundary":
                "boundary.png",

            "perspective":
                "perspective.png",

            "scan":
                "scan/scan.png",

            "ocr_regions":
                "ocr/regions.png",

            "ocr_text":
                "ocr/text.txt",

            "ocr_results":
                "ocr/results.csv",
        },
    }

    summary_path = (
        output_dir
        / "processing_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # =========================================================================
    # FINAL TERMINAL SUMMARY
    # =========================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "DOCUMENT PROCESSING COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"Detector route      : "
        f"{detector_route}"
    )

    print(
        f"OCR engine          : "
        f"PaddleOCR"
    )

    print(
        f"OCR route           : "
        f"{ocr_route}"
    )

    print(
        f"OCR quality         : "
        f"{chosen_quality}"
    )

    print(
        f"Recovery attempted  : "
        f"{recovery_attempted}"
    )

    print(
        f"Meaningful regions  : "
        f"{chosen_stats['meaningful_regions']}"
    )

    print(
        f"Words               : "
        f"{chosen_stats['words']}"
    )

    print(
        f"Characters          : "
        f"{chosen_stats['characters']}"
    )

    print(
        f"Mean confidence     : "
        f"{float(chosen_stats['mean_confidence']):.4f}"
    )

    if (
        chosen_quality
        == "INSUFFICIENT"
    ):
        print()

        print(
            "OCR warning         : "
            "Document text could not be reliably recovered."
        )

        print(
            "Recommendation      : "
            "Recapture the document closer to the camera "
            "with better focus and lighting."
        )

    elif (
        chosen_quality
        == "LIMITED"
    ):
        print()

        print(
            "OCR warning         : "
            "Only partial text was recovered."
        )

    else:
        print()

        print(
            "OCR status          : "
            "Text recognition acceptable."
        )

    print()

    print(
        f"Final scan:\n"
        f"{final_scan_path}"
    )

    print()

    print(
        f"OCR text:\n"
        f"{final_text_path}"
    )

    print()

    print(
        f"Processing summary:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()