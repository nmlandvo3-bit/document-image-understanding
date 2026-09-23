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
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_CONFIDENCE = 0.25
MIN_ALPHANUMERIC_CHARACTERS = 2

# Strong OCR gate.
#
# document1:
#   22 meaningful
#   219 chars
#
# letter001:
#   29 meaningful
#   981 chars
#
# These should pass immediately.
STRONG_MIN_REGIONS = 5
STRONG_MIN_CHARACTERS = 20
STRONG_MIN_MEAN_CONFIDENCE = 0.50

# Limited OCR gate.
#
# Useful for identifying partial recovery such as the magazine.
LIMITED_MIN_REGIONS = 2
LIMITED_MIN_CHARACTERS = 10


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 21 - adaptive PaddleOCR routing "
            "with OCR quality gating and recovery fallback."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Original document photograph.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Experiment 21 output directory.",
    )

    return parser.parse_args()


# =============================================================================
# TEXT HELPERS
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
    if confidence < MIN_CONFIDENCE:
        return False

    if (
        alphanumeric_count(text)
        < MIN_ALPHANUMERIC_CHARACTERS
    ):
        return False

    return True


# =============================================================================
# IMAGE RECOVERY
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

    sharpened = cv2.addWeighted(
        image,
        1.0 + amount,
        blurred,
        -amount,
        0,
    )

    return sharpened


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
    output_dir: Path,
    variant_name: str,
) -> list[dict[str, object]]:
    raw_dir = (
        output_dir
        / "raw_paddle_results"
    )

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
            / (
                f"{variant_name}_"
                f"{result_index:02d}.json"
            )
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
# STATISTICS
# =============================================================================

def summarize(
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
# QUALITY GATE
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

    mean_confidence = float(
        stats[
            "mean_confidence"
        ]
    )

    if (
        regions >= STRONG_MIN_REGIONS
        and characters >= STRONG_MIN_CHARACTERS
        and mean_confidence
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

def draw_results(
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
                row["meaningful"]
            ):
                continue

            file.write(
                str(
                    row["text"]
                ).strip()
            )

            file.write(
                "\n"
            )


def save_csv(
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
                        row["text"],

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


def save_variant_outputs(
    variant_name: str,
    image: np.ndarray,
    rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    variant_dir = (
        output_dir
        / variant_name
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            variant_dir
            / "ocr_input.png"
        ),
        image,
    )

    preview = draw_results(
        image,
        rows,
    )

    cv2.imwrite(
        str(
            variant_dir
            / "ocr_regions.png"
        ),
        preview,
    )

    save_text(
        rows,
        variant_dir
        / "recognized_text.txt",
    )

    save_csv(
        rows,
        variant_dir
        / "ocr_results.csv",
    )


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_stats(
    title: str,
    stats: dict[str, int | float],
    quality: str,
) -> None:
    print()

    print(
        "-" * 100
    )

    print(
        title
    )

    print(
        "-" * 100
    )

    print(
        f"Total regions      : "
        f"{stats['total_regions']}"
    )

    print(
        f"Meaningful regions : "
        f"{stats['meaningful_regions']}"
    )

    print(
        f"Words              : "
        f"{stats['words']}"
    )

    print(
        f"Characters         : "
        f"{stats['characters']}"
    )

    print(
        f"Mean confidence    : "
        f"{float(stats['mean_confidence']):.4f}"
    )

    print(
        f"OCR quality        : "
        f"{quality}"
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
            f"Input not found:\n"
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
        "=" * 100
    )

    print(
        "EXPERIMENT 21 — "
        "ADAPTIVE OCR ROUTING"
    )

    print(
        "=" * 100
    )

    print(
        f"Input: {input_path}"
    )

    # -------------------------------------------------------------------------
    # Frozen production document detector
    # -------------------------------------------------------------------------

    (
        document_points,
        _edges,
        detector_route,
    ) = detect_document(
        image
    )

    perspective = four_point_transform(
        image,
        document_points,
    )

    print(
        f"Detector route : "
        f"{detector_route}"
    )

    print(
        f"Document size  : "
        f"{perspective.shape[1]} x "
        f"{perspective.shape[0]}"
    )

    cv2.imwrite(
        str(
            output_dir
            / "00_perspective.png"
        ),
        perspective,
    )

    # -------------------------------------------------------------------------
    # PaddleOCR
    # -------------------------------------------------------------------------

    print()

    print(
        "Loading PaddleOCR..."
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

    # -------------------------------------------------------------------------
    # Stage 1: direct OCR
    # -------------------------------------------------------------------------

    print()

    print(
        "Stage 1: direct OCR"
    )

    direct_rows = run_paddleocr(
        pipeline,
        perspective,
        output_dir,
        "direct",
    )

    direct_stats = summarize(
        direct_rows
    )

    direct_quality = classify_quality(
        direct_stats
    )

    save_variant_outputs(
        "direct",
        perspective,
        direct_rows,
        output_dir,
    )

    print_stats(
        "DIRECT OCR",
        direct_stats,
        direct_quality,
    )

    # -------------------------------------------------------------------------
    # Routing decision
    # -------------------------------------------------------------------------

    recovery_attempted = False

    recovery_rows = []
    recovery_stats = None
    recovery_quality = None
    recovery_image = None

    chosen_rows = direct_rows
    chosen_stats = direct_stats
    chosen_quality = direct_quality

    ocr_route = "direct"

    if direct_quality == "STRONG":
        print()

        print(
            "Direct OCR passed strong quality gate."
        )

        print(
            "Recovery stage skipped."
        )

    else:
        recovery_attempted = True

        print()

        print(
            "Direct OCR did not pass strong quality gate."
        )

        print(
            "Stage 2: recovery fallback activated."
        )

        # ---------------------------------------------------------------------
        # Best practical recovery from Experiment 19
        # ---------------------------------------------------------------------

        recovery_image = (
            build_recovery_variant(
                perspective
            )
        )

        recovery_rows = run_paddleocr(
            pipeline,
            recovery_image,
            output_dir,
            "recovery_4x_unsharp",
        )

        recovery_stats = summarize(
            recovery_rows
        )

        recovery_quality = classify_quality(
            recovery_stats
        )

        save_variant_outputs(
            "recovery_4x_unsharp",
            recovery_image,
            recovery_rows,
            output_dir,
        )

        print_stats(
            "RECOVERY OCR — BICUBIC 4X + UNSHARP",
            recovery_stats,
            recovery_quality,
        )

        # ---------------------------------------------------------------------
        # Compare direct vs fallback
        # ---------------------------------------------------------------------

        if (
            result_score(
                recovery_stats
            )
            >
            result_score(
                direct_stats
            )
        ):
            chosen_rows = (
                recovery_rows
            )

            chosen_stats = (
                recovery_stats
            )

            chosen_quality = (
                recovery_quality
            )

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
            chosen_rows = direct_rows
            chosen_stats = direct_stats
            chosen_quality = direct_quality

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

    # -------------------------------------------------------------------------
    # Final production output
    # -------------------------------------------------------------------------

    final_text_path = (
        output_dir
        / "final_recognized_text.txt"
    )

    final_csv_path = (
        output_dir
        / "final_ocr_results.csv"
    )

    save_text(
        chosen_rows,
        final_text_path,
    )

    save_csv(
        chosen_rows,
        final_csv_path,
    )

    # -------------------------------------------------------------------------
    # Routing summary CSV
    # -------------------------------------------------------------------------

    summary_path = (
        output_dir
        / "routing_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "detector_route",
            "ocr_route",
            "recovery_attempted",
            "direct_quality",
            "direct_regions",
            "direct_words",
            "direct_characters",
            "direct_confidence",
            "recovery_quality",
            "recovery_regions",
            "recovery_words",
            "recovery_characters",
            "recovery_confidence",
            "final_quality",
            "final_regions",
            "final_words",
            "final_characters",
            "final_confidence",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerow(
            {
                "detector_route":
                    detector_route,

                "ocr_route":
                    ocr_route,

                "recovery_attempted":
                    recovery_attempted,

                "direct_quality":
                    direct_quality,

                "direct_regions":
                    direct_stats[
                        "meaningful_regions"
                    ],

                "direct_words":
                    direct_stats[
                        "words"
                    ],

                "direct_characters":
                    direct_stats[
                        "characters"
                    ],

                "direct_confidence":
                    (
                        f"{float(direct_stats['mean_confidence']):.6f}"
                    ),

                "recovery_quality":
                    (
                        recovery_quality
                        if recovery_quality
                        is not None
                        else ""
                    ),

                "recovery_regions":
                    (
                        recovery_stats[
                            "meaningful_regions"
                        ]
                        if recovery_stats
                        is not None
                        else ""
                    ),

                "recovery_words":
                    (
                        recovery_stats[
                            "words"
                        ]
                        if recovery_stats
                        is not None
                        else ""
                    ),

                "recovery_characters":
                    (
                        recovery_stats[
                            "characters"
                        ]
                        if recovery_stats
                        is not None
                        else ""
                    ),

                "recovery_confidence":
                    (
                        f"{float(recovery_stats['mean_confidence']):.6f}"
                        if recovery_stats
                        is not None
                        else ""
                    ),

                "final_quality":
                    chosen_quality,

                "final_regions":
                    chosen_stats[
                        "meaningful_regions"
                    ],

                "final_words":
                    chosen_stats[
                        "words"
                    ],

                "final_characters":
                    chosen_stats[
                        "characters"
                    ],

                "final_confidence":
                    (
                        f"{float(chosen_stats['mean_confidence']):.6f}"
                    ),
            }
        )

    # -------------------------------------------------------------------------
    # Final terminal summary
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 21 FINAL ROUTING DECISION"
    )

    print(
        "=" * 100
    )

    print(
        f"Detector route      : "
        f"{detector_route}"
    )

    print(
        f"OCR route           : "
        f"{ocr_route}"
    )

    print(
        f"Recovery attempted  : "
        f"{recovery_attempted}"
    )

    print(
        f"Final OCR quality   : "
        f"{chosen_quality}"
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
            "Recommendation      : "
            "Recapture document closer to the camera "
            "with better focus and lighting."
        )

    elif (
        chosen_quality
        == "LIMITED"
    ):
        print()

        print(
            "Recommendation      : "
            "Partial OCR recovered; some text may "
            "still be unreadable."
        )

    else:
        print()

        print(
            "Recommendation      : "
            "OCR quality acceptable."
        )

    print()

    print(
        f"Final text:\n"
        f"{final_text_path}"
    )

    print()

    print(
        f"Routing summary:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()