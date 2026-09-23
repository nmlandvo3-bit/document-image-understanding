from __future__ import annotations

# =============================================================================
# IMPORTANT PADDLE CPU WORKAROUND
# =============================================================================
#
# Must be set BEFORE importing PaddleOCR / PaddlePaddle.
#
# PaddlePaddle 3.3.x on Windows CPU can enter a PIR + oneDNN path that causes:
#
# ConvertPirAttribute2RuntimeAttribute not support
# [pir::ArrayAttribute<pir::DoubleAttribute>]
#
# We disable that path for this experiment.
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

EASYOCR_BASELINE_VARIANT = "raw_color_1x"


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 18 - compare PaddleOCR "
            "against the EasyOCR baseline."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Original document photograph.",
    )

    parser.add_argument(
        "--easyocr-ranking",
        required=True,
        help=(
            "Experiment 16 ranking.csv "
            "containing the EasyOCR baseline."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Experiment 18 output directory.",
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
# LOAD EASYOCR BASELINE
# =============================================================================

def load_easyocr_baseline(
    ranking_path: Path,
) -> dict[str, object]:
    if not ranking_path.exists():
        raise FileNotFoundError(
            f"EasyOCR ranking file not found:\n"
            f"{ranking_path}"
        )

    with ranking_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        for row in reader:
            if (
                row.get("variant")
                == EASYOCR_BASELINE_VARIANT
            ):
                return {
                    "variant":
                        row["variant"],

                    "total_regions":
                        int(
                            row[
                                "total_regions"
                            ]
                        ),

                    "meaningful_regions":
                        int(
                            row[
                                "meaningful_regions"
                            ]
                        ),

                    "mean_confidence":
                        float(
                            row[
                                "mean_confidence"
                            ]
                        ),

                    "median_confidence":
                        float(
                            row[
                                "median_confidence"
                            ]
                        ),

                    "word_count":
                        int(
                            row[
                                "word_count"
                            ]
                        ),

                    "character_count":
                        int(
                            row[
                                "character_count"
                            ]
                        ),
                }

    raise RuntimeError(
        f"Could not find "
        f"{EASYOCR_BASELINE_VARIANT} "
        f"in:\n"
        f"{ranking_path}"
    )


# =============================================================================
# PADDLEOCR
# =============================================================================

def run_paddleocr(
    pipeline: PaddleOCR,
    image: np.ndarray,
    output_dir: Path,
) -> list[dict[str, object]]:
    raw_dir = (
        output_dir
        / "paddle_raw"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Calling PaddleOCR predict()..."
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
            / f"result_{result_index:02d}.json"
        )

        # ---------------------------------------------------------------------
        # Save raw Paddle result
        # ---------------------------------------------------------------------

        result.save_to_json(
            save_path=str(
                json_path
            )
        )

        if not json_path.exists():
            print(
                f"Warning: expected Paddle result JSON "
                f"was not created:\n"
                f"{json_path}"
            )

            continue

        # ---------------------------------------------------------------------
        # Load result JSON
        # ---------------------------------------------------------------------

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(
                file
            )

        # PaddleX normally stores actual OCR
        # content under "res".
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

        # ---------------------------------------------------------------------
        # Parse OCR regions
        # ---------------------------------------------------------------------

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

            # -----------------------------------------------------------------
            # Parse bounding rectangle when available
            # -----------------------------------------------------------------

            if index < len(
                boxes
            ):
                box = boxes[index]

                try:
                    box_array = np.asarray(
                        box,
                        dtype=np.float32,
                    )

                    # rec_boxes is commonly:
                    # [x1, y1, x2, y2]
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

                    # Defensive support for
                    # four-corner polygons.
                    elif (
                        box_array.ndim == 2
                        and box_array.shape[1] >= 2
                    ):
                        xs = (
                            box_array[
                                :,
                                0
                            ]
                        )

                        ys = (
                            box_array[
                                :,
                                1
                            ]
                        )

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

                    else:
                        x1 = 0
                        y1 = 0
                        x2 = 0
                        y2 = 0

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
                    x = 0
                    y = 0
                    w = 0
                    h = 0

            meaningful = (
                is_meaningful(
                    text,
                    confidence,
                )
            )

            rows.append(
                {
                    "text":
                        text,

                    "confidence":
                        confidence,

                    "meaningful":
                        meaningful,

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

def summarize_rows(
    rows: list[dict[str, object]],
) -> dict[str, float | int]:
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

    word_count = sum(
        len(
            str(
                row["text"]
            ).split()
        )
        for row in meaningful_rows
    )

    character_count = sum(
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

        "mean_confidence":
            mean_confidence,

        "median_confidence":
            median_confidence,

        "word_count":
            word_count,

        "character_count":
            character_count,
    }


# =============================================================================
# VISUALIZATION
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

        meaningful = bool(
            row["meaningful"]
        )

        if meaningful:
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

        if (
            w <= 0
            or h <= 0
        ):
            continue

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


# =============================================================================
# SAVE CSV
# =============================================================================

def save_results_csv(
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
# SAVE TEXT
# =============================================================================

def save_recognized_text(
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

            if not text:
                continue

            file.write(
                text
            )

            file.write(
                "\n"
            )


# =============================================================================
# ENGINE COMPARISON
# =============================================================================

def determine_winner(
    easyocr: dict[str, object],
    paddle: dict[str, float | int],
) -> str:
    easy_chars = int(
        easyocr[
            "character_count"
        ]
    )

    paddle_chars = int(
        paddle[
            "character_count"
        ]
    )

    if (
        paddle_chars
        > easy_chars
    ):
        return "PADDLEOCR"

    if (
        easy_chars
        > paddle_chars
    ):
        return "EASYOCR"

    easy_regions = int(
        easyocr[
            "meaningful_regions"
        ]
    )

    paddle_regions = int(
        paddle[
            "meaningful_regions"
        ]
    )

    if (
        paddle_regions
        > easy_regions
    ):
        return "PADDLEOCR"

    if (
        easy_regions
        > paddle_regions
    ):
        return "EASYOCR"

    easy_confidence = float(
        easyocr[
            "mean_confidence"
        ]
    )

    paddle_confidence = float(
        paddle[
            "mean_confidence"
        ]
    )

    if (
        paddle_confidence
        > easy_confidence
    ):
        return "PADDLEOCR"

    if (
        easy_confidence
        > paddle_confidence
    ):
        return "EASYOCR"

    return "TIE"


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    input_path = Path(
        args.input
    ).resolve()

    easyocr_ranking_path = Path(
        args.easyocr_ranking
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    # -------------------------------------------------------------------------
    # Validate input
    # -------------------------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image not found:\n"
            f"{input_path}"
        )

    if not easyocr_ranking_path.exists():
        raise FileNotFoundError(
            f"EasyOCR ranking not found:\n"
            f"{easyocr_ranking_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Read image
    # -------------------------------------------------------------------------

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
        "EXPERIMENT 18 — "
        "PADDLEOCR VS EASYOCR"
    )

    print(
        "=" * 100
    )

    print(
        f"Input: "
        f"{input_path}"
    )

    # -------------------------------------------------------------------------
    # Frozen page detector
    # -------------------------------------------------------------------------

    (
        document_points,
        _edges,
        route,
    ) = detect_document(
        image
    )

    perspective = (
        four_point_transform(
            image,
            document_points,
        )
    )

    print(
        f"Detector route : "
        f"{route}"
    )

    print(
        f"Document size  : "
        f"{perspective.shape[1]} x "
        f"{perspective.shape[0]}"
    )

    cv2.imwrite(
        str(
            output_dir
            / "01_perspective.png"
        ),
        perspective,
    )

    # -------------------------------------------------------------------------
    # EasyOCR baseline
    # -------------------------------------------------------------------------

    easyocr_stats = (
        load_easyocr_baseline(
            easyocr_ranking_path
        )
    )

    # -------------------------------------------------------------------------
    # PaddleOCR
    # -------------------------------------------------------------------------

    print()

    print(
        "Loading PaddleOCR..."
    )

    print(
        "CPU oneDNN / MKL-DNN disabled "
        "for PaddlePaddle compatibility."
    )

    pipeline = PaddleOCR(
        lang="en",

        use_doc_orientation_classify=False,

        use_doc_unwarping=False,

        use_textline_orientation=False,

        device="cpu",

        # Critical workaround for the
        # PaddlePaddle 3.3.x Windows CPU
        # oneDNN/PIR inference issue.
        enable_mkldnn=False,
    )

    print(
        "PaddleOCR loaded."
    )

    print(
        "Running PaddleOCR..."
    )

    paddle_rows = run_paddleocr(
        pipeline,
        perspective,
        output_dir,
    )

    paddle_stats = summarize_rows(
        paddle_rows
    )

    # -------------------------------------------------------------------------
    # Save PaddleOCR outputs
    # -------------------------------------------------------------------------

    preview = draw_results(
        perspective,
        paddle_rows,
    )

    cv2.imwrite(
        str(
            output_dir
            / "02_paddleocr_regions.png"
        ),
        preview,
    )

    paddle_csv_path = (
        output_dir
        / "paddleocr_results.csv"
    )

    save_results_csv(
        paddle_rows,
        paddle_csv_path,
    )

    paddle_text_path = (
        output_dir
        / "paddleocr_recognized_text.txt"
    )

    save_recognized_text(
        paddle_rows,
        paddle_text_path,
    )

    # -------------------------------------------------------------------------
    # Print PaddleOCR text
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "PADDLEOCR RECOGNIZED TEXT"
    )

    print(
        "=" * 100
    )

    if not paddle_rows:
        print(
            "No PaddleOCR regions returned."
        )

    for index, row in enumerate(
        paddle_rows,
        start=1,
    ):
        if bool(
            row["meaningful"]
        ):
            status = "ACCEPT"

        else:
            status = "REJECT"

        print(
            f"P{index:03d} "
            f"| {status:6s} "
            f"| conf="
            f"{float(row['confidence']):.4f} "
            f"| {row['text']}"
        )

    # -------------------------------------------------------------------------
    # Engine comparison
    # -------------------------------------------------------------------------

    winner = determine_winner(
        easyocr_stats,
        paddle_stats,
    )

    print()

    print(
        "=" * 100
    )

    print(
        "ENGINE COMPARISON"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "EASYOCR — raw_color_1x"
    )

    print(
        f"Total regions       : "
        f"{easyocr_stats['total_regions']}"
    )

    print(
        f"Meaningful regions  : "
        f"{easyocr_stats['meaningful_regions']}"
    )

    print(
        f"Mean confidence     : "
        f"{float(easyocr_stats['mean_confidence']):.4f}"
    )

    print(
        f"Median confidence   : "
        f"{float(easyocr_stats['median_confidence']):.4f}"
    )

    print(
        f"Words               : "
        f"{easyocr_stats['word_count']}"
    )

    print(
        f"Characters          : "
        f"{easyocr_stats['character_count']}"
    )

    print()

    print(
        "PADDLEOCR"
    )

    print(
        f"Total regions       : "
        f"{paddle_stats['total_regions']}"
    )

    print(
        f"Meaningful regions  : "
        f"{paddle_stats['meaningful_regions']}"
    )

    print(
        f"Mean confidence     : "
        f"{float(paddle_stats['mean_confidence']):.4f}"
    )

    print(
        f"Median confidence   : "
        f"{float(paddle_stats['median_confidence']):.4f}"
    )

    print(
        f"Words               : "
        f"{paddle_stats['word_count']}"
    )

    print(
        f"Characters          : "
        f"{paddle_stats['character_count']}"
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 18 SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Detector route      : "
        f"{route}"
    )

    print(
        f"Recommended engine  : "
        f"{winner}"
    )

    print()

    print(
        f"PaddleOCR CSV:\n"
        f"{paddle_csv_path}"
    )

    print()

    print(
        f"PaddleOCR text:\n"
        f"{paddle_text_path}"
    )

    print()

    print(
        f"Outputs:\n"
        f"{output_dir}"
    )

    print()

    print(
        "Most important files:"
    )

    print(
        "  01_perspective.png"
    )

    print(
        "  02_paddleocr_regions.png"
    )

    print(
        "  paddleocr_results.csv"
    )

    print(
        "  paddleocr_recognized_text.txt"
    )


if __name__ == "__main__":
    main()