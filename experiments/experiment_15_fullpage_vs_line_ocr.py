from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import easyocr
import numpy as np

from document_scanner import (
    detect_document,
    enhance_document,
    four_point_transform,
)


OCR_LANGUAGES = ["en"]

MIN_CONFIDENCE = 0.25
MIN_ALPHANUMERIC_CHARACTERS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 15 - compare full-page OCR "
            "against Experiment 13 line-crop OCR."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Original document photograph.",
    )

    parser.add_argument(
        "--experiment14-csv",
        required=True,
        help="Experiment 14 OCR results CSV.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use EasyOCR GPU mode.",
    )

    return parser.parse_args()


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


def load_line_ocr_results(
    csv_path: Path,
) -> list[dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Experiment 14 CSV not found:\n"
            f"{csv_path}"
        )

    rows = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        for row in reader:
            text = str(
                row.get(
                    "selected_text",
                    "",
                )
            ).strip()

            confidence = float(
                row.get(
                    "selected_confidence",
                    0.0,
                )
            )

            meaningful_value = str(
                row.get(
                    "meaningful",
                    "",
                )
            ).lower()

            meaningful = (
                meaningful_value
                in {
                    "true",
                    "1",
                    "yes",
                }
            )

            rows.append(
                {
                    "line":
                        int(
                            row["line"]
                        ),

                    "text":
                        text,

                    "confidence":
                        confidence,

                    "meaningful":
                        meaningful,
                }
            )

    return rows


def prepare_full_page(
    corrected: np.ndarray,
) -> np.ndarray:
    height, width = (
        corrected.shape[:2]
    )

    minimum_height = 1200

    if height < minimum_height:
        scale = (
            minimum_height
            / float(height)
        )

        new_width = int(
            round(
                width * scale
            )
        )

        corrected = cv2.resize(
            corrected,
            (
                new_width,
                minimum_height,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

    corrected = cv2.copyMakeBorder(
        corrected,
        30,
        30,
        30,
        30,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    return corrected


def run_fullpage_ocr(
    reader: easyocr.Reader,
    image: np.ndarray,
) -> list[
    dict[str, object]
]:
    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
    )

    parsed = []

    for item in results:
        box = item[0]

        text = str(
            item[1]
        ).strip()

        confidence = float(
            item[2]
        )

        if not text:
            continue

        xs = [
            float(point[0])
            for point in box
        ]

        ys = [
            float(point[1])
            for point in box
        ]

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

        meaningful = (
            is_meaningful(
                text,
                confidence,
            )
        )

        parsed.append(
            {
                "text":
                    text,

                "confidence":
                    confidence,

                "meaningful":
                    meaningful,

                "x":
                    x1,

                "y":
                    y1,

                "w":
                    x2 - x1,

                "h":
                    y2 - y1,
            }
        )

    parsed.sort(
        key=lambda row: (
            float(row["y"]),
            float(row["x"]),
        )
    )

    return parsed


def draw_fullpage_results(
    image: np.ndarray,
    rows: list[
        dict[str, object]
    ],
) -> np.ndarray:
    if image.ndim == 2:
        preview = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    else:
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

        cv2.rectangle(
            preview,
            (x, y),
            (
                x + w,
                y + h,
            ),
            color,
            2,
        )

        cv2.putText(
            preview,
            f"F{index}",
            (
                x,
                max(
                    18,
                    y - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return preview


def save_fullpage_csv(
    rows: list[
        dict[str, object]
    ],
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
                        row["meaningful"],

                    "x":
                        f"{float(row['x']):.2f}",

                    "y":
                        f"{float(row['y']):.2f}",

                    "w":
                        f"{float(row['w']):.2f}",

                    "h":
                        f"{float(row['h']):.2f}",
                }
            )


def save_text(
    rows: list[
        dict[str, object]
    ],
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

            text = str(
                row["text"]
            ).strip()

            if not text:
                continue

            file.write(
                text
            )

            file.write(
                "\n"
            )


def summarize_strategy(
    name: str,
    rows: list[
        dict[str, object]
    ],
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

    character_count = sum(
        alphanumeric_count(
            str(
                row["text"]
            )
        )
        for row in meaningful_rows
    )

    word_count = sum(
        len(
            str(
                row["text"]
            ).split()
        )
        for row in meaningful_rows
    )

    print(
        name
    )

    print(
        f"Total regions       : "
        f"{len(rows)}"
    )

    print(
        f"Meaningful regions  : "
        f"{len(meaningful_rows)}"
    )

    print(
        f"Mean confidence     : "
        f"{mean_confidence:.4f}"
    )

    print(
        f"Median confidence   : "
        f"{median_confidence:.4f}"
    )

    print(
        f"Recognized words    : "
        f"{word_count}"
    )

    print(
        f"Alphanumeric chars  : "
        f"{character_count}"
    )

    return {
        "total":
            len(rows),

        "meaningful":
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


def main() -> None:
    args = parse_args()

    input_path = Path(
        args.input
    ).resolve()

    experiment14_csv = Path(
        args.experiment14_csv
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

    image = cv2.imread(
        str(
            input_path
        )
    )

    if image is None:
        raise RuntimeError(
            f"Could not load:\n"
            f"{input_path}"
        )

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 15 — "
        "FULL-PAGE OCR VS LINE-CROP OCR"
    )

    print(
        "=" * 100
    )

    print(
        f"Input         : "
        f"{input_path}"
    )

    print(
        f"GPU requested : "
        f"{args.gpu}"
    )

    # -------------------------------------------------------------------------
    # Frozen production pipeline
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

    enhanced = (
        enhance_document(
            perspective
        )
    )

    corrected = enhanced[
        "contrast"
    ]

    prepared_page = (
        prepare_full_page(
            corrected
        )
    )

    cv2.imwrite(
        str(
            output_dir
            / "01_fullpage_ocr_input.png"
        ),
        prepared_page,
    )

    # -------------------------------------------------------------------------
    # Load line-crop OCR results
    # -------------------------------------------------------------------------

    line_rows = (
        load_line_ocr_results(
            experiment14_csv
        )
    )

    # -------------------------------------------------------------------------
    # EasyOCR full-page
    # -------------------------------------------------------------------------

    print()

    print(
        "Loading EasyOCR..."
    )

    reader = easyocr.Reader(
        OCR_LANGUAGES,
        gpu=args.gpu,
    )

    print(
        "Running full-page OCR..."
    )

    fullpage_rows = (
        run_fullpage_ocr(
            reader,
            prepared_page,
        )
    )

    # -------------------------------------------------------------------------
    # Save full-page outputs
    # -------------------------------------------------------------------------

    preview = (
        draw_fullpage_results(
            prepared_page,
            fullpage_rows,
        )
    )

    cv2.imwrite(
        str(
            output_dir
            / "02_fullpage_ocr_regions.png"
        ),
        preview,
    )

    fullpage_csv = (
        output_dir
        / "fullpage_ocr_results.csv"
    )

    save_fullpage_csv(
        fullpage_rows,
        fullpage_csv,
    )

    fullpage_text_path = (
        output_dir
        / "fullpage_recognized_text.txt"
    )

    save_text(
        fullpage_rows,
        fullpage_text_path,
    )

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "STRATEGY COMPARISON"
    )

    print(
        "=" * 100
    )

    line_stats = (
        summarize_strategy(
            "A. LINE-CROP OCR",
            line_rows,
        )
    )

    print()

    fullpage_stats = (
        summarize_strategy(
            "B. FULL-PAGE OCR",
            fullpage_rows,
        )
    )

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 15 SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Detector route      : "
        f"{route}"
    )

    print()

    line_chars = int(
        line_stats[
            "character_count"
        ]
    )

    fullpage_chars = int(
        fullpage_stats[
            "character_count"
        ]
    )

    line_mean = float(
        line_stats[
            "mean_confidence"
        ]
    )

    fullpage_mean = float(
        fullpage_stats[
            "mean_confidence"
        ]
    )

    if (
        fullpage_chars
        > line_chars * 1.5
    ):
        recommendation = (
            "FULL_PAGE"
        )

    elif (
        line_chars
        > fullpage_chars * 1.5
    ):
        recommendation = (
            "LINE_CROPS"
        )

    elif (
        fullpage_mean
        > line_mean
    ):
        recommendation = (
            "FULL_PAGE"
        )

    else:
        recommendation = (
            "LINE_CROPS"
        )

    print(
        f"Recommended strategy: "
        f"{recommendation}"
    )

    print()

    print(
        "Line-crop characters : "
        f"{line_chars}"
    )

    print(
        "Full-page characters : "
        f"{fullpage_chars}"
    )

    print()

    print(
        f"Full-page text:\n"
        f"{fullpage_text_path}"
    )

    print(
        f"Full-page CSV:\n"
        f"{fullpage_csv}"
    )

    print()

    print(
        "Most important files:"
    )

    print(
        "  01_fullpage_ocr_input.png"
    )

    print(
        "  02_fullpage_ocr_regions.png"
    )

    print(
        "  fullpage_recognized_text.txt"
    )


if __name__ == "__main__":
    main()