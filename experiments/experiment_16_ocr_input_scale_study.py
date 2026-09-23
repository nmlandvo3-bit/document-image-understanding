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

SCALE_FACTORS = [
    1,
    2,
    4,
]

MIN_CONFIDENCE = 0.25
MIN_ALPHANUMERIC_CHARACTERS = 2

PAGE_BORDER = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 16 - compare OCR input "
            "representations and scale factors."
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
        help="Experiment 16 output directory.",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU mode for EasyOCR.",
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


def resize_image(
    image: np.ndarray,
    scale: int,
) -> np.ndarray:
    if scale == 1:
        result = image.copy()

    else:
        height, width = (
            image.shape[:2]
        )

        result = cv2.resize(
            image,
            (
                width * scale,
                height * scale,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

    if result.ndim == 2:
        border_value = 255
    else:
        border_value = (
            255,
            255,
            255,
        )

    result = cv2.copyMakeBorder(
        result,
        PAGE_BORDER,
        PAGE_BORDER,
        PAGE_BORDER,
        PAGE_BORDER,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )

    return result


def run_ocr(
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

    rows = []

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

        rows.append(
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

    rows.sort(
        key=lambda item: (
            float(item["y"]),
            float(item["x"]),
        )
    )

    return rows


def summarize_rows(
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


def draw_ocr_regions(
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
            f"R{index}",
            (
                x,
                max(
                    18,
                    y - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return preview


def save_variant_csv(
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
                        f"{float(row['confidence']):.6f}",

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


def save_variant_text(
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

            file.write(
                str(
                    row["text"]
                ).strip()
            )

            file.write(
                "\n"
            )


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
        "EXPERIMENT 16 — "
        "OCR INPUT REPRESENTATION + SCALE STUDY"
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
    # Frozen production detector + perspective
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

    # -------------------------------------------------------------------------
    # OCR representations
    # -------------------------------------------------------------------------

    raw_color = (
        perspective
    )

    raw_gray = cv2.cvtColor(
        perspective,
        cv2.COLOR_BGR2GRAY,
    )

    illumination_corrected = (
        enhanced[
            "illumination_corrected"
        ]
    )

    contrast = (
        enhanced[
            "contrast"
        ]
    )

    binary_scan = (
        enhanced[
            "scan"
        ]
    )

    representations = {
        "raw_color":
            raw_color,

        "raw_gray":
            raw_gray,

        "illumination_corrected":
            illumination_corrected,

        "contrast":
            contrast,

        "binary_scan":
            binary_scan,
    }

    # -------------------------------------------------------------------------
    # Load OCR
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
        "EasyOCR loaded."
    )

    all_results = []

    # -------------------------------------------------------------------------
    # Run every representation × scale
    # -------------------------------------------------------------------------

    for representation_name, source in (
        representations.items()
    ):
        for scale in SCALE_FACTORS:
            variant_name = (
                f"{representation_name}_"
                f"{scale}x"
            )

            print()

            print(
                "-" * 100
            )

            print(
                f"Running: "
                f"{variant_name}"
            )

            prepared = resize_image(
                source,
                scale,
            )

            variant_dir = (
                output_dir
                / variant_name
            )

            variant_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            input_output_path = (
                variant_dir
                / "ocr_input.png"
            )

            cv2.imwrite(
                str(
                    input_output_path
                ),
                prepared,
            )

            rows = run_ocr(
                reader,
                prepared,
            )

            stats = summarize_rows(
                rows
            )

            preview = (
                draw_ocr_regions(
                    prepared,
                    rows,
                )
            )

            cv2.imwrite(
                str(
                    variant_dir
                    / "ocr_regions.png"
                ),
                preview,
            )

            save_variant_csv(
                rows,
                variant_dir
                / "ocr_results.csv",
            )

            save_variant_text(
                rows,
                variant_dir
                / "recognized_text.txt",
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
                f"Mean confidence    : "
                f"{stats['mean_confidence']:.4f}"
            )

            print(
                f"Median confidence  : "
                f"{stats['median_confidence']:.4f}"
            )

            print(
                f"Words              : "
                f"{stats['word_count']}"
            )

            print(
                f"Characters         : "
                f"{stats['character_count']}"
            )

            all_results.append(
                {
                    "variant":
                        variant_name,

                    "representation":
                        representation_name,

                    "scale":
                        scale,

                    **stats,
                }
            )

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    all_results.sort(
        key=lambda row: (
            int(
                row[
                    "character_count"
                ]
            ),
            int(
                row[
                    "meaningful_regions"
                ]
            ),
            float(
                row[
                    "mean_confidence"
                ]
            ),
        ),
        reverse=True,
    )

    ranking_path = (
        output_dir
        / "ranking.csv"
    )

    ranking_fields = [
        "rank",
        "variant",
        "representation",
        "scale",
        "total_regions",
        "meaningful_regions",
        "mean_confidence",
        "median_confidence",
        "word_count",
        "character_count",
    ]

    with ranking_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=ranking_fields,
        )

        writer.writeheader()

        for rank, row in enumerate(
            all_results,
            start=1,
        ):
            writer.writerow(
                {
                    "rank":
                        rank,

                    "variant":
                        row[
                            "variant"
                        ],

                    "representation":
                        row[
                            "representation"
                        ],

                    "scale":
                        row[
                            "scale"
                        ],

                    "total_regions":
                        row[
                            "total_regions"
                        ],

                    "meaningful_regions":
                        row[
                            "meaningful_regions"
                        ],

                    "mean_confidence":
                        f"{float(row['mean_confidence']):.6f}",

                    "median_confidence":
                        f"{float(row['median_confidence']):.6f}",

                    "word_count":
                        row[
                            "word_count"
                        ],

                    "character_count":
                        row[
                            "character_count"
                        ],
                }
            )

    # -------------------------------------------------------------------------
    # Console ranking
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 16 RANKING"
    )

    print(
        "=" * 100
    )

    for rank, row in enumerate(
        all_results,
        start=1,
    ):
        print(
            f"{rank:02d}. "
            f"{str(row['variant']):30s} "
            f"| regions="
            f"{int(row['meaningful_regions']):3d} "
            f"| words="
            f"{int(row['word_count']):4d} "
            f"| chars="
            f"{int(row['character_count']):5d} "
            f"| mean_conf="
            f"{float(row['mean_confidence']):.4f}"
        )

    # -------------------------------------------------------------------------
    # Winner
    # -------------------------------------------------------------------------

    if all_results:
        winner = (
            all_results[0]
        )

        print()

        print(
            "=" * 100
        )

        print(
            "EXPERIMENT 16 SUMMARY"
        )

        print(
            "=" * 100
        )

        print(
            f"Detector route       : "
            f"{route}"
        )

        print(
            f"Document size        : "
            f"{perspective.shape[1]} x "
            f"{perspective.shape[0]}"
        )

        print()

        print(
            f"Winning variant      : "
            f"{winner['variant']}"
        )

        print(
            f"Meaningful regions   : "
            f"{winner['meaningful_regions']}"
        )

        print(
            f"Recognized words     : "
            f"{winner['word_count']}"
        )

        print(
            f"Recognized chars     : "
            f"{winner['character_count']}"
        )

        print(
            f"Mean confidence      : "
            f"{float(winner['mean_confidence']):.4f}"
        )

        print()

        print(
            f"Ranking CSV:\n"
            f"{ranking_path}"
        )


if __name__ == "__main__":
    main()