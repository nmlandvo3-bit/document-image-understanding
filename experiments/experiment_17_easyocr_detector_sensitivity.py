from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import easyocr
import numpy as np

from document_scanner import (
    detect_document,
    four_point_transform,
)


OCR_LANGUAGES = ["en"]

MIN_CONFIDENCE = 0.25
MIN_ALPHANUMERIC_CHARACTERS = 2


# =============================================================================
# DETECTOR PROFILES
# =============================================================================

PROFILES = [
    {
        "name": "default",
        "text_threshold": 0.70,
        "low_text": 0.40,
        "link_threshold": 0.40,
        "mag_ratio": 1.0,
        "canvas_size": 2560,
    },
    {
        "name": "sensitive_1",
        "text_threshold": 0.60,
        "low_text": 0.30,
        "link_threshold": 0.30,
        "mag_ratio": 1.5,
        "canvas_size": 3200,
    },
    {
        "name": "sensitive_2",
        "text_threshold": 0.50,
        "low_text": 0.25,
        "link_threshold": 0.25,
        "mag_ratio": 2.0,
        "canvas_size": 3840,
    },
    {
        "name": "very_sensitive_1",
        "text_threshold": 0.40,
        "low_text": 0.20,
        "link_threshold": 0.20,
        "mag_ratio": 2.5,
        "canvas_size": 4096,
    },
    {
        "name": "very_sensitive_2",
        "text_threshold": 0.30,
        "low_text": 0.15,
        "link_threshold": 0.20,
        "mag_ratio": 3.0,
        "canvas_size": 4096,
    },
    {
        "name": "extreme",
        "text_threshold": 0.20,
        "low_text": 0.10,
        "link_threshold": 0.15,
        "mag_ratio": 3.5,
        "canvas_size": 4096,
    },
]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 17 - EasyOCR detector "
            "sensitivity study for small text."
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
        help="Experiment 17 output directory.",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use EasyOCR GPU mode.",
    )

    return parser.parse_args()


# =============================================================================
# TEXT VALIDATION
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
# OCR
# =============================================================================

def run_profile(
    reader: easyocr.Reader,
    image: np.ndarray,
    profile: dict[str, object],
) -> list[dict[str, object]]:
    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,

        text_threshold=float(
            profile["text_threshold"]
        ),

        low_text=float(
            profile["low_text"]
        ),

        link_threshold=float(
            profile["link_threshold"]
        ),

        mag_ratio=float(
            profile["mag_ratio"]
        ),

        canvas_size=int(
            profile["canvas_size"]
        ),

        min_size=3,
    )

    rows = []

    for result in results:
        box = result[0]

        text = str(
            result[1]
        ).strip()

        confidence = float(
            result[2]
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

        meaningful = is_meaningful(
            text,
            confidence,
        )

        rows.append(
            {
                "text": text,
                "confidence": confidence,
                "meaningful": meaningful,
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
            }
        )

    rows.sort(
        key=lambda row: (
            float(row["y"]),
            float(row["x"]),
        )
    )

    return rows


# =============================================================================
# SUMMARY
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
            np.mean(confidences)
        )

        median_confidence = float(
            np.median(confidences)
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
            len(meaningful_rows),

        "mean_confidence":
            mean_confidence,

        "median_confidence":
            median_confidence,

        "words":
            words,

        "characters":
            characters,
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
# SAVE
# =============================================================================

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
            f"Could not load:\n"
            f"{input_path}"
        )

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 17 — "
        "EASYOCR SMALL-TEXT DETECTOR SENSITIVITY"
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
    # Frozen detector / perspective correction
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
        f"Detector route: "
        f"{route}"
    )

    print(
        f"Document size : "
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
    # OCR
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

    all_profiles = []

    # -------------------------------------------------------------------------
    # Profiles
    # -------------------------------------------------------------------------

    for profile in PROFILES:
        name = str(
            profile["name"]
        )

        print()

        print(
            "-" * 100
        )

        print(
            f"Running profile: {name}"
        )

        print(
            f"text_threshold = "
            f"{profile['text_threshold']}"
        )

        print(
            f"low_text       = "
            f"{profile['low_text']}"
        )

        print(
            f"link_threshold = "
            f"{profile['link_threshold']}"
        )

        print(
            f"mag_ratio      = "
            f"{profile['mag_ratio']}"
        )

        print(
            f"canvas_size    = "
            f"{profile['canvas_size']}"
        )

        profile_dir = (
            output_dir
            / name
        )

        profile_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows = run_profile(
            reader,
            perspective,
            profile,
        )

        stats = summarize(
            rows
        )

        preview = draw_results(
            perspective,
            rows,
        )

        cv2.imwrite(
            str(
                profile_dir
                / "ocr_regions.png"
            ),
            preview,
        )

        save_csv(
            rows,
            profile_dir
            / "ocr_results.csv",
        )

        save_text(
            rows,
            profile_dir
            / "recognized_text.txt",
        )

        print()

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
            f"{stats['words']}"
        )

        print(
            f"Characters         : "
            f"{stats['characters']}"
        )

        all_profiles.append(
            {
                "name":
                    name,

                "text_threshold":
                    profile[
                        "text_threshold"
                    ],

                "low_text":
                    profile[
                        "low_text"
                    ],

                "link_threshold":
                    profile[
                        "link_threshold"
                    ],

                "mag_ratio":
                    profile[
                        "mag_ratio"
                    ],

                "canvas_size":
                    profile[
                        "canvas_size"
                    ],

                **stats,
            }
        )

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    all_profiles.sort(
        key=lambda row: (
            int(
                row[
                    "characters"
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

    fieldnames = [
        "rank",
        "name",
        "text_threshold",
        "low_text",
        "link_threshold",
        "mag_ratio",
        "canvas_size",
        "total_regions",
        "meaningful_regions",
        "mean_confidence",
        "median_confidence",
        "words",
        "characters",
    ]

    with ranking_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for rank, row in enumerate(
            all_profiles,
            start=1,
        ):
            writer.writerow(
                {
                    "rank":
                        rank,

                    "name":
                        row["name"],

                    "text_threshold":
                        row[
                            "text_threshold"
                        ],

                    "low_text":
                        row[
                            "low_text"
                        ],

                    "link_threshold":
                        row[
                            "link_threshold"
                        ],

                    "mag_ratio":
                        row[
                            "mag_ratio"
                        ],

                    "canvas_size":
                        row[
                            "canvas_size"
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
                        (
                            f"{float(row['mean_confidence']):.6f}"
                        ),

                    "median_confidence":
                        (
                            f"{float(row['median_confidence']):.6f}"
                        ),

                    "words":
                        row["words"],

                    "characters":
                        row[
                            "characters"
                        ],
                }
            )

    # -------------------------------------------------------------------------
    # Print ranking
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 17 RANKING"
    )

    print(
        "=" * 100
    )

    for rank, row in enumerate(
        all_profiles,
        start=1,
    ):
        print(
            f"{rank:02d}. "
            f"{str(row['name']):20s} "
            f"| total="
            f"{int(row['total_regions']):3d} "
            f"| meaningful="
            f"{int(row['meaningful_regions']):3d} "
            f"| words="
            f"{int(row['words']):4d} "
            f"| chars="
            f"{int(row['characters']):5d} "
            f"| conf="
            f"{float(row['mean_confidence']):.4f}"
        )

    # -------------------------------------------------------------------------
    # Winner
    # -------------------------------------------------------------------------

    if all_profiles:
        winner = (
            all_profiles[0]
        )

        print()

        print(
            "=" * 100
        )

        print(
            "EXPERIMENT 17 SUMMARY"
        )

        print(
            "=" * 100
        )

        print(
            f"Winning profile     : "
            f"{winner['name']}"
        )

        print(
            f"Meaningful regions  : "
            f"{winner['meaningful_regions']}"
        )

        print(
            f"Recognized words    : "
            f"{winner['words']}"
        )

        print(
            f"Recognized chars    : "
            f"{winner['characters']}"
        )

        print(
            f"Mean confidence     : "
            f"{float(winner['mean_confidence']):.4f}"
        )

        print()

        print(
            f"Ranking CSV:\n"
            f"{ranking_path}"
        )


if __name__ == "__main__":
    main()