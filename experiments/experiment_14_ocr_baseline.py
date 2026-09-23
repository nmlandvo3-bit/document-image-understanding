from __future__ import annotations

import argparse
import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import easyocr
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

OCR_LANGUAGES = [
    "en",
]

MIN_CONFIDENCE = 0.25
MIN_ALPHANUMERIC_CHARACTERS = 2

UPSCALE_FACTOR = 3


CLASSROOM_GROUND_TRUTH = [
    "Schedule",
    "Week",
    "Date",
    "Topic",
    "1",
    "Sep 9",
    "Course introduction",
    "2",
    "Sep 16",
    "Generative Modeling",
    "3",
    "Sep 23",
    "Deep Learning",
    "4",
    "Sep 30",
    "Variational Autoencoders",
    "5",
    "Oct 7",
    "Generative Adversarial Networks",
    "6",
    "Oct 14",
    "Diffusion Models",
    "7",
    "Oct 21",
    "Attention Mechanism",
    "8",
    "Oct 28",
    "Transformers",
    "9",
    "Nov 4",
    "Mid-term exam",
]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 14 - OCR baseline on "
            "Experiment 13 text-line crops."
        )
    )

    parser.add_argument(
        "--experiment13-dir",
        required=True,
        help=(
            "Experiment 13 output directory "
            "containing ocr_lines_gray and "
            "ocr_lines_binary."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for OCR results.",
    )

    parser.add_argument(
        "--classroom",
        action="store_true",
        help=(
            "Enable classroom ground-truth "
            "similarity evaluation."
        ),
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use CUDA if EasyOCR supports it.",
    )

    return parser.parse_args()


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(
    text: str,
) -> str:
    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def alphanumeric_count(
    text: str,
) -> int:
    return sum(
        character.isalnum()
        for character in text
    )


# =============================================================================
# IMAGE PREPARATION
# =============================================================================

def prepare_crop(
    image: np.ndarray,
) -> np.ndarray:
    if image is None:
        raise RuntimeError(
            "Invalid OCR crop."
        )

    height, width = (
        image.shape[:2]
    )

    enlarged = cv2.resize(
        image,
        (
            width * UPSCALE_FACTOR,
            height * UPSCALE_FACTOR,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    border = 20

    if enlarged.ndim == 2:
        padded = cv2.copyMakeBorder(
            enlarged,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    else:
        padded = cv2.copyMakeBorder(
            enlarged,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=(
                255,
                255,
                255,
            ),
        )

    return padded


# =============================================================================
# OCR
# =============================================================================

def run_ocr(
    reader: easyocr.Reader,
    image: np.ndarray,
) -> tuple[
    str,
    float,
    int,
]:
    prepared = prepare_crop(
        image
    )

    results = reader.readtext(
        prepared,
        detail=1,
        paragraph=False,
    )

    if not results:
        return (
            "",
            0.0,
            0,
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

        left_x = min(
            float(
                point[0]
            )
            for point in box
        )

        parsed.append(
            (
                left_x,
                text,
                confidence,
            )
        )

    if not parsed:
        return (
            "",
            0.0,
            0,
        )

    parsed.sort(
        key=lambda item: item[0]
    )

    text_parts = [
        item[1]
        for item in parsed
    ]

    confidences = [
        item[2]
        for item in parsed
    ]

    combined_text = " ".join(
        text_parts
    ).strip()

    mean_confidence = float(
        np.mean(
            confidences
        )
    )

    return (
        combined_text,
        mean_confidence,
        len(parsed),
    )


# =============================================================================
# TEXT QUALITY
# =============================================================================

def is_meaningful_text(
    text: str,
    confidence: float,
) -> bool:
    if confidence < MIN_CONFIDENCE:
        return False

    if (
        alphanumeric_count(
            text
        )
        < MIN_ALPHANUMERIC_CHARACTERS
    ):
        return False

    return True


# =============================================================================
# GROUND-TRUTH MATCHING
# =============================================================================

def text_similarity(
    a: str,
    b: str,
) -> float:
    normalized_a = normalize_text(
        a
    )

    normalized_b = normalize_text(
        b
    )

    if (
        not normalized_a
        or not normalized_b
    ):
        return 0.0

    return SequenceMatcher(
        None,
        normalized_a,
        normalized_b,
    ).ratio()


def find_best_ground_truth_match(
    recognized_text: str,
) -> tuple[
    str,
    float,
]:
    best_text = ""
    best_score = 0.0

    for target in (
        CLASSROOM_GROUND_TRUTH
    ):
        score = text_similarity(
            recognized_text,
            target,
        )

        if score > best_score:
            best_score = score
            best_text = target

    return (
        best_text,
        best_score,
    )


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def get_line_files(
    directory: Path,
) -> dict[
    str,
    Path,
]:
    files = {}

    for path in sorted(
        directory.glob(
            "line_*.png"
        )
    ):
        files[
            path.stem
        ] = path

    return files


# =============================================================================
# VISUAL SUMMARY
# =============================================================================

def create_text_report_image(
    rows: list[
        dict[str, object]
    ],
) -> np.ndarray:
    width = 1600

    row_height = 52

    height = max(
        300,
        100
        + len(rows)
        * row_height,
    )

    canvas = np.full(
        (
            height,
            width,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    cv2.putText(
        canvas,
        "Experiment 14 OCR Results",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    y = 90

    for row in rows:
        line_number = int(
            row["line"]
        )

        source = str(
            row["selected_source"]
        )

        confidence = float(
            row[
                "selected_confidence"
            ]
        )

        text = str(
            row["selected_text"]
        )

        display_text = (
            f"L{line_number:02d} "
            f"[{source}] "
            f"conf={confidence:.3f} "
            f"{text}"
        )

        if len(
            display_text
        ) > 120:
            display_text = (
                display_text[:117]
                + "..."
            )

        cv2.putText(
            canvas,
            display_text,
            (
                30,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        y += row_height

    return canvas


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    experiment13_dir = Path(
        args.experiment13_dir
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    gray_dir = (
        experiment13_dir
        / "ocr_lines_gray"
    )

    binary_dir = (
        experiment13_dir
        / "ocr_lines_binary"
    )

    if not gray_dir.exists():
        raise FileNotFoundError(
            f"Gray line directory not found:\n"
            f"{gray_dir}"
        )

    if not binary_dir.exists():
        raise FileNotFoundError(
            f"Binary line directory not found:\n"
            f"{binary_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gray_files = get_line_files(
        gray_dir
    )

    binary_files = get_line_files(
        binary_dir
    )

    line_names = sorted(
        set(
            gray_files.keys()
        )
        | set(
            binary_files.keys()
        )
    )

    if not line_names:
        raise RuntimeError(
            "No Experiment 13 OCR crops found."
        )

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 14 — "
        "OCR BASELINE"
    )

    print(
        "=" * 100
    )

    print(
        f"Experiment 13 dir : "
        f"{experiment13_dir}"
    )

    print(
        f"Lines found       : "
        f"{len(line_names)}"
    )

    print(
        f"GPU requested     : "
        f"{args.gpu}"
    )

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

    print()

    rows = []

    gray_wins = 0
    binary_wins = 0
    ties = 0

    meaningful_count = 0

    selected_confidences = []

    classroom_similarities = []

    # -------------------------------------------------------------------------
    # OCR each line
    # -------------------------------------------------------------------------

    for index, line_name in enumerate(
        line_names,
        start=1,
    ):
        gray_text = ""
        gray_confidence = 0.0
        gray_segments = 0

        binary_text = ""
        binary_confidence = 0.0
        binary_segments = 0

        if line_name in gray_files:
            gray_image = cv2.imread(
                str(
                    gray_files[
                        line_name
                    ]
                ),
                cv2.IMREAD_GRAYSCALE,
            )

            (
                gray_text,
                gray_confidence,
                gray_segments,
            ) = run_ocr(
                reader,
                gray_image,
            )

        if line_name in binary_files:
            binary_image = cv2.imread(
                str(
                    binary_files[
                        line_name
                    ]
                ),
                cv2.IMREAD_GRAYSCALE,
            )

            (
                binary_text,
                binary_confidence,
                binary_segments,
            ) = run_ocr(
                reader,
                binary_image,
            )

        # ---------------------------------------------------------------------
        # Select gray vs binary
        # ---------------------------------------------------------------------

        gray_meaningful = (
            is_meaningful_text(
                gray_text,
                gray_confidence,
            )
        )

        binary_meaningful = (
            is_meaningful_text(
                binary_text,
                binary_confidence,
            )
        )

        if (
            gray_meaningful
            and not binary_meaningful
        ):
            selected_source = "gray"
            selected_text = gray_text
            selected_confidence = (
                gray_confidence
            )

            gray_wins += 1

        elif (
            binary_meaningful
            and not gray_meaningful
        ):
            selected_source = "binary"
            selected_text = binary_text
            selected_confidence = (
                binary_confidence
            )

            binary_wins += 1

        elif (
            gray_meaningful
            and binary_meaningful
        ):
            if (
                gray_confidence
                > binary_confidence
                + 0.02
            ):
                selected_source = "gray"
                selected_text = gray_text
                selected_confidence = (
                    gray_confidence
                )

                gray_wins += 1

            elif (
                binary_confidence
                > gray_confidence
                + 0.02
            ):
                selected_source = (
                    "binary"
                )

                selected_text = (
                    binary_text
                )

                selected_confidence = (
                    binary_confidence
                )

                binary_wins += 1

            else:
                # Confidence nearly equal.
                # Prefer the version with more
                # alphanumeric information.

                gray_chars = (
                    alphanumeric_count(
                        gray_text
                    )
                )

                binary_chars = (
                    alphanumeric_count(
                        binary_text
                    )
                )

                if (
                    gray_chars
                    >= binary_chars
                ):
                    selected_source = (
                        "gray"
                    )

                    selected_text = (
                        gray_text
                    )

                    selected_confidence = (
                        gray_confidence
                    )

                else:
                    selected_source = (
                        "binary"
                    )

                    selected_text = (
                        binary_text
                    )

                    selected_confidence = (
                        binary_confidence
                    )

                ties += 1

        else:
            # Neither passed our confidence rule.
            # Retain the higher-confidence output
            # for diagnostics, but mark it rejected.

            if (
                gray_confidence
                >= binary_confidence
            ):
                selected_source = (
                    "gray_rejected"
                )

                selected_text = (
                    gray_text
                )

                selected_confidence = (
                    gray_confidence
                )

            else:
                selected_source = (
                    "binary_rejected"
                )

                selected_text = (
                    binary_text
                )

                selected_confidence = (
                    binary_confidence
                )

        meaningful = (
            is_meaningful_text(
                selected_text,
                selected_confidence,
            )
        )

        if meaningful:
            meaningful_count += 1

            selected_confidences.append(
                selected_confidence
            )

        # ---------------------------------------------------------------------
        # Classroom GT
        # ---------------------------------------------------------------------

        gt_match = ""
        gt_similarity = 0.0

        if args.classroom:
            (
                gt_match,
                gt_similarity,
            ) = (
                find_best_ground_truth_match(
                    selected_text
                )
            )

            if meaningful:
                classroom_similarities.append(
                    gt_similarity
                )

        line_number_match = re.search(
            r"(\d+)$",
            line_name,
        )

        if line_number_match:
            line_number = int(
                line_number_match.group(1)
            )
        else:
            line_number = index

        row = {
            "line":
                line_number,

            "gray_text":
                gray_text,

            "gray_confidence":
                gray_confidence,

            "gray_segments":
                gray_segments,

            "binary_text":
                binary_text,

            "binary_confidence":
                binary_confidence,

            "binary_segments":
                binary_segments,

            "selected_source":
                selected_source,

            "selected_text":
                selected_text,

            "selected_confidence":
                selected_confidence,

            "meaningful":
                meaningful,

            "ground_truth_match":
                gt_match,

            "ground_truth_similarity":
                gt_similarity,
        }

        rows.append(
            row
        )

        print(
            f"L{line_number:02d}"
        )

        print(
            f"    gray   : "
            f"{gray_confidence:.3f} "
            f"| {gray_text}"
        )

        print(
            f"    binary : "
            f"{binary_confidence:.3f} "
            f"| {binary_text}"
        )

        print(
            f"    select : "
            f"{selected_source} "
            f"| {selected_confidence:.3f} "
            f"| {selected_text}"
        )

        if args.classroom:
            print(
                f"    GT     : "
                f"{gt_similarity:.3f} "
                f"| {gt_match}"
            )

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------

    csv_path = (
        output_dir
        / "ocr_results.csv"
    )

    fieldnames = [
        "line",
        "gray_text",
        "gray_confidence",
        "gray_segments",
        "binary_text",
        "binary_confidence",
        "binary_segments",
        "selected_source",
        "selected_text",
        "selected_confidence",
        "meaningful",
        "ground_truth_match",
        "ground_truth_similarity",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            formatted = row.copy()

            formatted[
                "gray_confidence"
            ] = (
                f"{float(row['gray_confidence']):.6f}"
            )

            formatted[
                "binary_confidence"
            ] = (
                f"{float(row['binary_confidence']):.6f}"
            )

            formatted[
                "selected_confidence"
            ] = (
                f"{float(row['selected_confidence']):.6f}"
            )

            formatted[
                "ground_truth_similarity"
            ] = (
                f"{float(row['ground_truth_similarity']):.6f}"
            )

            writer.writerow(
                formatted
            )

    # -------------------------------------------------------------------------
    # Text export
    # -------------------------------------------------------------------------

    text_path = (
        output_dir
        / "recognized_text.txt"
    )

    with text_path.open(
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
                    row["selected_text"]
                ).strip()
            )

            file.write(
                "\n"
            )

    # -------------------------------------------------------------------------
    # Diagnostic report image
    # -------------------------------------------------------------------------

    report_image = (
        create_text_report_image(
            rows
        )
    )

    cv2.imwrite(
        str(
            output_dir
            / "ocr_report.png"
        ),
        report_image,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    if selected_confidences:
        mean_confidence = float(
            np.mean(
                selected_confidences
            )
        )

        median_confidence = float(
            np.median(
                selected_confidences
            )
        )

    else:
        mean_confidence = 0.0
        median_confidence = 0.0

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 14 SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Total line crops      : "
        f"{len(rows)}"
    )

    print(
        f"Meaningful OCR lines  : "
        f"{meaningful_count}"
    )

    print(
        f"Rejected OCR lines    : "
        f"{len(rows) - meaningful_count}"
    )

    print()

    print(
        f"Gray wins             : "
        f"{gray_wins}"
    )

    print(
        f"Binary wins           : "
        f"{binary_wins}"
    )

    print(
        f"Near-confidence ties  : "
        f"{ties}"
    )

    print()

    print(
        f"Mean selected conf    : "
        f"{mean_confidence:.4f}"
    )

    print(
        f"Median selected conf  : "
        f"{median_confidence:.4f}"
    )

    if (
        args.classroom
        and classroom_similarities
    ):
        print()

        print(
            "CLASSROOM GROUND TRUTH"
        )

        print(
            f"Mean best similarity  : "
            f"{np.mean(classroom_similarities):.4f}"
        )

        print(
            f"Median similarity     : "
            f"{np.median(classroom_similarities):.4f}"
        )

        print(
            f"Similarity >= 0.90    : "
            f"{np.sum(np.asarray(classroom_similarities) >= 0.90)}"
            f"/{len(classroom_similarities)}"
        )

        print(
            f"Similarity >= 0.75    : "
            f"{np.sum(np.asarray(classroom_similarities) >= 0.75)}"
            f"/{len(classroom_similarities)}"
        )

    print()

    print(
        f"CSV:\n"
        f"{csv_path}"
    )

    print()

    print(
        f"Recognized text:\n"
        f"{text_path}"
    )

    print()

    print(
        f"Report image:\n"
        f"{output_dir / 'ocr_report.png'}"
    )


if __name__ == "__main__":
    main()