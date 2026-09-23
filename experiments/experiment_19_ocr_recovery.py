from __future__ import annotations

import os

os.environ["FLAGS_enable_pir_api"] = "0"

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


MIN_CONFIDENCE = 0.25
MIN_ALPHANUMERIC_CHARACTERS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 19 - OCR-specific recovery "
            "for difficult small-text documents."
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
        help="Experiment 19 output directory.",
    )

    return parser.parse_args()


def alphanumeric_count(text: str) -> int:
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


def upscale(
    image: np.ndarray,
    scale: int,
) -> np.ndarray:
    if scale == 1:
        return image.copy()

    height, width = image.shape[:2]

    return cv2.resize(
        image,
        (
            width * scale,
            height * scale,
        ),
        interpolation=cv2.INTER_CUBIC,
    )


def apply_clahe_color(
    image: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    l_channel, a_channel, b_channel = (
        cv2.split(lab)
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    l_channel = clahe.apply(
        l_channel
    )

    merged = cv2.merge(
        (
            l_channel,
            a_channel,
            b_channel,
        )
    )

    return cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR,
    )


def apply_unsharp(
    image: np.ndarray,
    amount: float = 1.5,
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


def apply_laplacian_sharpen(
    image: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_16S,
        ksize=3,
    )

    abs_laplacian = cv2.convertScaleAbs(
        laplacian
    )

    sharpened_gray = cv2.subtract(
        gray,
        abs_laplacian,
    )

    return cv2.cvtColor(
        sharpened_gray,
        cv2.COLOR_GRAY2BGR,
    )


def build_variants(
    perspective: np.ndarray,
) -> dict[str, np.ndarray]:
    variants = {}

    # Baselines
    variants["raw_1x"] = (
        perspective.copy()
    )

    variants["raw_2x"] = upscale(
        perspective,
        2,
    )

    variants["raw_4x"] = upscale(
        perspective,
        4,
    )

    # CLAHE
    clahe = apply_clahe_color(
        perspective
    )

    variants["clahe_2x"] = upscale(
        clahe,
        2,
    )

    variants["clahe_4x"] = upscale(
        clahe,
        4,
    )

    # Unsharp before scaling
    unsharp = apply_unsharp(
        perspective,
        amount=1.2,
        sigma=1.0,
    )

    variants["unsharp_2x"] = upscale(
        unsharp,
        2,
    )

    variants["unsharp_4x"] = upscale(
        unsharp,
        4,
    )

    # Upscale first, then sharpen
    raw_2x = upscale(
        perspective,
        2,
    )

    raw_4x = upscale(
        perspective,
        4,
    )

    variants[
        "2x_then_unsharp"
    ] = apply_unsharp(
        raw_2x,
        amount=1.2,
        sigma=1.0,
    )

    variants[
        "4x_then_unsharp"
    ] = apply_unsharp(
        raw_4x,
        amount=1.2,
        sigma=1.0,
    )

    # CLAHE + unsharp
    clahe_unsharp = apply_unsharp(
        clahe,
        amount=1.0,
        sigma=1.0,
    )

    variants[
        "clahe_unsharp_2x"
    ] = upscale(
        clahe_unsharp,
        2,
    )

    variants[
        "clahe_unsharp_4x"
    ] = upscale(
        clahe_unsharp,
        4,
    )

    # Laplacian sharpen
    laplacian = apply_laplacian_sharpen(
        perspective
    )

    variants[
        "laplacian_2x"
    ] = upscale(
        laplacian,
        2,
    )

    variants[
        "laplacian_4x"
    ] = upscale(
        laplacian,
        4,
    )

    return variants


def run_paddle(
    pipeline: PaddleOCR,
    image: np.ndarray,
    raw_dir: Path,
    variant_name: str,
) -> list[dict[str, object]]:
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
            / f"{variant_name}_"
              f"{result_index:02d}.json"
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

        for index in range(count):
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

            if index < len(boxes):
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

            rows.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "meaningful":
                        is_meaningful(
                            text,
                            confidence,
                        ),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

    return rows


def summarize(
    rows: list[dict[str, object]],
) -> dict[str, int | float]:
    meaningful = [
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
        for row in meaningful
    ]

    if confidences:
        mean_conf = float(
            np.mean(
                confidences
            )
        )

        median_conf = float(
            np.median(
                confidences
            )
        )
    else:
        mean_conf = 0.0
        median_conf = 0.0

    words = sum(
        len(
            str(
                row["text"]
            ).split()
        )
        for row in meaningful
    )

    chars = sum(
        alphanumeric_count(
            str(
                row["text"]
            )
        )
        for row in meaningful
    )

    return {
        "total_regions":
            len(rows),

        "meaningful_regions":
            len(meaningful),

        "mean_confidence":
            mean_conf,

        "median_confidence":
            median_conf,

        "words":
            words,

        "characters":
            chars,
    }


def draw_results(
    image: np.ndarray,
    rows: list[dict[str, object]],
) -> np.ndarray:
    preview = image.copy()

    for row in rows:
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
            (x, y),
            (
                x + w,
                y + h,
            ),
            color,
            2,
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
        "EXPERIMENT 19 — "
        "OCR-SPECIFIC RECOVERY"
    )

    print(
        "=" * 100
    )

    print(
        f"Input: {input_path}"
    )

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
        f"Detector route : {route}"
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

    print()

    print(
        "Loading PaddleOCR..."
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

    variants = build_variants(
        perspective
    )

    raw_dir = (
        output_dir
        / "raw_results"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    for variant_name, variant in (
        variants.items()
    ):
        print()

        print(
            "-" * 100
        )

        print(
            f"Running: {variant_name}"
        )

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
            variant,
        )

        rows = run_paddle(
            pipeline,
            variant,
            raw_dir,
            variant_name,
        )

        stats = summarize(
            rows
        )

        preview = draw_results(
            variant,
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
            f"{stats['mean_confidence']:.4f}"
        )

        summary_rows.append(
            {
                "variant":
                    variant_name,

                **stats,
            }
        )

    summary_rows.sort(
        key=lambda row: (
            int(
                row["characters"]
            ),
            int(
                row["meaningful_regions"]
            ),
            float(
                row["mean_confidence"]
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
        "variant",
        "total_regions",
        "meaningful_regions",
        "words",
        "characters",
        "mean_confidence",
        "median_confidence",
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
            summary_rows,
            start=1,
        ):
            writer.writerow(
                {
                    "rank": rank,
                    "variant":
                        row["variant"],
                    "total_regions":
                        row[
                            "total_regions"
                        ],
                    "meaningful_regions":
                        row[
                            "meaningful_regions"
                        ],
                    "words":
                        row["words"],
                    "characters":
                        row[
                            "characters"
                        ],
                    "mean_confidence":
                        f"{float(row['mean_confidence']):.6f}",
                    "median_confidence":
                        f"{float(row['median_confidence']):.6f}",
                }
            )

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 19 RANKING"
    )

    print(
        "=" * 100
    )

    for rank, row in enumerate(
        summary_rows,
        start=1,
    ):
        print(
            f"{rank:02d}. "
            f"{str(row['variant']):24s} "
            f"| meaningful="
            f"{int(row['meaningful_regions']):3d} "
            f"| words="
            f"{int(row['words']):4d} "
            f"| chars="
            f"{int(row['characters']):5d} "
            f"| conf="
            f"{float(row['mean_confidence']):.4f}"
        )

    if summary_rows:
        winner = (
            summary_rows[0]
        )

        print()

        print(
            "=" * 100
        )

        print(
            "EXPERIMENT 19 SUMMARY"
        )

        print(
            "=" * 100
        )

        print(
            f"Winning variant     : "
            f"{winner['variant']}"
        )

        print(
            f"Meaningful regions  : "
            f"{winner['meaningful_regions']}"
        )

        print(
            f"Words               : "
            f"{winner['words']}"
        )

        print(
            f"Characters          : "
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