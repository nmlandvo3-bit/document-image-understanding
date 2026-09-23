from __future__ import annotations

# =============================================================================
# PADDLEPADDLE WINDOWS CPU COMPATIBILITY
# =============================================================================
#
# These environment variables MUST be set before importing PaddleOCR/Paddle.
#
# Without this, PaddlePaddle 3.3.x can enter the PIR + oneDNN execution path
# and fail with:
#
# ConvertPirAttribute2RuntimeAttribute not support
# [pir::ArrayAttribute<pir::DoubleAttribute>]
#
# =============================================================================

import os

os.environ["FLAGS_enable_pir_api"] = "0"

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import csv
import json
import subprocess
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

REALESRGAN_MODEL_NAME = "realesrgan-x4plus"


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 20 - Real-ESRGAN learned super-resolution "
            "before PaddleOCR."
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
        help="Experiment 20 output directory.",
    )

    parser.add_argument(
        "--realesrgan-exe",
        default=(
            "./external/realesrgan-ncnn/"
            "realesrgan-ncnn-vulkan.exe"
        ),
        help="Path to Real-ESRGAN NCNN executable.",
    )

    parser.add_argument(
        "--realesrgan-models",
        default=(
            "./external/realesrgan-ncnn/models"
        ),
        help="Path to Real-ESRGAN model directory.",
    )

    return parser.parse_args()


# =============================================================================
# TEXT HELPERS
# =============================================================================

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


# =============================================================================
# IMAGE HELPERS
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


def apply_clahe_color(
    image: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_l = clahe.apply(
        l_channel
    )

    merged = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel,
        )
    )

    return cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR,
    )


# =============================================================================
# REAL-ESRGAN
# =============================================================================

def run_realesrgan(
    exe_path: Path,
    model_dir: Path,
    input_path: Path,
    output_path: Path,
    scale: int = 4,
) -> None:
    command = [
        str(exe_path),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-n",
        REALESRGAN_MODEL_NAME,
        "-s",
        str(scale),
        "-m",
        str(model_dir),
        "-f",
        "png",
        "-g",
        "0",
        "-v",
    ]

    print()
    print(
        f"Running Real-ESRGAN {scale}x..."
    )

    print(
        " ".join(
            f'"{item}"'
            if " " in item
            else item
            for item in command
        )
    )

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if completed.stdout:
        print(
            completed.stdout.strip()
        )

    if completed.stderr:
        print(
            completed.stderr.strip()
        )

    if completed.returncode != 0:
        raise RuntimeError(
            "Real-ESRGAN failed with "
            f"exit code {completed.returncode}."
        )

    if not output_path.exists():
        raise RuntimeError(
            "Real-ESRGAN completed but output "
            "was not created:\n"
            f"{output_path}"
        )


# =============================================================================
# PADDLEOCR
# =============================================================================

def run_paddleocr(
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

        "words":
            word_count,

        "characters":
            character_count,

        "mean_confidence":
            mean_confidence,

        "median_confidence":
            median_confidence,
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


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

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


def save_detail_csv(
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
                        row["meaningful"],

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

    realesrgan_exe = Path(
        args.realesrgan_exe
    ).resolve()

    realesrgan_models = Path(
        args.realesrgan_models
    ).resolve()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found:\n"
            f"{input_path}"
        )

    if not realesrgan_exe.exists():
        raise FileNotFoundError(
            "Real-ESRGAN executable not found:\n"
            f"{realesrgan_exe}"
        )

    if not realesrgan_models.exists():
        raise FileNotFoundError(
            "Real-ESRGAN model directory not found:\n"
            f"{realesrgan_models}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load original image
    # -------------------------------------------------------------------------

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
        "EXPERIMENT 20 — "
        "REALESRGAN OCR RECOVERY"
    )

    print(
        "=" * 100
    )

    print(
        f"Input: {input_path}"
    )

    # -------------------------------------------------------------------------
    # Frozen production detector
    # -------------------------------------------------------------------------

    (
        document_points,
        _edges,
        route,
    ) = detect_document(
        image
    )

    perspective = four_point_transform(
        image,
        document_points,
    )

    print(
        f"Detector route : {route}"
    )

    print(
        f"Document size  : "
        f"{perspective.shape[1]} x "
        f"{perspective.shape[0]}"
    )

    perspective_path = (
        output_dir
        / "00_perspective.png"
    )

    cv2.imwrite(
        str(
            perspective_path
        ),
        perspective,
    )

    # -------------------------------------------------------------------------
    # Classical baselines
    # -------------------------------------------------------------------------

    bicubic_4x = resize_bicubic(
        perspective,
        4,
    )

    bicubic_4x_unsharp = apply_unsharp(
        bicubic_4x,
        amount=1.2,
        sigma=1.0,
    )

    # -------------------------------------------------------------------------
    # Real-ESRGAN
    # -------------------------------------------------------------------------

    realesrgan_4x_path = (
        output_dir
        / "01_realesrgan_4x.png"
    )

    run_realesrgan(
        realesrgan_exe,
        realesrgan_models,
        perspective_path,
        realesrgan_4x_path,
        scale=4,
    )

    realesrgan_4x = cv2.imread(
        str(
            realesrgan_4x_path
        )
    )

    if realesrgan_4x is None:
        raise RuntimeError(
            "Could not load Real-ESRGAN output."
        )

    realesrgan_4x_unsharp = apply_unsharp(
        realesrgan_4x,
        amount=1.2,
        sigma=1.0,
    )

    realesrgan_4x_clahe = apply_clahe_color(
        realesrgan_4x
    )

    realesrgan_4x_clahe_unsharp = apply_unsharp(
        realesrgan_4x_clahe,
        amount=1.0,
        sigma=1.0,
    )

    # -------------------------------------------------------------------------
    # Variants
    # -------------------------------------------------------------------------

    variants = {
        "raw_1x":
            perspective,

        "bicubic_4x":
            bicubic_4x,

        "bicubic_4x_unsharp":
            bicubic_4x_unsharp,

        "realesrgan_4x":
            realesrgan_4x,

        "realesrgan_4x_unsharp":
            realesrgan_4x_unsharp,

        "realesrgan_4x_clahe":
            realesrgan_4x_clahe,

        "realesrgan_4x_clahe_unsharp":
            realesrgan_4x_clahe_unsharp,
    }

    variants_dir = (
        output_dir
        / "variants"
    )

    variants_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, variant in variants.items():
        cv2.imwrite(
            str(
                variants_dir
                / f"{name}.png"
            ),
            variant,
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
        "CPU oneDNN / MKL-DNN disabled "
        "for PaddlePaddle compatibility."
    )

    pipeline = PaddleOCR(
        lang="en",

        use_doc_orientation_classify=False,

        use_doc_unwarping=False,

        use_textline_orientation=False,

        device="cpu",

        # Critical Windows CPU workaround.
        enable_mkldnn=False,
    )

    print(
        "PaddleOCR loaded."
    )

    raw_json_dir = (
        output_dir
        / "raw_paddle_results"
    )

    raw_json_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    # -------------------------------------------------------------------------
    # OCR each variant
    # -------------------------------------------------------------------------

    for variant_name, variant in variants.items():
        print()

        print(
            "-" * 100
        )

        print(
            f"Running: {variant_name}"
        )

        rows = run_paddleocr(
            pipeline,
            variant,
            raw_json_dir,
            variant_name,
        )

        stats = summarize(
            rows
        )

        variant_output_dir = (
            output_dir
            / variant_name
        )

        variant_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        preview = draw_results(
            variant,
            rows,
        )

        cv2.imwrite(
            str(
                variant_output_dir
                / "ocr_regions.png"
            ),
            preview,
        )

        save_text(
            rows,
            variant_output_dir
            / "recognized_text.txt",
        )

        save_detail_csv(
            rows,
            variant_output_dir
            / "ocr_results.csv",
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

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    summary_rows.sort(
        key=lambda row: (
            int(
                row["characters"]
            ),
            int(
                row["meaningful_regions"]
            ),
            int(
                row["words"]
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

    with ranking_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "variant",
                "total_regions",
                "meaningful_regions",
                "words",
                "characters",
                "mean_confidence",
                "median_confidence",
            ],
        )

        writer.writeheader()

        for rank, row in enumerate(
            summary_rows,
            start=1,
        ):
            writer.writerow(
                {
                    "rank":
                        rank,

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
                        (
                            f"{float(row['mean_confidence']):.6f}"
                        ),

                    "median_confidence":
                        (
                            f"{float(row['median_confidence']):.6f}"
                        ),
                }
            )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 20 RANKING"
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
            f"{str(row['variant']):32s} "
            f"| meaningful="
            f"{int(row['meaningful_regions']):3d} "
            f"| words="
            f"{int(row['words']):4d} "
            f"| chars="
            f"{int(row['characters']):5d} "
            f"| conf="
            f"{float(row['mean_confidence']):.4f}"
        )

    winner = summary_rows[0]

    print()

    print(
        "=" * 100
    )

    print(
        "EXPERIMENT 20 SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"Detector route      : "
        f"{route}"
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