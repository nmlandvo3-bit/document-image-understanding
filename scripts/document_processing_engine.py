from __future__ import annotations

import os

# =============================================================================
# PADDLE WINDOWS COMPATIBILITY
# =============================================================================

# MUST be set before importing PaddleOCR / PaddlePaddle.
os.environ["FLAGS_enable_pir_api"] = "0"


# =============================================================================
# IMPORTS
# =============================================================================

import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

from config import (
    LIMITED_MIN_CHARACTERS,
    LIMITED_MIN_REGIONS,
    MIN_ALPHANUMERIC_CHARACTERS,
    MIN_CONFIDENCE,
    OCR_DEVICE,
    OCR_ENABLE_MKLDNN,
    OCR_LANGUAGE,
    RECOVERY_SCALE,
    RECOVERY_UNSHARP_AMOUNT,
    RECOVERY_UNSHARP_SIGMA,
    SINGLE_CHARACTER_MIN_CONFIDENCE,
    STRONG_MIN_CHARACTERS,
    STRONG_MIN_MEAN_CONFIDENCE,
    STRONG_MIN_REGIONS,
    configure_logging,
)


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

try:
    from .document_scanner import (
        detect_document,
        draw_document_boundary,
        enhance_document,
        four_point_transform,
    )

except ImportError:
    from document_scanner import (
        detect_document,
        draw_document_boundary,
        enhance_document,
        four_point_transform,
    )


logger = configure_logging()


# =============================================================================
# OCR FILTERING
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
    text = text.strip()

    if not text:
        return False

    if confidence < MIN_CONFIDENCE:
        return False

    alnum_count = alphanumeric_count(
        text
    )

    if (
        alnum_count
        >= MIN_ALPHANUMERIC_CHARACTERS
    ):
        return True

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
    amount: float,
    sigma: float,
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
        RECOVERY_SCALE,
    )

    return apply_unsharp(
        upscaled,
        amount=RECOVERY_UNSHARP_AMOUNT,
        sigma=RECOVERY_UNSHARP_SIGMA,
    )


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
# OCR OUTPUT
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

            if not text:
                continue

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
                        row[
                            "x"
                        ],

                    "y":
                        row[
                            "y"
                        ],

                    "w":
                        row[
                            "w"
                        ],

                    "h":
                        row[
                            "h"
                        ],
                }
            )


# =============================================================================
# DOCUMENT PROCESSING ENGINE
# =============================================================================

class DocumentProcessingEngine:
    def __init__(
        self,
        device: str = OCR_DEVICE,
        verbose: bool = True,
    ):
        self.device = device
        self.verbose = verbose

        logger.info(
            "Initializing OCR engine. device=%s",
            self.device,
        )

        self._print(
            f"Requested device: "
            f"{self.device}"
        )

        self._print(
            "Loading PaddleOCR production engine..."
        )

        # ---------------------------------------------------------------------
        # Validated production configuration
        # ---------------------------------------------------------------------
        #
        # DO NOT automatically enable MKL-DNN for CPU.
        #
        # The working Windows production environment requires:
        #
        # FLAGS_enable_pir_api = 0
        # enable_mkldnn = False
        # ---------------------------------------------------------------------

        self.pipeline = PaddleOCR(
            lang=OCR_LANGUAGE,

            use_doc_orientation_classify=False,

            use_doc_unwarping=False,

            use_textline_orientation=False,

            device=self.device,

            enable_mkldnn=OCR_ENABLE_MKLDNN,
        )

        logger.info(
            "OCR engine initialized successfully."
        )

        self._print(
            f"PaddleOCR ready on "
            f"{self.device}."
        )


    def _print(
        self,
        message: str = "",
    ) -> None:
        if self.verbose:
            print(
                message
            )


    # =========================================================================
    # OCR INFERENCE
    # =========================================================================

    def run_paddleocr(
        self,
        image: np.ndarray,
        raw_dir: Path,
        variant_name: str,
    ) -> list[dict[str, object]]:
        raw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Running OCR variant=%s image=%sx%s",
            variant_name,
            image.shape[1],
            image.shape[0],
        )

        result_objects = (
            self.pipeline.predict(
                input=image,
                text_rec_score_thresh=0.0,
            )
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

                        elif (
                            box_array.ndim == 2
                            and box_array.shape[1] >= 2
                        ):
                            xs = box_array[:, 0]
                            ys = box_array[:, 1]

                            x1 = int(
                                round(
                                    float(
                                        np.min(
                                            xs
                                        )
                                    )
                                )
                            )

                            y1 = int(
                                round(
                                    float(
                                        np.min(
                                            ys
                                        )
                                    )
                                )
                            )

                            x2 = int(
                                round(
                                    float(
                                        np.max(
                                            xs
                                        )
                                    )
                                )
                            )

                            y2 = int(
                                round(
                                    float(
                                        np.max(
                                            ys
                                        )
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
                        logger.exception(
                            "Could not parse OCR bounding box."
                        )

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
                    row[
                        "y"
                    ]
                ),
                int(
                    row[
                        "x"
                    ]
                ),
            )
        )

        return rows


    # =========================================================================
    # PROCESS DOCUMENT
    # =========================================================================

    def process(
        self,
        input_path: str | Path,
        output_dir: str | Path,
    ) -> dict:
        input_path = Path(
            input_path
        ).resolve()

        output_dir = Path(
            output_dir
        ).resolve()

        logger.info(
            "Processing document: %s",
            input_path,
        )

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

        try:
            # =================================================================
            # DOCUMENT DETECTION
            # =================================================================

            (
                document_points,
                edge_preview,
                detector_route,
            ) = detect_document(
                image
            )

            logger.info(
                "Detector route=%s",
                detector_route,
            )

            boundary = (
                draw_document_boundary(
                    image,
                    document_points,
                )
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


            # =================================================================
            # PERSPECTIVE
            # =================================================================

            perspective = (
                four_point_transform(
                    image,
                    document_points,
                )
            )

            cv2.imwrite(
                str(
                    output_dir
                    / "perspective.png"
                ),
                perspective,
            )


            # =================================================================
            # SCAN BRANCH
            # =================================================================

            enhanced = enhance_document(
                perspective
            )

            scan_outputs = {
                "gray.png":
                    enhanced[
                        "gray"
                    ],

                "illumination_background.png":
                    enhanced[
                        "illumination_background"
                    ],

                "illumination_corrected.png":
                    enhanced[
                        "illumination_corrected"
                    ],

                "contrast.png":
                    enhanced[
                        "contrast"
                    ],

                "scan.png":
                    enhanced[
                        "scan"
                    ],
            }

            for filename, scan_image in scan_outputs.items():
                cv2.imwrite(
                    str(
                        scan_dir
                        / filename
                    ),
                    scan_image,
                )


            # =================================================================
            # DIRECT OCR
            # =================================================================

            direct_rows = (
                self.run_paddleocr(
                    perspective,
                    raw_ocr_dir,
                    "direct",
                )
            )

            direct_stats = (
                summarize_ocr(
                    direct_rows
                )
            )

            direct_quality = (
                classify_quality(
                    direct_stats
                )
            )


            # =================================================================
            # ADAPTIVE OCR ROUTING
            # =================================================================

            recovery_attempted = False

            recovery_stats = None
            recovery_quality = None

            chosen_rows = direct_rows
            chosen_stats = direct_stats
            chosen_quality = direct_quality

            final_ocr_image = perspective

            ocr_route = "direct"


            if direct_quality != "STRONG":
                recovery_attempted = True

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

                recovery_rows = (
                    self.run_paddleocr(
                        recovery_image,
                        raw_ocr_dir,
                        "recovery",
                    )
                )

                recovery_stats = (
                    summarize_ocr(
                        recovery_rows
                    )
                )

                recovery_quality = (
                    classify_quality(
                        recovery_stats
                    )
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


            # =================================================================
            # SAVE OCR OUTPUTS
            # =================================================================

            regions_preview = (
                draw_ocr_regions(
                    final_ocr_image,
                    chosen_rows,
                )
            )

            cv2.imwrite(
                str(
                    ocr_dir
                    / "regions.png"
                ),
                regions_preview,
            )

            save_text(
                chosen_rows,
                ocr_dir
                / "text.txt",
            )

            save_ocr_csv(
                chosen_rows,
                ocr_dir
                / "results.csv",
            )


            # =================================================================
            # SUMMARY
            # =================================================================

            summary = {
                "input_file":
                    str(
                        input_path
                    ),

                "device":
                    self.device,

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

                    "edges":
                        "edges.png",

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

            logger.info(
                (
                    "Processing complete | "
                    "detector=%s | "
                    "route=%s | "
                    "quality=%s | "
                    "regions=%s | "
                    "words=%s | "
                    "confidence=%.4f"
                ),
                detector_route,
                ocr_route,
                chosen_quality,
                chosen_stats[
                    "meaningful_regions"
                ],
                chosen_stats[
                    "words"
                ],
                float(
                    chosen_stats[
                        "mean_confidence"
                    ]
                ),
            )

            return summary

        except Exception:
            logger.exception(
                "Document processing failed: %s",
                input_path,
            )

            raise