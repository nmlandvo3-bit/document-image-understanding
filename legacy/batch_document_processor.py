from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from document_processing_engine import DocumentProcessingEngine


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch Document Processor V3 - "
            "shared PaddleOCR engine with CPU/GPU selection."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing document images.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where batch results will be saved.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Processing device, for example cpu or gpu:0.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input subdirectories recursively.",
    )

    return parser.parse_args()


def find_images(
    input_dir: Path,
    recursive: bool,
) -> list[Path]:
    if recursive:
        candidates = input_dir.rglob("*")
    else:
        candidates = input_dir.glob("*")

    images = [
        path
        for path in candidates
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    ]

    images.sort(
        key=lambda path: str(path).lower()
    )

    return images


def main() -> None:
    args = parse_args()

    input_dir = Path(
        args.input_dir
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    device = args.device

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found:\n{input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory:\n{input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = find_images(
        input_dir,
        args.recursive,
    )

    print("=" * 100)
    print("BATCH DOCUMENT PROCESSOR V3")
    print("=" * 100)

    print(
        f"Input directory : {input_dir}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print(
        f"Device          : {device}"
    )

    print(
        f"Images found    : {len(images)}"
    )

    if not images:
        print()
        print("No supported images found.")
        return

    # =========================================================================
    # LOAD SHARED OCR ENGINE ONCE
    # =========================================================================

    print()
    print("=" * 100)
    print("LOADING SHARED PADDLEOCR ENGINE")
    print("=" * 100)

    engine_load_start = time.perf_counter()

    engine = DocumentProcessingEngine(
        device=device,
        verbose=False,
    )

    engine_load_seconds = (
        time.perf_counter()
        - engine_load_start
    )

    print(
        f"Device selected : {device}"
    )

    print(
        f"Engine load time: "
        f"{engine_load_seconds:.2f}s"
    )

    # =========================================================================
    # PROCESS DOCUMENTS
    # =========================================================================

    rows = []

    batch_start = time.perf_counter()

    for index, image_path in enumerate(
        images,
        start=1,
    ):
        print()
        print("=" * 100)

        print(
            f"[{index}/{len(images)}] "
            f"{image_path.name}"
        )

        print("=" * 100)

        document_output = (
            output_dir
            / image_path.stem
        )

        start_time = time.perf_counter()

        try:
            summary = engine.process(
                image_path,
                document_output,
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            row = {
                "index":
                    index,

                "filename":
                    image_path.name,

                "status":
                    "SUCCESS",

                "device":
                    device,

                "detector_route":
                    summary.get(
                        "detector_route",
                        "",
                    ),

                "ocr_route":
                    summary.get(
                        "ocr_route",
                        "",
                    ),

                "ocr_quality":
                    summary.get(
                        "ocr_quality",
                        "",
                    ),

                "recovery_attempted":
                    summary.get(
                        "recovery_attempted",
                        False,
                    ),

                "meaningful_regions":
                    summary.get(
                        "meaningful_regions",
                        0,
                    ),

                "words":
                    summary.get(
                        "words",
                        0,
                    ),

                "characters":
                    summary.get(
                        "characters",
                        0,
                    ),

                "mean_confidence":
                    summary.get(
                        "mean_confidence",
                        0.0,
                    ),

                "processing_seconds":
                    elapsed,

                "output_directory":
                    str(document_output),

                "error":
                    "",
            }

            print(
                "Status              : SUCCESS"
            )

            print(
                f"Device              : {device}"
            )

            print(
                "Detector route      : "
                f"{row['detector_route']}"
            )

            print(
                "OCR route           : "
                f"{row['ocr_route']}"
            )

            print(
                "OCR quality         : "
                f"{row['ocr_quality']}"
            )

            print(
                "Meaningful regions  : "
                f"{row['meaningful_regions']}"
            )

            print(
                "Words               : "
                f"{row['words']}"
            )

            print(
                "Characters          : "
                f"{row['characters']}"
            )

            print(
                "Mean confidence     : "
                f"{float(row['mean_confidence']):.4f}"
            )

            print(
                "Processing time     : "
                f"{elapsed:.2f}s"
            )

        except Exception as error:
            elapsed = (
                time.perf_counter()
                - start_time
            )

            row = {
                "index":
                    index,

                "filename":
                    image_path.name,

                "status":
                    "FAILED",

                "device":
                    device,

                "detector_route":
                    "",

                "ocr_route":
                    "",

                "ocr_quality":
                    "",

                "recovery_attempted":
                    "",

                "meaningful_regions":
                    0,

                "words":
                    0,

                "characters":
                    0,

                "mean_confidence":
                    0.0,

                "processing_seconds":
                    elapsed,

                "output_directory":
                    str(document_output),

                "error":
                    str(error),
            }

            print(
                "Status              : FAILED"
            )

            print(
                f"Error               : {error}"
            )

        rows.append(row)

    batch_processing_seconds = (
        time.perf_counter()
        - batch_start
    )

    total_seconds_with_engine_load = (
        engine_load_seconds
        + batch_processing_seconds
    )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    csv_path = (
        output_dir
        / "batch_summary.csv"
    )

    fieldnames = [
        "index",
        "filename",
        "status",
        "device",
        "detector_route",
        "ocr_route",
        "ocr_quality",
        "recovery_attempted",
        "meaningful_regions",
        "words",
        "characters",
        "mean_confidence",
        "processing_seconds",
        "output_directory",
        "error",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            saved_row = dict(row)

            saved_row[
                "mean_confidence"
            ] = (
                f"{float(saved_row['mean_confidence']):.6f}"
            )

            saved_row[
                "processing_seconds"
            ] = (
                f"{float(saved_row['processing_seconds']):.3f}"
            )

            writer.writerow(
                saved_row
            )

    # =========================================================================
    # AGGREGATE SUMMARY
    # =========================================================================

    successful = [
        row
        for row in rows
        if row["status"] == "SUCCESS"
    ]

    failed = (
        len(rows)
        - len(successful)
    )

    strong = sum(
        1
        for row in successful
        if row["ocr_quality"] == "STRONG"
    )

    limited = sum(
        1
        for row in successful
        if row["ocr_quality"] == "LIMITED"
    )

    insufficient = sum(
        1
        for row in successful
        if row["ocr_quality"] == "INSUFFICIENT"
    )

    recovery_attempted = sum(
        1
        for row in successful
        if bool(
            row["recovery_attempted"]
        )
    )

    if successful:
        mean_confidence = sum(
            float(
                row["mean_confidence"]
            )
            for row in successful
        ) / len(successful)
    else:
        mean_confidence = 0.0

    average_seconds_per_image = (
        batch_processing_seconds
        / len(images)
    )

    summary = {
        "input_directory":
            str(input_dir),

        "output_directory":
            str(output_dir),

        "device":
            device,

        "total_images":
            len(images),

        "successful":
            len(successful),

        "failed":
            failed,

        "strong":
            strong,

        "limited":
            limited,

        "insufficient":
            insufficient,

        "recovery_attempted":
            recovery_attempted,

        "mean_ocr_confidence":
            mean_confidence,

        "engine_load_seconds":
            engine_load_seconds,

        "batch_processing_seconds":
            batch_processing_seconds,

        "total_seconds_with_engine_load":
            total_seconds_with_engine_load,

        "average_seconds_per_image":
            average_seconds_per_image,

        "shared_ocr_engine":
            True,
    }

    json_path = (
        output_dir
        / "batch_summary.json"
    )

    with json_path.open(
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
    # FINAL REPORT
    # =========================================================================

    print()
    print("=" * 100)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 100)

    print(
        f"Device               : {device}"
    )

    print(
        f"Total images         : {len(images)}"
    )

    print(
        f"Successful           : {len(successful)}"
    )

    print(
        f"Failed               : {failed}"
    )

    print(
        f"STRONG               : {strong}"
    )

    print(
        f"LIMITED              : {limited}"
    )

    print(
        f"INSUFFICIENT         : {insufficient}"
    )

    print(
        f"Recovery attempted   : {recovery_attempted}"
    )

    print(
        "Mean confidence      : "
        f"{mean_confidence:.4f}"
    )

    print(
        "Engine load time     : "
        f"{engine_load_seconds:.2f}s"
    )

    print(
        "Batch processing     : "
        f"{batch_processing_seconds:.2f}s"
    )

    print(
        "Total incl. loading  : "
        f"{total_seconds_with_engine_load:.2f}s"
    )

    print(
        "Average/image        : "
        f"{average_seconds_per_image:.2f}s"
    )

    print()

    print(
        f"Batch CSV:\n{csv_path}"
    )

    print()

    print(
        f"Batch JSON:\n{json_path}"
    )


if __name__ == "__main__":
    main()