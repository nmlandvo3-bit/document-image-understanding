from __future__ import annotations

import csv
import traceback
from pathlib import Path

import cv2

from document_scanner import (
    detect_document,
    draw_document_boundary,
    enhance_document,
    four_point_transform,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "smartdoc_frames"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "experiment_2_batch_baseline"
)

SUMMARY_CSV = (
    OUTPUT_ROOT
    / "experiment_2_summary.csv"
)


def process_image(
    image_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not load image: {image_path}"
        )

    points, edges = detect_document(
        image
    )

    boundary = draw_document_boundary(
        image,
        points,
    )

    perspective = four_point_transform(
        image,
        points,
    )

    enhanced = enhance_document(
        perspective
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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
        edges,
    )

    cv2.imwrite(
        str(
            output_dir
            / "perspective.png"
        ),
        perspective,
    )

    cv2.imwrite(
        str(
            output_dir
            / "scan.png"
        ),
        enhanced["scan"],
    )

    return {
        "width":
            image.shape[1],

        "height":
            image.shape[0],

        "perspective_width":
            perspective.shape[1],

        "perspective_height":
            perspective.shape[0],
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        INPUT_ROOT.rglob("*.png")
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No SmartDoc frames found in:\n"
            f"{INPUT_ROOT}"
        )

    print("=" * 90)
    print(
        "EXPERIMENT 2 — SMARTDOC "
        "BATCH BASELINE"
    )
    print("=" * 90)

    print(
        f"Images found: {len(image_paths)}"
    )

    print()

    rows: list[dict[str, object]] = []

    success_count = 0
    failure_count = 0

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        relative_path = (
            image_path.relative_to(
                INPUT_ROOT
            )
        )

        video_name = (
            relative_path.parts[0]
        )

        sample_name = (
            image_path.stem
        )

        output_dir = (
            OUTPUT_ROOT
            / video_name
            / sample_name
        )

        print(
            f"[{index:02d}/{len(image_paths):02d}] "
            f"{relative_path}"
        )

        row: dict[str, object] = {
            "sample":
                str(relative_path),

            "video":
                video_name,

            "status":
                "FAIL",

            "error":
                "",

            "width":
                "",

            "height":
                "",

            "perspective_width":
                "",

            "perspective_height":
                "",
        }

        try:
            information = process_image(
                image_path=image_path,
                output_dir=output_dir,
            )

            row.update(
                information
            )

            row["status"] = "PASS"

            success_count += 1

            print(
                "    [PASS] "
                "four-corner document detected"
            )

        except Exception as error:
            failure_count += 1

            row["error"] = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                f"    [FAIL] {row['error']}"
            )

            traceback.print_exc()

        rows.append(
            row
        )

    fieldnames = [
        "sample",
        "video",
        "status",
        "error",
        "width",
        "height",
        "perspective_width",
        "perspective_height",
    ]

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    total = len(
        image_paths
    )

    success_rate = (
        success_count
        / total
        if total
        else 0.0
    )

    print()
    print("=" * 90)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 90)

    print(
        f"Total   : {total}"
    )

    print(
        f"Passed  : {success_count}"
    )

    print(
        f"Failed  : {failure_count}"
    )

    print(
        f"Pass rate: "
        f"{success_rate * 100:.2f}%"
    )

    print(
        f"Summary : {SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()