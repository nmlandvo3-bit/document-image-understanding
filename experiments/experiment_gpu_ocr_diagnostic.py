from __future__ import annotations

import os

os.environ["FLAGS_enable_pir_api"] = "0"

import argparse
import json
from pathlib import Path

import cv2
from paddleocr import PaddleOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--device",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(
        str(input_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {input_path}"
        )

    print("=" * 100)
    print("GPU OCR DIAGNOSTIC")
    print("=" * 100)

    print(
        f"Input  : {input_path}"
    )

    print(
        f"Device : {args.device}"
    )

    print(
        f"Size   : {image.shape[1]} x {image.shape[0]}"
    )

    pipeline = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=args.device,
        enable_mkldnn=False,
    )

    results = pipeline.predict(
        input=image,
        text_rec_score_thresh=0.0,
    )

    all_texts = []
    all_scores = []

    for result_index, result in enumerate(
        results,
        start=1,
    ):
        json_path = (
            output_dir
            / f"result_{result_index:02d}.json"
        )

        result.save_to_json(
            save_path=str(json_path)
        )

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

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

        all_texts.extend(texts)
        all_scores.extend(scores)

    print()
    print(
        f"Detected regions: {len(all_texts)}"
    )

    for index, (
        text,
        score,
    ) in enumerate(
        zip(
            all_texts,
            all_scores,
        ),
        start=1,
    ):
        print(
            f"{index:03d} | "
            f"{float(score):.6f} | "
            f"{text}"
        )


if __name__ == "__main__":
    main()