
"""
Experiment 26: Recover Missing Text from Cell 24

Tests several image variants of the existing male-application
cell crop using the production CPU PaddleOCR model.

Does not alter the original OCR results or table-cell JSON.
"""

import os

os.environ["FLAGS_enable_pir_api"] = "0"

import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "cell_ocr_recovery"
    / "cell_24_male_application.png"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "cell_24_recovery"
)


def add_white_padding(image: np.ndarray, padding: int) -> np.ndarray:
    return cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def create_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Create OCR input variants without changing the source image.
    """

    height, width = image.shape[:2]

    resized_2x = cv2.resize(
        image,
        (width * 2, height * 2),
        interpolation=cv2.INTER_CUBIC,
    )

    resized_3x = cv2.resize(
        image,
        (width * 3, height * 3),
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_3x = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )
    gray_3x = cv2.cvtColor(
    gray_3x,
    cv2.COLOR_GRAY2BGR,
    )

    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    sharpened_3x = cv2.filter2D(
        resized_3x,
        -1,
        sharpen_kernel,
    )

    return {
        "01_original": image,
        "02_resized_2x": resized_2x,
        "03_resized_3x": resized_3x,
        "04_grayscale_3x": gray_3x,
        "05_sharpened_3x": sharpened_3x,
        "06_resized_3x_extra_padding": add_white_padding(
            resized_3x,
            32,
        ),
    }


def extract_recognition(prediction) -> list[dict]:
    if not prediction:
        return []

    payload = prediction[0].json

    if callable(payload):
        payload = payload()

    if isinstance(payload, str):
        payload = json.loads(payload)

    data = payload.get("res", payload)

    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])

    return [
        {
            "text": str(text),
            "confidence": (
                float(scores[index])
                if index < len(scores)
                else None
            ),
        }
        for index, text in enumerate(texts)
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(
        str(INPUT_PATH),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read cell crop: {INPUT_PATH}"
        )

    variants = create_variants(image)

    print("=" * 65)
    print("EXPERIMENT 26: CELL 24 OCR RECOVERY")
    print("=" * 65)
    print("Loading CPU PaddleOCR...")

    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        enable_mkldnn=False,
    )

    results = []

    for variant_name, variant_image in variants.items():
        output_image = OUTPUT_DIR / f"{variant_name}.png"

        if not cv2.imwrite(
            str(output_image),
            variant_image,
        ):
            raise RuntimeError(
                f"Could not save: {output_image}"
            )

        prediction = ocr.predict(variant_image)
        recognized = extract_recognition(prediction)

        print("-" * 65)
        print(f"Variant: {variant_name}")
        print(
            "Image size:",
            f"{variant_image.shape[1]} x {variant_image.shape[0]}",
        )

        if recognized:
            for item in recognized:
                print(
                    f"  {item['text']!r} "
                    f"(confidence: {item['confidence']})"
                )
        else:
            print("  No text recognized.")

        results.append(
            {
                "variant": variant_name,
                "image_path": str(output_image),
                "recognized_regions": recognized,
                "recognized_text": "\n".join(
                    item["text"]
                    for item in recognized
                ),
            }
        )

    output_json = OUTPUT_DIR / "cell_24_variant_results.json"

    with output_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "experiment": 26,
                "target_cell_id": 24,
                "source_crop": str(INPUT_PATH),
                "results": results,
                "note": (
                    "Diagnostic comparison only. "
                    "No original OCR results were modified."
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 65)
    print("EXPERIMENT 26 COMPLETE")
    print(f"Results: {output_json}")
    print("=" * 65)


if __name__ == "__main__":
    main()