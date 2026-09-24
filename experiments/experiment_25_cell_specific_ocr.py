
"""
Experiment 25: Cell-Specific OCR Recovery

Investigates two OCR regions that span neighboring timetable cells.

Reads:
    - Existing perspective-corrected document
    - Existing candidate table-cell JSON

Produces:
    - Individual cell-crop images
    - OCR results for each crop
    - Diagnostic JSON

Does not overwrite original OCR results or automatically replace
the text assigned to any table cell.
"""

import os

# Keep the same PaddlePaddle compatibility setting as the
# existing working CPU OCR engine.
os.environ["FLAGS_enable_pir_api"] = "0"

import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PERSPECTIVE_PATH = (
    PROJECT_ROOT
    / "results"
    / "gui_v4"
    / "single"
    / "pic1"
    / "perspective.png"
)

TABLE_JSON_PATH = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "header_recovery"
    / "candidate_table_cells_with_header.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "cell_ocr_recovery"
)

TARGET_CELLS = {
    19: "female_category",
    18: "female_application",
    23: "male_category",
    24: "male_application",
}

# Inset the crop slightly to avoid including printed table borders.
CROP_INSET = 2

# Add a white margin around the crop so text is not positioned
# directly against the image boundary during OCR.
WHITE_PADDING = 16


def load_table_cells() -> dict:
    with TABLE_JSON_PATH.open("r", encoding="utf-8") as file:
        document = json.load(file)

    return {
        cell["cell_id"]: cell
        for cell in document["cells"]
    }


def crop_cell(image: np.ndarray, box: dict) -> np.ndarray:
    image_height, image_width = image.shape[:2]

    x1 = max(0, box["x"] + CROP_INSET)
    y1 = max(0, box["y"] + CROP_INSET)

    x2 = min(image_width, box["x2"] - CROP_INSET)
    y2 = min(image_height, box["y2"] - CROP_INSET)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid cell crop: {box}")

    crop = image[y1:y2, x1:x2].copy()

    return cv2.copyMakeBorder(
        crop,
        WHITE_PADDING,
        WHITE_PADDING,
        WHITE_PADDING,
        WHITE_PADDING,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def extract_recognition(result) -> list[dict]:
    """
    Extract recognized text and scores from a PaddleOCR 3.x
    prediction result.
    """

    if not result:
        return []

    first_result = result[0]

    # PaddleOCR 3.x result objects expose a JSON-compatible
    # representation through their json property.
    payload = first_result.json

    if callable(payload):
        payload = payload()

    if isinstance(payload, str):
        payload = json.loads(payload)

    data = payload.get("res", payload)

    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])

    recognized = []

    for index, text in enumerate(texts):
        score = (
            float(scores[index])
            if index < len(scores)
            else None
        )

        recognized.append(
            {
                "text": str(text),
                "confidence": score,
            }
        )

    return recognized


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(
        str(PERSPECTIVE_PATH),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(PERSPECTIVE_PATH)

    cells = load_table_cells()

    missing_cells = [
        cell_id
        for cell_id in TARGET_CELLS
        if cell_id not in cells
    ]

    if missing_cells:
        raise ValueError(
            f"Missing target cells: {missing_cells}"
        )

    print("=" * 65)
    print("EXPERIMENT 25: CELL-SPECIFIC OCR RECOVERY")
    print("=" * 65)
    print("Initializing the existing CPU PaddleOCR model...")

    # Initialize once and reuse the model for all four crops.
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        enable_mkldnn=False,
    )

    results = []

    for cell_id, label in TARGET_CELLS.items():
        cell = cells[cell_id]

        crop = crop_cell(
            image,
            cell["bounding_box"],
        )

        crop_filename = f"cell_{cell_id}_{label}.png"
        crop_path = OUTPUT_DIR / crop_filename

        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError(
                f"Could not save crop: {crop_path}"
            )

        print("-" * 65)
        print(f"Cell {cell_id}: {label}")
        print(f"Crop saved: {crop_filename}")
        print(f"Previous OCR: {cell['text']!r}")

        prediction = ocr.predict(crop)
        recognized = extract_recognition(prediction)

        print("Cell-specific OCR:")

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
                "cell_id": cell_id,
                "label": label,
                "crop_image": str(crop_path),
                "cell_bounding_box": cell["bounding_box"],
                "previous_cell_text": cell["text"],
                "recognized_regions": recognized,
                "recovered_text": "\n".join(
                    item["text"]
                    for item in recognized
                ),
                "status": "requires_visual_verification",
            }
        )

    output_json = OUTPUT_DIR / "cell_ocr_diagnostics.json"

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "experiment": 25,
                "purpose": "cell_specific_ocr_recovery",
                "source_image": str(PERSPECTIVE_PATH),
                "target_unassigned_region_ids": [55, 70],
                "results": results,
                "note": (
                    "Diagnostic results only. Original OCR and "
                    "table-cell assignments have not been modified."
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 65)
    print("EXPERIMENT 25 COMPLETE")
    print(f"JSON output: {output_json}")
    print(f"Cell images: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()