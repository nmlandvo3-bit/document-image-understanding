
"""
Experiment 24: Recover Rejected Header Cells

Uses Experiment 23's candidate cells, rejected components, and
unassigned OCR regions to recover legitimate table-header cells
that touch the top image boundary.

This experiment does not rerun OCR, modify production code,
or overwrite Experiment 23's outputs.
"""

import json
from copy import deepcopy
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIC1_DIR = (
    PROJECT_ROOT
    / "results"
    / "gui_v4"
    / "single"
    / "pic1"
)

PARSING_DIR = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
)

INPUT_JSON = (
    PARSING_DIR
    / "cell_reconstruction"
    / "candidate_table_cells.json"
)

OCR_JSON = PARSING_DIR / "structured_document.json"

PERSPECTIVE_IMAGE = PIC1_DIR / "perspective.png"

OUTPUT_DIR = PARSING_DIR / "header_recovery"

# A rejected component must be within the document's top band.
MAX_HEADER_TOP_FRACTION = 0.03
MAX_HEADER_HEIGHT_FRACTION = 0.05

# At least this fraction of an OCR bounding box must overlap
# the candidate header component.
MIN_OCR_OVERLAP = 0.60


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def intersection_area(box_a: dict, box_b: dict) -> int:
    left = max(box_a["x"], box_b["x"])
    top = max(box_a["y"], box_b["y"])

    right = min(box_a["x2"], box_b["x2"])
    bottom = min(box_a["y2"], box_b["y2"])

    return max(0, right - left) * max(0, bottom - top)


def overlap_fraction(ocr_box: dict, candidate_box: dict) -> float:
    ocr_area = ocr_box["width"] * ocr_box["height"]

    if ocr_area <= 0:
        return 0.0

    return (
        intersection_area(ocr_box, candidate_box)
        / ocr_area
    )


def is_header_candidate(
    component: dict,
    image_width: int,
    image_height: int,
) -> bool:
    box = component["bounding_box"]

    if component.get("rejection_reason") != "touches_image_border":
        return False

    if box["y"] > image_height * MAX_HEADER_TOP_FRACTION:
        return False

    if box["height"] > image_height * MAX_HEADER_HEIGHT_FRACTION:
        return False

    # Exclude narrow artifacts and large exterior/background
    # components that do not resemble table cells.
    if box["width"] < 20 or box["height"] < 8:
        return False

    if box["width"] > image_width * 0.90:
        return False

    return True


def recover_header_cells(
    reconstruction: dict,
    ocr_document: dict,
    image_width: int,
    image_height: int,
) -> tuple[dict, list[dict]]:
    result = deepcopy(reconstruction)

    ocr_lookup = {
        region["region_id"]: region
        for region in ocr_document["regions"]
    }

    existing_ids = {
        cell["cell_id"]
        for cell in result["cells"]
    }

    next_cell_id = max(existing_ids, default=0) + 1

    remaining_unassigned = []
    recovered_cells = []
    used_component_labels = set()

    for unassigned in result["unassigned_regions"]:
        region_id = unassigned["region_id"]

        region = ocr_lookup.get(region_id)

        if region is None:
            remaining_unassigned.append(unassigned)
            continue

        matches = []

        for component in result["rejected_components"]:
            if not is_header_candidate(
                component,
                image_width,
                image_height,
            ):
                continue

            score = overlap_fraction(
                region["bounding_box"],
                component["bounding_box"],
            )

            if score >= MIN_OCR_OVERLAP:
                matches.append((score, component))

        # Recover only when exactly one rejected header-like
        # component is a plausible match. Otherwise retain the
        # OCR region for manual inspection.
        if len(matches) != 1:
            remaining_unassigned.append(unassigned)
            continue

        score, component = matches[0]
        component_label = component["component_label"]

        if component_label in used_component_labels:
            remaining_unassigned.append(unassigned)
            continue

        used_component_labels.add(component_label)

        recovered_cell = {
            "component_label": component_label,
            "bounding_box": deepcopy(
                component["bounding_box"]
            ),
            "interior_area_pixels": component[
                "interior_area_pixels"
            ],
            "cell_id": next_cell_id,
            "recovery_method": "rejected_top_boundary_component",
            "requires_visual_verification": True,
            "ocr_regions": [
                {
                    "region_id": region["region_id"],
                    "text": region["text"],
                    "confidence": region["confidence"],
                    "meaningful": region["meaningful"],
                    "source_order": region["source_order"],
                    "bounding_box": deepcopy(
                        region["bounding_box"]
                    ),
                    "cell_overlap_fraction": round(
                        score,
                        4,
                    ),
                }
            ],
            "text": (
                region["text"]
                if region["meaningful"]
                else ""
            ),
        }

        recovered_cells.append(recovered_cell)
        result["cells"].append(recovered_cell)

        next_cell_id += 1

    result["unassigned_regions"] = remaining_unassigned

    result["rejected_components"] = [
        component
        for component in result["rejected_components"]
        if component["component_label"]
        not in used_component_labels
    ]

    result["statistics"]["candidate_cells"] = len(
        result["cells"]
    )

    result["statistics"]["rejected_components"] = len(
        result["rejected_components"]
    )

    result["statistics"]["unassigned_ocr_regions"] = len(
        remaining_unassigned
    )

    result["statistics"]["assigned_ocr_regions"] = sum(
        len(cell["ocr_regions"])
        for cell in result["cells"]
    )

    result["statistics"]["empty_candidate_cells"] = sum(
        not cell["ocr_regions"]
        for cell in result["cells"]
    )

    result["statistics"]["recovered_header_candidates"] = len(
        recovered_cells
    )

    result["schema_version"] = "0.2-experimental"

    result["recovery_notes"] = [
        "Top-boundary recovery is a geometric heuristic.",
        "Recovered header cells require visual verification.",
        "Logical row and column indices have not yet been inferred.",
        "OCR regions crossing multiple cells remain unresolved.",
    ]

    return result, recovered_cells


def save_overlay(
    image,
    recovered_cells: list[dict],
    output_path: Path,
) -> None:
    overlay = image.copy()

    for cell in recovered_cells:
        box = cell["bounding_box"]

        x1 = box["x"]
        y1 = box["y"]
        x2 = box["x2"]
        y2 = box["y2"]

        # Yellow rectangle: recovered header-cell candidate.
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

        cv2.putText(
            overlay,
            f"Recovered cell {cell['cell_id']}",
            (x1 + 4, max(15, y2 + 17)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(
            f"Could not save overlay: {output_path}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reconstruction = load_json(INPUT_JSON)
    ocr_document = load_json(OCR_JSON)

    image = cv2.imread(
        str(PERSPECTIVE_IMAGE),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(PERSPECTIVE_IMAGE)

    height, width = image.shape[:2]

    result, recovered_cells = recover_header_cells(
        reconstruction,
        ocr_document,
        width,
        height,
    )

    output_json = (
        OUTPUT_DIR
        / "candidate_table_cells_with_header.json"
    )

    with output_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    overlay_path = (
        OUTPUT_DIR
        / "recovered_header_overlay.png"
    )

    save_overlay(
        image,
        recovered_cells,
        overlay_path,
    )

    print("=" * 65)
    print("EXPERIMENT 24: HEADER CELL RECOVERY")
    print("=" * 65)

    print(
        "Original candidate cells:",
        reconstruction["statistics"]["candidate_cells"],
    )

    print(
        "Recovered header candidates:",
        len(recovered_cells),
    )

    for cell in recovered_cells:
        print(
            f"  Cell {cell['cell_id']}: "
            f"{cell['text']!r}"
        )

    print(
        "Updated candidate cells:",
        result["statistics"]["candidate_cells"],
    )

    print(
        "Assigned OCR regions:",
        result["statistics"]["assigned_ocr_regions"],
    )

    print(
        "Unassigned OCR regions:",
        result["statistics"]["unassigned_ocr_regions"],
    )

    print(f"JSON:    {output_json}")
    print(f"Overlay: {overlay_path}")

    print("=" * 65)


if __name__ == "__main__":
    main()