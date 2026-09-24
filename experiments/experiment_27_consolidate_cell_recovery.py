
"""
Experiment 27: Consolidate Verified Cell-Specific OCR Recovery

Combines:
    Experiment 24: Table cells with recovered header
    Experiment 25: Cell-specific OCR diagnostic results
    Experiment 26: Improved OCR for cell 24

Creates a new table JSON without modifying previous outputs.

The original OCR region assignments and unassigned regions are
preserved for traceability.
"""

import json
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARSING_DIR = (
    PROJECT_ROOT / "results" / "parsing" / "pic1"
)

TABLE_PATH = (
    PARSING_DIR
    / "header_recovery"
    / "candidate_table_cells_with_header.json"
)

CELL_RECOVERY_PATH = (
    PARSING_DIR
    / "cell_ocr_recovery"
    / "cell_ocr_diagnostics.json"
)

VARIANT_PATH = (
    PARSING_DIR
    / "cell_24_recovery"
    / "cell_24_variant_results.json"
)

OUTPUT_DIR = PARSING_DIR / "consolidated_table"

# These are the four cells we inspected in Experiments 25 and 26.
TARGET_CELL_IDS = (18, 19, 23, 24)

# Experiment 26 showed that the 3x variant recognizes all
# three visible lines in cell 24.
SELECTED_CELL_24_VARIANT = "03_resized_3x"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_recognized_text(records: list[dict]) -> str:
    return "\n".join(
        str(record["text"])
        for record in records
        if str(record.get("text", "")).strip()
    )


def main() -> None:
    table = deepcopy(load_json(TABLE_PATH))
    cell_diagnostics = load_json(CELL_RECOVERY_PATH)
    variant_diagnostics = load_json(VARIANT_PATH)

    cells_by_id = {
        cell["cell_id"]: cell
        for cell in table["cells"]
    }

    recovered_by_id = {
        result["cell_id"]: result
        for result in cell_diagnostics["results"]
    }

    variants_by_name = {
        result["variant"]: result
        for result in variant_diagnostics["results"]
    }

    selected_variant = variants_by_name.get(
        SELECTED_CELL_24_VARIANT
    )

    if selected_variant is None:
        raise ValueError(
            "The selected cell 24 OCR variant was not found."
        )

    changes = []

    for cell_id in TARGET_CELL_IDS:
        if cell_id not in cells_by_id:
            raise ValueError(
                f"Cell {cell_id} not found in the table."
            )

        if cell_id not in recovered_by_id:
            raise ValueError(
                f"Cell {cell_id} not found in Experiment 25."
            )

        cell = cells_by_id[cell_id]
        diagnostic = recovered_by_id[cell_id]

        if cell_id == 24:
            records = selected_variant["recognized_regions"]
            source_experiment = 26
            source_variant = SELECTED_CELL_24_VARIANT
            crop_image = selected_variant["image_path"]

        else:
            records = diagnostic["recognized_regions"]
            source_experiment = 25
            source_variant = "original_cell_crop"
            crop_image = diagnostic["crop_image"]

        recovered_text = get_recognized_text(records)

        if not recovered_text:
            raise ValueError(
                f"No recovered OCR text for cell {cell_id}."
            )

        previous_text = cell["text"]

        # Preserve the original cell text and its source OCR
        # regions. The new text is stored separately, with
        # an explicit record of where it came from.
        cell["original_cell_text"] = previous_text

        cell["recovered_cell_text"] = recovered_text

        cell["text"] = recovered_text

        cell["text_source"] = "cell_specific_ocr"

        cell["text_recovery"] = {
            "source_experiment": source_experiment,
            "source_variant": source_variant,
            "crop_image": crop_image,
            "recognized_regions": deepcopy(records),
            "visual_review": "reviewed_against_uploaded_cell_crop",
            "note": (
                "Original full-page OCR assignments are retained "
                "in ocr_regions. Cell text was recovered separately."
            ),
        }

        changes.append(
            {
                "cell_id": cell_id,
                "original_text": previous_text,
                "recovered_text": recovered_text,
                "source_experiment": source_experiment,
                "source_variant": source_variant,
            }
        )

    # Preserve original assignment statistics. Cell-specific
    # recovery did not assign the two original merged OCR boxes.
    table["schema_version"] = "0.3-experimental"

    table["statistics"]["cells_with_recovered_text"] = len(
        changes
    )

    table["statistics"]["original_cross_boundary_regions"] = 2

    table["recovery_summary"] = {
        "method": "cell_specific_ocr",
        "updated_cell_ids": list(TARGET_CELL_IDS),
        "original_merged_region_ids": [55, 70],
        "original_merged_regions_retained": True,
        "note": (
            "Missing content was recovered using separate "
            "cell crops. This does not validate all table cells "
            "or establish complete logical table structure."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_json = (
        OUTPUT_DIR / "consolidated_table_cells.json"
    )

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(
            table,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 65)
    print("EXPERIMENT 27: CONSOLIDATED CELL RECOVERY")
    print("=" * 65)

    for change in changes:
        print(f"\nCell {change['cell_id']}")
        print(f"  Original:  {change['original_text']!r}")
        print(f"  Recovered: {change['recovered_text']!r}")
        print(
            "  Source:    "
            f"Experiment {change['source_experiment']}, "
            f"{change['source_variant']}"
        )

    print("-" * 65)
    print(f"Table cells: {len(table['cells'])}")
    print(f"Updated cell texts: {len(changes)}")
    print(
        "Original unassigned OCR regions retained:",
        len(table["unassigned_regions"]),
    )
    print(f"Output: {output_json}")
    print("=" * 65)
    print("EXPERIMENT 27 COMPLETE")


if __name__ == "__main__":
    main()