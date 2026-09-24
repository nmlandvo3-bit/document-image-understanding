"""
Experiment 40: Unified Document Representation

Combines validated outputs from:
- pic1 table/timetable branch
- pic13 form branch

into a common document schema.

This experiment does NOT rerun OCR or parsing.

The unified schema is intended to become the interface for:
- retrieval
- LLM grounding
- agent workflows
- APIs
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIC1_NORMALIZED = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "normalized_records"
    / "normalized_timetable_records.json"
)

PIC1_LOGICAL = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "logical_table"
    / "logical_table.json"
)

PIC13_FORM = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic13"
    / "semantic_form"
    / "semantic_form_structure.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "unified_documents"
)

PIC1_OUTPUT = (
    OUTPUT_DIR
    / "pic1_unified_document.json"
)

PIC13_OUTPUT = (
    OUTPUT_DIR
    / "pic13_unified_document.json"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "unified_document_summary.txt"
)


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def make_common_source(
    document_id,
    document_type,
    source_image,
):
    return {
        "document_id": document_id,
        "schema_version": "1.0-unified-experimental",
        "document_type": document_type,
        "source": {
            "image": source_image,
        },
        "sections": [],
        "fields": [],
        "records": [],
        "controls": [],
        "free_text": [],
        "evidence": [],
        "review_flags": [],
        "metadata": {},
    }


def extract_cell_bbox(cell):
    """
    Extract the exact bounding-box structure used by
    pic1 logical_table.json.
    """

    bbox = cell.get(
        "bounding_box"
    )

    if not isinstance(
        bbox,
        dict,
    ):
        return None

    x = bbox.get("x")
    y = bbox.get("y")
    width = bbox.get("width")
    height = bbox.get("height")

    if None in (
        x,
        y,
        width,
        height,
    ):
        return None

    return {
        "x": x,
        "y": y,
        "w": width,
        "h": height,
        "x2": bbox.get(
            "x2",
            x + width,
        ),
        "y2": bbox.get(
            "y2",
            y + height,
        ),
    }


def extract_logical_position(cell):
    """
    Extract logical table position.
    """

    position = cell.get(
        "logical_position"
    )

    if isinstance(
        position,
        dict,
    ):
        return position

    return None


def extract_ocr_region_ids(cell):
    """
    Extract OCR region IDs from the nested ocr_regions
    structure used by pic1 logical_table.json.
    """

    regions = cell.get(
        "ocr_regions",
        [],
    )

    region_ids = []

    for region in regions:
        if not isinstance(
            region,
            dict,
        ):
            continue

        region_id = region.get(
            "region_id"
        )

        if region_id is not None:
            region_ids.append(
                region_id
            )

    return region_ids


def extract_cell_text(cell):
    """
    Safely obtain cell text.
    """

    value = cell.get(
        "text",
        "",
    )

    if value is None:
        return ""

    return str(value)


def make_pic1():
    normalized = load_json(
        PIC1_NORMALIZED
    )

    logical = load_json(
        PIC1_LOGICAL
    )

    source_image = (
        normalized
        .get("source", {})
        .get(
            "image",
            (
                "results/gui_v4/single/"
                "pic1/perspective.png"
            ),
        )
    )

    unified = make_common_source(
        document_id="pic1",
        document_type="table",
        source_image=source_image,
    )

    table_geometry = logical.get(
        "table_geometry",
        {},
    )

    logical_cells = logical.get(
        "cells",
        [],
    )

    unified["metadata"] = {
        "semantic_type": "timetable",
        "table": {
            "rows": table_geometry.get(
                "row_count"
            ),
            "columns": table_geometry.get(
                "column_count"
            ),
            "cells": len(
                logical_cells
            ),
        },
    }

    records = normalized.get(
        "records",
        [],
    )

    for record in records:
        category_data = record.get(
            "category",
            {},
        )

        event_data = record.get(
            "event",
            {},
        )

        detail_data = record.get(
            "detail",
            {},
        )

        explanation_data = record.get(
            "explanation",
            {},
        )

        unified_record = {
            "record_id": record.get(
                "record_id"
            ),
            "source_row": record.get(
                "source_row"
            ),
            "category": (
                category_data.get(
                    "text"
                )
            ),
            "event": (
                event_data.get(
                    "text"
                )
            ),
            "time": record.get(
                "normalized_time"
            ),
            "detail": (
                detail_data.get(
                    "text"
                )
            ),
            "explanation": (
                explanation_data.get(
                    "text"
                )
            ),
            "review_flags": record.get(
                "review_flags",
                [],
            ),
            "source_cells": {},
        }

        for field_name in (
            "category",
            "event",
            "time",
            "detail",
            "explanation",
        ):
            field_data = record.get(
                field_name,
                {},
            )

            if not isinstance(
                field_data,
                dict,
            ):
                continue

            source_cell = field_data.get(
                "source_cell"
            )

            if source_cell is not None:
                unified_record[
                    "source_cells"
                ][
                    field_name
                ] = source_cell

        unified[
            "records"
        ].append(
            unified_record
        )

        for flag in record.get(
            "review_flags",
            [],
        ):
            unified[
                "review_flags"
            ].append(
                {
                    "scope": "record",
                    "record_id": (
                        record.get(
                            "record_id"
                        )
                    ),
                    "flag": flag,
                }
            )

    missing_bbox_count = 0
    missing_position_count = 0

    for cell in logical_cells:
        bbox = extract_cell_bbox(
            cell
        )

        logical_position = (
            extract_logical_position(
                cell
            )
        )

        if bbox is None:
            missing_bbox_count += 1

        if logical_position is None:
            missing_position_count += 1

        unified[
            "evidence"
        ].append(
            {
                "evidence_type": (
                    "table_cell"
                ),
                "cell_id": cell.get(
                    "cell_id"
                ),
                "text": extract_cell_text(
                    cell
                ),
                "bbox": bbox,
                "logical_position": (
                    logical_position
                ),
                "ocr_region_ids": (
                    extract_ocr_region_ids(
                        cell
                    )
                ),
            }
        )

    unified[
        "metadata"
    ][
        "conversion"
    ] = {
        "cells_missing_bbox": (
            missing_bbox_count
        ),
        "cells_missing_logical_position": (
            missing_position_count
        ),
    }

    return unified


def make_pic13():
    form = load_json(
        PIC13_FORM
    )

    source_image = (
        form
        .get("source", {})
        .get(
            "image",
            (
                "results/gui_v4/single/"
                "pic13/scan/scan.png"
            ),
        )
    )

    unified = make_common_source(
        document_id="pic13",
        document_type="form",
        source_image=source_image,
    )

    unified[
        "metadata"
    ] = {
        "semantic_type": (
            "key_value_form"
        ),
        "statistics": (
            form.get(
                "statistics",
                {}
            )
        ),
    }

    for field in form.get(
        "fields",
        [],
    ):
        unified[
            "fields"
        ].append(
            {
                "field_id": field.get(
                    "field_id"
                ),
                "label": field.get(
                    "label"
                ),
                "value": field.get(
                    "value"
                ),
                "label_region_id": (
                    field.get(
                        "label_region_id"
                    )
                ),
                "value_region_id": (
                    field.get(
                        "value_region_id"
                    )
                ),
                "label_confidence": (
                    field.get(
                        "label_confidence"
                    )
                ),
                "value_confidence": (
                    field.get(
                        "value_confidence"
                    )
                ),
            }
        )

    for control in form.get(
        "checkboxes",
        [],
    ):
        bbox = None

        if all(
            key in control
            for key in (
                "x",
                "y",
                "w",
                "h",
            )
        ):
            x = control["x"]
            y = control["y"]
            w = control["w"]
            h = control["h"]

            bbox = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "x2": control.get(
                    "x2",
                    x + w,
                ),
                "y2": control.get(
                    "y2",
                    y + h,
                ),
            }

        unified[
            "controls"
        ].append(
            {
                "control_id": (
                    control.get(
                        "control_id"
                    )
                ),
                "control_type": (
                    "checkbox"
                ),
                "label": (
                    control.get(
                        "label"
                    )
                ),
                "checked": bool(
                    control.get(
                        "checked",
                        False,
                    )
                ),
                "state": (
                    control.get(
                        "state"
                    )
                ),
                "label_region_id": (
                    control.get(
                        "label_region_id"
                    )
                ),
                "bbox": bbox,
                "interior_dark_ratio": (
                    float(
                        control.get(
                            "interior_dark_ratio",
                            0.0,
                        )
                    )
                ),
            }
        )

    for item in form.get(
        "followup_text",
        [],
    ):
        unified[
            "free_text"
        ].append(
            {
                "source_control_id": (
                    item.get(
                        "control_id"
                    )
                ),
                "source_control_label": (
                    item.get(
                        "control_label"
                    )
                ),
                "text_region_id": (
                    item.get(
                        "text_region_id"
                    )
                ),
                "text": (
                    item.get(
                        "text"
                    )
                ),
            }
        )

    for role in form.get(
        "region_roles",
        [],
    ):
        unified[
            "evidence"
        ].append(
            {
                "evidence_type": (
                    "ocr_region"
                ),
                "region_id": (
                    role.get(
                        "region_id"
                    )
                ),
                "text": (
                    role.get(
                        "text"
                    )
                ),
                "semantic_role": (
                    role.get(
                        "role"
                    )
                ),
            }
        )

    return unified


def validate_common_schema(
    document,
):
    required = {
        "document_id",
        "schema_version",
        "document_type",
        "source",
        "sections",
        "fields",
        "records",
        "controls",
        "free_text",
        "evidence",
        "review_flags",
        "metadata",
    }

    missing = (
        required
        - set(
            document.keys()
        )
    )

    if missing:
        raise ValueError(
            "Unified schema missing keys: "
            + ", ".join(
                sorted(missing)
            )
        )


def validate_pic1(
    document,
):
    errors = []

    if document.get(
        "document_type"
    ) != "table":
        errors.append(
            "pic1 document type is not table."
        )

    if len(
        document.get(
            "records",
            [],
        )
    ) != 18:
        errors.append(
            "pic1 should contain 18 records."
        )

    if len(
        document.get(
            "evidence",
            [],
        )
    ) != 83:
        errors.append(
            "pic1 should contain 83 evidence cells."
        )

    conversion = (
        document
        .get(
            "metadata",
            {}
        )
        .get(
            "conversion",
            {}
        )
    )

    if conversion.get(
        "cells_missing_bbox"
    ) != 0:
        errors.append(
            "pic1 should contain bbox geometry "
            "for all 83 cells."
        )

    if conversion.get(
        "cells_missing_logical_position"
    ) != 0:
        errors.append(
            "pic1 should contain logical positions "
            "for all 83 cells."
        )

    return errors


def validate_pic13(
    document,
):
    errors = []

    if document.get(
        "document_type"
    ) != "form":
        errors.append(
            "pic13 document type is not form."
        )

    if len(
        document.get(
            "fields",
            [],
        )
    ) != 7:
        errors.append(
            "pic13 should contain 7 fields."
        )

    if len(
        document.get(
            "controls",
            [],
        )
    ) != 7:
        errors.append(
            "pic13 should contain 7 controls."
        )

    if len(
        document.get(
            "free_text",
            [],
        )
    ) != 1:
        errors.append(
            "pic13 should contain 1 free-text item."
        )

    if len(
        document.get(
            "evidence",
            [],
        )
    ) != 37:
        errors.append(
            "pic13 should contain 37 OCR evidence items."
        )

    return errors


def make_summary(
    pic1,
    pic13,
    pic1_errors,
    pic13_errors,
):
    pic1_conversion = (
        pic1
        .get(
            "metadata",
            {}
        )
        .get(
            "conversion",
            {}
        )
    )

    lines = [
        "=" * 78,
        "EXPERIMENT 40: UNIFIED DOCUMENT REPRESENTATION",
        "=" * 78,
        "",
        "PIC1",
        "-" * 78,
        (
            f"Document type: "
            f"{pic1['document_type']}"
        ),
        (
            f"Records: "
            f"{len(pic1['records'])}"
        ),
        (
            f"Fields: "
            f"{len(pic1['fields'])}"
        ),
        (
            f"Controls: "
            f"{len(pic1['controls'])}"
        ),
        (
            f"Evidence items: "
            f"{len(pic1['evidence'])}"
        ),
        (
            f"Review flags: "
            f"{len(pic1['review_flags'])}"
        ),
        (
            "Cells missing bbox: "
            f"{pic1_conversion.get('cells_missing_bbox')}"
        ),
        (
            "Cells missing logical position: "
            f"{pic1_conversion.get('cells_missing_logical_position')}"
        ),
        (
            "Validation: "
            + (
                "PASS"
                if not pic1_errors
                else "FAIL"
            )
        ),
        "",
        "PIC13",
        "-" * 78,
        (
            f"Document type: "
            f"{pic13['document_type']}"
        ),
        (
            f"Records: "
            f"{len(pic13['records'])}"
        ),
        (
            f"Fields: "
            f"{len(pic13['fields'])}"
        ),
        (
            f"Controls: "
            f"{len(pic13['controls'])}"
        ),
        (
            f"Free-text items: "
            f"{len(pic13['free_text'])}"
        ),
        (
            f"Evidence items: "
            f"{len(pic13['evidence'])}"
        ),
        (
            "Validation: "
            + (
                "PASS"
                if not pic13_errors
                else "FAIL"
            )
        ),
        "",
        "COMMON SCHEMA",
        "-" * 78,
        "Both documents expose:",
        "  document_id",
        "  schema_version",
        "  document_type",
        "  source",
        "  sections",
        "  fields",
        "  records",
        "  controls",
        "  free_text",
        "  evidence",
        "  review_flags",
        "  metadata",
        "",
        "=" * 78,
        (
            "UNIFIED REPRESENTATION COMPLETE"
            if (
                not pic1_errors
                and not pic13_errors
            )
            else (
                "UNIFIED REPRESENTATION "
                "COMPLETED WITH VALIDATION ERRORS"
            )
        ),
        "=" * 78,
    ]

    if pic1_errors:
        lines.extend(
            [
                "",
                "PIC1 VALIDATION ERRORS",
                "-" * 78,
            ]
        )

        for error in pic1_errors:
            lines.append(
                f"- {error}"
            )

    if pic13_errors:
        lines.extend(
            [
                "",
                "PIC13 VALIDATION ERRORS",
                "-" * 78,
            ]
        )

        for error in pic13_errors:
            lines.append(
                f"- {error}"
            )

    return "\n".join(
        lines
    ) + "\n"


def main():
    print("=" * 72)
    print(
        "EXPERIMENT 40: "
        "UNIFIED DOCUMENT REPRESENTATION"
    )
    print("=" * 72)

    pic1 = make_pic1()
    pic13 = make_pic13()

    validate_common_schema(
        pic1
    )

    validate_common_schema(
        pic13
    )

    pic1_errors = validate_pic1(
        pic1
    )

    pic13_errors = validate_pic13(
        pic13
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PIC1_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            pic1,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with PIC13_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            pic13,
            file,
            ensure_ascii=False,
            indent=2,
        )

    summary = make_summary(
        pic1,
        pic13,
        pic1_errors,
        pic13_errors,
    )

    SUMMARY_OUTPUT.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        f"pic1 type: "
        f"{pic1['document_type']}"
    )

    print(
        f"pic1 records: "
        f"{len(pic1['records'])}"
    )

    print(
        f"pic1 evidence: "
        f"{len(pic1['evidence'])}"
    )

    print(
        "pic1 missing bbox:",
        pic1[
            "metadata"
        ][
            "conversion"
        ][
            "cells_missing_bbox"
        ],
    )

    print(
        "pic1 missing logical position:",
        pic1[
            "metadata"
        ][
            "conversion"
        ][
            "cells_missing_logical_position"
        ],
    )

    print()

    print(
        f"pic13 type: "
        f"{pic13['document_type']}"
    )

    print(
        f"pic13 fields: "
        f"{len(pic13['fields'])}"
    )

    print(
        f"pic13 controls: "
        f"{len(pic13['controls'])}"
    )

    print(
        f"pic13 free text: "
        f"{len(pic13['free_text'])}"
    )

    print()

    print(
        "pic1 validation:",
        (
            "PASS"
            if not pic1_errors
            else "FAIL"
        ),
    )

    print(
        "pic13 validation:",
        (
            "PASS"
            if not pic13_errors
            else "FAIL"
        ),
    )

    print()

    print(
        f"pic1 unified: "
        f"{PIC1_OUTPUT}"
    )

    print(
        f"pic13 unified: "
        f"{PIC13_OUTPUT}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_OUTPUT}"
    )

    print("=" * 72)

    if (
        not pic1_errors
        and not pic13_errors
    ):
        print(
            "EXPERIMENT 40 COMPLETE"
        )
    else:
        print(
            "EXPERIMENT 40 COMPLETE "
            "WITH VALIDATION ERRORS"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()